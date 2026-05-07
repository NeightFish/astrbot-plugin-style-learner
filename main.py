"""
AstrBot Style Learner - 学习他人说话风格并模仿回复的插件

命令列表:
  /sklearn <技能名> <文件路径>    从 .jsonl 文件学习风格
  /sklist                         列出所有已学技能
  /imitate <技能名> <消息>        以指定风格回复
  /imitate <技能名> --on          开启持续扮演
  /imitate --off                  关闭持续扮演
  /imitate --status               查看扮演状态
  /skdelete <技能名>              删除一个技能
  /skinfo <技能名>                查看技能详情
  /skupdate <技能名> [新路径]     更新技能
  /skrename <旧名> <新名>         重命名技能
  /skmerge <技能A> <技能B> <新名> 合并两个技能
  /skstats                        查看技能使用排行
  /sklearn_active <技能名> [数量] 从 buffer 手动创建技能
  /skbuffer                       查看 buffer 状态
  /skbuffer_clear                 清空 buffer

/imitate 支持风格混合: /imitate 技能1:0.3+技能2:0.7 消息内容
支持持续扮演、默认风格自动应用、jieba+TF-IDF 示例检索

--- 本插件由 AI (DeepSeek-V4-Pro) 生成 ---
"""

import asyncio
import json
import math
import random
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.message import TextPart
from astrbot.core.star.config import put_config

try:
    import jieba

    HAS_JIEBA = True
except ImportError:
    jieba = None
    HAS_JIEBA = False

try:
    from openai import AsyncOpenAI

    HAS_OPENAI = True
except ImportError:
    AsyncOpenAI = None
    HAS_OPENAI = False

PLUGIN_NAMESPACE = "astrbot_plugin_style_learner"
FEEDBACK_LOG = "feedback_log.jsonl"


@register(PLUGIN_NAMESPACE, "AI (DeepSeek-V4-Pro)",
          "从对话记录学习说话风格，支持风格混合与模仿回复", "1.0.0")
class StyleLearner(Star):

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config
        self.plugin_dir = Path(__file__).parent
        self.skills_file = self.plugin_dir / "skills.json"
        self.feedback_file = self.plugin_dir / FEEDBACK_LOG
        self.buffer_file = self.plugin_dir / "active_buffer.jsonl"
        self._skills_lock = asyncio.Lock()
        self._skills: dict[str, dict] = {}
        # 主动学习: ring buffer (max 200 turns)
        self._buffer: deque[dict] = deque(maxlen=200)
        # pending: session_id → (user_msg, sender_id) (等待 bot 回复后配对)
        self._pending: dict[str, tuple[str, str, datetime]] = {}
        self._last_auto_analyze_count: int = 0
        # 持续扮演: session_id → skill_name
        self._roleplay: dict[str, str] = {}
        self._register_config()

    # ==================== Config ====================

    def _register_config(self):
        put_config(PLUGIN_NAMESPACE, "DeepSeek API Key", "deepseek_api_key", "",
                   "DeepSeek API 密钥（必填）")
        put_config(PLUGIN_NAMESPACE, "DeepSeek Model", "deepseek_model",
                   "deepseek-chat", "使用的 DeepSeek 模型名称")
        put_config(PLUGIN_NAMESPACE, "DeepSeek Base URL", "deepseek_base_url",
                   "https://api.deepseek.com", "DeepSeek API 地址")
        put_config(PLUGIN_NAMESPACE, "Max Sample Chars", "max_sample_chars",
                   3000, "学习时单次分析最大字符数")

        # 主动学习
        put_config(PLUGIN_NAMESPACE, "主动学习模式", "active_learning_mode",
                   "off", "off=关闭 / all=学习所有人 / specific=学习指定用户")
        put_config(PLUGIN_NAMESPACE, "指定学习QQ号", "specific_qq",
                   "", "当主动学习模式为 specific 时，指定要学习的 QQ 号")
        put_config(PLUGIN_NAMESPACE, "自动分析阈值", "batch_size",
                   30, "缓冲区满多少条后自动触发风格分析")

        # 消息过滤
        put_config(PLUGIN_NAMESPACE, "启用消息过滤", "enable_message_filter",
                   True, "是否过滤涉政/色色等敏感消息，防止污染数据集")
        put_config(PLUGIN_NAMESPACE, "过滤关键词", "filter_keywords",
                   "习近平,习总书记,国务院,共产党,台独,港独,六四,法轮功,"
                   "裸体,做爱,性交,色情,黄色,成人网站,赌博,赌场",
                   "敏感词列表，逗号分隔。命中任意关键词的消息将被跳过")
        put_config(PLUGIN_NAMESPACE, "语义审核 Prompt", "filter_prompt",
                   "", "留空则不启用。填写后每次分析前调 DeepSeek 做语义审核，消耗 token")
        put_config(PLUGIN_NAMESPACE, "显示隐私提示", "show_filter_notice",
                   False, "开启后在 WebUI 显示隐私提示文字")
        put_config(PLUGIN_NAMESPACE, "默认风格", "default_skill",
                   "", "留空则不启用。填写技能名后，所有消息自动以该风格回复（持续扮演优先）")

    def _cfg(self, key, default=None):
        if self.config and hasattr(self.config, key):
            v = getattr(self.config, key)
            if v is not None:
                return v
        return default

    # ==================== Lifecycle ====================

    async def initialize(self):
        await self._load_skills()
        await self._load_buffer()
        logger.info(
            f"[StyleLearner] 已加载 {len(self._skills)} 个技能, "
            f"buffer 中有 {len(self._buffer)} 条对话")

    async def terminate(self):
        pass

    # ==================== Active Learning ====================

    def _active_skill_name(self) -> str | None:
        """根据当前模式返回主动学习的技能名，返回 None 表示不启用"""
        mode = self._cfg("active_learning_mode", "off")
        if mode == "off":
            return None
        if mode == "specific":
            qq = self._cfg("specific_qq", "").strip()
            return f"@{qq}" if qq else None
        if mode == "all":
            return "@全局"
        return None

    def _should_learn_from(self, sender_id: str) -> bool:
        """判断是否应该学习该发送者的消息"""
        mode = self._cfg("active_learning_mode", "off")
        if mode == "off":
            return False
        if mode == "all":
            return True
        if mode == "specific":
            target = self._cfg("specific_qq", "").strip()
            return target and sender_id == target
        return False

    def _clean_stale_pending(self):
        """清理超过 5 分钟的 stale pending entry，仅在 pending 较多时触发"""
        if len(self._pending) < 20:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        stale = [sid for sid, (_, _, ts) in self._pending.items() if ts < cutoff]
        for sid in stale:
            self._pending.pop(sid, None)
        if stale:
            logger.debug(f"[StyleLearner] 清理 {len(stale)} 条 stale pending")

    def _filter_message(self, msg: str) -> bool:
        """消息过滤。返回 True 表示应该跳过（被过滤）。
        只有启用过滤且消息命中关键词时才返回 True。
        """
        if not self._cfg("enable_message_filter", True):
            return False
        keywords = self._cfg("filter_keywords", "")
        if not keywords or not isinstance(keywords, str):
            return False
        msg_lower = msg.lower()
        for kw in keywords.split(","):
            kw = kw.strip()
            if kw and kw.lower() in msg_lower:
                logger.debug(f"[StyleLearner] 消息被过滤，命中关键词: {kw}")
                return True
        return False

    @filter.on_llm_request()
    async def _on_llm_request(self, event: AstrMessageEvent, req):
        """被动监听：主动学习记录 + 持续扮演/默认风格注入"""
        self._clean_stale_pending()

        # === 主动学习：记录用户消息 ===
        active_skill = self._active_skill_name()
        if active_skill:
            sender_id = event.get_sender_id()
            if self._should_learn_from(sender_id):
                user_msg = event.get_message_str().strip()
                if user_msg and not self._filter_message(user_msg):
                    session_id = event.get_session_id()
                    self._pending[session_id] = (
                        user_msg, sender_id, datetime.now(timezone.utc))

        # === 持续扮演 / 默认风格注入 ===
        self._inject_roleplay_prompt(event, req)

    def _inject_roleplay_prompt(self, event: AstrMessageEvent, req):
        """根据持续扮演或默认风格，向 LLM 请求注入风格 prompt"""
        session_id = event.get_session_id()
        skill_name = None

        # 优先：session 级持续扮演
        if session_id in self._roleplay:
            skill_name = self._roleplay[session_id]
        else:
            # 其次：全局默认风格
            default = self._cfg("default_skill", "").strip()
            if default and default in self._skills:
                skill_name = default

        if not skill_name or skill_name not in self._skills:
            return

        user_msg = event.get_message_str().strip()
        if not user_msg or user_msg.startswith("/"):
            return

        skill = self._skills[skill_name]
        prompt = self._build_imitation_prompt(
            [(skill_name, 1.0)], user_msg)
        req.extra_user_content_parts.append(TextPart(text=prompt))
        # 统计使用（异步递增，不阻塞请求）
        asyncio.create_task(self._increment_usage([(skill_name, 1.0)]))
        logger.debug(
            f"[StyleLearner] 注入风格 [{skill_name}] → session={session_id}")

    @filter.after_message_sent()
    async def _on_after_message_sent(self, event: AstrMessageEvent):
        """被动监听：bot 回复后，与 pending 中的用户消息配对写入 buffer"""
        skill_name = self._active_skill_name()
        if not skill_name:
            return

        session_id = event.get_session_id()
        pending = self._pending.pop(session_id, None)
        if not pending:
            return
        user_msg, sender_id, _ = pending

        bot_msg = event.get_message_str()
        if not bot_msg:
            return

        item = {
            "sender": sender_id,
            "user_msg": user_msg,
            "bot_msg": bot_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._buffer.append(item)
        await self._save_buffer_item(item)

        # 检查是否达到自动分析阈值（节流：至少增长 batch_size/2 才再次触发）
        batch_size = int(self._cfg("batch_size", 30))
        if len(self._buffer) >= batch_size and \
                len(self._buffer) - self._last_auto_analyze_count >= max(batch_size // 2, 5):
            self._last_auto_analyze_count = len(self._buffer)
            await self._auto_analyze(skill_name)

    async def _auto_analyze(self, skill_name: str):
        """从 buffer 采样并自动创建/更新技能"""
        if len(self._buffer) < 10:
            return

        try:
            # 从 buffer 提取对话采样
            pool = list(self._buffer)
            random.shuffle(pool)
            sampled = []
            total_chars = 0
            max_chars = int(self._cfg("max_sample_chars", 3000))
            for item in pool[:50]:
                text = f"{item['user_msg']}|||{item['bot_msg']}"
                if total_chars + len(text) > max_chars and sampled:
                    break
                sampled.append([item["user_msg"], item["bot_msg"]])
                total_chars += len(text)

            sample_text = "\n".join(
                f"{s[0]}|||{s[1]}" for s in sampled)

            # 可选的语义审核
            filter_prompt = self._cfg("filter_prompt", "").strip()
            if filter_prompt:
                ok = await self._semantic_filter(sample_text, filter_prompt)
                if not ok:
                    logger.info(
                        f"[StyleLearner] 语义审核未通过，跳过自动分析 [{skill_name}]")
                    return

            result = await self._analyze_style(sample_text)

            prev_count = self._skills.get(skill_name, {}).get("usage_count", 0)
            skill_data = {
                "label": result.get("label", skill_name),
                "summary": result.get("summary", ""),
                "examples": sampled[:10],
                "source_file": "@active_learning",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "auto_update": True,
                "buffer_count": len(self._buffer),
                "usage_count": prev_count,
            }

            async with self._skills_lock:
                self._skills[skill_name] = skill_data
            await self._save_skills()

            logger.info(
                f"[StyleLearner] 自动分析完成: [{skill_name}] "
                f"标签={result.get('label')}, buffer={len(self._buffer)}条")
        except Exception as e:
            logger.error(f"[StyleLearner] 自动分析失败 [{skill_name}]: {e}", exc_info=True)

    async def _semantic_filter(self, sample_text: str, custom_prompt: str) -> bool:
        """用 DeepSeek 做语义审核，返回 True 表示通过"""
        system_prompt = (
            "你是一个内容安全审核助手。请判断以下对话样本是否包含"
            "涉政、色情、暴力或其他严重违规内容。只输出 PASS 或 FAIL。"
        )
        user_prompt = f"{custom_prompt}\n\n对话样本:\n{sample_text[:3000]}"
        try:
            resp = await self._call_llm(
                system_prompt, user_prompt, max_tokens=16, temperature=0.1)
            return "PASS" in resp.upper()
        except Exception:
            # API 调用失败时不阻塞学习流程
            return True

    # ==================== Persistence ====================

    async def _load_skills(self):
        async with self._skills_lock:
            try:
                if self.skills_file.exists():
                    raw = await asyncio.to_thread(
                        self.skills_file.read_text, encoding="utf-8")
                    self._skills = json.loads(raw)
                else:
                    self._skills = {}
            except Exception as e:
                logger.error(f"[StyleLearner] 加载 skills.json 失败: {e}")
                self._skills = {}

    async def _save_skills(self):
        async with self._skills_lock:
            try:
                data = json.dumps(self._skills, ensure_ascii=False, indent=2)
                await asyncio.to_thread(
                    self.skills_file.write_text, data, encoding="utf-8")
            except Exception as e:
                logger.error(f"[StyleLearner] 保存 skills.json 失败: {e}")

    async def _load_buffer(self):
        """从 active_buffer.jsonl 恢复 ring buffer"""
        try:
            if self.buffer_file.exists():
                raw = await asyncio.to_thread(
                    self.buffer_file.read_text, encoding="utf-8")
                for line in raw.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        self._buffer.append(item)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"[StyleLearner] 加载 buffer 失败: {e}")

    async def _save_buffer_item(self, item: dict):
        """追加一条对话到 active_buffer.jsonl"""
        def _write():
            with self.buffer_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        try:
            await asyncio.to_thread(_write)
        except Exception as e:
            logger.debug(f"[StyleLearner] buffer 持久化失败: {e}")

    # ==================== LLM ====================

    def _get_client(self):
        if not HAS_OPENAI:
            raise RuntimeError("请安装 openai 库: pip install openai")
        api_key = self._cfg("deepseek_api_key", "")
        if not api_key:
            raise RuntimeError("请先在插件配置中设置 deepseek_api_key")
        base_url = self._cfg("deepseek_base_url", "https://api.deepseek.com")
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _call_llm(self, system_prompt: str, user_prompt: str,
                        max_tokens: int = 1024, temperature: float = 0.7) -> str:
        model = self._cfg("deepseek_model", "deepseek-chat")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = self._get_client()
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                delay = 2 ** attempt
                logger.warning(
                    f"[StyleLearner] API 重试 {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"API 调用失败: {e}")
        return ""

    # ==================== /sklearn ====================

    @filter.command("sklearn")
    async def sklearn(self, event: AstrMessageEvent):
        """从 .jsonl 文件学习风格。用法: /sklearn <技能名> <文件路径>"""
        msg = event.get_message_str().strip()
        args = msg[len("/sklearn"):].strip().split(" ", 1)
        if len(args) < 2 or not args[0] or not args[1]:
            yield event.plain_result(
                "用法: /sklearn <技能名> <文件路径>\n"
                "例如: /sklearn 温柔学姐 D:/data/style.jsonl")
            return

        skill_name = args[0]
        file_path = args[1].strip()

        if skill_name in self._skills:
            yield event.plain_result(
                f"技能 [{skill_name}] 已存在。\n"
                f"如需重新学习请先用 /skdelete {skill_name} 删除。")
            return

        path = Path(file_path)
        if not path.exists():
            yield event.plain_result(f"文件不存在: {file_path}")
            return

        yield event.plain_result(f"正在分析 [{skill_name}] 的对话风格，请稍候...")

        try:
            conversations = await self._read_jsonl(path)
            if not conversations:
                yield event.plain_result("文件中没有有效的对话数据（每行需为长度>=2的JSON数组）")
                return

            sampled = self._sample_conversations(conversations)
            sample_text = self._format_samples(sampled)
            result = await self._analyze_style(sample_text)

            skill = {
                "label": result.get("label", skill_name),
                "summary": result.get("summary", ""),
                "examples": sampled[:10],
                "source_file": str(path.absolute()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "usage_count": 0,
            }

            self._skills[skill_name] = skill
            await self._save_skills()

            yield event.plain_result(
                f"技能 [{skill_name}] 学习完成！\n"
                f"风格标签: {skill['label']}\n"
                f"风格描述: {skill['summary']}")
        except Exception as e:
            logger.error(f"[StyleLearner] /sklearn 失败: {e}", exc_info=True)
            yield event.plain_result(f"学习失败: {e}")

    # ==================== /sklist ====================

    @filter.command("sklist")
    async def sklist(self, event: AstrMessageEvent):
        """列出所有已学习的技能。"""
        if not self._skills:
            yield event.plain_result(
                "暂无已学习的技能。使用 /sklearn <技能名> <文件路径> 开始学习。")
            return

        lines = ["已学习的技能:\n"]
        for name, skill in self._skills.items():
            label = skill.get("label", "未标注")
            created = skill.get("created_at", "未知")[:10]
            count = skill.get("usage_count", 0)
            lines.append(f"  [{name}] {label} (创建于 {created}, 使用 {count} 次)")

        yield event.plain_result("\n".join(lines))

    # ==================== /skstats ====================

    @filter.command("skstats")
    async def skstats(self, event: AstrMessageEvent):
        """显示技能使用排行。"""
        if not self._skills:
            yield event.plain_result("暂无已学习的技能。")
            return

        ranked = sorted(
            self._skills.items(),
            key=lambda x: x[1].get("usage_count", 0),
            reverse=True,
        )
        lines = ["技能使用排行:\n"]
        for i, (name, skill) in enumerate(ranked, 1):
            label = skill.get("label", "未标注")
            count = skill.get("usage_count", 0)
            bar = "█" * min(count, 20) if count > 0 else ""
            lines.append(f"  {i}. [{name}] {label} — {count} 次 {bar}")

        yield event.plain_result("\n".join(lines))

    # ==================== /imitate ====================

    @filter.command("imitate")
    async def imitate(self, event: AstrMessageEvent):
        """模仿指定风格回复。
        用法:
          /imitate <技能名> <消息>    一次性模仿
          /imitate <技能名> --on      开启持续扮演
          /imitate --off              关闭持续扮演
          /imitate --status           查看扮演状态
        支持风格混合: /imitate 技能1:0.3+技能2:0.7 消息
        """
        msg = event.get_message_str().strip()
        rest = msg[len("/imitate"):].strip()
        session_id = event.get_session_id()

        # --off: 关闭持续扮演
        if rest == "--off":
            removed = self._roleplay.pop(session_id, None)
            if removed:
                yield event.plain_result(f"已关闭持续扮演 [{removed}]。")
            else:
                yield event.plain_result("当前会话未开启持续扮演。")
            return

        # --status: 查看扮演状态
        if rest == "--status":
            rp = self._roleplay.get(session_id)
            default = self._cfg("default_skill", "").strip()
            lines = []
            if rp:
                skill = self._skills.get(rp, {})
                lines.append(f"持续扮演: [{rp}] {skill.get('label', '')}")
            else:
                lines.append("持续扮演: 未开启")
            if default:
                lines.append(f"默认风格: [{default}]")
            else:
                lines.append("默认风格: 未设置")
            yield event.plain_result("\n".join(lines))
            return

        # --on: 开启持续扮演
        if rest.endswith(" --on"):
            skill_name = rest[:-4].strip()
            if not skill_name or skill_name not in self._skills:
                yield event.plain_result(
                    f"技能 [{skill_name}] 不存在。使用 /sklist 查看可用技能。")
                return
            self._roleplay[session_id] = skill_name
            label = self._skills[skill_name].get("label", skill_name)
            yield event.plain_result(
                f"已开启持续扮演 [{skill_name}] {label}。\n"
                f"此会话的后续消息将自动以该风格回复。\n"
                f"使用 /imitate --off 关闭。")
            return

        # 一次性模仿（原有逻辑）
        parts = rest.split(" ", 1)
        if len(parts) < 2:
            yield event.plain_result(
                "用法:\n"
                "  /imitate <技能名> <消息>        一次性模仿\n"
                "  /imitate <技能名> --on           开启持续扮演\n"
                "  /imitate --off                   关闭持续扮演\n"
                "  /imitate --status                查看状态\n"
                "支持混合: /imitate 技能1:0.3+技能2:0.7 消息")
            return

        spec = parts[0]
        message = parts[1]

        skill_refs = self._parse_skill_spec(spec)
        if not skill_refs:
            yield event.plain_result("无效的技能指定格式。")
            return

        for name, _ in skill_refs:
            if name not in self._skills:
                yield event.plain_result(
                    f"技能 [{name}] 不存在。使用 /sklist 查看可用技能。")
                return

        try:
            system_prompt = self._build_imitation_prompt(skill_refs, message)
            response = await self._call_llm(
                system_prompt=system_prompt,
                user_prompt=message,
                max_tokens=200,
                temperature=0.9,
            )

            await self._log_feedback(skill_refs, message, response)
            await self._increment_usage(skill_refs)

            yield event.plain_result(response)

        except Exception as e:
            logger.error(f"[StyleLearner] /imitate 失败: {e}", exc_info=True)
            yield event.plain_result(f"模仿失败: {e}")

    def _parse_skill_spec(self, spec: str) -> list[tuple[str, float]]:
        """解析技能与权重: 'name' 或 'a:0.3+b:0.7'"""
        result = []
        for part in spec.split("+"):
            part = part.strip()
            if ":" in part:
                name, w_str = part.rsplit(":", 1)
                try:
                    w = float(w_str)
                except ValueError:
                    w = 1.0
            else:
                name = part
                w = 1.0
            name = name.strip()
            if name:
                result.append((name, w))
        return result

    def _build_imitation_prompt(
            self, skill_refs: list[tuple[str, float]], _message: str) -> str:
        """构造模仿 System Prompt，支持单技能和混合风格。"""
        parts = [
            "你现在要扮演一个角色，请严格遵循以下风格描述进行回复。"
            "不要跳出角色，不要解释自己是AI，直接给出回复内容。"
        ]

        if len(skill_refs) == 1:
            name, _ = skill_refs[0]
            skill = self._skills[name]
            parts.append(f"\n【角色风格】\n{skill['summary']}")

            examples = skill.get("examples", [])
            if examples:
                picked = self._retrieve_relevant_examples(
                    _message, examples, top_k=3)
                parts.append("\n【对话示例】")
                for ex in picked:
                    if len(ex) >= 2:
                        parts.append(f"用户: {ex[0]}")
                        parts.append(f"角色: {ex[-1]}")
        else:
            parts.append("\n【混合风格】")
            total_w = sum(w for _, w in skill_refs) or 1.0
            for name, w in skill_refs:
                skill = self._skills[name]
                pct = int(w / total_w * 100)
                parts.append(
                    f"\n{pct}% {skill.get('label', name)} 风格: {skill['summary']}")

        return "\n".join(parts)

    async def _log_feedback(self, skill_refs: list[tuple[str, float]],
                            message: str, response: str):
        """记录模仿输出到反馈日志，供后续手动优化。"""
        def _write():
            with self.feedback_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "skills": [(n, w) for n, w in skill_refs],
                    "user_message": message,
                    "bot_response": response,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
        try:
            await asyncio.to_thread(_write)
        except Exception as e:
            logger.debug(f"[StyleLearner] 反馈日志写入失败: {e}")

    async def _increment_usage(self, skill_refs: list[tuple[str, float]]):
        """递增技能使用次数"""
        async with self._skills_lock:
            for name, _ in skill_refs:
                if name in self._skills:
                    self._skills[name]["usage_count"] = \
                        self._skills[name].get("usage_count", 0) + 1
        # 不立即保存，避免频繁 I/O；下次 save 时一起写入

    # ==================== /skdelete ====================

    @filter.command("skdelete")
    async def skdelete(self, event: AstrMessageEvent):
        """删除一个技能。用法: /skdelete <技能名>"""
        msg = event.get_message_str().strip()
        name = msg[len("/skdelete"):].strip()
        if not name:
            yield event.plain_result("用法: /skdelete <技能名>")
            return

        if name not in self._skills:
            yield event.plain_result(f"技能 [{name}] 不存在。")
            return

        del self._skills[name]
        await self._save_skills()
        yield event.plain_result(f"技能 [{name}] 已删除。")

    # ==================== /skinfo ====================

    @filter.command("skinfo")
    async def skinfo(self, event: AstrMessageEvent):
        """查看技能详情。用法: /skinfo <技能名>"""
        msg = event.get_message_str().strip()
        name = msg[len("/skinfo"):].strip()
        if not name:
            yield event.plain_result("用法: /skinfo <技能名>")
            return

        if name not in self._skills:
            yield event.plain_result(
                f"技能 [{name}] 不存在。使用 /sklist 查看可用技能。")
            return

        skill = self._skills[name]
        count = skill.get("usage_count", 0)
        info = (
            f"【{name}】\n"
            f"风格标签: {skill.get('label', '未知')}\n"
            f"风格描述: {skill.get('summary', '无')}\n"
            f"来源文件: {skill.get('source_file', '未知')}\n"
            f"创建时间: {skill.get('created_at', '未知')[:10]}\n"
            f"示例数量: {len(skill.get('examples', []))} 条\n"
            f"使用次数: {count} 次"
        )
        yield event.plain_result(info)

    # ==================== /skupdate ====================

    @filter.command("skupdate")
    async def skupdate(self, event: AstrMessageEvent):
        """更新一个技能。用法: /skupdate <技能名> [新文件路径]"""
        msg = event.get_message_str().strip()
        args = msg[len("/skupdate"):].strip().split(" ", 1)
        if not args or not args[0]:
            yield event.plain_result(
                "用法: /skupdate <技能名> [新文件路径]")
            return

        skill_name = args[0]
        if skill_name not in self._skills:
            yield event.plain_result(f"技能 [{skill_name}] 不存在。")
            return

        if len(args) > 1:
            file_path = args[1].strip()
        else:
            file_path = self._skills[skill_name].get("source_file", "")
            if not file_path:
                yield event.plain_result(
                    f"技能 [{skill_name}] 没有记录源文件路径，请指定文件路径。")
                return

        path = Path(file_path)
        if not path.exists():
            yield event.plain_result(f"文件不存在: {file_path}")
            return

        yield event.plain_result(f"正在更新 [{skill_name}]，请稍候...")

        try:
            conversations = await self._read_jsonl(path)
            if not conversations:
                yield event.plain_result("文件中没有有效的对话数据")
                return

            sampled = self._sample_conversations(conversations)
            sample_text = self._format_samples(sampled)
            result = await self._analyze_style(sample_text)

            self._skills[skill_name].update({
                "label": result.get("label",
                                   self._skills[skill_name].get("label", skill_name)),
                "summary": result.get("summary", ""),
                "examples": sampled[:10],
                "source_file": str(path.absolute()),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })

            await self._save_skills()

            yield event.plain_result(
                f"技能 [{skill_name}] 更新完成！\n"
                f"风格标签: {self._skills[skill_name]['label']}\n"
                f"风格描述: {self._skills[skill_name]['summary']}")
        except Exception as e:
            logger.error(f"[StyleLearner] /skupdate 失败: {e}", exc_info=True)
            yield event.plain_result(f"更新失败: {e}")

    # ==================== /skrename ====================

    @filter.command("skrename")
    async def skrename(self, event: AstrMessageEvent):
        """重命名技能。用法: /skrename <旧名> <新名>"""
        msg = event.get_message_str().strip()
        args = msg[len("/skrename"):].strip().split()
        if len(args) < 2:
            yield event.plain_result("用法: /skrename <旧名> <新名>")
            return

        old_name, new_name = args[0], args[1]
        if old_name not in self._skills:
            yield event.plain_result(f"技能 [{old_name}] 不存在。")
            return
        if new_name in self._skills:
            yield event.plain_result(
                f"技能 [{new_name}] 已存在。请先删除或换一个名字。")
            return

        async with self._skills_lock:
            self._skills[new_name] = self._skills.pop(old_name)
        await self._save_skills()
        yield event.plain_result(f"技能 [{old_name}] → [{new_name}] 重命名完成。")

    # ==================== /skmerge ====================

    @filter.command("skmerge")
    async def skmerge(self, event: AstrMessageEvent):
        """合并两个技能。用法: /skmerge <技能A> <技能B> <新技能名>"""
        msg = event.get_message_str().strip()
        args = msg[len("/skmerge"):].strip().split()
        if len(args) < 3:
            yield event.plain_result(
                "用法: /skmerge <技能A> <技能B> <新技能名>\n"
                "例如: /skmerge 温柔学姐 沙雕群友 混合风格")
            return

        name_a, name_b, new_name = args[0], args[1], args[2]
        if name_a not in self._skills:
            yield event.plain_result(f"技能 [{name_a}] 不存在。")
            return
        if name_b not in self._skills:
            yield event.plain_result(f"技能 [{name_b}] 不存在。")
            return
        if new_name in self._skills:
            yield event.plain_result(
                f"技能 [{new_name}] 已存在。请先删除或换一个名字。")
            return

        yield event.plain_result(
            f"正在合并 [{name_a}] + [{name_b}] → [{new_name}]，请稍候...")

        try:
            skill_a = self._skills[name_a]
            skill_b = self._skills[name_b]

            # 合并 examples 并去重
            examples_a = skill_a.get("examples", [])
            examples_b = skill_b.get("examples", [])
            seen = set()
            merged_examples = []
            for ex in examples_a + examples_b:
                key = "|||".join(str(s) for s in ex)
                if key not in seen:
                    seen.add(key)
                    merged_examples.append(ex)

            # 重新分析合并后的样本
            max_chars = int(self._cfg("max_sample_chars", 3000))
            sampled = []
            total_chars = 0
            random.shuffle(merged_examples)
            for ex in merged_examples[:50]:
                text = "|||".join(str(s) for s in ex)
                if total_chars + len(text) > max_chars and sampled:
                    break
                sampled.append(ex)
                total_chars += len(text)

            sample_text = "\n".join(
                "|||".join(str(s) for s in c) for c in sampled)
            result = await self._analyze_style(sample_text)

            total_count = (skill_a.get("usage_count", 0) +
                           skill_b.get("usage_count", 0))
            skill_data = {
                "label": result.get("label", new_name),
                "summary": result.get("summary", ""),
                "examples": sampled[:10],
                "source_file": "@merged",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "usage_count": total_count,
                "merged_from": [name_a, name_b],
            }

            async with self._skills_lock:
                self._skills[new_name] = skill_data
            await self._save_skills()

            yield event.plain_result(
                f"合并完成！新技能: [{new_name}]\n"
                f"风格标签: {skill_data['label']}\n"
                f"风格描述: {skill_data['summary']}\n"
                f"合并 {len(examples_a)}+{len(examples_b)} 条示例, "
                f"使用 {len(sampled)} 条分析\n"
                f"原技能 [{name_a}]、[{name_b}] 已保留。")
        except Exception as e:
            logger.error(f"[StyleLearner] /skmerge 失败: {e}", exc_info=True)
            yield event.plain_result(f"合并失败: {e}")

    # ==================== /sklearn_active ====================

    @filter.command("sklearn_active")
    async def sklearn_active(self, event: AstrMessageEvent):
        """手动从当前 buffer 创建技能。用法: /sklearn_active <技能名> [数量]"""
        msg = event.get_message_str().strip()
        args = msg[len("/sklearn_active"):].strip().split()
        if not args:
            yield event.plain_result(
                "用法: /sklearn_active <技能名> [数量]\n"
                "例如: /sklearn_active @全局 30")
            return

        skill_name = args[0]
        count = int(args[1]) if len(args) > 1 else int(self._cfg("batch_size", 30))

        if not self._buffer:
            yield event.plain_result("Buffer 为空，没有对话数据。请先开启主动学习并聊天。")
            return

        yield event.plain_result(f"正在从 buffer ({len(self._buffer)}条) 分析 [{skill_name}]，请稍候...")

        try:
            pool = list(self._buffer)[-count:]
            sampled = []
            total_chars = 0
            max_chars = int(self._cfg("max_sample_chars", 3000))
            for item in reversed(pool):
                text = f"{item['user_msg']}|||{item['bot_msg']}"
                if total_chars + len(text) > max_chars and sampled:
                    break
                sampled.append([item["user_msg"], item["bot_msg"]])
                total_chars += len(text)

            sample_text = "\n".join(f"{s[0]}|||{s[1]}" for s in sampled)
            result = await self._analyze_style(sample_text)

            prev_count = self._skills.get(skill_name, {}).get("usage_count", 0)
            skill_data = {
                "label": result.get("label", skill_name),
                "summary": result.get("summary", ""),
                "examples": sampled[:10],
                "source_file": "@active_learning",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "auto_update": True,
                "buffer_count": len(self._buffer),
                "usage_count": prev_count,
            }

            async with self._skills_lock:
                self._skills[skill_name] = skill_data
            await self._save_skills()

            yield event.plain_result(
                f"技能 [{skill_name}] 创建完成！\n"
                f"风格标签: {skill_data['label']}\n"
                f"风格描述: {skill_data['summary']}\n"
                f"使用 {len(sampled)} 条对话样本")
        except Exception as e:
            logger.error(f"[StyleLearner] /sklearn_active 失败: {e}", exc_info=True)
            yield event.plain_result(f"学习失败: {e}")

    # ==================== /skbuffer ====================

    @filter.command("skbuffer")
    async def skbuffer(self, event: AstrMessageEvent):
        """查看当前 buffer 状态。用法: /skbuffer"""
        mode = self._cfg("active_learning_mode", "off")
        if mode == "off":
            yield event.plain_result("主动学习已关闭。在 WebUI 中开启后自动积累对话数据。")
            return

        skill_name = self._active_skill_name()
        batch_size = int(self._cfg("batch_size", 30))

        # 按 sender 统计
        sender_counts: dict[str, int] = {}
        for item in self._buffer:
            s = item.get("sender", "unknown")
            sender_counts[s] = sender_counts.get(s, 0) + 1

        lines = [
            f"主动学习模式: {mode}",
            f"目标技能名: {skill_name}",
            f"Buffer 总量: {len(self._buffer)} / 自动分析阈值: {batch_size}",
            f"进度: {min(len(self._buffer), batch_size)}/{batch_size} ({min(100, int(len(self._buffer)/batch_size*100))}%)",
            "\n按发送者统计:",
        ]
        for s, c in sorted(sender_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {s}: {c} 条")

        yield event.plain_result("\n".join(lines))

    # ==================== /skbuffer_clear ====================

    @filter.command("skbuffer_clear")
    async def skbuffer_clear(self, event: AstrMessageEvent):
        """清空 buffer。用法: /skbuffer_clear [技能名]（不指定则清空全部）"""
        msg = event.get_message_str().strip()
        arg = msg[len("/skbuffer_clear"):].strip()

        count = len(self._buffer)
        self._buffer.clear()
        self._pending.clear()
        self._last_auto_analyze_count = 0

        # 清空 buffer 文件
        try:
            await asyncio.to_thread(
                lambda: self.buffer_file.write_text("", encoding="utf-8"))
        except Exception:
            pass

        yield event.plain_result(f"已清空 buffer（原 {count} 条对话）。")

    # ==================== Helpers ====================

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """分词。jieba 可用则用 jieba，否则用字符 2-gram"""
        if HAS_JIEBA:
            return [w for w in jieba.cut(text) if len(w.strip()) > 1]
        # 降级：字符 bigram
        return [text[i:i+2] for i in range(len(text) - 1)]

    @staticmethod
    def _build_tfidf_index(examples: list[list[str]]) -> dict:
        """构建轻量 TF-IDF 索引。
        返回 {"idf": {term: idf}, "docs": [{"tf": {term: tf}, "raw": ex}]}
        """
        if not examples:
            return {"idf": {}, "docs": []}

        # 每篇文档 = 拼接 example 的所有句子
        docs_text = [" ".join(str(s) for s in ex) for ex in examples]

        # 分词 + TF
        tokenized = [StyleLearner._tokenize(d) for d in docs_text]
        doc_count = len(tokenized)

        # DF (document frequency)
        df: dict[str, int] = {}
        for tokens in tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        # IDF
        idf = {term: math.log((doc_count + 1) / (freq + 1)) + 1
               for term, freq in df.items()}

        # TF per doc
        docs = []
        for i, tokens in enumerate(tokenized):
            tf: dict[str, float] = {}
            total = max(len(tokens), 1)
            for term in tokens:
                tf[term] = tf.get(term, 0) + 1.0 / total
            docs.append({"tf": tf, "raw": examples[i]})

        return {"idf": idf, "docs": docs}

    @staticmethod
    def _retrieve_relevant_examples(
            message: str, examples: list[list[str]], top_k: int = 3
    ) -> list[list[str]]:
        """用 TF-IDF 余弦相似度检索与 message 最相关的 examples"""
        if len(examples) <= top_k:
            return examples[:]

        index = StyleLearner._build_tfidf_index(examples)
        idf = index["idf"]
        docs = index["docs"]

        # query vector
        query_tokens = StyleLearner._tokenize(message)
        query_tf: dict[str, float] = {}
        total = max(len(query_tokens), 1)
        for term in query_tokens:
            query_tf[term] = query_tf.get(term, 0) + 1.0 / total

        # 对每篇 doc 算余弦相似度
        scored = []
        for doc in docs:
            dot = 0.0
            norm_q = 0.0
            norm_d = 0.0
            all_terms = set(query_tf.keys()) | set(doc["tf"].keys())
            for term in all_terms:
                q_w = query_tf.get(term, 0) * idf.get(term, 1.0)
                d_w = doc["tf"].get(term, 0) * idf.get(term, 1.0)
                dot += q_w * d_w
                norm_q += q_w * q_w
                norm_d += d_w * d_w
            norm_q = math.sqrt(max(norm_q, 1e-8))
            norm_d = math.sqrt(max(norm_d, 1e-8))
            scored.append((dot / (norm_q * norm_d), doc["raw"]))

        scored.sort(key=lambda x: -x[0])
        return [raw for _, raw in scored[:top_k]]

    def _sample_conversations(self, conversations: list[list[str]]) -> list[list[str]]:
        max_chars = int(self._cfg("max_sample_chars", 3000))
        max_count = min(len(conversations), 50)

        pool = conversations.copy()
        random.shuffle(pool)

        result = []
        total = 0
        for conv in pool[:max_count]:
            text = "|||".join(conv)
            if total + len(text) > max_chars and result:
                break
            result.append(conv)
            total += len(text)

        return result

    def _format_samples(self, conversations: list[list[str]]) -> str:
        return "\n".join("|||".join(c) for c in conversations)

    async def _analyze_style(self, sample_text: str) -> dict:
        system_prompt = (
            "你是一个专业的对话风格分析师。仔细阅读对话样本，总结说话人的语言风格。"
            "只输出JSON，不要包含其他内容。"
        )
        user_prompt = (
            "以下是同一个人的多段对话记录，请总结该说话人的语言风格。\n\n"
            "要求：\n"
            "1. 起一个简洁的中文风格标签（如\"温柔学姐\"、\"幽默吐槽\"等）。\n"
            "2. 用一段话（100字以内）描述风格：语气特征、常用语气词、句子长度、"
            "表情符号习惯、常见话题等。\n\n"
            f"对话样本：\n{sample_text}\n\n"
            "输出JSON: {\"label\": \"风格标签\", \"summary\": \"风格描述\"}"
        )

        response = await self._call_llm(
            system_prompt, user_prompt, max_tokens=512, temperature=0.7)

        try:
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                return json.loads(m.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        logger.warning(f"[StyleLearner] 无法解析风格JSON: {response[:200]}")
        return {"label": "未知风格", "summary": response[:200]}
