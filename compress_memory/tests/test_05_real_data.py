"""
测试 5：使用真实 long_session 数据的完整流程测试
- 加载真实对话数据
- 模拟对话流程
- 详细的压缩统计和可视化
- 全面的性能指标
- 向量检索详细日志
- Rerank 支持
- 强化测试：事实核查、细节召回
"""

import os
import sys
import json
import asyncio
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# 设置路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 设置 API Key
os.environ["OPENROUTER_API_KEY"] = "your-api-key-here"

# 日志文件路径
LOG_FILE = os.path.join(PROJECT_ROOT, "compress_memory", "tests", "test_05_real_data.log")

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.memory import Memory
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from compress_memory.memory import create_memory
from compress_memory.config import MemoryConfig
from compress_memory.models import create_llm, create_embedding
from compress_memory.blocks.rolling_summary import RollingSummaryBlock
from compress_memory.blocks.fine_grained_vector import FineGrainedVectorBlock


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging():
    """配置详细日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s',
        datefmt='%H:%M:%S'
    )
    # 降低一些噪声日志级别
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)

logger = logging.getLogger('test_real_data')


# ============================================================================
# 样式和格式化工具
# ============================================================================

class Style:
    """终端样式"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(title: str, width: int = 80):
    """打印大标题"""
    print()
    print(Style.BOLD + Style.CYAN + "=" * width + Style.END)
    print(Style.BOLD + Style.CYAN + f"  {title}".center(width) + Style.END)
    print(Style.BOLD + Style.CYAN + "=" * width + Style.END)
    print()


def print_section(title: str, width: int = 80):
    """打印小节标题"""
    print()
    print(Style.BOLD + Style.BLUE + "-" * width + Style.END)
    print(Style.BOLD + Style.BLUE + f"  {title}" + Style.END)
    print(Style.BOLD + Style.BLUE + "-" * width + Style.END)


def print_subsection(title: str):
    """打印子小节"""
    print()
    print(f"  {Style.BOLD}{Style.MAGENTA}>>> {title}{Style.END}")


def print_kv(key: str, value, indent: int = 2, key_width: int = 30):
    """打印键值对"""
    spaces = " " * indent
    formatted_key = f"{key}:".ljust(key_width)
    print(f"{spaces}{Style.DIM}{formatted_key}{Style.END}{value}")


def print_progress_bar(current: int, total: int, width: int = 40, prefix: str = ""):
    """打印进度条"""
    progress = current / total
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    percent = progress * 100
    print(f"\r{prefix}[{Style.GREEN}{bar}{Style.END}] {percent:5.1f}% ({current}/{total})", end="", flush=True)


def format_size(chars: int) -> str:
    """格式化字符数为可读形式"""
    if chars < 1000:
        return f"{chars} 字符"
    elif chars < 1000000:
        return f"{chars/1000:.1f}K 字符"
    else:
        return f"{chars/1000000:.2f}M 字符"


def format_duration(seconds: float) -> str:
    """格式化时间"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def print_box(lines: List[str], title: str = "", width: int = 60):
    """打印美观的框"""
    print()
    print(f"  ┌{'─' * width}┐")
    if title:
        print(f"  │{Style.BOLD}{title.center(width)}{Style.END}│")
        print(f"  ├{'─' * width}┤")
    for line in lines:
        # 处理颜色代码的显示宽度
        visible_len = len(line.replace(Style.BOLD, '').replace(Style.END, '').replace(Style.GREEN, '').replace(Style.YELLOW, '').replace(Style.RED, ''))
        padding = width - visible_len
        print(f"  │ {line}{' ' * (padding - 1)}│")
    print(f"  └{'─' * width}┘")


# ============================================================================
# 统计数据类
# ============================================================================

@dataclass
class CompressionEvent:
    """单次压缩事件"""
    message_index: int
    old_snapshot_len: int
    new_snapshot_len: int
    timestamp: float
    messages_since_last: int
    llm_call_duration: float = 0


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    query_time: float
    vector_results: List[Dict[str, Any]] = field(default_factory=list)
    reranked_results: List[Dict[str, Any]] = field(default_factory=list)
    final_context_len: int = 0


@dataclass
class TestStats:
    """测试统计数据"""
    # 数据信息
    session_file: str = ""
    user_language: str = ""
    total_messages: int = 0
    processed_messages: int = 0

    # 字符统计
    total_input_chars: int = 0
    user_chars: int = 0
    assistant_chars: int = 0
    final_snapshot_chars: int = 0

    # 压缩统计
    compression_events: List[CompressionEvent] = field(default_factory=list)

    # 检索统计
    retrieval_results: List[RetrievalResult] = field(default_factory=list)

    # 时间统计
    start_time: float = 0
    end_time: float = 0

    # 向量存储统计
    vector_store_node_count: int = 0

    @property
    def compression_count(self) -> int:
        return len(self.compression_events)

    @property
    def compression_ratio(self) -> float:
        if self.final_snapshot_chars == 0:
            return 0
        return self.total_input_chars / self.final_snapshot_chars

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def avg_messages_per_compression(self) -> float:
        if self.compression_count == 0:
            return 0
        return self.processed_messages / self.compression_count


# ============================================================================
# 数据加载
# ============================================================================

def load_session_data(
    file_path: str,
    max_messages: int = 100,
    session_id: str = None,
) -> Tuple[List[ChatMessage], dict, dict]:
    """
    加载 session 数据

    Returns:
        chat_messages: 对话消息列表
        user_info: 用户信息
        raw_stats: 原始数据统计
    """
    logger.info(f"Loading session data from {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = data['data']['sessions'][0]['messages']
    user_info = data['data']['user']

    # 转换为 ChatMessage
    chat_messages = []
    user_chars = 0
    assistant_chars = 0
    skipped = 0

    for msg in messages[:max_messages]:
        role_str = msg['message']['role']
        role = MessageRole.USER if role_str == 'user' else MessageRole.ASSISTANT
        content = msg['message']['content']['text']

        if content:  # 跳过空消息
            # 添加 session_id 到 additional_kwargs，确保 VectorMemoryBlock 能正确过滤
            additional_kwargs = {}
            if session_id:
                additional_kwargs['session_id'] = session_id

            chat_messages.append(ChatMessage(
                role=role,
                content=content,
                additional_kwargs=additional_kwargs,
            ))
            if role == MessageRole.USER:
                user_chars += len(content)
            else:
                assistant_chars += len(content)
        else:
            skipped += 1

    raw_stats = {
        'total_in_file': len(messages),
        'loaded': len(chat_messages),
        'skipped_empty': skipped,
        'user_chars': user_chars,
        'assistant_chars': assistant_chars,
    }

    logger.info(f"Loaded {len(chat_messages)} messages, skipped {skipped} empty")

    return chat_messages, user_info, raw_stats


# ============================================================================
# 自定义 Memory 创建（带 Rerank）
# ============================================================================

_chroma_client = chromadb.Client()


def create_memory_with_rerank(
    config: MemoryConfig,
    session_id: str,
    enable_rerank: bool = False,
    rerank_top_n: int = 3,
    window_size: int = 3,
    stride: int = 2,
) -> Tuple[Memory, 'RollingSummaryBlock', 'FineGrainedVectorBlock']:
    """创建带细粒度向量检索的 Memory"""
    logger.info(f"Creating memory with FineGrainedVectorBlock, session={session_id}")
    logger.info(f"  window_size={window_size}, stride={stride}, rerank={enable_rerank}")

    llm = create_llm(model=config.llm_model, api_key=config.openrouter_api_key)
    embed_model = create_embedding(model=config.embed_model, api_key=config.openrouter_api_key)

    # Chroma 向量存储
    collection = _chroma_client.get_or_create_collection(
        name=f"memory_{session_id}_{int(time.time())}",  # 唯一名称避免冲突
        metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Summary Block
    summary_block = RollingSummaryBlock(
        name="ConversationSummary",
        llm=llm,
        max_snapshot_tokens=config.summary_max_tokens,
        priority=0,
    )

    # 配置 Reranker
    node_postprocessors = []
    if enable_rerank:
        reranker = LLMRerank(
            llm=llm,
            top_n=rerank_top_n,
            choice_batch_size=10,
        )
        node_postprocessors.append(reranker)
        logger.info(f"Reranker enabled with top_n={rerank_top_n}")

    # FineGrained Vector Block（以 QA 对为单位的滑动窗口）
    vector_block = FineGrainedVectorBlock(
        name="FineGrainedHistory",
        vector_store=vector_store,
        embed_model=embed_model,
        window_size=window_size,  # 每个窗口包含 3 个 turn（QA对）
        stride=stride,            # 步长 2，重叠 1 个 turn
        similarity_top_k=config.vector_similarity_top_k * 2 if enable_rerank else config.vector_similarity_top_k,
        retrieval_context_window=config.vector_retrieval_context_window,
        node_postprocessors=node_postprocessors,
        priority=1,
    )

    # Memory
    memory = Memory.from_defaults(
        session_id=session_id,
        token_limit=config.token_limit,
        token_flush_size=config.token_flush_size,
        chat_history_token_ratio=config.chat_history_token_ratio,
        memory_blocks=[summary_block, vector_block],
    )

    return memory, summary_block, vector_block


# ============================================================================
# 检索测试
# ============================================================================

def write_log(content: str, mode: str = "a"):
    """写入日志文件"""
    with open(LOG_FILE, mode, encoding="utf-8") as f:
        f.write(content + "\n")


async def test_retrieval_detailed(
    memory: Memory,
    summary_block: RollingSummaryBlock,
    vector_block: FineGrainedVectorBlock,
    test_queries: List[Dict[str, Any]],
    stats: TestStats,
    session_id: str,
):
    """详细的检索测试"""
    print_section("详细检索测试")

    # 打印 FineGrained 统计
    fg_stats = vector_block.get_stats()
    print(f"  {Style.CYAN}FineGrained 统计:{Style.END}")
    print(f"    - 存储的 node 数: {fg_stats['node_count']}")
    print(f"    - 处理的 turn 数: {fg_stats['turn_count']}")
    print(f"    - 窗口大小: {fg_stats['window_size']} turns")
    print(f"    - 滑动步长: {fg_stats['stride']} turns")
    print(f"    - 重叠: {fg_stats['overlap']} turns")
    print()

    # 初始化日志文件
    write_log(f"\n{'='*80}", mode="w")
    write_log(f"RAG 检索详细日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(f"{'='*80}\n")
    write_log(f"FineGrained 配置:")
    write_log(f"  - node_count: {fg_stats['node_count']}")
    write_log(f"  - turn_count: {fg_stats['turn_count']}")
    write_log(f"  - window_size: {fg_stats['window_size']}")
    write_log(f"  - stride: {fg_stats['stride']}")
    write_log(f"  - overlap: {fg_stats['overlap']}")
    write_log("")

    for i, query_info in enumerate(test_queries, 1):
        query = query_info['query']
        expected_keywords = query_info.get('expected_keywords', [])
        description = query_info.get('description', '')

        print_subsection(f"Query #{i}: {description or query[:50]}")
        print()
        print(f"    {Style.BOLD}查询:{Style.END} {query}")

        write_log(f"\n{'='*60}")
        write_log(f"Query #{i}: {description}")
        write_log(f"查询: {query}")
        write_log(f"{'='*60}\n")

        # ========== 直接调用 FineGrainedVectorBlock 检索 ==========
        write_log(">>> FineGrainedVectorBlock 直接检索结果:")

        # 构造查询消息
        query_messages = [ChatMessage(role=MessageRole.USER, content=query)]

        start_time = time.time()
        # 直接调用 vector_block._aget 获取原始检索结果
        raw_retrieved = await vector_block._aget(
            messages=query_messages,
            session_id=session_id,
        )
        vector_time = time.time() - start_time

        write_log(f"检索耗时: {format_duration(vector_time)}")
        write_log(f"返回内容长度: {len(raw_retrieved)} 字符")
        write_log("")

        if raw_retrieved:
            write_log("--- 检索到的原始 block 内容 ---")
            write_log(raw_retrieved)
            write_log("--- 原始 block 结束 ---")
            print(f"    {Style.GREEN}VectorBlock 检索到 {len(raw_retrieved)} 字符{Style.END}")
        else:
            write_log("!!! 未检索到任何内容 !!!")
            print(f"    {Style.RED}VectorBlock 未检索到内容{Style.END}")

        write_log("")

        # ========== 完整 Memory.aget 调用 ==========
        write_log(">>> Memory.aget 完整调用结果:")

        start_time = time.time()
        context = await memory.aget(input=query)
        query_time = time.time() - start_time

        result = RetrievalResult(query=query, query_time=query_time)

        print()
        print_kv("Memory.aget 耗时", format_duration(query_time), indent=4)
        print_kv("返回消息数", len(context), indent=4)

        write_log(f"耗时: {format_duration(query_time)}")
        write_log(f"返回消息数: {len(context)}")

        # 分析返回的上下文
        for msg in context:
            role = msg.role.value
            content = str(msg.content)

            write_log(f"\n[{role}] 消息长度: {len(content)} 字符")

            if role == "system":
                result.final_context_len = len(content)
                print_kv("System 消息长度", format_size(len(content)), indent=4)

                # 写入完整 system 消息到日志
                write_log("\n--- System 消息完整内容 ---")
                write_log(content)
                write_log("--- System 消息结束 ---\n")

                # 检查是否包含检索结果
                if "FineGrainedHistory" in content or "RetrievedHistory" in content:
                    retrieved_count = content.count("<turn index=")
                    print(f"    {Style.GREEN}包含 RAG 检索结果: {retrieved_count} 个 turns{Style.END}")
                else:
                    print(f"    {Style.YELLOW}未包含 RAG 检索结果{Style.END}")

                # 检查摘要
                if "<snapshot>" in content:
                    print(f"    {Style.CYAN}包含摘要{Style.END}")

        stats.retrieval_results.append(result)
        print()

    write_log(f"\n{'='*80}")
    write_log("日志结束")
    write_log(f"{'='*80}")

    print()
    print(f"  {Style.BOLD}{Style.GREEN}详细日志已写入: {LOG_FILE}{Style.END}")


# ============================================================================
# 主测试流程
# ============================================================================

async def test_with_real_data():
    """使用真实数据测试"""
    stats = TestStats()

    print_header("真实数据完整流程测试（增强版）", width=80)

    # -------------------------------------------------------------------------
    # 1. 加载数据
    # -------------------------------------------------------------------------
    print_section("1. 数据加载")

    # 使用技术对话数据（Ollama 故障排除）- 更容易验证 RAG 准确性
    session_file = os.path.join(
        PROJECT_ROOT,
        "long_session",
        "019a5835-ee72-7f63-ae7a-75d026d2291d.json"  # Ollama 技术支持对话
    )

    stats.session_file = os.path.basename(session_file)

    if not os.path.exists(session_file):
        print(f"{Style.RED}找不到测试数据文件: {session_file}{Style.END}")
        return None

    # 定义 session_id（需要在加载数据时使用，确保向量检索能匹配）
    SESSION_ID = "real_session_test"

    # 加载更多消息进行强化测试
    messages, user_info, raw_stats = load_session_data(
        session_file,
        max_messages=150,
        session_id=SESSION_ID,  # 传入 session_id，确保向量检索能正确过滤
    )

    stats.user_language = user_info.get('language', 'unknown')
    stats.total_messages = raw_stats['total_in_file']
    stats.processed_messages = len(messages)
    stats.total_input_chars = raw_stats['user_chars'] + raw_stats['assistant_chars']
    stats.user_chars = raw_stats['user_chars']
    stats.assistant_chars = raw_stats['assistant_chars']

    print_kv("数据文件", stats.session_file)
    print_kv("用户语言", stats.user_language)
    print_kv("文件中总消息数", stats.total_messages)
    print_kv("加载消息数", f"{Style.BOLD}{stats.processed_messages}{Style.END}")
    print_kv("跳过空消息", raw_stats['skipped_empty'])
    print()
    print_kv("用户消息字符数", format_size(stats.user_chars))
    print_kv("助手消息字符数", format_size(stats.assistant_chars))
    print_kv("总输入字符数", f"{Style.BOLD}{format_size(stats.total_input_chars)}{Style.END}")

    # 消息角色分布
    user_count = sum(1 for m in messages if m.role == MessageRole.USER)
    assistant_count = len(messages) - user_count
    print()
    print_kv("用户消息数", user_count)
    print_kv("助手消息数", assistant_count)
    print_kv("平均用户消息长度", f"{stats.user_chars / max(user_count, 1):.0f} 字符")
    print_kv("平均助手消息长度", f"{stats.assistant_chars / max(assistant_count, 1):.0f} 字符")

    # 预览首尾消息
    print()
    print(f"  {Style.DIM}首条消息预览:{Style.END}")
    first_content = messages[0].content[:100].replace('\n', ' ')
    print(f"    [{messages[0].role.value}] {first_content}...")
    print(f"  {Style.DIM}末条消息预览:{Style.END}")
    last_content = messages[-1].content[:100].replace('\n', ' ')
    print(f"    [{messages[-1].role.value}] {last_content}...")

    # -------------------------------------------------------------------------
    # 2. 配置 Memory（带 Rerank）
    # -------------------------------------------------------------------------
    print_section("2. Memory 配置（启用 Rerank）")

    config = MemoryConfig(
        token_limit=20000,      # 10k 上下文
        token_flush_size=5000,  # 每次弹出约 2000 tokens
        chat_history_token_ratio=0.5,
        vector_similarity_top_k=5,  # 初始检索更多
    )

    memory, summary_block, vector_block = create_memory_with_rerank(
        config=config,
        session_id=SESSION_ID,
        enable_rerank=True,   # 启用 LLM Rerank
        rerank_top_n=3,       # rerank 后保留 top 3
        window_size=3,        # 每个窗口 3 个 QA 对
        stride=2,             # 步长 2，重叠 1 个 QA 对
    )

    print_kv("token_limit", f"{config.token_limit:,} tokens")
    print_kv("token_flush_size", f"{config.token_flush_size:,} tokens")
    print_kv("chat_history_token_ratio", f"{config.chat_history_token_ratio} ({int(config.token_limit * config.chat_history_token_ratio):,} tokens for buffer)")
    print_kv("summary_max_tokens", f"{config.summary_max_tokens:,} tokens")
    print()
    print_kv("vector_similarity_top_k", f"{config.vector_similarity_top_k * 2} (before rerank)")
    print_kv("rerank_top_n", "3 (after rerank)")
    print_kv("retrieval_context_window", config.vector_retrieval_context_window)
    print()
    print_kv("LLM 模型", config.llm_model)
    print_kv("Embedding 模型", config.embed_model)
    print_kv("Embedding 维度", config.embed_dimension)

    # -------------------------------------------------------------------------
    # 3. 模拟对话流程
    # -------------------------------------------------------------------------
    print_section("3. 模拟对话流程")

    stats.start_time = time.time()
    last_snapshot_len = 0
    last_compression_idx = 0
    compression_start_time = None

    print()
    print(f"  {Style.DIM}开始处理 {len(messages)} 条消息...{Style.END}")
    print()

    for i, msg in enumerate(messages):
        # 记录压缩前的状态
        pre_snapshot_len = len(summary_block.snapshot)

        await memory.aput(msg)

        # 检查是否触发压缩
        current_snapshot_len = len(summary_block.snapshot)
        if current_snapshot_len > last_snapshot_len:
            event = CompressionEvent(
                message_index=i + 1,
                old_snapshot_len=last_snapshot_len,
                new_snapshot_len=current_snapshot_len,
                timestamp=time.time() - stats.start_time,
                messages_since_last=i - last_compression_idx,
            )
            stats.compression_events.append(event)

            # 清除进度条并打印压缩事件
            print(f"\r{' ' * 100}\r", end="")
            growth = current_snapshot_len - last_snapshot_len
            print(f"  {Style.YELLOW}[压缩 #{len(stats.compression_events):2d}]{Style.END} "
                  f"@ 消息 {i+1:3d} | "
                  f"摘要: {last_snapshot_len:5d} → {current_snapshot_len:5d} ({growth:+5d} 字符) | "
                  f"间隔: {event.messages_since_last:2d} 条 | "
                  f"耗时: {format_duration(event.timestamp)}")

            last_snapshot_len = current_snapshot_len
            last_compression_idx = i

        # 更新进度条
        print_progress_bar(i + 1, len(messages), prefix="  处理进度: ")

    stats.end_time = time.time()
    stats.final_snapshot_chars = len(summary_block.snapshot)

    print()
    print()
    print(f"  {Style.GREEN}处理完成!{Style.END}")
    print_kv("总耗时", format_duration(stats.duration))
    print_kv("平均处理速度", f"{stats.processed_messages / stats.duration:.1f} 条/秒")

    # -------------------------------------------------------------------------
    # 4. 压缩统计
    # -------------------------------------------------------------------------
    print_section("4. 压缩统计")

    print_kv("压缩触发次数", stats.compression_count)
    print_kv("平均压缩间隔", f"{stats.avg_messages_per_compression:.1f} 条消息")
    print_kv("最终摘要大小", format_size(stats.final_snapshot_chars))
    print_kv("压缩比", f"{Style.BOLD}{Style.GREEN}{stats.compression_ratio:.1f}x{Style.END}")

    if stats.compression_events:
        print()
        print(f"  {Style.DIM}压缩事件时间线:{Style.END}")
        print(f"    {'#':>3} {'时间':>10} {'消息':>6} {'摘要大小':>12} {'增长':>8} {'间隔':>6}")
        print(f"    {'-'*50}")
        for i, event in enumerate(stats.compression_events):
            growth = event.new_snapshot_len - event.old_snapshot_len
            print(f"    {i+1:3d} {format_duration(event.timestamp):>10} {event.message_index:>6} "
                  f"{event.new_snapshot_len:>12} {growth:>+8} {event.messages_since_last:>6}")

    # -------------------------------------------------------------------------
    # 5. 最终摘要
    # -------------------------------------------------------------------------
    print_section("5. 最终摘要内容")

    if summary_block.snapshot:
        snapshot = summary_block.snapshot
        print()
        print(f"  {Style.DIM}摘要长度: {format_size(len(snapshot))}{Style.END}")
        print(f"  {Style.DIM}摘要行数: {len(snapshot.splitlines())}{Style.END}")
        print()

        # 按行打印，带行号
        lines = snapshot.split('\n')
        max_lines = 60
        for i, line in enumerate(lines[:max_lines]):
            line_num = f"{i+1:3d}"
            # 截断过长的行
            if len(line) > 100:
                line = line[:100] + "..."
            print(f"  {Style.DIM}{line_num}{Style.END} │ {line}")
        if len(lines) > max_lines:
            print(f"  {Style.DIM}... (共 {len(lines)} 行，显示前 {max_lines} 行){Style.END}")
    else:
        print(f"  {Style.YELLOW}摘要为空{Style.END}")

    # -------------------------------------------------------------------------
    # 6. 强化检索测试 - 针对 Ollama 技术对话
    # -------------------------------------------------------------------------
    test_queries = [
        {
            "query": "What error did the user encounter with Ollama?",
            "description": "错误信息查询",
            "expected_keywords": ["error", "manifest", "file does not exist"],
        },
        {
            "query": "How to fix the modelfile FROM directive issue?",
            "description": "FROM 指令修复方案",
            "expected_keywords": ["FROM", "modelfile", "llama2", "path"],
        },
        {
            "query": "What commands were suggested for debugging?",
            "description": "调试命令查询",
            "expected_keywords": ["ollama", "list", "pull", "cat"],
        },
        {
            "query": "What GPU or memory issues were discussed?",
            "description": "GPU/内存问题",
            "expected_keywords": ["GPU", "memory", "VRAM"],
        },
        {
            "query": "What model names or versions were mentioned?",
            "description": "模型名称查询",
            "expected_keywords": ["llama", "shadow", "mistral"],
        },
    ]

    await test_retrieval_detailed(memory, summary_block, vector_block, test_queries, stats, SESSION_ID)

    # -------------------------------------------------------------------------
    # 7. 向量存储统计
    # -------------------------------------------------------------------------
    print_section("7. 向量存储统计")

    # 获取 Chroma collection 信息
    try:
        collection = vector_block.vector_store._collection
        stats.vector_store_node_count = collection.count()
        print_kv("存储的向量节点数", stats.vector_store_node_count)
        print_kv("每节点平均包含消息", f"{stats.processed_messages / max(stats.vector_store_node_count, 1):.1f} 条")
    except Exception as e:
        print(f"  {Style.DIM}无法获取向量存储统计: {e}{Style.END}")

    # -------------------------------------------------------------------------
    # 8. 总结报告
    # -------------------------------------------------------------------------
    print_section("8. 测试总结报告")

    report_lines = [
        f"数据文件:        {stats.session_file}",
        f"处理消息数:      {stats.processed_messages}",
        f"原始输入大小:    {format_size(stats.total_input_chars)}",
        "",
        f"压缩触发次数:    {stats.compression_count}",
        f"最终摘要大小:    {format_size(stats.final_snapshot_chars)}",
        f"压缩比:          {Style.GREEN}{stats.compression_ratio:.1f}x{Style.END}",
        "",
        f"向量节点数:      {stats.vector_store_node_count}",
        f"检索测试数:      {len(stats.retrieval_results)}",
        "",
        f"总耗时:          {format_duration(stats.duration)}",
        f"处理速度:        {stats.processed_messages / stats.duration:.1f} 条/秒",
    ]
    print_box(report_lines, title="测试结果", width=60)

    # 检索效果评估
    if stats.retrieval_results:
        print()
        print(f"  {Style.BOLD}检索性能:{Style.END}")
        avg_query_time = sum(r.query_time for r in stats.retrieval_results) / len(stats.retrieval_results)
        print_kv("平均检索耗时", format_duration(avg_query_time), indent=4)
        avg_context_len = sum(r.final_context_len for r in stats.retrieval_results) / len(stats.retrieval_results)
        print_kv("平均上下文长度", format_size(int(avg_context_len)), indent=4)

    return stats


async def main():
    setup_logging()

    print()
    print(f"{Style.BOLD}{Style.CYAN}╔════════════════════════════════════════════════════════════════════════════════╗{Style.END}")
    print(f"{Style.BOLD}{Style.CYAN}║             Long Context Compression - Enhanced Real Data Test                 ║{Style.END}")
    print(f"{Style.BOLD}{Style.CYAN}║                         compress_memory v1.0                                   ║{Style.END}")
    print(f"{Style.BOLD}{Style.CYAN}║                                                                                ║{Style.END}")
    print(f"{Style.BOLD}{Style.CYAN}║  Features: Rolling Summary + Vector RAG + LLM Rerank                          ║{Style.END}")
    print(f"{Style.BOLD}{Style.CYAN}╚════════════════════════════════════════════════════════════════════════════════╝{Style.END}")
    print()
    print(f"  {Style.DIM}测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.END}")
    print(f"  {Style.DIM}Python: {sys.version.split()[0]}{Style.END}")
    print(f"  {Style.DIM}工作目录: {os.getcwd()}{Style.END}")

    try:
        stats = await test_with_real_data()

        if stats:
            print()
            print(f"{Style.BOLD}{Style.GREEN}{'=' * 80}{Style.END}")
            print(f"{Style.BOLD}{Style.GREEN}  测试完成! 压缩比: {stats.compression_ratio:.1f}x{Style.END}")
            print(f"{Style.BOLD}{Style.GREEN}{'=' * 80}{Style.END}")
            print()

    except Exception as e:
        print()
        print(f"{Style.RED}{'=' * 80}{Style.END}")
        print(f"{Style.RED}  测试失败: {e}{Style.END}")
        print(f"{Style.RED}{'=' * 80}{Style.END}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
