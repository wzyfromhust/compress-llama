"""Memory 工厂函数"""

from typing import Optional

import chromadb
from llama_index.core.memory import Memory
from llama_index.core.memory.memory_blocks import VectorMemoryBlock
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from .blocks.rolling_summary import RollingSummaryBlock
from .config import MemoryConfig
from .models import create_llm, create_embedding


# 全局 Chroma client（内存模式）
_chroma_client = chromadb.Client()


def create_memory(
    config: Optional[MemoryConfig] = None,
    session_id: str = "default",
    llm: Optional[LLM] = None,
    embed_model: Optional[BaseEmbedding] = None,
) -> Memory:
    """
    创建配置好的 Memory 实例

    Args:
        config: 配置对象，默认使用 MemoryConfig()
        session_id: 会话 ID
        llm: 用于摘要的 LLM，默认创建新实例
        embed_model: Embedding 模型，默认创建新实例

    Returns:
        配置好的 Memory 实例，包含：
        - RollingSummaryBlock (priority=0, 永不截断)
        - VectorMemoryBlock (priority=1, 可截断)
    """
    config = config or MemoryConfig()

    # 创建模型（如果未提供）
    if llm is None:
        llm = create_llm(
            model=config.llm_model,
            api_key=config.openrouter_api_key,
        )

    if embed_model is None:
        embed_model = create_embedding(
            model=config.embed_model,
            api_key=config.openrouter_api_key,
        )

    # 创建 Chroma 向量存储（内存模式，按 session_id 分 collection）
    collection = _chroma_client.get_or_create_collection(
        name=f"memory_{session_id}",
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # 创建 Memory Blocks
    summary_block = RollingSummaryBlock(
        name="ConversationSummary",
        llm=llm,
        max_snapshot_tokens=config.summary_max_tokens,
        priority=0,  # 永不截断
    )

    vector_block = VectorMemoryBlock(
        name="RetrievedHistory",
        vector_store=vector_store,
        embed_model=embed_model,
        similarity_top_k=config.vector_similarity_top_k,
        retrieval_context_window=config.vector_retrieval_context_window,
        priority=1,  # 可截断
    )

    # 创建 Memory
    memory = Memory.from_defaults(
        session_id=session_id,
        token_limit=config.token_limit,
        token_flush_size=config.token_flush_size,
        chat_history_token_ratio=config.chat_history_token_ratio,
        memory_blocks=[
            summary_block,  # 摘要在前
            vector_block,   # 检索在后
        ],
    )

    return memory


def create_memory_simple(session_id: str = "default") -> Memory:
    """
    使用默认配置创建 Memory（简化版）

    Args:
        session_id: 会话 ID

    Returns:
        配置好的 Memory 实例
    """
    return create_memory(session_id=session_id)
