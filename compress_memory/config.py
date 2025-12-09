"""配置类"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryConfig:
    """Memory 配置"""

    # Token 限制
    token_limit: int = 8000
    token_flush_size: int = 2000
    chat_history_token_ratio: float = 0.5

    # Vector Block 配置
    vector_similarity_top_k: int = 3
    vector_retrieval_context_window: int = 5

    # Summary Block 配置
    summary_max_tokens: int = 1500

    # 模型配置
    llm_model: str = "google/gemini-2.5-flash"
    embed_model: str = "qwen/qwen3-embedding-8b"
    embed_dimension: int = 4096  # qwen3-embedding-8b 的实际维度

    # API 配置
    openrouter_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    def __post_init__(self):
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
