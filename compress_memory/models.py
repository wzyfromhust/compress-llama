"""模型封装：OpenRouter LLM 和 Embedding"""

import os
from typing import Optional

# LlamaIndex imports
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding


def create_llm(
    model: str = "google/gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> OpenRouter:
    """
    创建 OpenRouter LLM

    Args:
        model: 模型名称
        api_key: API key，默认从环境变量读取
        temperature: 温度
        max_tokens: 最大输出 token 数
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    return OpenRouter(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_embedding(
    model: str = "qwen/qwen3-embedding-8b",
    api_key: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> OpenAILikeEmbedding:
    """
    创建 OpenRouter Embedding

    Args:
        model: 模型名称
        api_key: API key，默认从环境变量读取
        dimensions: 嵌入维度（可选）
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    return OpenAILikeEmbedding(
        model_name=model,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        dimensions=dimensions,
        embed_batch_size=10,
        timeout=60.0,
    )
