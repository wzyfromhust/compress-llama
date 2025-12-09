"""
Compress Memory - 基于 LlamaIndex 的长上下文压缩方案

三层架构：
1. 短期 Buffer：最近 N 轮原始对话
2. RAG 召回：向量检索相关历史原文
3. Rolling 摘要：滚动更新的结构化摘要
"""

from .config import MemoryConfig

__all__ = ["MemoryConfig"]
