# CLAUDE.md

本项目基于 LlamaIndex 研究长上下文压缩方案。

## 方案概述

三层记忆架构：

1. **短期缓冲区**：保留最近的原始对话
2. **滚动摘要**：用淘汰的上下文 + 旧摘要更新摘要（参考 Gemini CLI、Codex）
3. **细粒度 RAG**：弹出的消息按 QA 对滑窗分 chunk，存入向量库检索

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM Context Window                      │
├─────────────────────────────────────────────────────────────┤
│  1. 短期缓冲区 - 最近原始对话（FIFO）                           │
├─────────────────────────────────────────────────────────────┤
│  2. 滚动摘要 - 全局视角，压缩历史                              │
├─────────────────────────────────────────────────────────────┤
│  3. 细粒度 RAG - 局部细节，按 QA 对滑窗检索                     │
└─────────────────────────────────────────────────────────────┘
```

## 测试结果

真实对话测试（149 条消息，179K 字符）：

| 指标 | 结果 |
|-----|------|
| 压缩比 | 39.5x |
| RAG 召回准确率 | 96%（启用 LLM Rerank） |
| 摘要质量 | 7.5/10 |

### 查询速度

| 阶段 | 耗时 | 说明 |
|-----|------|------|
| Embedding | ~600ms | API 网络延迟为主 |
| 向量检索 | ~7ms | Chroma 内存检索 |

## 项目结构

```
compress_memory/
├── config.py              # MemoryConfig 配置类
├── models.py              # OpenRouter LLM/Embedding 封装
├── memory.py              # Memory 工厂函数
├── blocks/
│   ├── rolling_summary.py # RollingSummaryBlock 滚动摘要
│   └── fine_grained_vector.py # FineGrainedVectorBlock 细粒度 RAG
├── prompts/
│   └── compression.py     # 压缩提示词模板
└── tests/
    ├── test_01_api.py           # API 连接测试
    ├── test_02_summary_block.py # RollingSummaryBlock 单元测试
    ├── test_03_memory.py        # Memory 集成测试
    ├── test_04_compression.py   # 压缩触发详细测试
    ├── test_05_real_data.py     # 真实数据完整流程测试
    └── test_rag_speed.py        # RAG 查询速度测试
```

## 快速使用

```python
import os
os.environ["OPENROUTER_API_KEY"] = "your-key"

from compress_memory.memory import create_memory
from compress_memory.config import MemoryConfig
from llama_index.core.base.llms.types import ChatMessage, MessageRole

config = MemoryConfig(
    token_limit=8000,
    token_flush_size=2000,
    chat_history_token_ratio=0.5,
)
memory = create_memory(config=config, session_id="user_123")

await memory.aput(ChatMessage(role=MessageRole.USER, content="我叫小明"))
await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content="你好小明！"))

context = await memory.aget(input="我叫什么名字？")
```

## 核心组件

### FineGrainedVectorBlock

以 QA 对为单位的滑动窗口向量检索：

```python
FineGrainedVectorBlock(
    vector_store=vector_store,
    embed_model=embed_model,
    window_size=3,   # 每个窗口 3 个 QA 对
    stride=2,        # 步长 2，重叠 1 个
    similarity_top_k=5,
    node_postprocessors=[reranker],  # 可选 LLM Rerank
)
```

### RollingSummaryBlock

滚动摘要，增量更新：

```python
RollingSummaryBlock(
    llm=llm,
    max_snapshot_tokens=1500,
    priority=0,  # 永不截断
)
```

## 多会话隔离

通过 metadata 过滤实现，不影响查询速度（瓶颈在 Embedding API）：

```python
node.metadata = {"session_id": "user_123"}
# 检索时自动按 session_id 过滤
```

## 运行测试

```bash
# 完整流程测试（含 RAG + 摘要质量评估）
python compress_memory/tests/test_05_real_data.py

# RAG 查询速度测试
python compress_memory/tests/test_rag_speed.py
```

## LlamaIndex Memory 并发模式

| 操作 | 方式 | 说明 |
|-----|------|------|
| aput（写入） | asyncio.gather 并行 | 多 block 同时写入 |
| aget（读取） | for 循环串行 | 按 priority 顺序 |
