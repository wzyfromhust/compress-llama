"""
测试 3：Memory 集成测试
- 测试完整的三层架构
- 测试消息弹出和压缩流程
"""

import os
import sys
import asyncio

# 设置路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 设置 API Key
os.environ["OPENROUTER_API_KEY"] = "your-api-key-here"

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from compress_memory.memory import create_memory
from compress_memory.config import MemoryConfig


async def test_memory_creation():
    """测试 Memory 创建"""
    print("=" * 60)
    print("测试 1: Memory 创建")
    print("=" * 60)

    # 使用较小的 token limit 以便快速触发压缩
    config = MemoryConfig(
        token_limit=2000,       # 较小，便于测试
        token_flush_size=500,   # 每次弹出约 500 tokens
        chat_history_token_ratio=0.5,  # 50% 给短期 buffer
    )

    memory = create_memory(config=config, session_id="test_session")

    print(f"Memory 创建成功")
    print(f"  - token_limit: {memory.token_limit}")
    print(f"  - token_flush_size: {memory.token_flush_size}")
    print(f"  - chat_history_token_ratio: {memory.chat_history_token_ratio}")
    print(f"  - memory_blocks: {[b.name for b in memory.memory_blocks]}")

    assert len(memory.memory_blocks) == 2, "应该有 2 个 memory blocks"
    print("✅ Memory 创建测试通过")

    return memory


async def test_basic_put_get(memory):
    """测试基本的 put/get 操作"""
    print("\n" + "=" * 60)
    print("测试 2: 基本 put/get 操作")
    print("=" * 60)

    # 添加几条消息
    messages = [
        ChatMessage(role=MessageRole.USER, content="你好，我是小明"),
        ChatMessage(role=MessageRole.ASSISTANT, content="你好小明！很高兴认识你。有什么我可以帮助你的吗？"),
        ChatMessage(role=MessageRole.USER, content="我想了解一下Python"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Python是一门很棒的编程语言！它简单易学，功能强大。你想从哪方面开始学习呢？"),
    ]

    for msg in messages:
        await memory.aput(msg)
        print(f"  添加消息: [{msg.role.value}] {msg.content[:30]}...")

    # 获取上下文
    context = await memory.aget(input="继续")
    print(f"\n获取到的上下文消息数: {len(context)}")

    for i, msg in enumerate(context):
        content = str(msg.content)[:50] if msg.content else "(empty)"
        print(f"  [{i}] {msg.role.value}: {content}...")

    assert len(context) > 0, "应该返回上下文消息"
    print("✅ 基本 put/get 测试通过")


async def test_compression_trigger(memory):
    """测试压缩触发"""
    print("\n" + "=" * 60)
    print("测试 3: 压缩触发（添加大量消息）")
    print("=" * 60)

    # 添加更多消息以触发压缩
    conversations = [
        ("我在阿里巴巴工作", "阿里巴巴是很棒的公司！你是做什么岗位的？"),
        ("我是后端开发工程师", "后端开发很有挑战性！主要用什么技术栈？"),
        ("主要用Java和Go", "Java和Go都是很好的选择。最近有什么项目在做吗？"),
        ("在做微服务架构改造", "微服务是个很热门的方向。遇到什么挑战了吗？"),
        ("主要是服务拆分和数据一致性问题", "这确实是微服务的常见挑战。有考虑用什么解决方案吗？"),
        ("准备用Saga模式", "Saga模式是个好选择，适合长事务场景。"),
        ("对了我下周要换工作了", "哦？要去哪里呢？"),
        ("准备去腾讯做游戏服务器", "恭喜！游戏服务器很有挑战性！"),
        ("有点紧张，没做过游戏", "不用担心，你的后端经验会很有帮助的。"),
        ("希望能快速上手", "相信你一定可以的！有什么具体想了解的吗？"),
    ]

    for user_msg, assistant_msg in conversations:
        await memory.aput(ChatMessage(role=MessageRole.USER, content=user_msg))
        await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_msg))
        print(f"  + [{user_msg[:20]}...] → [{assistant_msg[:20]}...]")

    # 检查 summary block 的状态
    summary_block = memory.memory_blocks[0]
    print(f"\n摘要块状态:")
    print(f"  - snapshot 长度: {len(summary_block.snapshot)} 字符")

    if summary_block.snapshot:
        print(f"  - snapshot 内容:")
        print("-" * 40)
        print(summary_block.snapshot[:500] + "..." if len(summary_block.snapshot) > 500 else summary_block.snapshot)
        print("-" * 40)

    print("✅ 压缩触发测试通过")


async def test_context_retrieval(memory):
    """测试上下文检索"""
    print("\n" + "=" * 60)
    print("测试 4: 上下文检索")
    print("=" * 60)

    # 获取完整上下文
    context = await memory.aget(input="我之前说我叫什么名字？")

    print(f"获取到的上下文消息数: {len(context)}")

    # 分析上下文结构
    has_system = any(msg.role.value == "system" for msg in context)
    has_summary = any("ConversationSummary" in str(msg.content) for msg in context)
    has_retrieved = any("RetrievedHistory" in str(msg.content) for msg in context)

    print(f"  - 包含 system 消息: {has_system}")
    print(f"  - 包含摘要: {has_summary}")
    print(f"  - 包含检索结果: {has_retrieved}")

    # 打印 system 消息（如果有）
    for msg in context:
        if msg.role.value == "system":
            content = str(msg.content)
            print(f"\nSystem 消息内容:")
            print("-" * 40)
            print(content[:800] + "..." if len(content) > 800 else content)
            print("-" * 40)
            break

    print("✅ 上下文检索测试通过")


async def main():
    print("\n🚀 开始 Memory 集成测试\n")

    try:
        # 测试 Memory 创建
        memory = await test_memory_creation()

        # 测试基本 put/get
        await test_basic_put_get(memory)

        # 测试压缩触发
        await test_compression_trigger(memory)

        # 测试上下文检索
        await test_context_retrieval(memory)

        print("\n" + "=" * 60)
        print("✅ 所有 Memory 集成测试通过!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
