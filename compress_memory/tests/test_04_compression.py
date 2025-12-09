"""
测试 4：压缩触发详细测试
- 验证 token 计算
- 验证压缩触发条件
- 验证三层架构协作
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


async def test_compression_with_long_messages():
    """使用长消息测试压缩触发"""
    print("=" * 60)
    print("测试: 压缩触发（长消息）")
    print("=" * 60)

    # 更激进的配置
    config = MemoryConfig(
        token_limit=1000,       # 非常小
        token_flush_size=200,   # 每次弹出约 200 tokens
        chat_history_token_ratio=0.5,  # 50% 给短期 buffer = 500 tokens
    )

    memory = create_memory(config=config, session_id="test_compression")

    print(f"配置:")
    print(f"  - token_limit: {config.token_limit}")
    print(f"  - token_flush_size: {config.token_flush_size}")
    print(f"  - 短期 buffer 上限: {config.token_limit * config.chat_history_token_ratio} tokens")

    # 获取 summary block 引用
    summary_block = memory.memory_blocks[0]

    # 生成较长的消息
    long_text = "这是一段很长的测试文本。" * 50  # 约 500 字符

    messages = [
        ChatMessage(role=MessageRole.USER, content=f"我叫小明，今年25岁，在北京工作。{long_text[:200]}"),
        ChatMessage(role=MessageRole.ASSISTANT, content=f"你好小明！很高兴认识你。{long_text[:200]}"),
        ChatMessage(role=MessageRole.USER, content=f"我在阿里巴巴做后端开发。{long_text[:200]}"),
        ChatMessage(role=MessageRole.ASSISTANT, content=f"阿里巴巴是个很棒的公司！{long_text[:200]}"),
        ChatMessage(role=MessageRole.USER, content=f"我下周要去腾讯了。{long_text[:200]}"),
        ChatMessage(role=MessageRole.ASSISTANT, content=f"恭喜你！腾讯也很好。{long_text[:200]}"),
    ]

    print(f"\n添加 {len(messages)} 条消息...")

    for i, msg in enumerate(messages):
        print(f"\n--- 添加消息 {i+1} ---")
        print(f"  [{msg.role.value}]: {msg.content[:50]}...")
        print(f"  消息长度: {len(msg.content)} 字符")

        await memory.aput(msg)

        # 检查摘要状态
        snapshot_len = len(summary_block.snapshot)
        print(f"  摘要长度: {snapshot_len} 字符")

        # 获取当前 active 消息数
        active_msgs = await memory.aget_all()
        print(f"  Active 消息数: {len(active_msgs)}")

    print("\n" + "=" * 60)
    print("最终状态")
    print("=" * 60)

    # 最终摘要
    if summary_block.snapshot:
        print(f"\n摘要内容:")
        print("-" * 40)
        print(summary_block.snapshot)
        print("-" * 40)
    else:
        print("\n⚠️ 摘要为空 - 压缩可能未触发")

    # 获取完整上下文
    context = await memory.aget(input="我叫什么名字？")
    print(f"\n上下文消息数: {len(context)}")

    # 检查是否有 memory 内容
    for msg in context:
        if msg.role.value == "system":
            print(f"\n找到 System 消息:")
            print("-" * 40)
            content = str(msg.content)
            print(content[:1000] + "..." if len(content) > 1000 else content)
            print("-" * 40)


async def test_manual_trigger():
    """手动测试压缩流程"""
    print("\n" + "=" * 60)
    print("测试: 手动触发压缩流程")
    print("=" * 60)

    config = MemoryConfig()
    memory = create_memory(config=config, session_id="test_manual")

    summary_block = memory.memory_blocks[0]
    vector_block = memory.memory_blocks[1]

    # 直接调用 summary block 的 _aput
    messages = [
        ChatMessage(role=MessageRole.USER, content="我叫张三，在上海工作"),
        ChatMessage(role=MessageRole.ASSISTANT, content="你好张三！上海是个很棒的城市"),
        ChatMessage(role=MessageRole.USER, content="我喜欢吃火锅"),
        ChatMessage(role=MessageRole.ASSISTANT, content="火锅很美味！有什么特别喜欢的口味吗？"),
    ]

    print("直接调用 summary_block._aput()...")
    await summary_block._aput(messages)

    print(f"\n摘要结果:")
    print("-" * 40)
    print(summary_block.snapshot)
    print("-" * 40)

    # 直接调用 vector block 的 _aput
    print("\n直接调用 vector_block._aput()...")
    await vector_block._aput(messages)

    # 测试检索
    print("\n测试向量检索...")
    retrieved = await vector_block._aget(messages=messages[-2:])
    print(f"检索结果:")
    print("-" * 40)
    print(retrieved if retrieved else "(空)")
    print("-" * 40)


async def main():
    print("\n🚀 开始压缩触发详细测试\n")

    try:
        await test_compression_with_long_messages()
        await test_manual_trigger()

        print("\n" + "=" * 60)
        print("✅ 测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
