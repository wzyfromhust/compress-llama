"""
测试 2：RollingSummaryBlock 单元测试
- 测试摘要生成
- 测试滚动更新
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

from compress_memory.models import create_llm
from compress_memory.blocks.rolling_summary import RollingSummaryBlock


async def test_first_compression():
    """测试首次压缩（无历史摘要）"""
    print("=" * 60)
    print("测试 1: 首次压缩")
    print("=" * 60)

    llm = create_llm(model="google/gemini-2.5-flash")
    block = RollingSummaryBlock(llm=llm)

    # 模拟一批对话
    messages = [
        ChatMessage(role=MessageRole.USER, content="我叫小明，今年25岁，在北京工作"),
        ChatMessage(role=MessageRole.ASSISTANT, content="你好小明！很高兴认识你。在北京工作感觉怎么样？"),
        ChatMessage(role=MessageRole.USER, content="还不错，我在阿里巴巴做后端开发"),
        ChatMessage(role=MessageRole.ASSISTANT, content="阿里巴巴是个很棒的公司！后端开发用的什么技术栈呢？"),
        ChatMessage(role=MessageRole.USER, content="主要用Java和Go，最近在学习Rust"),
    ]

    print(f"输入消息数: {len(messages)}")
    print(f"初始摘要: '{block.snapshot}'")

    # 执行压缩
    await block._aput(messages)

    print(f"\n压缩后摘要:")
    print("-" * 40)
    print(block.snapshot)
    print("-" * 40)

    assert block.snapshot, "摘要不应为空"
    assert "<snapshot>" in block.snapshot or "小明" in block.snapshot, "摘要应包含用户信息"
    print("✅ 首次压缩测试通过")

    return block


async def test_rolling_update(block: RollingSummaryBlock):
    """测试滚动更新（追加新对话）"""
    print("\n" + "=" * 60)
    print("测试 2: 滚动更新")
    print("=" * 60)

    old_snapshot = block.snapshot
    print(f"已有摘要长度: {len(old_snapshot)} 字符")

    # 新一批对话（包含更新信息）
    new_messages = [
        ChatMessage(role=MessageRole.USER, content="对了，我下周要离职了，准备去腾讯"),
        ChatMessage(role=MessageRole.ASSISTANT, content="恭喜！腾讯也是很好的选择。是什么岗位呢？"),
        ChatMessage(role=MessageRole.USER, content="还是后端，不过是做游戏服务器的"),
        ChatMessage(role=MessageRole.ASSISTANT, content="游戏服务器很有挑战性！高并发场景会比较多。"),
        ChatMessage(role=MessageRole.USER, content="是啊，有点紧张，毕竟没做过游戏"),
    ]

    print(f"新消息数: {len(new_messages)}")

    # 执行滚动更新
    await block._aput(new_messages)

    print(f"\n更新后摘要:")
    print("-" * 40)
    print(block.snapshot)
    print("-" * 40)

    # 验证更新
    assert block.snapshot != old_snapshot, "摘要应该被更新"
    # 检查是否包含新信息（腾讯）
    assert "腾讯" in block.snapshot or "游戏" in block.snapshot, "摘要应包含新信息"

    print("✅ 滚动更新测试通过")

    return block


async def test_get_snapshot(block: RollingSummaryBlock):
    """测试获取摘要"""
    print("\n" + "=" * 60)
    print("测试 3: 获取摘要 (_aget)")
    print("=" * 60)

    result = await block._aget()
    print(f"_aget 返回: {result[:100]}..." if len(result) > 100 else f"_aget 返回: {result}")

    assert result == block.snapshot, "_aget 应返回当前摘要"
    print("✅ 获取摘要测试通过")


async def test_empty_block():
    """测试空摘要块"""
    print("\n" + "=" * 60)
    print("测试 4: 空摘要块")
    print("=" * 60)

    llm = create_llm(model="google/gemini-2.5-flash")
    block = RollingSummaryBlock(llm=llm)

    result = await block._aget()
    print(f"空块 _aget 返回: '{result}'")

    assert result == "", "空块应返回空字符串"
    print("✅ 空摘要块测试通过")


async def main():
    print("\n🚀 开始测试 RollingSummaryBlock\n")

    try:
        # 测试首次压缩
        block = await test_first_compression()

        # 测试滚动更新
        block = await test_rolling_update(block)

        # 测试获取摘要
        await test_get_snapshot(block)

        # 测试空摘要块
        await test_empty_block()

        print("\n" + "=" * 60)
        print("✅ 所有 RollingSummaryBlock 测试通过!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
