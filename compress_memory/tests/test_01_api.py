"""
测试 1：验证 OpenRouter API 连接
- 测试 LLM 调用
- 测试 Embedding 调用
"""

import os
import sys
import asyncio

# 设置路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# 设置 API Key
os.environ["OPENROUTER_API_KEY"] = "your-api-key-here"

from compress_memory.models import create_llm, create_embedding


def test_llm():
    """测试 LLM 调用"""
    print("=" * 50)
    print("测试 LLM (google/gemini-2.5-flash)")
    print("=" * 50)

    llm = create_llm(model="google/gemini-2.5-flash")

    # 同步调用
    response = llm.complete("请用一句话介绍你自己")
    print(f"Response: {response.text[:200]}...")
    print(f"✅ LLM 测试通过")
    return True


def test_embedding():
    """测试 Embedding 调用"""
    print("\n" + "=" * 50)
    print("测试 Embedding (qwen/qwen3-embedding-8b)")
    print("=" * 50)

    embed_model = create_embedding(model="qwen/qwen3-embedding-8b")

    # 同步调用
    text = "这是一个测试文本"
    embedding = embed_model.get_text_embedding(text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Embedding[:5]: {embedding[:5]}")
    print(f"✅ Embedding 测试通过")

    return len(embedding)


async def test_async():
    """测试异步调用"""
    print("\n" + "=" * 50)
    print("测试异步调用")
    print("=" * 50)

    llm = create_llm(model="google/gemini-2.5-flash")
    embed_model = create_embedding(model="qwen/qwen3-embedding-8b")

    # 并行异步调用
    llm_task = llm.acomplete("说'异步测试成功'")
    embed_task = embed_model.aget_text_embedding("异步测试文本")

    llm_response, embedding = await asyncio.gather(llm_task, embed_task)

    print(f"LLM async response: {llm_response.text[:100]}...")
    print(f"Embedding async dimension: {len(embedding)}")
    print(f"✅ 异步测试通过")


def main():
    print("\n🚀 开始测试 OpenRouter API 连接\n")

    try:
        # 测试 LLM
        test_llm()

        # 测试 Embedding
        embed_dim = test_embedding()

        # 测试异步
        asyncio.run(test_async())

        print("\n" + "=" * 50)
        print("✅ 所有测试通过!")
        print(f"📊 Embedding 维度: {embed_dim}")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
