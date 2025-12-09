"""RAG 查询速度测试 - 真实数据"""

import os
import sys
import json
import time
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

os.environ["OPENROUTER_API_KEY"] = "your-api-key-here"

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from compress_memory.models import create_embedding
from compress_memory.blocks.fine_grained_vector import FineGrainedVectorBlock


def load_real_data(max_messages=150):
    """加载真实对话数据"""
    file_path = os.path.join(PROJECT_ROOT, "long_session", "019a5835-ee72-7f63-ae7a-75d026d2291d.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages = []
    for msg in data['data']['sessions'][0]['messages'][:max_messages]:
        role_str = msg['message']['role']
        role = MessageRole.USER if role_str == 'user' else MessageRole.ASSISTANT
        content = msg['message']['content']['text']
        if content:
            messages.append(ChatMessage(role=role, content=content))
    return messages


async def main():
    print("=" * 60)
    print("RAG 查询速度测试（真实数据）")
    print("=" * 60)

    # 初始化
    embed_model = create_embedding()
    client = chromadb.Client()
    collection = client.create_collection("speed_test_real", metadata={"hnsw:space": "cosine"})
    vector_store = ChromaVectorStore(chroma_collection=collection)

    vector_block = FineGrainedVectorBlock(
        vector_store=vector_store,
        embed_model=embed_model,
        window_size=3,
        stride=2,
        similarity_top_k=5,
    )

    # 加载真实数据
    print("\n1. 加载真实数据...")
    messages = load_real_data(150)
    print(f"   消息数: {len(messages)}")

    # 写入
    print("\n2. 写入向量库...")
    start = time.time()
    await vector_block._aput(messages)
    write_time = time.time() - start

    stats = vector_block.get_stats()
    print(f"   写入耗时: {write_time:.2f}s")
    print(f"   节点数: {stats['node_count']}")

    # 查询测试
    print("\n3. 查询速度测试...")
    queries = [
        "What error did the user encounter with Ollama?",
        "How to fix the modelfile FROM directive issue?",
        "What commands were suggested for debugging?",
        "What GPU or memory issues were discussed?",
        "What model names or versions were mentioned?",
    ]

    print(f"\n   {'查询':<45} {'Embed':<10} {'Search':<10} {'Total':<10}")
    print("   " + "-" * 75)

    total_embed = 0
    total_search = 0

    for query in queries:
        # Embedding
        start = time.time()
        query_embedding = await embed_model.aget_query_embedding(query)
        embed_time = time.time() - start

        # 检索
        start = time.time()
        vq = VectorStoreQuery(query_str=query, query_embedding=query_embedding, similarity_top_k=5)
        await vector_store.aquery(vq)
        search_time = time.time() - start

        total_embed += embed_time
        total_search += search_time

        print(f"   {query:<45} {embed_time*1000:>6.0f}ms   {search_time*1000:>6.0f}ms   {(embed_time+search_time)*1000:>6.0f}ms")

    n = len(queries)
    print("   " + "-" * 75)
    print(f"   {'平均':<45} {total_embed/n*1000:>6.0f}ms   {total_search/n*1000:>6.0f}ms   {(total_embed+total_search)/n*1000:>6.0f}ms")

    print("\n" + "=" * 60)
    print(f"结论：Embedding {total_embed/n*1000:.0f}ms + 检索 {total_search/n*1000:.0f}ms = 总计 {(total_embed+total_search)/n*1000:.0f}ms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
