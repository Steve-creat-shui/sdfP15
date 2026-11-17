import asyncio
import httpx
import os
from openai import AsyncOpenAI

api_key = os.environ.get("ZENMUX_API_KEY", "")
print(f"环境变量 ZENMUX_API_KEY: {'已设置' if api_key else '未设置'}")

async def test():
    # 测试1: 基础连通性
    print("\n=== 测试1: 基础连通性 ===")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://zenmux.ai/api/v1/models")
            print(f"HTTP 状态码: {resp.status_code}")
            print(f"响应前200字符: {resp.text[:200]}")
    except Exception as e:
        print(f"请求失败: {type(e).__name__}: {e}")

    # 测试2: OpenAI 兼容客户端（不调用生成，只测 models）
    print("\n=== 测试2: OpenAI 兼容客户端 ===")
    test_key = api_key or "test_key"
    try:
        client = AsyncOpenAI(
            api_key=test_key,
            base_url="https://zenmux.ai/api/v1",
            timeout=httpx.Timeout(30.0),
        )
        try:
            models = await client.models.list()
            print(f"模型列表获取成功，共 {len(models.data)} 个模型:")
            for m in models.data[:10]:
                print(f"  - {m.id}")
        except Exception as e:
            print(f"获取模型列表失败: {type(e).__name__}: {str(e)[:200]}")
    except Exception as e:
        print(f"初始化客户端失败: {e}")

asyncio.run(test())
