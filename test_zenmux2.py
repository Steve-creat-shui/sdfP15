import asyncio
import httpx
import os
from openai import AsyncOpenAI

async def test():
    # 获取所有模型，筛选图像模型
    print("=== 获取所有模型，筛选图像模型 ===")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get("https://zenmux.ai/api/v1/models")
            data = resp.json()
            all_models = data.get("data", [])
            print(f"总模型数: {len(all_models)}")
            
            # 筛选图像模型（output_modalities 包含 image）
            image_models = []
            for m in all_models:
                out_mod = m.get("output_modalities", [])
                cap = m.get("capabilities", {})
                if "image" in out_mod or "image_generation" in str(cap).lower() or "image" in str(cap).lower():
                    image_models.append(m)
            
            print(f"\n图像模型（output contains image）: {len(image_models)} 个")
            for m in image_models:
                print(f"  - {m['id']} | output: {m.get('output_modalities')} | cap: {m.get('capabilities')}")
            
            # 也搜索包含 image 的模型 id
            print("\n=== 模型名包含 image/gpt-image/dall/flux 关键词 ===")
            for m in all_models:
                mid = m["id"].lower()
                if any(k in mid for k in ["image", "dall", "flux", "midjourney", "stable", "sdxl"]):
                    print(f"  - {m['id']} | input: {m.get('input_modalities')} | output: {m.get('output_modalities')}")
    except Exception as e:
        print(f"请求失败: {type(e).__name__}: {e}")

asyncio.run(test())
