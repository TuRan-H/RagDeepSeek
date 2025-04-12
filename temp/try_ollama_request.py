import aiohttp
import json


async def generate_text_with_ollama(
    prompt,
    model="llama3.1",
    num_predict=100,
    history=None,
):
    """
    使用Ollama API生成文本

    Args:
        prompt: 用户的query
        model (str): 使用的模型名称
        num_predict (int): 最多生成的token
        history (list): 历史消息
    """

    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}

    request_data = {
        "model": model,
        "messages": [],  # ! 这里请求体中message是复数messages
        "stream": False,
        "num_predict": num_predict,  # ! 使用http的response响应方式, 能够指定max_tokens
    }

    if history:
        request_data["messages"] = history.copy()
    request_data['messages'].append({"role": "user", "content": prompt})

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=request_data) as response:
                if response.status != 200:
                    response.raise_for_status()

                response_text = await response.text()
                response_result = json.loads(response_text)
                response_result = response_result['message']['content']
    except aiohttp.ClientError as e:
        raise RuntimeError(f"api请求失败: {e}")

    return response_result


if __name__ == "__main__":

    async def main():
        data = await generate_text_with_ollama("你好", model="llama3.1", num_predict=100)
        print(data)

    # 需要运行异步函数的方式也需要改变
    import asyncio

    asyncio.run(main())
