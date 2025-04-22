import re
import asyncio
import aiohttp
import json

from transformers import AutoTokenizer, AutoModel
from lightrag.llm.hf import hf_embed, initialize_hf_model
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, locate_json_string_body_from_string

from RAGDeepSeek.utils import MultiQAConfig, delete_deepseek_thinking


async def ollama_request_model_generate(
    prompt: str,
    model: str,
    system_prompt: str = None,
    num_predict: int = 100,
    history=None,
):
    """
    使用Ollama API生成文本

    Args:
        prompt: 用户的query
        model (str): 使用的模型名称
        system_prompt (str): 系统提示词
        num_predict (int): 最多生成的token
        history (list): 历史消息
    """

    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}

    request_data = {
        "model": model,
        "messages": [],  # ! 注意typo问题, 这里请求体中message是复数messages
        "stream": False,
        "options": {"num_predict": num_predict},
    }

    if system_prompt:
        request_data["messages"].append({"role": "system", "content": system_prompt})
    if history:
        request_data["messages"].extend(history)
    request_data['messages'].append({"role": "user", "content": prompt})

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=request_data) as response:
                if response.status != 200:
                    response.raise_for_status()

                response_result = await response.text()
                response_result = json.loads(response_result)
                response_text = response_result['message']['content']
    except aiohttp.ClientError as e:
        raise RuntimeError(f"api请求失败: {e}")

    if "deepseek" in model:
        response_text = delete_deepseek_thinking(response_text)

    return response_text


async def api_ollama_model_generate(**kwargs):
    """
    包装openai_complete_if_cache函数,
    处理DeepSeek模型的输出格式
    """
    model_name = kwargs.get("model", "")
    result = await openai_complete_if_cache(**kwargs)

    # 删除DeepSeek中深度思考的内容, 仅保存正文
    if "deepseek" in model_name.lower():
        result = delete_deepseek_thinking(result)

    return result


async def hf_model_generate(
    model: str, prompt: str, system_prompt: str | None = None, history_message: list = [], **kwargs
):
    """
    使用huggingface模型进行文本生成

    Args:
        model (str): 模型名称
        prompt (str): 用户输入的prompt
        system_prompt (str | None): 系统提示词
        history_message (list): 历史消息列表
        kwargs: 其他参数
    """
    # 使用lru_cache初始化模型, 避免重复实例化
    hf_model, hf_tokenizer = initialize_hf_model(model)
    # "Setting `pad_token_id` to `eos_token_id` for open-end generation."
    # https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
    hf_model.generation_config.pad_token_id = hf_tokenizer.pad_token_id

    # 构造输入
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_message)
    messages.append({"role": "user", "content": prompt})
    input_prompt = hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 模型inference
    # input_ids = hf_tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True).to(
    #     "cuda"
    # )
    input_ids = hf_tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = {k: v.to(hf_model.device) for k, v in input_ids.items()}
    output = hf_model.generate(
        **input_ids,
        max_new_tokens=kwargs.get("max_answer_length", 2048),
        num_beams=3,
        num_return_sequences=1,
        early_stopping=True,
    )
    response_text = hf_tokenizer.decode(
        output[0][len(input_ids["input_ids"][0]) :], skip_special_tokens=True
    )

    # 删除DeepSeek中深度思考的内容, 仅保存正文
    if "deepseek" in model.lower():
        response_text = delete_deepseek_thinking(response_text)

    response_text = response_text.strip()

    return response_text


def load_model_hf(config: MultiQAConfig):
    """
    使用huffingface方式本地加载模型语言模型

    Args:
        config (Config): 配置文件

    Returns:
        (语言模型调用函数, 嵌入模型调用函数)
    """
    emb_tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
    emb_model = AutoModel.from_pretrained(config.embedding_model)

    embedding_func = EmbeddingFunc(
        embedding_dim=config.embedding_dim,
        max_token_size=config.max_token_size,
        func=lambda texts: hf_embed(
            texts=texts,
            tokenizer=emb_tokenizer,
            embed_model=emb_model,
        ),
    )

    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ):
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if kwargs.get("hashing_kv") is not None:
            model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
        else:
            model_name = config.language_model

        # 调用hf_model_generate进行文本生成
        result = await hf_model_generate(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )  # type: ignore

        if keyword_extraction:
            return locate_json_string_body_from_string(result)
        return result

    return llm_model_func, embedding_func


def load_model_api_ollama(config: MultiQAConfig):
    """
    使用api加载模型语言模型

    Args:
        config (Config): 配置文件

    Returns:
        (语言模型调用函数, 嵌入模型调用函数)
    """
    tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
    model = AutoModel.from_pretrained(config.embedding_model)

    async def llm_model_func(
        prompt: str,
        system_prompt: str = None,
        history_messages: list = [],
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        return await api_ollama_model_generate(
            model=config.language_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=config.api_key,
            base_url=config.base_url,
        )  # type: ignore

    embedding_func = EmbeddingFunc(
        embedding_dim=config.embedding_dim,
        max_token_size=config.max_token_size,
        func=lambda texts: hf_embed(
            texts=texts,
            tokenizer=tokenizer,
            embed_model=model,
        ),
    )

    return llm_model_func, embedding_func


if __name__ == "__main__":
    # # 测试hf模型
    # config = Config(
    #     loading_method="hf",
    #     language_model="./models/DeepSeek-R1-Distill-Llama-8B",
    #     working_dir="temp",
    # )
    # llm_model_func, _ = load_model_hf(config)

    # # 测试ollama模型
    # config = Config(
    #     working_dir="temp",
    #     loading_method="ollama",
    #     language_model="deepseek-r1:14b",
    # )
    # llm_model_func, _ = load_model_api_ollama(config)

    # # 测试单轮对话
    # async def test():
    #     task_1 = asyncio.create_task(llm_model_func("你是谁"))
    #     task_2 = asyncio.create_task(llm_model_func("你好"))
    #     task_3 = asyncio.create_task(llm_model_func("你好"))
    #     results = await asyncio.gather(task_1, task_2, task_3)
    #     print(results)
    #     return results

    # results = asyncio.run(test())

    # # 测试多轮对话
    # history_message = []
    # history_message.append({"role": "user", "content": "你好"})
    # history_message.append(
    #     {
    #         "role": "assistant",
    #         "content": "你好我是一个人工智能客服助手, 请问有什么问题吗?",
    #     }
    # )
    # async def test():
    #     task = asyncio.create_task(
    #         llm_model_func(prompt="你是谁?", history_messages=history_message)
    #     )
    #     return await task

    # print(asyncio.run(test()))

    # 测试ollma的request生成
    config = MultiQAConfig(
        working_dir="temp",
        loading_method="ollama",
        companyid="0",
        language_model="llama3.1:latest",
        embedding_model="0",
        max_token_size=5000,
    )
    from RAGDeepSeek.prompt import SYSTEMMESSAGE

    history_message = [
        {"role": "system", "content": SYSTEMMESSAGE.format(max_answer_length=100, RAG_response="")},
    ]

    async def test():
        task = asyncio.create_task(
            ollama_request_model_generate(
                prompt="你是谁?",
                model=config.language_model,
                num_predict=100,
                history=history_message,
            )
        )
        return await task

    print(asyncio.run(test()))
