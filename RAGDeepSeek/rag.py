import asyncio
import re
from pathlib import Path

from lightrag.lightrag import LightRAG, QueryParam
from lightrag.utils import setup_logger, always_get_an_event_loop
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.base import StoragesStatus

from RAGDeepSeek.llm import load_model_hf, load_model_api_ollama, ollama_request_model_generate
from RAGDeepSeek.utils import MultiQAConfig
from RAGDeepSeek.prompt import SYSTEMMESSAGE

from lightrag.prompt import PROMPTS
from RAGDeepSeek.prompt import KEYWORDS_EXTRACTION

PROMPTS['keywords_extraction'] = KEYWORDS_EXTRACTION


async def load_RAG(config: MultiQAConfig):
    """
    加载RAG模型

    Args:
            config (Config): 配置文件

    Returns:
            LightRAG: RAG实例
    """
    setup_logger("lightrag", log_file_path=config.log_file_path, level="INFO")

    if config.loading_method == "hf":
        llm_model_func, embedding_func = load_model_hf(config)
    else:
        llm_model_func, embedding_func = load_model_api_ollama(config)

    rag = LightRAG(
        working_dir=config.working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=config.language_model,
        embedding_func=embedding_func,
        addon_params={"language": config.responsed_language},
        enable_llm_cache=False,
    )  # type: ignore

    # Error mesaage when doing an insert: "del pipeline_status["history_messages"][:] KeyError: 'history_messages'
    # https://github.com/HKUDS/LightRAG/issues/1104
    await initialize_pipeline_status()

    if rag._storages_status != StoragesStatus.INITIALIZED:
        await rag.initialize_storages()

    return rag


async def index_documents(rag: LightRAG, documents: list):
    """
    索引文档, 根据外部文档内容构建本地知识图谱

    Args:
            rag (LightRAG): RAG实例
            documents (list): 文档列表
    """
    tasks = []
    for doc in documents:
        with open(doc, "r", encoding="utf-8") as f:
            content = f.read()
        task = asyncio.create_task(rag.ainsert(content, file_paths=str(Path(doc).absolute())))
        tasks.append(task)

    await asyncio.gather(*tasks)


async def query_RAG(rag: LightRAG, query: str, config: MultiQAConfig, history_messages: list = []):
    """
    使用RAG处理query, 返回结果

    Args:
        rag (LightRAG): RAG实例
        query_method (str): 查询方法, `local`, `global` 和 `hybrid`
        config (Config): 配置文件
    """
    rag_response = await rag.aquery(
        query=query,
        param=QueryParam(
            mode=config.query_mode, only_need_context=not config.rag_process_context_with_llm,
            conversation_history=config.history
        ),
    )

    if rag_response != "":
        match_result = re.search(
            pattern=r"```\n\s\s\s\s-----Sources-----\n.+```", string=rag_response, flags=re.DOTALL
        )  # type: ignore
        if match_result:
            rag_response = rag_response[: match_result.start()] + rag_response[match_result.end() :]  # type: ignore

    system_prompt = SYSTEMMESSAGE.format(
        RAG_response=rag_response,
        max_answer_length=config.max_answer_length,
        companyid=config.companyid,
    )

    # 不同的loading_method会从llm_model_func中导入不同的处理函数
    if config.loading_method in ["hf", "api"]:
        model_response = await rag.llm_model_func(
            prompt=query,
            system_prompt=system_prompt,
            max_answer_length=config.max_answer_length,
            history_messages=history_messages,
        )  # type: ignore
    # 当使用ollama调用方式时, 需要使用request方式进行调用
    # 只有request的时候, 可以使用num_predict来控制生成的长度
    else:
        model_response = await ollama_request_model_generate(
            prompt=query,
            model=config.language_model,
            system_prompt=system_prompt,
            num_predict=config.max_answer_length,
            history=history_messages,
        )

    return model_response


if __name__ == "__main__":
    # # 测试hf
    # config = Config(
    #     loading_method="hf",
    #     language_model="./models/DeepSeek-R1-Distill-Llama-8B",
    #     working_dir="./knowledge_base/meadjohnson",
    # )

    # 测试ollama
    config = MultiQAConfig(
        working_dir="./knowledge_base/meadjohnson",
        loading_method="ollama",
        companyid="meadjohnson",
        language_model="gemma3:27b",
        embedding_model="/mnt/sdb/TuRan/Downloads/models/all-MiniLM-L6-v2",
        query_mode="local",
        max_answer_length=-1,
    )

    # # 测试api
    # config = Config(
    #     working_dir="./knowledge_base/meadjohnson",
    #     loading_method="api",
    #     companyid="meadjohnson",
    #     language_model="qwen-plus",
    #     embedding_model="/mnt/sdb/TuRan/Downloads/models/all-MiniLM-L6-v2",
    #     rag_process_context_with_llm=False,
    #     query_mode="local",
    #     max_answer_length=100,
    # )

    # 运行协程
    async def main():
        rag = await load_RAG(config)  # type: ignore
        # await index_documents(rag, ["./data/meadjohnson.txt"])
        result = await query_RAG(
            rag,
            "我想买一罐奶粉, 有什么推荐的吗?",
            config=config,
        )
        return result

    loop = always_get_an_event_loop()
    result = loop.run_until_complete(main())
    print(result)
