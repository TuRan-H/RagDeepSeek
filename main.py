import os, sys
import asyncio
from collections import defaultdict

import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM

from RAGDeepSeek.utils import (
    IntentRecognitionConfig,
    IntentRecognitionItem,
    IntentRecognitionResult,
    MultiQAConfig,
    MultiQAItem,
    MultiQAResult,
    MultiIntentDetectionItem,
    MultiIntentDetectionConfig,
)
from RAGDeepSeek.intent_detection import MultiIntentDetection
from RAGDeepSeek.rag import load_RAG, query_RAG
from RAGDeepSeek.SFTDeepSeek import inference


# rag实例字典, rag实例锁
rag_instances, rag_instances_lock = dict(), defaultdict(asyncio.Lock)
user, user_instance_lock = dict(), defaultdict(asyncio.Lock)
app = FastAPI()


@app.post("/nlp/multiQA/")
async def multi_qa(item: MultiQAItem):
    """
    基于Rag+deepseek实现语义匹配（实现客户提问的问题答案匹配）
    """
    # 测试
    if item.query == "":
        return MultiQAResult(
            success=True,
            model_response="TEST",
            error_message="",
        )

    # RAG实例, 和用户对话历史记录实例
    global rag_instances, rag_instances_lock
    global user, user_instance_lock

    # 根据userid动态创建Config实例
    async with user_instance_lock[item.userid]:
        user[item.userid] = MultiQAConfig(
            working_dir=os.path.join("knowledge_base", item.companyid),
            loading_method="ollama",
            language_model="gemma3:27b",
            embedding_model="/mnt/sdb/TuRan/Downloads/models/all-MiniLM-L6-v2",
            companyid=item.companyid,
            query_mode="local",
            max_answer_length=item.max_answer_length,
            # rag_process_context_with_llm=False,
        )

    # 根据companyid动态创建RAG实例
    async with rag_instances_lock[item.companyid]:
        try:
            rag_instances[item.companyid] = await load_RAG(user[item.userid])
        except Exception as e:
            return MultiQAResult(
                success=False,
                model_response="",
                error_message="Failed to instantiate RAG\n" + str(e),
            )

    # 如果对应公司的外部知识库没有被建模, 则返回错误信息
    if not os.path.exists(os.path.join(user[item.userid].working_dir, "kv_store_doc_status.json")):
        return MultiQAResult(
            success=False,
            model_response="",
            error_message="Failed to load external knowledge database",
        )

    # 使用rag进行query
    try:
        model_response = await query_RAG(
            rag_instances[item.companyid],
            item.query,
            user[item.userid],
            history_messages=user[item.userid].history,
        )
    except Exception as e:
        return MultiQAResult(
            success=False,
            model_response="",
            error_message="Failed to query RAG\n" + str(e),
        )

    # 添加历史信息
    async with user_instance_lock[item.userid]:
        user[item.userid].history.append(
            {
                "role": "user",
                "content": item.query,
            }
        )
        user[item.userid].history.append(
            {
                "role": "assistant",
                "content": model_response,
            }
        )
    return MultiQAResult(
        success=True,
        model_response=model_response,
        error_message="",
    )


@app.post("/nlp/multiQAIntent/")
async def multi_intent_detection(item: MultiIntentDetectionItem):
    config = MultiIntentDetectionConfig(
        loading_method="ollama",
        language_model="deepseek-r1:14b",
        hyper_alpha=0.3,
        hyper_beta=0.7,
    )
    intent_detection = MultiIntentDetection(config)
    return intent_detection.multi_predict(file_name=item.filename)


@app.on_event("startup")
async def intent_recognition_start_up():
    """
    针对 `intent_recognition` 的startup事件, 在应用启动时加载模型
    """
    # ! 注意修改模型地址, max_length保持256
    # ! 默认有GPU使用GPU, 没有GPU使用CPU. 如果想使用GPU则指定 device_map="auto" (auto代表模型并行, 即模型的不同部分被分片到不同的GPU上)
    app.state.intent_recognition_config = IntentRecognitionConfig(
        model_path="./results/lr_0.001_epoch_3_model_Llama-3.1-8B-Instruct_time_20_30_38",
        max_length=256,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    app.state.intent_recognition_model = AutoModelForCausalLM.from_pretrained(
        app.state.intent_recognition_config.model_path,
        device_map=app.state.intent_recognition_config.device_map,
    )
    app.state.intent_recognition_tokenizer = AutoTokenizer.from_pretrained(
        app.state.intent_recognition_config.model_path
    )


@app.post("/nlp/IntentRecognition/")
async def intent_recognition(item: IntentRecognitionItem):

    try:
        intent = inference(
            query=item.query,
            model_path=app.state.intent_recognition_config.model_path,
            max_length=app.state.intent_recognition_config.max_length,
            model=app.state.intent_recognition_model,
            tokenizer=app.state.intent_recognition_tokenizer,
        )
    except Exception as e:
        return IntentRecognitionResult(
            success=False,
            intent="",
        )

    if not intent:
        return IntentRecognitionResult(
            success=False,
            intent="",
        )

    return IntentRecognitionResult(success=True, intent=intent)


if __name__ == '__main__':
    uvicorn.run(app='main:app', host='127.0.0.1', port=8080, reload=True, workers=3)
