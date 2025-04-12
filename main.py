import os, sys
import asyncio
from collections import defaultdict

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import torch
import numpy as np

from RAGDeepSeek.utils import Config, MultiQaItem
from RAGDeepSeek.rag import load_RAG, query_RAG


# rag实例字典, rag实例锁
rag_instances, rag_instances_lock = dict(), defaultdict(asyncio.Lock)
user, user_instance_lock = dict(), defaultdict(asyncio.Lock)
app = FastAPI()


class nlp_Item(BaseModel):
    question: str


class clf_asr_Item(BaseModel):
    audioname: str


class tts_Item(BaseModel):
    text: str
    spk_id: str
    speed: str
    volume: str


class Items(BaseModel):
    companyid: str  # 公司id
    taskid: str  # 任务id
    customerphone: str  # 用户电话号码
    value: str  # 选择路线
    question: str  # 问题
    isall: bool  # 是否为音频输入
    spk_id: str  # 发音人id
    speed: str  # 语速
    volume: str  # 音量
    file_list: list  # 音频文件列表
    audio_stream: str  # 音频流
    counter: str  # 对话轮次计数器


@app.post("/nlp/multiQA/")
async def multi_qa(item: MultiQaItem):
    """
    基于Rag+deepseek实现语义匹配（实现客户提问的问题答案匹配）
    """
    # 测试
    if item.query == "":
        return {
            "sucess": True,
            "model_response": "TEST",
            "error_message": "",
        }

    # RAG实例, 和用户对话历史记录实例
    global rag_instances, rag_instances_lock
    global user, user_instance_lock

    # 根据userid动态创建Config实例
    async with user_instance_lock[item.userid]:
        user[item.userid] = Config(
            working_dir=os.path.join("knowledge_base", item.companyid),
            language_model="qwen-plus",
            loading_method="api",
            companyid=item.companyid,
            max_answer_length=item.max_answer_length,
            query_mode="local",
        )

    # 根据companyid动态创建RAG实例
    async with rag_instances_lock[item.companyid]:
        try:
            rag_instances[item.companyid] = await load_RAG(user[item.userid])
        except Exception as e:
            return {
                "success": False,
                "model_response": "",
                "error_message": "Failed to instantiate RAG\n" + str(e),
            }

    # 如果对应公司的外部知识库没有被建模, 则返回错误信息
    if not os.path.exists(os.path.join(user[item.userid].working_dir, "kv_store_doc_status.json")):
        return {
            "success": False,
            "model_response": "",
            "error_message": "Failed to load external knowledge database",
        }

    # 使用rag进行query
    try:
        model_response = await query_RAG(
            rag_instances[item.companyid],
            item.query,
            user[item.userid],
            history_messages=user[item.userid].history,
        )
    except Exception as e:
        return {
            "success": False,
            "model_response": "",
            "error_message": "Failed to query RAG\n" + str(e),
        }

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
    return {
        "success": True,
        "model_response": model_response,
        "error_message": "",
    }


if __name__ == '__main__':
    uvicorn.run(app='main_bert_base:app', host='127.0.0.1', port=8080, reload=True, workers=3)