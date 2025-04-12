import sys
sys.path.append(".")
# from metric import mean_reciprocal_rank,mean_average_precision

# from flask import Flask, render_template, session, request, redirect, url_for, jsonify

import asyncio
from typing import Union, Callable
# from ASRApi import predict_audio, recording, filenameRandomizing, save_to_db, interruption_judging
from fastapi import FastAPI, Form, Request, Query, UploadFile, File, Depends
from fastapi.responses import PlainTextResponse,HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import shutil
from pathlib import Path
from collections import defaultdict

import argparse
import glob
import json
import logging
import logging.handlers
import os
import random
# from ASRApi import ASRapi
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
# from paddlespeech.server.bin.paddlespeech_client import TTSOnlineClientExecutor
from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    BertModel,
    BertTokenizer,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from transformers.data.processors.utils import InputExample, DataProcessor

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

import code
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
# from sklearn_Classification.clf_model import CLFModel
# from bert_intent_recognition.app import BertIntentModel    #nlp方法意图识别


# from paddlespeech.server.bin.paddlespeech_client import TTSClientExecutor
# from playsound import playsound

# from replace_word import tihuan
# from synthesis_record import bohao

# from scipy import fftpack
# import pyaudio
import wave
import time
import threading
# from merge import merge_wav
import base64

from RAGDeepSeek.utils import Config, MultiQaItem
from RAGDeepSeek.rag import load_RAG, query_RAG


# CHUNK = 1024  # 块大小
# FORMAT = pyaudio.paInt16  # 每次采集的位数
# CHANNELS = 1  # 声道数
# RATE = 16000  # 采样率：每秒采集数据的次数
# p = pyaudio.PyAudio()
# stream_2 = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
# frames = []
# threshold=12000
# stopflag = 0
# stopflag2 = 0
# oneSecond = int(RATE / CHUNK)
file_list=[]

# global variables
flag = True      # track the value of the flag

# # define a function to toggle the flag based on user input
# def toggle_flag():
#     global flag
#     while True: # loop until the program ends
#         data = stream_2.read(CHUNK)
#         rt_data = np.frombuffer(data, np.dtype('<i2'))
#         # print(rt_data*10)
#         # 傅里叶变换
#         fft_temp_data = fftpack.fft(rt_data, rt_data.size, overwrite_x=True)
#         fft_data = np.abs(fft_temp_data)[0:fft_temp_data.size // 2 + 1]
#
#         # 测试阈值，输出值用来判断阈值
#         # print(sum(fft_data) // len(fft_data))
#
#         # 判断麦克风是否停止，判断说话是否结束，# 麦克风阈值，默认7000
#         if sum(fft_data) // len(fft_data) > threshold:
#             flag= False
#             # print("flag======"+str(flag))
#             # print("* 对话被打断")


# nlp意图识别用
# BIM_model=BertIntentModel()
# 融合方法意图识别
# clf_model = CLFModel('./sklearn_Classification/model_file/')

# rag实例字典, rag实例锁
rag_instances, rag_instances_lock = dict(), defaultdict(asyncio.Lock)
user, user_instance_lock = dict(), defaultdict(asyncio.Lock)

logger = logging.getLogger(__name__)
def set_log(logger):
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename="BERT.log")
    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.INFO)
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)
    logger.addHandler(handler1)


MODEL_CLASS = {"bert":(BertConfig,BertModel,BertTokenizer)}
# 设置训练种子
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
# 定义问答类
class FAQProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_candidates(self,file_dir):
        train_df = pd.read_csv(file_dir,sep='\t')
        self.candidate_title = train_df['best_title'].tolist()
        self.candidate_reply = train_df["reply"].tolist()
        return self._create_examples(self.candidate_title,"train")

    def _create_examples(self,lines,set_type):
        examples = []
        for (i,line) in enumerate(lines):
            guid = "%s-%s" % (set_type,i)
            examples.append(InputExample(guid=guid,text_a=line,text_b=None,label=1))
        return examples
# 验证问题
def evaluate(args,model,dataset):
    results = []
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1,args.n_gpu)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,sampler=sampler,batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    for batch in tqdm(dataloader,desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids":batch[0],"attention_mask":batch[1],"token_type_ids":batch[2]}
            outputs = model(**inputs)
            sequence_output,pooled_output = outputs[:2]
            if args.output_type == "pooled":
                results.append(pooled_output)
            elif args.output_type == "avg":
                results.append(sequence_output.mean(1))
    # list of tensor
    # tensor : [batch,outputsize]
    return results

# 下面两个函数用于获取训练数据
# data_dir="right_samples.csv"
data_dir="test.txt"
def load_examples(args,tokenizer):
    """获取原数据库里的问题features"""
    processor = FAQProcessor()
    #examples = processor.get_candidates(args.data_dir)
    examples=processor.get_candidates("./data/"+data_dir)
    print(data_dir)
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=[1],
        output_mode="classification",
        max_length=args.max_seq_length,
        pad_on_left=bool(args.model_type in ["xlnet"]),
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )
    all_input_ids = torch.tensor([f.input_ids for f in features],dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features],dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features],dtype=torch.long)
    dataset = TensorDataset(all_input_ids,all_attention_mask,all_token_type_ids)
    return dataset,processor.candidate_title,processor.candidate_reply

def from_examples2dataset(args,examples,tokenizer):
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=[1],
        output_mode="classification",
        max_length=args.max_seq_length,
        pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids)
    return dataset


# nlp服务开启以及七条线路主要运行函数，内部可选七条线路。
# 参数传入value：路线选择；question：问题输入（主要针对文本问答）；ifall：是否为语音输入；spk_id：语音合成线路；file_list：音频文件列表（用以三号线路融合）
# 输出为反馈文本与问题，三号线路附加输入音频文件名与输出音频文件名输出
def main(
    value,
    question="",
    ifall=True,
    spk_id="0",
    file_list=[],
    customerphone="12333333333",
    audio_data="./work/test.wav",
    speed=1.0,
    volume=1.0,
):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="./data/right_samples.csv",
        type=str,
        required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--evaluate_dir",
        default="./data/eval_touzi.xlsx",
        type=str,
        required=False,
        help="The evaluate data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        required=False,
        help="Model type selected in the list:",
    )
    parser.add_argument(
        "--model_name_or_path",
        #default='D:\\NLP\\my-wholes-models\\chinese_wwm_pytorch\\',
        default='./models/chines-wmm-original',
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--output_type",
        default="avg",
        type=str,
        required=False,
        choices=["pooled", "avg"],
        help="the type of choice output vector",
    )

    parser.add_argument(
        "--task_name",
        default="faq",
        type=str,
        required=False,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_predict",default=True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--do_lower_case",default=True, action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    # Setup logging
    # set_log(logger)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        filename="BERT.log",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args)
    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASS[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        finetuning_task = args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else "./cache"
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else "./cache")
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else "./cache"
    )
    model = model.to(device)

    if not os.path.exists("embeddings.pkl"):
        eval_dataset, candidate_title, candidate_reply = load_examples(args, tokenizer)
        outputs = evaluate(args, model, eval_dataset)
        candidate_embeddings = torch.cat([o.cpu().data for o in outputs]).numpy()
        torch.save([candidate_title,candidate_reply,candidate_embeddings],"embeddings.pkl")
    else:
        candidate_title, candidate_reply, candidate_embeddings = torch.load("embeddings.pkl")
        print("加载模型")

    if question=="p":
        if file_list != []:
            output_wav = filenameRandomizing(True) + ".wav"
            output_wav = "./work/" + output_wav
            merge_wav(file_list,output_wav)
            print(output_wav)
            file_list.clear()
        end="kill"
        return end,question,
    ttsclient_executor = TTSClientExecutor()
    # tts_executor = TTSExecutor()
    counter = 0
    phone_id="12333333333"
    if value=="1":
        text,audioname = ASRapi(if_punc=True,audioname=audio_data)
        print(text)
        return text
    elif value=="2":
        while True:
            if ifall:
                question,audioname = ASRapi(False,audio_data)
            nlp_result=""
            quest_return=""
            quest_return=question
            counter = counter + 1
            if len(str(question)) == 0:
                nlp_result = "请给出需要咨询的问题"
                print(nlp_result)
            else:
                if clf_model.predict(question)=='greet':
                    nlp_result = "你好呀"
                    print(nlp_result)
                elif clf_model.predict(question)=='accept':
                    nlp_result="用户意图为肯定"
                    print(nlp_result)
                elif clf_model.predict(question)=='deny':
                    nlp_result="用户意图为否定"
                    print(nlp_result)
                else:
                    examples = [InputExample(guid=0, text_a=question, text_b=None, label=1)]
                    dataset = from_examples2dataset(args,examples,tokenizer)
                    outputs = evaluate(args, model, dataset)

                    question_embedding = torch.cat([o.cpu().data for o in outputs]).numpy()
                    scores = cosine_similarity(question_embedding, candidate_embeddings)[0]
                    top1 = scores.argsort()[-1:][::-1]
                    for index in top1:
                        # nlp_result=("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))+nlp_result
                        # print("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))
                        if scores[index]>=0.92:
                            nlp_result = ("{}".format(candidate_reply[index])) + nlp_result
                            nlp_result = tihuan(nlp_result, phone_id)
                            print("{}".format(candidate_reply[index]))
                            print(nlp_result)
                        else:
                            nlp_result="对不起，我还不清楚您的问题"
                            print(nlp_result)
            return nlp_result,quest_return
    elif value=="3":
        # create a thread to run the toggle_flag function
        # t = threading.Thread(target=toggle_flag)
        # t.start()  # start the thread
        while True:
            if ifall:
                question,audioname = ASRapi(False,audio_data)
                audioname = str(audioname)
                file_list.append(audioname)
                print(question)

            nlp_result = ""
            quest_return=""
            quest_return=question
            counter = counter + 1
            if len(str(question)) == 0:
                nlp_result = "请给出需要咨询的问题"
                print(nlp_result)
            else:
                if clf_model.predict(question) == 'greet':
                    nlp_result = "你好呀你你你你你你你你你牛牛牛牛牛牛牛牛牛牛牛牛牛牛牛牛牛牛牛你你你你你你你你你"
                    print(nlp_result)
                elif clf_model.predict(question) == 'accept':
                    nlp_result = "用户意图为肯定"
                    print(nlp_result)
                elif clf_model.predict(question) == 'deny':
                    nlp_result = "用户意图为否定"
                    print(nlp_result)
                else:
                    examples = [InputExample(guid=0, text_a=question, text_b=None, label=1)]
                    dataset = from_examples2dataset(args, examples, tokenizer)
                    outputs = evaluate(args, model, dataset)

                    question_embedding = torch.cat([o.cpu().data for o in outputs]).numpy()
                    scores = cosine_similarity(question_embedding, candidate_embeddings)[0]
                    top1 = scores.argsort()[-1:][::-1]
                    for index in top1:
                        # nlp_result=("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))+nlp_result
                        # print("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))
                        if scores[index] >= 0.92:
                            nlp_result = ("{}".format(candidate_reply[index])) + nlp_result
                            nlp_result = tihuan(nlp_result, phone_id)
                            print("{}".format(candidate_reply[index]))
                            print(nlp_result)
                        else:
                            nlp_result = "对不起，我还不清楚您的问题"
                            print(nlp_result)
            output_name = filenameRandomizing(True) + ".wav"
            output_name = "./work/" + output_name
            file_list.append(output_name)
            res = ttsclient_executor(
                input=nlp_result,
                server_ip="127.0.0.1",
                port=8090,
                spk_id=int(spk_id),
                speed=speed,
                volume=volume,
                sample_rate=16000,
                output=output_name)
            # # you audio here
            # wf = wave.open(output_name, 'rb')
            #
            # # instantiate PyAudio
            # p = pyaudio.PyAudio()
            #
            # # define callback
            # def callback(in_data, frame_count, time_info, status):
            #     data = wf.readframes(frame_count)
            #     return (data, pyaudio.paContinue)
            #
            # # open stream using callback
            # stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
            #                 channels=wf.getnchannels(),
            #                 rate=wf.getframerate(),
            #                 output=True,
            #                 stream_callback=callback)
            #
            # # start the stream
            # stream.start_stream()
            # global flag
            # flag=True
            #
            # # main loop
            # while stream.is_active():
            #     print(flag)
            #     if not flag:  # if flag is False and audio is playing, pause it
            #         stream.stop_stream()
            #         flag = True
            #     time.sleep(0.1)
            #
            #     # stop stream
            # stream.stop_stream()
            # stream.close()
            # wf.close()
            #
            # # close PyAudio
            # p.terminate()
            return nlp_result,quest_return,audioname,output_name
    elif value=="4":
        # create a thread to run the toggle_flag function
        # t = threading.Thread(target=toggle_flag)
        # t.start()  # start the thread
        while True:
            if ifall:
                question,audioname = ASRapi(False,audio_data)
            nlp_result = ""
            quest_return = ""
            quest_return = question
            counter = counter + 1
            if len(str(question)) == 0:
                nlp_result = "请给出需要咨询的问题"
                print(nlp_result)
            else:
                if clf_model.predict(question) == 'greet':
                    nlp_result = "你好呀"
                    print(nlp_result)
                elif clf_model.predict(question) == 'accept':
                    nlp_result = "用户意图为肯定"
                    print(nlp_result)
                elif clf_model.predict(question) == 'deny':
                    nlp_result = "用户意图为否定"
                    print(nlp_result)
                else:
                    examples = [InputExample(guid=0, text_a=question, text_b=None, label=1)]
                    dataset = from_examples2dataset(args, examples, tokenizer)
                    outputs = evaluate(args, model, dataset)

                    question_embedding = torch.cat([o.cpu().data for o in outputs]).numpy()
                    scores = cosine_similarity(question_embedding, candidate_embeddings)[0]
                    top1 = scores.argsort()[-1:][::-1]
                    for index in top1:
                        # nlp_result=("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))+nlp_result
                        # print("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))
                        if scores[index] >= 0.92:
                            nlp_result = ("{}".format(candidate_reply[index])) + nlp_result
                            nlp_result = tihuan(nlp_result, phone_id)
                            print("{}".format(candidate_reply[index]))
                            print(nlp_result)
                        else:
                            nlp_result = "对不起，我还不清楚您的问题"
                            print(nlp_result)

            output_name = filenameRandomizing(True) + ".wav"
            output_name = "./work/" + output_name
            res = ttsclient_executor(
                input=nlp_result,
                server_ip="127.0.0.1",
                port=8090,
                spk_id=int(spk_id),
                speed=speed,
                volume=volume,
                sample_rate=16000,
                output=output_name)
            # playsound(output_name)
            return nlp_result,quest_return
    elif value=="5":
        text = question

        output_name = filenameRandomizing(True) + ".wav"
        output_name = "./work/" + output_name
        res = ttsclient_executor(
            input=text,
            server_ip="127.0.0.1",
            port=8090,
            spk_id=int(spk_id),
            speed=speed,
            volume=volume,
            sample_rate=16000,
            output=output_name)
        # playsound(output_name)
    elif value=="6":
        ifall=False
        while True:
            if ifall:
                question,audioname = ASRapi(False,audio_data)
            # else:
            #     question = input("请输入问题")
            nlp_result = ""
            quest_return = ""
            quest_return = question
            counter = counter + 1
            if len(str(question)) == 0:
                nlp_result = "请给出需要咨询的问题"
                print(nlp_result)
            else:
                if clf_model.predict(question) == 'greet':
                    nlp_result = "你好呀"
                    print(nlp_result)
                elif clf_model.predict(question) == 'accept':
                    nlp_result = "用户意图为肯定"
                    print(nlp_result)
                elif clf_model.predict(question) == 'deny':
                    nlp_result = "用户意图为否定"
                    print(nlp_result)
                else:
                    examples = [InputExample(guid=0, text_a=question, text_b=None, label=1)]
                    dataset = from_examples2dataset(args, examples, tokenizer)
                    outputs = evaluate(args, model, dataset)

                    question_embedding = torch.cat([o.cpu().data for o in outputs]).numpy()
                    scores = cosine_similarity(question_embedding, candidate_embeddings)[0]
                    top1 = scores.argsort()[-1:][::-1]
                    for index in top1:
                        # nlp_result=("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))+nlp_result
                        # print("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))
                        if scores[index] >= 0.92:
                            nlp_result = ("{}".format(candidate_reply[index])) + nlp_result
                            nlp_result = tihuan(nlp_result, phone_id)
                            print("{}".format(candidate_reply[index]))
                            print(nlp_result)
                        else:
                            nlp_result = "对不起，我还不清楚您的问题"
                            print(nlp_result)

            output_name = filenameRandomizing(True) + ".wav"
            output_name = "./work/" + output_name
            res = ttsclient_executor(
                input=nlp_result,
                server_ip="127.0.0.1",
                port=8090,
                spk_id=int(spk_id),
                speed=speed,
                volume=volume,
                sample_rate=16000,
                output=output_name)
            # playsound(output_name)
            return nlp_result,quest_return
    elif value=='7':
        ifall = False
        while True:
            if ifall:
                question,audioname = ASRapi(False,audio_data)
            # else:
            #     question = input("请输入问题")
            nlp_result = ""
            quest_return = ""
            quest_return = question
            counter = counter + 1
            if len(str(question)) == 0:
                nlp_result = "请给出需要咨询的问题"
                print(nlp_result)
            else:
                if clf_model.predict(question) == 'greet':
                    nlp_result = "你好呀"
                    print(nlp_result)
                elif clf_model.predict(question) == 'accept':
                    nlp_result = "用户意图为肯定"
                    print(nlp_result)
                elif clf_model.predict(question) == 'deny':
                    nlp_result = "用户意图为否定"
                    print(nlp_result)
                else:
                    examples = [InputExample(guid=0, text_a=question, text_b=None, label=1)]
                    dataset = from_examples2dataset(args, examples, tokenizer)
                    outputs = evaluate(args, model, dataset)

                    question_embedding = torch.cat([o.cpu().data for o in outputs]).numpy()
                    scores = cosine_similarity(question_embedding, candidate_embeddings)[0]
                    top1 = scores.argsort()[-1:][::-1]
                    for index in top1:
                        # nlp_result=("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))+nlp_result
                        # print("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))
                        if scores[index] >= 0.92:
                            nlp_result = ("{}".format(candidate_reply[index])) + nlp_result
                            nlp_result=tihuan(nlp_result, phone_id)
                            print("{}".format(candidate_reply[index]))
                            print(nlp_result)
                        else:
                            nlp_result = "对不起，我还不清楚您的问题"
                            print(nlp_result)
                return nlp_result,quest_return
    else:
        ifall = False
        while True:
            if ifall:
                question,audioname = ASRapi(False,audio_data)
            # else:
            #     question = input("请输入问题")
            nlp_result = ""
            quest_return = ""
            quest_return = question
            counter = counter + 1
            if len(str(question)) == 0:
                nlp_result = "请给出需要咨询的问题"
                print(nlp_result)
            else:
                if clf_model.predict(question) == 'greet':
                    nlp_result = "你好呀"
                    print(nlp_result)
                elif clf_model.predict(question) == 'accept':
                    nlp_result = "用户意图为肯定"
                    print(nlp_result)
                elif clf_model.predict(question) == 'deny':
                    nlp_result = "用户意图为否定"
                    print(nlp_result)
                else:
                    examples = [InputExample(guid=0, text_a=question, text_b=None, label=1)]
                    dataset = from_examples2dataset(args, examples, tokenizer)
                    outputs = evaluate(args, model, dataset)

                    question_embedding = torch.cat([o.cpu().data for o in outputs]).numpy()
                    scores = cosine_similarity(question_embedding, candidate_embeddings)[0]
                    top1 = scores.argsort()[-1:][::-1]
                    for index in top1:
                        # nlp_result=("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))+nlp_result
                        # print("可能得答案，参考问题为:{},答案:{},得分:{}\n".format(candidate_title[index], candidate_reply[index],
                        #                                           str(scores[index])))
                        if scores[index] >= 0.92:
                            nlp_result = ("{}".format(candidate_reply[index])) + nlp_result
                            nlp_result = tihuan(nlp_result, phone_id)
                            print("{}".format(candidate_reply[index]))
                            print(nlp_result)
                        else:
                            nlp_result = "对不起，我还不清楚您的问题"
                            print(nlp_result)
                return nlp_result,quest_return


# 录音+合成调用函数，参数phone_id：电话号码；输出为结果文本
def main_2(phone_id,audio_name):
    while True:
        question,audioname=ASRapi(False,audioname=audio_name)
        nlp_result=bohao(phone_id,question)
        output_audio='./merge/daikuan.wav'
        # playsound('./merge/daikuan.wav')
        return nlp_result,output_audio,question


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


@app.post("/clf_A/")
def A_nlp(item: nlp_Item):
    ret = clf_model.predict(item.question)
    return {
        "success": True,
        "code": 0,
        "message": "One-cycle NLP proceeded successfully.",
        "question": item.question,
        "ret": ret,
    }


# @app.post('/clf_asr/')
# def A_nlp(item:clf_asr_Item):
#     A_question,audioname=ASRapi(False,item.audioname)
#     ret=clf_model.predict(A_question)
#     return {
#         "success": True,
#         "code": 0,
#         "message": "One-cycle NLP proceeded successfully.",
#         "question":A_question,
#         "audioname":audioname,
#         "ret": ret}

# @app.post("/nlp_A/")
# async def ans_nlp(request:Request,A_question=Query(None)):
#     print(A_question)
#     if len(A_question)==0:
#         ret="请提供需要解答的问题"
#         return {"request": request,"message": ret}
#     else:
#         ret=main('8',A_question,False)
#         return {"request": request,"message": ret}


@app.post("/nlp_A/")
def ans_nlp(item: nlp_Item):
    """
    纯文本单轮对话模型

    Args:
        item (nlp_Item): 仅有一个 `question` 成员变量, 用户输入的query
    """
    if len(item.question)==0:
        ans="请提供需要解答的问题"
        return {
            "success": True,
            "code": 0,
            "message": "One-cycle NLP proceeded successfully.",
            "ans": ans,
            "question": item.question
        }
    else:
        ans,question=main('8', item.question, False)
        return {
            "success": True,
            "code": 0,
            "message": "One-cycle NLP proceeded successfully.",
            "ans": ans,
            "question": question
        }


@app.post("/nlp_All/")
def ans_all(item: Items):
    # x = open("./work/test.wav","w")
    print(item.audio_stream)
    data = item.audio_stream
    audio_data = base64.b64decode(data)
    record_name = filenameRandomizing(True) + ".wav"
    record_name = "./work/" + record_name
    with wave.open(record_name,"wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)
    if item.value == "3":
        if item.question == "p":
            ans, que = main(item.value, item.question, item.isall, item.spk_id, item.file_list,item.customerphone,audio_data=record_name)
            return {
                "success": True,
                "code": 0,
                "message": "One-cycle ANT proceeded successfully.",
                "ans": ans,
                "question": que,
                "counter": item.counter,
                "customerphone": item.customerphone,
                "companyid": item.companyid,
                "taskid": item.taskid,
            }
        else:
            ans, que, audio_name, output_name = main(item.value, item.question, item.isall, item.spk_id, item.file_list,item.customerphone,audio_data="./work/test.wav",speed=float(item.speed),volume=float(item.volume))

            with open(output_name,"rb") as wav_file:
                # audio_data = wav_file.readframes(wav_file.getnframes())
                audio_data = wav_file.read()
                audio_back_stream = base64.b64encode(audio_data).decode()
            print("output")

            return {
                "success": True,
                "code": 0,
                "message": "One-cycle ANT proceeded successfully.",
                "ans": ans,
                "question": que,
                "counter": item.counter,
                "audio_name": audio_name,
                "output_name": output_name,
                "customerphone": item.customerphone,
                "companyid": item.companyid,
                "taskid": item.taskid,
                "audio_back_stream": audio_back_stream,
            }

    else:
        ans,que=main(item.value,item.question,item.isall,item.spk_id,customerphone=item.customerphone, audio_data=record_name,speed=float(item.speed),volume=float(item.volume))
        return {
            "success": True,
            "code": 0,
            "message": "One-cycle ANT proceeded successfully.",
            "ans": ans,
            "question": que,
            "counter": item.counter,
            "customerphone": item.customerphone,
            "companyid": item.companyid,
            "taskid": item.taskid,
        }
    # ans, que, audio_name, output_name = main(item.value, item.question, item.isall, item.spk_id, item.file_list)


@app.post("/tts/streaming")
def streamingTTS(item: tts_Item):
    tts = TTSClientExecutor()
    output_name = filenameRandomizing(True) + ".wav"
    output_name = "/mnt/shared/" + output_name
    tts(
        input=item.text,
        server_ip="127.0.0.1",
        port=8090,
        spk_id=int(item.spk_id),
        speed=float(item.speed),
        volume=float(item.volume),
        sample_rate=16000,
        output=output_name)

    with open(output_name, "rb") as wav_file:
        # audio_data = wav_file.readframes(wav_file.getnframes())
        audio_data = wav_file.read()
        audio_stream = base64.b64encode(audio_data).decode()

    return {
            "success": True,
            "code": 0,
            "message": {"global": "success" },
            "result": {
                "lang": "zh",
                "spk_id": item.spk_id,
                "speed": item.speed,
                "volume": item.volume,
                "sample_rate": 16000,
                "output_name": output_name,
                "audio_stream": audio_stream
            }
    }

@app.post("/tts/streaming2")
def streamingTTS2(item: tts_Item):
    tts = TTSClientExecutor()
    output_name = filenameRandomizing(True) + ".wav"
    output_name = "./work/" + output_name
    tts(
        input=item.text,
        server_ip="127.0.0.1",
        port=8090,
        spk_id=int(item.spk_id),
        speed=float(item.speed),
        volume=float(item.volume),
        sample_rate=16000,
        output=output_name)

    with open(output_name, "rb") as wav_file:
        # audio_data = wav_file.readframes(wav_file.getnframes())
        audio_data = wav_file.read()
        audio_stream = base64.b64encode(audio_data).decode()
    os.remove(output_name)
    return {
            "success": True,
            "code": 0,
            "message": {"global": "success" },
            "result": {
                "lang": "zh",
                "spk_id": item.spk_id,
                "speed": item.speed,
                "volume": item.volume,
                "sample_rate": 16000,
                "audio_stream": audio_stream
            }
    }

@app.post("/lyAhc/")
def lyAhc(item:Items):
    data = item.audio_stream
    audio_data = base64.b64decode(data)
    record_name = filenameRandomizing(True) + ".wav"
    record_name = "./work/" + record_name
    with wave.open(record_name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)
    ans,output_name,que=main_2(item.customerphone,record_name)

    with open(output_name, "rb") as wav_file:
        # audio_data = wav_file.readframes(wav_file.getnframes())
        audio_data = wav_file.read()
        audio_back_stream = base64.b64encode(audio_data).decode()

    return {"success": True,
            "code": 0,
            "message": "One-cycle ANT proceeded successfully.",
            "ans": ans,
            "question": que,
            "audio_name": record_name,
            "output_name": output_name,
            "customerphone": item.customerphone,
            "companyid": item.companyid,
            "taskid": item.taskid,
            "audio_back_stream": audio_back_stream}

class asr_Item(BaseModel):
    audiocode: str
    filename: str
    filepath: str
    punc_flag: bool
    save_flag: bool

@app.post("/asr/save_record")
async def save_record(filename: str, audioname: str):
    save_to_db(filename, audioname)

@app.post("/asr/complete_asr")
async def complete_asr(item:asr_Item):
    # 4.完整的一套语音识别流程，录音后直接输出
    # record_msg = record(filename=filepath, auto=auto, time=time, threshold=threshold)
    # if not record_msg["success"]:
    #     return record_msg
    # else:
    #     output = predict_audio(record_msg["result"]["output"], filepath, punc_flag, save_flag)
    if len(item.audiocode) >= 3:
        data = item.audiocode
        audio_data = base64.b64decode(data)
        item.filename = filenameRandomizing(True) + ".wav"
        item.filename = item.filepath +"/"+ item.filename
        with wave.open(item.filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)
    output = predict_audio(item.filename, item.filepath, item.punc_flag, item.save_flag)
    save_record(output["txtname"],output["audioname"])
    return {
        "success": True,
        "code": 0,
        "message": "One-cycle ASR proceeded successfully.",
        "result": {
            "transcription": output["text"],
            "timecost": output["timespent"],
            "audioname": output["audioname"],
            "txtname": output["txtname"],
        },
    }


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
            query_mode="local"
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
            history_messages=user[item.userid].history
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
    # value=input("选择路线")

    # main(value)
    # app.run(port=5000, debug=True)
    # uvicorn 中workers参数即为线程数
    # uvicorn.run(app='main_bert_base:app', host='202.192.46.104', port=8080, reload=True, workers=3)
    uvicorn.run(app='main_bert_base:app', host='127.0.0.1', port=8080, reload=True, workers=3)
    #main_2("12333333333")
    
