import os
import re
import json
from dataclasses import dataclass, field

import torch
from pydantic import BaseModel, Field


@dataclass
class IntentRecognitionConfig:
    """
    意图识别的配置文件
    """

    model_path: str = field(
        default="./results/lr_0.001_epoch_3_model_Llama-3.1-8B-Instruct_time_20_30_38",
        metadata={"help": "模型的存放路径"},
    )
    max_length: int = field(default=256)
    device_map: str = field(
        default="cpu",
        metadata={"help": "模型的设备映射, 包括 `auto`, `cpu`, `cuda`"},
    )


class IntentRecognitionItem(BaseModel):
    """
    意图识别的输入数据
    非 `中性`, `同意`, `不同意` 这种简单的意图识别
    而是在给定的多个标签中将用户的query分类到特定标签的意图识别
    """

    query: str = Field(description="用户的query")


class IntentRecognitionResult(BaseModel):
    """
    意图识别 (IntentRecognitionItem) 的返回数据
    """

    success: bool = Field(description="是否成功")
    intent: str = Field(description="意图识别的结果")


@dataclass
class MultiIntentDetectionConfig:
    """
    多轮意图识别的配置文件
    """

    loading_method: str = field(
        metadata={"help": "导入模型的方法, 包括 `ollama`, `api`"},
    )
    language_model: str = field(metadata={"help": "语言模型的模型名称"})
    hyper_alpha: float = field(
        default=0.3,
        metadata={"help": "加权sigmoid权重时的超参"},
    )
    hyper_beta: float = field(
        default=0.7,
        metadata={"help": "加权语言模型权重时的超参"},
    )

    def __post_init__(self):
        if self.loading_method == "api":
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
        else:
            self.base_url = "http://127.0.0.1:11434/v1"
            self.api_key = "ollama"


@dataclass
class SingleIntentDetectionResult:
    """
    用来存储单轮意图识别的输出结果
    """

    Confidence: float
    Weight: float
    Intent: str


class MultiIntentDetectionItem(BaseModel):
    """
    多轮意图识别的输入数据
    """

    filename: str = Field(description="包含多轮对话的意图识别文件名")


class MultiIntentDetectionResult(BaseModel):
    """
    多轮意图识别的返回数据
    """

    success: bool = Field(description="是否成功")
    content_intent: str = Field(description="每一句话的识别意图")
    global_intent: str = Field(description="全局的总意图")


@dataclass
class MultiQAConfig:
    """
    多轮问答的配置文件
    """

    working_dir: str = field(metadata={"help": "Working directory"})
    loading_method: str = field(
        metadata={"help": "导入模型的方法, 包括 `ollama`, `api`和`hf`"},
    )
    companyid: str = field(metadata={"help": "The company id"})
    language_model: str = field(metadata={"help": "语言模型的name或者path"})
    embedding_model: str = field(
        metadata={"help": "embedding模型的name或者path"},
    )
    embedding_dim: int = field(default=384)
    max_token_size: int = field(
        default=5000, metadata={"help": "The max input token size for embedding model"}
    )
    responsed_language: str = field(default="简体中文", metadata={"help": "语言模型使用的语言"})
    max_answer_length: int = field(
        default=100, metadata={"help": "The max length of model response"}
    )
    history: list = field(default_factory=list, metadata={"help": "Conversation history"})
    query_mode: str = field(
        default="hybrid",
        metadata={"help": "The query mode of RAG system, include `local`, `global` and `hybrid`"},
    )
    rag_process_context_with_llm: bool = field(
        default=True,
        metadata={"help": "By default, RAG system will process collected context with LLM"},
    )

    def __post_init__(self):
        from pathlib import Path

        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # 创建日志和图知识图谱存储实例路径
        self.log_file_path = str((Path(self.working_dir) / Path("log.txt")).absolute())
        self.graphml_path = str(
            (Path(self.working_dir) / Path("graph_chunk_entity_relation.graphml")).absolute()
        )

        # 为不同的加载方式设定不同的参数
        if self.loading_method == "api":
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.api_key = os.getenv("DASHSCOPE_API_KEY")
        elif self.loading_method == "hf":
            if not os.path.exists(self.language_model):
                raise FileExistsError(f"Language model path {self.language_model} does not exist.")
        else:
            self.base_url = "http://127.0.0.1:11434/v1"
            self.api_key = "ollama"


class MultiQAItem(BaseModel):
    """
    多轮问答的输入数据
    """

    userid: str = Field(description="user id")
    companyid: str = Field(description="user company id")
    query: str = Field(description="user query")
    max_answer_length: int = Field(default=100, description="模型输出的最大长度")


class MultiQAResult(BaseModel):
    """
    多轮问答的返回数据
    """

    success: bool = Field(description="是否成功")
    model_response: str = Field(description="模型的输出")
    error_message: str = Field(description="错误信息")


def delete_deepseek_thinking(input_text):
    """
    删除DeepSeek思考的提示词
    """
    think_delimiter = re.search(r"<\/think>", input_text)
    if think_delimiter:
        result = input_text[think_delimiter.end() :].lstrip()
        return result

    return input_text


def parse_llm_json_output(text: str):
    """
    从LLM输出的代码块中提取并解析JSON
    """
    # 匹配```json 和 ``` 之间的内容
    pattern = r'```(?:json)?\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)

    # 如果有代码块标记, 则提取其中的Josn字符串并解析
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return eval(json_str)
    # 如果没有代码块标记，尝试直接解析
    else:
        try:
            return json.loads(text)
        except:
            return eval(text)


if __name__ == "__main__":
    # intent = IntentDetectionExample(
    #     conversation_history="SalesPerson: 请问您想要订阅我们的产品吗? Customer: 抱歉, 我不感兴趣",
    #     Confidence=0.1,
    #     Weight=0.9,
    #     Intent="不感兴趣",
    # )
    # print(intent.__str__())

    string = """```json\n{"Confidence": 0.3, "Weight": 0.8, "Intention": "询问货物详情，尚未明确同意承运"}\n```"""
    print(string)
    output = parse_llm_json_output(string)
    print(output)
    type(output)
