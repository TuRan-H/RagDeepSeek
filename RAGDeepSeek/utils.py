import os
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class IntentDetectionExample:
    conversation_history: str
    Confidence: str
    Weight: str

    def __str__(self):
        return (
            "-----\n"
            f"Example:\n"
            f"Conversation History: \"{self.conversation_history}\"\n"
            f"Model Response: {{\"Confidence\": {self.Confidence}, \"Weight\": {self.Weight}}}\n"
            "-----"
        )


@dataclass
class IntentDetectionResult:
    """
    用来存储单轮意图识别的输出结果
    """

    Confidence: str
    Weight: str


class MultiIntentResult(BaseModel):
    """
    多轮意图识别的返回数据
    """

    success: bool = Field(description="是否成功")
    content_intent: str = Field(description="每一句话的识别意图")
    global_intent: str = Field(description="全局的总意图")


class MultiIntentItem(BaseModel):
    """
    多轮意图识别的输入数据
    """

    filename: str = Field(description="包含多轮对话的意图识别文件名")


@dataclass
class MultiIntentConfig:
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


class MultiQaItem(BaseModel):
    """
    多轮问答的输入数据
    """

    userid: str = Field(description="user id")
    companyid: str = Field(description="user company id")
    query: str = Field(description="user query")
    max_answer_length: int = Field(default=100, description="模型输出的最大长度")


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
        default=2048, metadata={"help": "The max length of model response"}
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


def delete_deepseek_thinking(input_text):
    """
    删除DeepSeek思考的提示词
    """
    think_delimiter = re.search(r"<\/think>", input_text)
    if think_delimiter:
        result = input_text[think_delimiter.end() :].lstrip()
        return result

    return input_text


if __name__ == "__main__":
    pass
