import os
import re
import subprocess
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class MultiQaItem(BaseModel):
    userid: str = Field(description="user id")
    companyid: str = Field(description="user company id")
    query: str = Field(description="user query")
    max_answer_length: int = Field(default="2048", description="the max length of model response")


@dataclass
class Config:
    working_dir: str = field(metadata={"help": "Working directory"})
    language_model: str = field(metadata={"help": "The name or path used in RAG framework"})
    loading_method: str = field(
        metadata={
            "help": "The method for loading language model, include `hf`, `api` and `ollama`"
        },
    )
    companyid: str = field(metadata={"help": "The company id for the user"})
    embedding_model: str = field(
        default="/newSSD/home/TuRan/Downloads/models/all-MiniLM-L6-v2",
        metadata={"help": "The name for embedding model"},
    )
    embedding_dim: int = field(default=384)
    max_token_size: int = field(
        default=5000, metadata={"help": "The max input token size for embedding model"}
    )
    responsed_language: str = field(
        default="简体中文", metadata={"help": "The language used in response"}
    )
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

        # 创建日志和图可视化文件路径
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
    config = Config(
        working_dir="temp",
        loading_method="ollama",
        language_model="deepseek-r1:14b",
    )
    print(config.working_dir)
    print(config.log_file_path)
    print(config.graphml_path)
    print(config.api_key)
