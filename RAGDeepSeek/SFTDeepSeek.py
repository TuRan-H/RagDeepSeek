"""
TODO 将LoRA矩阵合并到模型中
TODO 保存模型, 将训练得到的模型文件转化成ollama的gguf格式
"""

import os, sys
from datetime import datetime
from typing import cast, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
import re
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import wandb
from tqdm import tqdm


PROMPT = """You are an intelligent assistant for a banking app. You need to classify customer queries for intent recognition into the following types:
['greet', 'goodbye', 'deny', 'bot', 'accept', 'e-commerce', 'operator', 'bank', 'bridge', 'doubt']
Directly output the result, without any additional explanation.

### examples ###
User query: 我想查询订单的物流信息
Your response: e-commerce

User query: 如何开通国际漫游服务
Your response: operator

### Real Data ###
User query: {query}
Your response:
"""


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> substitute Tensor.__repr__
_ = __import__("sys")
if "torch" in _.modules:
    _ = __import__("torch")
    if not hasattr(_.Tensor, "_custom_repr_installed"):
        original_tensor_repr = _.Tensor.__repr__

        def custom_tensor_repr(self):
            return f"{tuple(self.shape)}{original_tensor_repr(self)}"

        setattr(_.Tensor, "__repr__", custom_tensor_repr)
        setattr(_.Tensor, "_custom_repr_installed", True)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< substitute Tensor.__repr__


@dataclass
class SFTTrainingArguments:
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=1e-4)
    remove_unused_columns: bool = field(default=True)
    run_name: str = field(default="fine_tune_deepseek")
    max_grad_norm: float = field(default=0.3)


@dataclass
class ModelArguments:
    # ! LoRA的超参, 建议保持默认, 如果显存不够可以修改target_modules
    lora_alpha: int = field(default=16, metadata={"help": "使用LoRA包装模型时的lora_alpha参数"})
    lora_dropout: float = field(
        default=0.1, metadata={"help": "使用LoRA包装模型时的lora_dropout参数"}
    )
    lora_r: int = field(default=8, metadata={"help": "使用LoRA包装模型时的r参数"})
    bias: str = field(default="none", metadata={"help": "使用LoRA包装模型时的bias参数"})
    task_type: str = field(
        default="CAUSAL_LM", metadata={"help": "使用LoRA包装模型时的task_type参数"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj"],
        metadata={"help": "使用LoRA包装模型时的target_modules参数"},
    )
    max_length: int = field(default=256, metadata={"help": "模型训练时允许模型生成的token最大长度"})
    # ! 如果显存还是不够, 将use_quantization改为True, 对模型启用量化, 但是量化可能会导致模型的性能下降
    use_quantization: bool = field(
        default=False, metadata={"help": "是否使用BitsAndBytes来进行量化"}
    )
    load_in_4bit: bool = field(default=True, metadata={"help": "BitsAndBytes中load_in_4bit参数"})
    bnb_4bit_use_double_quant: bool = field(
        default=True, metadata={"help": "BitsAndBytes中bnb_4bit_use_double_quant参数"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4", metadata={"help": "BitsAndBytes中bnb_4bit_quant_type参数"}
    )


@dataclass
class DataArguments:
    data_path: str = field(default="./data/intent_recog_data_v2.csv")
    model_path: str = field(default="./models/DeepSeek-R1-Distill-Llama-8B")
    save_dir: str = field(default="./results", metadata={"help": "模型保存的目录"})


def get_model_tokenizer(
    training_args: SFTTrainingArguments, model_args: ModelArguments, data_args: DataArguments
):
    """
    加载模型和tokenizer

    Args:
        training_args (SFTTrainingArguments): 训练参数
        model_args (ModelArguments): 模型参数
        data_args (DataArguments): 数据参数

    Returns:
        model: 模型
        tokenizer: 分词器
    """
    # 导入model
    lora_config = LoraConfig(
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        r=model_args.lora_r,
        bias=model_args.bias,
        task_type=model_args.task_type,
        target_modules=model_args.target_modules,
    )  # type: ignore

    if model_args.use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # device_map="auto"会将模型自动分片到可用的GPU上
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=data_args.model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            use_cache=False,
            quantization_config=bnb_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=data_args.model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            use_cache=False,
        )

    # ! 如果使用LoRA包装模型, 模型的embedding层会被冻结, 需要显式的启用输入的require_grads()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 使用LoRA包装模型
    model = get_peft_model(model=model, peft_config=lora_config)

    # 导入tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=data_args.model_path,
        trust_remote_code=True,
    )
    # 设定tokenizer的pad_token和padding方向 (右向padding)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def get_trainer(
    training_args: SFTTrainingArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    train_set: Dataset,
    test_set: Dataset,
):
    """
    获取训练器

    Args:
        training_args (SFTTrainingArguments): 训练参数
        model_args (ModelArguments): 模型参数
        data_args (DataArguments): 数据参数
        model: 模型
        tokenizer: 分词器
        train_dataset: 训练集
        test_set: 测试集

    Returns:
        trainer: 训练器
    """
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_set,
        eval_dataset=test_set,
        processing_class=tokenizer,
        args=TrainingArguments(
            run_name=training_args.run_name,
            output_dir=data_args.save_dir,
            num_train_epochs=training_args.num_train_epochs,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            learning_rate=training_args.learning_rate,
            max_grad_norm=training_args.max_grad_norm,
            weight_decay=0.01,
            warmup_ratio=0.03,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            fp16=False,
            bf16=False,
            optim="adamw_torch_fused",
            report_to="wandb",
            gradient_checkpointing=True,
            group_by_length=True,
            dataloader_num_workers=0,
            remove_unused_columns=True,  # type: ignore
            deepspeed=None,
            local_rank=-1,
            ddp_find_unused_parameters=None,
            torch_compile=False,
            disable_tqdm=False,
        ),
    )  # type: ignore

    return trainer


def get_dataset(data_args: DataArguments):
    """
    获取数据集
    Args:
        data_args (DataArguments): 数据参数

    Returns:
        train_set: 训练集
        test_set: 测试集
    """
    # ! 这里需要提供 RAGDeepSeek/SFTDataset.py 的路径, 如果相对路径不正确, 使用绝对路径
    all_dataset = load_dataset(
        "./RAGDeepSeek/SFTDataset.py", data_dir=data_args.data_path, trust_remote_code=True
    )
    dataset: Dataset = all_dataset["train"]  # type: ignore

    train_test_split_dict = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_set = train_test_split_dict["train"]
    test_set = train_test_split_dict["test"]

    return train_set, test_set


def inference(
    query: str,
    model_path: str,
    max_length: int,
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
):
    """
    生成文本
    do_sample = True, num_beams = 1 (多项式采样)

    Args:
        prompt: 模型输入
        model: 模型
        tokenizer: 分词器

    Returns:
        生成的文本
    """

    query = PROMPT.format(query=query)

    inputs = tokenizer(query, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    genreation_config = GenerationConfig.from_pretrained(model_path)
    genreation_config.pad_token_id = tokenizer.eos_token_id
    genreation_config.max_length = max_length

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=genreation_config)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r"### Real Data ###.+Your response:(.+)", response, re.DOTALL)
    if match:
        result = match.group(1).strip()
    else:
        result = ""

    return result


def evaluate(
    model_args: ModelArguments,
    data_args: DataArguments,
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    test_dataset,
):
    """
    评估模型, 使用accuracy作为评估指标

    Args:
        model_args (ModelArguments): 模型参数
        data_args (DataArguments): 数据参数
        model: 模型
        tokenizer: 分词器

    Returns:
        评估结果
    """
    # 批量inference需要将padding side改为左侧, 因为模型会在处理到eos_token是停止输出
    tokenizer.padding_side = "left"
    model.eval()
    data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    generation_config = GenerationConfig.from_pretrained(data_args.model_path)
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.max_length = model_args.max_length

    labels = []
    predictions = []
    with torch.no_grad():
        bar = tqdm(total=len(data_loader), desc="Evaluating")
        for data in data_loader:
            # 获取模型的输出
            labels.extend(data["completion"])
            inputs = tokenizer(data["prompt"], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            p = model.generate(**inputs, generation_config=generation_config)
            p = tokenizer.batch_decode(p, skip_special_tokens=True)
            predictions.extend(p)
            bar.update(1)

    # 处理模型的输出
    processed_predictions = []
    for p in predictions:
        match = re.search(r"### Real Data ###.+Your response:(.+)", p, re.DOTALL)
        processed_predictions.append(match.group(1).strip() if match else " ")

    # 计算accuracy
    correct_predictions = 0
    for p, l in zip(processed_predictions, labels):
        if p == l:
            correct_predictions += 1
    accuracy = correct_predictions / len(labels)

    tokenizer.padding_side = "right"
    return accuracy


if __name__ == "__main__":
    # 获取参数
    parser = HfArgumentParser((SFTTrainingArguments, ModelArguments, DataArguments))
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = ""

    if json_file:
        (training_args, model_args, data_args) = parser.parse_json_file(json_file=json_file)
    else:
        (training_args, model_args, data_args) = parser.parse_args_into_dataclasses()
    (training_args, model_args, data_args) = (
        cast(SFTTrainingArguments, training_args),
        cast(ModelArguments, model_args),
        cast(DataArguments, data_args),
    )

    now = datetime.now().timestamp()
    formatted_time = datetime.fromtimestamp(now).strftime("%H_%M_%S")
    training_args.run_name = f"lr_{training_args.learning_rate}_epoch_{training_args.num_train_epochs}_model_{os.path.basename(data_args.model_path)}_time_{formatted_time}"
    data_args.save_dir = os.path.join(data_args.save_dir, training_args.run_name)

    # ! 如果不想用wandb观察训练状态, 则直接注释掉下面这段代码
    wandb.init(
        project=os.getenv("WANDB_PROJECT", None),
        entity=os.getenv("WANDB_ENTITY", None),
        name=training_args.run_name,
        config={
            "_data_args": asdict(data_args),
            "_model_args": asdict(model_args),
            "_training_args": asdict(training_args),
        },
    )

    # 加载模型和tokenizer
    model, tokenizer = get_model_tokenizer(training_args, model_args, data_args)

    # train_set, test_set = get_dataset(data_args=data_args)
    train_set, test_set = get_dataset(data_args=data_args)

    trainer = get_trainer(
        training_args=training_args,
        model_args=model_args,
        data_args=data_args,
        model=model,
        tokenizer=tokenizer,
        train_set=train_set,
        test_set=test_set,
    )


    print("**************************************************inference before training")
    print("query: 你好啊\nresult: ")
    print(
        inference(
            model_path=data_args.model_path,
            max_length=model_args.max_length,
            model=model,
            tokenizer=tokenizer,
            query="你好啊",
        ),
        "\n",
    )

    # 训练
    trainer.train()

    # 将LoRA的权重合并到模型中
    merged_model: LlamaForCausalLM = model.merge_and_unload()  # type: ignore

    print("**************************************************inference after training")
    print("query: 不知道\nresult: ")
    print(
        inference(
            model_path=data_args.model_path,
            max_length=model_args.max_length,
            model=merged_model,
            tokenizer=tokenizer,
            query="不知道",
        ),
        "\n",
    )

    accuracy = evaluate(model_args, data_args, merged_model, tokenizer, test_set)
    print("**************************************************accuracy")
    print(f"accuracy: {accuracy}\n")
    wandb.log(data={"accuracy": accuracy})

    # 保存模型
    merged_model.save_pretrained(data_args.save_dir)
    tokenizer.save_pretrained(data_args.save_dir)