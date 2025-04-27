"""
TODO 改成开放式的意图识别
"""

from typing import List
import json

import numpy as np
from openai import OpenAI


from RAGDeepSeek.utils import (
    MultiIntentConfig,
    IntentDetectionResult,
    MultiIntentResult,
    delete_deepseek_thinking,
    parse_llm_json_output,
)
from RAGDeepSeek.prompt import INTENT_DETECTION_EXAMPLES, INTENT_DETECTION_TEMPLATE


def map_confidence_to_intent(confidence: float) -> str:
    if confidence >= 0.8:
        return "强烈同意"
    elif confidence >= 0.6:
        return "同意"
    elif confidence >= 0.4:
        return "中性"
    elif confidence >= 0.2:
        return "不同意"
    else:
        return "强烈不同意"


class IntentDetection:
    def __init__(
        self,
        config: MultiIntentConfig,
    ):
        """
        生成式意图识别

        Args:
            config (MultiIntentConfig): 参数配置
        """
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model_name = config.language_model
        self.few_shot_examples = INTENT_DETECTION_EXAMPLES.copy()
        self.alpha = config.hyper_alpha
        self.beta = config.hyper_beta

    def round_predict(self, text: str) -> IntentDetectionResult:
        """
        单轮意图识别

        Args:
            text (str): 输入文本

        Returns:
            IntentDetectionResult: 单轮意图识别的结果
        """
        examples_str = ""
        for example in self.few_shot_examples:
            examples_str += str(example) + "\n\n"
        examples_str = examples_str.strip()

        prompt = INTENT_DETECTION_TEMPLATE.format(examples=examples_str, text=text)

        if "deepseek" in self.config.language_model:
            completion = (
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "text"},
                )
                .choices[0]
                .message.content
            )
            completion = delete_deepseek_thinking(completion)
            response_dict = parse_llm_json_output(completion)
        else:
            completion = (
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                .choices[0]
                .message.content
            )
            response_dict = json.loads(completion)

        return IntentDetectionResult(
            response_dict["Confidence"], response_dict["Weight"], response_dict["Intent"]
        )

    def multi_predict(self, file_name) -> MultiIntentResult:
        """
        多轮意图识别

        Args:
            file_path (str): 文件路径

        Returns:
            MultiIntentResult: 多轮意图识别的结果
        """
        with open(file_name, 'r') as fp:
            data = [json.loads(line) for line in fp.readlines()]

        # 将文件内容解析为多轮对话列表
        conversation_history: List[str] = []
        sales_person, customer = ("", "")
        i = 0
        while i < len(data):
            if data[i]['name'] == "AI":
                sales_person = data[i]["content"]
                customer = data[i + 1]["content"]
                conversation_history.append(f"SalesPerson: {sales_person} Customer: {customer}")
                i += 2
            else:
                customer = data[i]["content"]
                conversation_history.append(f"Customer: {customer}")
                i += 1

        # 计算单轮对话的意图和语言模型输出的权重
        llm_weight_list, confidence_list, intention_list = list(), list(), list()
        try:
            for conversation in conversation_history:
                result = self.round_predict(conversation)
                confidence_list.append(result.Confidence)
                llm_weight_list.append(result.Weight)
                intention_list.append(result.Intent)
        except Exception as e:
            return MultiIntentResult(
                success=False,
                content_intent="[]",
                global_intent="",
            )
        global_weight_list = self.compute_global_weight_list(llm_weight_list)

        # 计算全局意图
        confidence = 0
        for c, w in zip(confidence_list, global_weight_list):
            confidence += float(c) * float(w)

        # 将float类型的confidence转化为str类型的intent
        confidence = map_confidence_to_intent(float(confidence))

        return MultiIntentResult(
            success=True,
            content_intent=intention_list.__str__(),
            global_intent=confidence,
        )

    def compute_global_weight_list(self, llm_weight_list):
        """
        计算全局权重列表

        Args:
            weight_list_llm: 权重列表

        Returns:
            list[float]: 全局权重列表
        """
        weight_list_length = len(llm_weight_list)

        # 计算sigmoid权重
        indices = np.arange(weight_list_length)
        shifted_indices = indices - (weight_list_length - 1) / 2
        sigmoid_weight_list = 1 / (1 + np.exp(-shifted_indices))
        sigmoid_weight_list = [
            np.exp(i) / np.sum(np.exp(sigmoid_weight_list)) for i in sigmoid_weight_list
        ]

        # 重构语言模型输出的权重
        llm_weight_list = np.array(llm_weight_list)
        llm_weight_list = [np.exp(i) / np.sum(np.exp(llm_weight_list)) for i in llm_weight_list]

        # 计算全局权重
        global_weight_list = [
            self.alpha * a + self.beta * b for a, b in zip(sigmoid_weight_list, llm_weight_list)
        ]
        return global_weight_list


if __name__ == "__main__":
    config = MultiIntentConfig(
        loading_method="ollama",
        language_model="deepseek-r1:14b",
    )
    intent_detection = IntentDetection(config=config)
    print(intent_detection.multi_predict("./data/intent_detection.jsonl"))
