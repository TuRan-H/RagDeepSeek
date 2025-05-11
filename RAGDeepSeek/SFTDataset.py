import datasets
import pandas


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


class SFTDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = datasets.BuilderConfig
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=DEFAULT_CONFIG_NAME,
            version=datasets.Version("1.0.0"),
            description="SFT dataset for fine-tuning.",
            data_dir=None,
        )
    ]

    def _info(self):
        features = datasets.Features(
            {
                "prompt": datasets.Value("string"),
                "completion": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        print(self.config.data_dir)
        if hasattr(self.config, "data_dir") and self.config.data_dir:
            file_path = self.config.data_dir
        else:
            raise ValueError("Please provide a valid data_dir in load_dataset()")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": file_path,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, **kwargs):
        file_path = kwargs.get("filepath", None)
        if not file_path:
            raise ValueError("File path is required")

        df = pandas.read_csv(file_path, header=None)
        for index, row in df.iterrows():
            prompt = PROMPT.format(query=row[0])

            yield index, {
                "prompt": prompt,
                "completion": row[1],
            }
