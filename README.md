# 配置python环境
安装python
```
conda create -n RAGDeepSeek python==3.10.*
```

安装依赖
```
bash install.sh
```

# 运行微调代码
首先在配置文件目录 `./config/default_config.json` 中填写 `model_path` 字段<br>
填写上从huggingface上下载的模型的地址, 服务器文件位于 `/newSSD/home/TuRan/Downloads/models/DeepSeek-R1-Distill-Llama-8B`


在代码的工作目录 `/newSSD/home/TuRan/Projects/RagDeepSeek` 上, 运行下面代码
```
conda activate RAGDeepSeek
python RAGDeepSeek/SFTDeepSeek.py config/default_config.json
```

# 运行主文件
## 意图识别
在运行 `main.py` 之前, 需要对一些字段进行特殊配置

在 `main.py` 的 `intent_recognition_start_up` 函数中, 填写已经训练好的模型的地址

已经训练好了两个模型, 存放在 `/newSSD/home/TuRan/Downloads/models/lr_0.001_epoch_3_model_DeepSeek-R1-Distill-Llama-8B_time_23_02_54` 目录中 <br>
一个是根据 `Llama3.1` 微调得来, 另一个是根据 `DeepSeek-R1` 微调得来. `fine-tuned Llama3.1` 最终accuracy在89%左右, `fine-tuned DeepSeek-R1` 最终accuracy在85%左右

```
bash run.sh
```

## 多轮意图识别
```
bash run.sh
```