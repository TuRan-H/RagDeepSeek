# 配置ollama
安装ollama
```
curl https://ollama.ai/install.sh | sh
```

ollama 下载 deepseek-r1:14b模型
```
ollama pull deepseek-r1:14b
```
deepseek-r1:14b模型已经下载好了, 存放在 `/newSSD/home/TuRan/Downloads/ollama` 中

# 下载embedding模型
embedding模型已经下载好了, 存放在 `/newSSD/home/TuRan/Downloads/models/all-MiniLM-L6-v2`
在使用时, 将config的 `embedding_model` 字段填上模型目录即可

或者在huggingface中下载该模型

# 配置python环境
安装python
```
conda create -n RAGDeepSeek python==3.10.*
```

安装依赖
```
bash install.sh
```