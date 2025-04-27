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
embedding模型已经下载好了, 存放在 `/newSSD/home/TuRan/Downloads/models/`
在使用时, 将config的 `embedding_model` 字段填上模型目录即可

# 配置python环境
安装python
```
conda create -n RAGDeepSeek python==3.10.*
```

安装依赖
```
pip install fastapi-0.87.0-py3-none-any.whl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.50.1
pip install lightrag-hku==1.2.6
pip install open-intent-classifier==0.0.8
pip install pandas dotenv tiktoken pipmaster future uvicorn
pip install openai==1.69.0
pip install aiohttp
```



在安装完所有依赖之后, 可能会出现numpy版本错误的问题
```
pip install numpy==1.24.4
```