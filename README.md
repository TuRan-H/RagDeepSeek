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


# 配置python环境
创建python环境
```
conda create -n RAGDeepSeek python==3.10.*
```
安装pytorch
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

安装transformers
```
pip install transformers==4.50.1
```

安装lightrag
```
pip install lightrag-hku==1.2.6
```

其他
```
pip install pandas, dotenv, tiktoken, pipmaster, future, uvicorn
```

在安装完所有依赖之后, 可能会出现numpy版本错误的问题
```
pip install numpy==1.24.4
```

# 下载embedding模型
embedding模型已经下载好了, 存放在 `/newSSD/home/TuRan/Downloads/models/`
在使用时, 将config的 `embedding_model` 字段填上模型目录即可