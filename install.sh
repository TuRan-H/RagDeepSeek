#!/bin/bash

source ~/.anaconda3/bin/activate
conda activate RAGDeepSeek

if [ -f .env ]; then
	source .env
fi


pip install fastapi-0.87.0-py3-none-any.whl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.0
pip install lightrag-hku==1.2.6
pip install openai==1.69.0
pip install pydantic==1.10.21
pip install trl==0.17.0
pip install peft==0.15.2
pip install datasets==3.5.1
pip install accelerate==1.6.0
pip install pandas dotenv tiktoken pipmaster future uvicorn aiohttp wandb