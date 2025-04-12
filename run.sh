#!/bin/bash

source /newSSD/home/TuRan/.anaconda3/bin/activate
conda activate RAGDeepSeek

if [ -f .env ]; then
    export $(cat .env | grep -v '^[#;]' | xargs)
fi

python main_bert_base.py