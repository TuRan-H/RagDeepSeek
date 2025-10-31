# Configure Python Environment
Install Python
```
conda create -n RAGDeepSeek python==3.10.*
```

Install dependencies
```
bash install.sh
```

# Run Fine-tuning Code
First, fill in the `model_path` field in the configuration file `./config/default_config.json`.
Fill in the address of the model downloaded from Hugging Face. The server file is located at `/newSSD/home/TuRan/Downloads/models/DeepSeek-R1-Distill-Llama-8B`.


In the working directory of the code `/newSSD/home/TuRan/Projects/RagDeepSeek`, run the following code:
```
conda activate RAGDeepSeek
python RAGDeepSeek/SFTDeepSeek.py config/default_config.json
```

# Run Main File
## Intent Recognition
Before running `main.py`, some fields need to be specially configured.

In the `intent_recognition_start_up` function of `main.py`, fill in the address of the trained model.

Two models have been trained and are stored in the `/newSSD/home/TuRan/Downloads/models/lr_0.001_epoch_3_model_DeepSeek-R1-Distill-Llama-8B_time_23_02_54` directory.
One is fine-tuned based on `Llama3.1`, and the other is fine-tuned based on `DeepSeek-R1`. The final accuracy of `fine-tuned Llama3.1` is about 89%, and the final accuracy of `fine-tuned DeepSeek-R1` is about 85%.

```
bash run.sh
```

## Multi-turn Intent Recognition
```
bash run.sh
```