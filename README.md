# 基於LLaMA-7B及LoRA之醫學數據問答集
本專案將使用LLaMA-7B模型，並使用LoRA之方式進行微調，其訓練資料集PubMedQ&A Dataset

## 如何使用
請先下載requirement.txt
```
pip install -r requirement.txt
```
之後執行
```
python app.py
```
即可進入實際測試畫面，並輸入Instruction和Input即可得到Output

#### 注意：請確定自身設備之GPU vRAM是否足夠，本專案之設備為nVidia 2070×2

## 如何微調
請先確定自身的資料集為['instruction','input','output']之格式，input可為空，之後調整下列參數
```
MICRO_BATCH_SIZE = 4    
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  
LEARNING_RATE = 3e-4  
CUTOFF_LEN = 128  
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE=10
```
參數可自行依照設備來進行微調，本次使用資料集共211.3K筆，在GPU為2070×2之情況下共訓練8ours

## 如何使用訓練後之模型