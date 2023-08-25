# 基於LLaMA-7B及LoRA之醫學數據問答集
本專案旨在利用 LLaMA-7B 模型，並透過 LoRA 的微調方式，應用於醫學數據問答。我們使用了 PubMedQ&A 資料集進行訓練。
## 如何使用
請先下載 `requirement.txt`中所列的相依套件：
```
pip install -r requirement.txt
```
安裝完套件後，執行以下指令：
```
python app.py
```
即可進入實際測試界面。您只需輸入指示（Instruction）與輸入（Input），即可獲得相對應的回覆（Output）。

#### 注意：請確認您的 GPU 設備是否擁有足夠的 vRAM。本專案開發時使用了 nVidia 2070×2。

## 如何微調
若想進行模型的微調，請打開`llama.ipynb`。首先，確保資料集格式為 ['instruction', 'input', 'output']。其中，輸入（input）部分可為空。接著，可以調整以下參數：
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
可以根據本身的硬體設備和需求，調整這些參數。在我們的實驗中，使用了包含 211,300 筆資料的資料集，並在搭載 nVidia 2070×2 的 GPU 上進行訓練，共耗時約 8 小時。

## 如何使用訓練後之模型
當訓練完成後，可以在指定的資料夾中找到儲存的模型檔案，檔案名稱應為`adapter_model.bin`和 `adapter_config.json`。在我們的實驗中，我們將這些模型檔案保存在名為`lora-alpaca-1000`的資料夾中，您可以在該資料夾中查閱我們訓練完成的模型檔案及其相關格式。接著，需編輯`app.py`，將`lora_weights`參數指向儲存模型的資料夾路徑，然後即可執行程式。
