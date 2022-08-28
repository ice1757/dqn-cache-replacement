# dqn-cache-replacement
## Installation

* These instructions were tested in ubuntu 20.04 or Window 10 with 16GB memory.

  ```bash
  git clone https://github.com/ice1757/dqn-cache-replacement.git
  ```
## Generate Request Traces
- Generate dynamic-zipf trace
  - change rank(Dyna-zipf-1)
  ```
  python gen_dyna_zipf_1.py \
    --dataset_name=<檔案名稱> \
    --req_kind=<request種類> \
    --length=<request長度> \
    --zipf_para=<zipf參數> \
    --change_rank=<rank變化週期>
  ```
  - change zipf parameter(Dyna-zipf-2)
  ```
  python gen_dyna_zipf_2.py \    --dataset_name=<檔案名稱> \
    --req_kind=<request種類> \
    --length=<request長度> \
    --zipf_para=<zipf起始參數>
  ```
- Generate snm trace
  ```
  python snm_1.py 
    --dataset_name='snm' \
    --req_kind=50
  ```
- Generate random content size
  ```
  python gen_req_content.py <檔案名稱> <max content size> <content size 1 比例> <content size 2 比例> <content size 3 比例>
  ```
- Trace will be saved in dataset/req_trace
- Content size will be saved in dataset/req_data
## Evaluation
- Evaluation for Traditional Algorithm
  ```
  python main.py \
    --cache_size=10 \
    --experiment_name='traditional' \
    --dataset='snm_50_l9666.txt' \
    --req_to_size_dataset='size_50_' \
    --feature_filepath="./dataset/parameter/features/Ffqr_norm/full_new_norm.json"
  ```
- Evaluation for DQN
  ```
  python main.py \
    --cache_size=10 \
    --experiment_name='DQN' \
    --dataset='snm_50_l9666.txt' \
    --req_to_size_dataset='size_50_' \
    --model_filepath="./dataset/parameter/model/DQN_our.json" \
    --feature_filepath="./dataset/parameter/features/Ffqr_norm/full_new_norm.json"
  ```
- 相關結果會產生在資料夾檔名為 dqn-cache-replacement/logs 底下
