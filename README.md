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
  python gen_zipf_c_rank.py \
    --dataset_name=<檔案名稱> \
    --req_kind=<request種類> \
    --length=<request長度> \
    --zipf_para=<zipf參數> \
    --change_rank=<rank變化週期>
  ```
  - change zipf parameter(Dyna-zipf-2)
  ```
  python gen_zipf_c_para.py \
    --dataset_name=<檔案名稱> \
    --req_kind=<request種類> \
    --length=<request長度> \
    --zipf_para=<zipf起始參數>
  ```
- Generate snm trace
  ```
  python snm_1.py
  ```
- Generate random content size
  ```
  python gen_req_content.py <檔案名稱> <request種類> <最大content size>
  ```
- Trace will be saved in dataset/req_trace
- Content size will be saved in dataset/req_data
## Testing
- Testing Command
  ```
  python main_2.py 
    --cache_size=10 \ 
    --experiment_name='Test' \
    --dataset='snm_50_l9666.txt' \
    --req_to_size_dataset='size_50_' \
    --model_filepath="./dataset/parameter/model/DQN_our_015_paper.json" \
    --feature_filepath="./dataset/parameter/features/Ffqr_norm/full_new_norm.json"
  ```