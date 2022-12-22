This repository includes the codebase for adapting EgoVLP features to NLQ & MQ, Ego4D challenges. 

# NLQ (VSLNet)

- Extract video features `python3 run/test_nlq.py --subsample 'video'`
  - We have uploaded EgoVLP clip-level features on G Drive, you can download them directly and do not need to extract them by yourself.
    - train & val clip features: https://drive.google.com/file/d/1TXBlLDqDuL_XPCuXlgiikEfVXO8Ly6eM/view
    - test video features: https://drive.google.com/file/d/1-CGZg9t-kpW5bmg9M62VHk5eYTllsPKV/view
- For the text branch, I recommend directly loading the pretrained text encoder, first download the egovlp checkpoints: https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7/view
- Set up VSLNet codebase.
  - Install environment dependency:  `pip install environment.txt`
  - The metadata has been included in the codebase.
  - Place the downloaded pertained checkpoint to the `VSLNet/utils/frozen_ego_minimum/pretrained`
  - Run the training script: `bash VSLNet/scripts/03_egovlp_egonce.sh`
  - You can monitor the training log in the such path: `VSLNet/checkpoints/egovlp_egonce_default/egovlp_egonce_batch32/vslnet_nlq_official_v1_egovlp_egonce_256_EgoVLP/model/eval_results.txt`

- Model configs / training log / fine-tuned checkpoints

  - default setting: `VSLNet/checkpoints/egovlp_egonce_default`

  - the best setting, w/ hyper-parameters searching (bs, LR, max_pos_len): `VSLNet/checkpoints/hps_search_egovlp_egonce`

    - ```bash
      # how we perform hps
      EXP_NUMBER = 0
      
      for BSIZE in [4, 8, 16, 32, 128, 512]:
          for LR_RATE in [0.0005,0.0001,0.00005,0.00001]:
              for MAXPOSLEN in [64,128,256,512]:
                  create_sbatches(EXP_NUMBER, BSIZE, LR_RATE, MAXPOSLEN)
                  EXP_NUMBER+=1
                              
      print(f'Total Number of experiments {EXP_NUMBER}.')
      ```

- Experimental results:

| Model  | Video-Text Pre-extrated Features        | R@1, IoU=0.3 | R@5, IoU=0.3  | R@1, IoU=0.5 | R@5, IoU=0.5   |
| ------ | ----------- | -------- | ----------------- | ----------------- | ---- |
| [VSLNet](https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet) | SlowFast + BERT  | 5.45 | 10.74 | 3.12 | 6.63
| [VSLNet](https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet) | EgoVLP | **10.84** | **18.84** | **6.81** | **13.45**


# MQ (VSGN)

- Extract video features `python3 run/test_mq.py --subsample 'video'`

  - We have uploaded EgoVLP clip-level features on G Drive, you can download them directly and do not need to extract them by yourself.

    - train & val clip features: https://drive.google.com/file/d/1-HEUCdyfNX7CBZhz40yiyTr7to_p7wUi/view

    - test video features: https://drive.google.com/file/d/1-JmezY3eIkHKJ1JBA_AA8QWBoY3W2HpS/view

- Set up VSGN codebase

  - Install environment dependency:  `pip install environment.txt`

    ```bash
    conda create -n pytorch160 python=3.7 
    conda activate pytorch160   
    conda install pytorch=1.6.0 torchvision cudatoolkit=10.1.243 -c pytorch   
    conda install -c anaconda pandas    
    conda install -c anaconda h5py  
    conda install -c anaconda scipy 
    conda install -c conda-forge tensorboardx   
    conda install -c anaconda joblib    
    conda install -c conda-forge matplotlib 
    conda install -c conda-forge urllib3
    ```

  - training script: `bash VSGN/script/train_infer_eval_ego_nce.sh` by setting `FEATURES=PATH downloaded video features`

- Model configs / training log / fine-tuned checkpoints

  - default setting: `VSGN/outputs/egovlp_egonce_default_features`

  - the best setting, w/ hyper-parameters searching (bs, LR, step_size, step_gamma): `VSLNet/outputs/hps_search_egovlp_egonce_features`

    - ```bash
      # how we perform hps
      EXP_NUMBER = 0
      INPUT_DIM=256
      
      for BSIZE in [8,16,32]:
          for LR_RATE in [0.0005,0.0001,0.00005,0.00001]:
              for STEP in [5,15,30]:
                  for GAMMA in [0.5,0.25,0.1,0.05]:
                      create_sbatches(EXP_NUMBER, INPUT_DIM, BSIZE, LR_RATE, STEP, GAMMA)
                      EXP_NUMBER+=1
                              
      print(f'Total Number of experiments {EXP_NUMBER}.')
      ```

- Experimental results:

| Model  | Video Pre-extrated Features         | R@1, IoU=0.5 | R@5, IoU=0.5   | mAP
| ------ | ----------- | --------  | ---- | ---- |
| [VSGN](https://github.com/EGO4D/episodic-memory/tree/main/MQ) | SlowFast  | 25.16 | 46.18 | 6.03
| [VSGN](https://github.com/EGO4D/episodic-memory/tree/main/MQ) | EgoVLP | **30.14** | **51.98** | **11.39** |

