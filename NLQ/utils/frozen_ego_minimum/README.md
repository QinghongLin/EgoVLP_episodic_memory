# dependency

- following this page: https://github.com/m-bain/frozen-in-time/blob/main/environment.yml, but I recommend you install according to based on what you lack in your environment when run the code

# Setup path

- dataset

  - MQ

    `data_loader/Ego4d_MQ_V.py: metadata_dir`  (where you save the JSON file)

    `data_loader/Ego4d_MQ_L.py: metadata_dir`

    `configs/ego4d-mq-v.json: data_dir` (where you save the **rgb video**)

  - NLQ

    `data_loader/Ego4d_NLQ_V.py: metadata_dir (line 28)`  (where you save the JSON file)

    `data_loader/Ego4d_NLQ_T.py: metadata_dir (line 28)`

    `configs/ego4d-nlq-v.json: data_dir` (where you save the **rgb video**)

- model checkpoints

  - install and put EgoVLP checkpoints in pertained_dir `frozen_pt_egoclip`  (you also can replace it with another cp such as cc3m+webvid here: https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/cc-webvid2m-4f_stformer_b_16_224.pth.tar)

# What you need to pay attention

1. **"video-level" vs "clip-level" videos**: since I extract visual features from the raw Ego4d video, and you may extract features from clip video, so you need to modify the code of data_loader, mainly the start/end timestamp. e.g., lines 33-45 & lines 66 of `Ego4d_MQ/NLQ_V.py`
2. **"Cls feature" vs "tokens feature" of sentence**: I have define two func `compute_text`/ `compute_text_tokens` respectively in `model/model.py`, you just need to turn `--token` as true in `test_ego4d_mq/nlq_l.py`
3. do not change the `batchsize=1` of `text_xx.py`
4. **besides, can you help me to double-check the details of video feature extraction**? mainly the line 81-92 of `test_ego4d_mq/nlq_v`
   - our model pt on 4f, but maximizing to 16f is possible.

# Extract feature

- NLQ
  - visual features: `test_ego4d_mq_v.py --r $egovlp_checkpoint --split train/val/test --save_feats xxx --gpu x` (one run per split)
  - text features: `test_ego4d_mq_l.py --r $egovlp_checkpoint --tokens --save_feats xxx --gpu x`  (one run for three splits)
- MQ: similar to NLQ

# If you want to fine-tune the model

- such as FT our model on MQ/NLQ tasks by aligning text-video, you need to re-implement the data_loader.