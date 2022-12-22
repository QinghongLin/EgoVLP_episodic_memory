export CUDA_VISIBLE_DEVICES=2
FEATURES=./Evaluation/ego4d/ht100m
CHECKPOINT=./outputs/hps_search_ht100m_features/59
INPUT_DIM=256

python Infer.py --use_xGPN --is_train false --dataset ego4d --feature_path $FEATURES --checkpoint_path $CHECKPOINT --input_feat_dim $INPUT_DIM --output_path $CHECKPOINT --infer_datasplit test --batch_size 4

python Eval.py --dataset ego4d --output_path $CHECKPOINT --out_prop_map true --eval_stage all --infer_datasplit test

python Merge_detection_retrieval.py --output_path $CHECKPOINT
