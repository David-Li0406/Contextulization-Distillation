export CUDA_VISIBLE_DEVICES=5

CONTEXTUALIZATION=true
RECONSTRUCTION=false
DATASET=FB15k-237N
LR=0.001
EPOCH=50
BATCH_SIZE=16
SKIP_N_VAL_EPOCH=30

# FB15K-237N
python3 main.py -dataset $DATASET \
                -contextualization $CONTEXTUALIZATION \
                -reconstruction $RECONSTRUCTION \
                -lr $LR \
                -epoch $EPOCH \
                -batch_size $BATCH_SIZE \
                -src_descrip_max_length 80 \
                -tgt_descrip_max_length 80 \
                -use_soft_prompt \
                -use_rel_prompt_emb \
                -seq_dropout 0.2 \
                -num_beams 40 \
                -eval_tgt_max_length 30 \
                -skip_n_val_epoch $SKIP_N_VAL_EPOCH \
                -save_dir checkpoint/dataset-${DATASET}_contextualization-${CONTEXTUALIZATION}_reconstruction-${RECONSTRUCTION}_lr-${LR}_epoch-${EPOCH}_batch_size-${BATCH_SIZE}_skip_n_val_epoch-${SKIP_N_VAL_EPOCH}

python3 main.py -dataset $DATASET \
                -contextualization false \
                -reconstruction $RECONSTRUCTION \
                -src_descrip_max_length 80 \
                -tgt_descrip_max_length 80 \
                -use_soft_prompt \
                -use_rel_prompt_emb \
                -num_beams 40 \
                -eval_tgt_max_length 30 \
                -model_path  \
                -use_prefix_search 