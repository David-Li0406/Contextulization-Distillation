export CUDA_VISIBLE_DEVICES=6

CONTEXTUALIZATION=true
RECONSTRUCTION=false
DATASET=WN18RR
LR=0.001
EPOCH=100
BATCH_SIZE=32
SKIP_N_VAL_EPOCH=30


# WN18RR
python3 main.py -dataset $DATASET \
                -small_dataset \
                -lr $LR \
                -epoch $EPOCH \
                -batch_size $BATCH_SIZE \
                -src_descrip_max_length 40 \
                -tgt_descrip_max_length 10 \
                -use_soft_prompt \
                -use_rel_prompt_emb \
                -seq_dropout 0.1 \
                -num_beams 40 \
                -eval_tgt_max_length 30 \
                -skip_n_val_epoch $SKIP_N_VAL_EPOCH \
                -contextualization $CONTEXTUALIZATION \
                -save_dir checkpoint/dataset-${DATASET}_contextualization-${CONTEXTUALIZATION}_reconstruction-${RECONSTRUCTION}_lr-${LR}_epoch-${EPOCH}_batch_size-${BATCH_SIZE}_skip_n_val_epoch-${SKIP_N_VAL_EPOCH}

# evaluation commandline:
python3 main.py -dataset 'WN18RR' \
                -src_descrip_max_length 40 \
                -tgt_descrip_max_length 10 \
                -use_soft_prompt \
                -use_rel_prompt_emb \
                -num_beams 40 \
                -eval_tgt_max_length 30 \
                -model_path  \
                -use_prefix_search