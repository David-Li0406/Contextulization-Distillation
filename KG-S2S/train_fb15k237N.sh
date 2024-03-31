export CUDA_VISIBLE_DEVICES=6

# FB15K-237N
python3 main.py -dataset 'FB15k-237N' \
                -lr 0.001 \
                -epoch 50 \
                -batch_size 32 \
                -src_descrip_max_length 80 \
                -tgt_descrip_max_length 80 \
                -use_soft_prompt \
                -use_rel_prompt_emb \
                -seq_dropout 0.2 \
                -num_beams 40 \
                -eval_tgt_max_length 30 \
                -skip_n_val_epoch 30

# evaluation commandline:
python3 main.py -dataset 'FB15k-237N' \
                -src_descrip_max_length 80 \
                -tgt_descrip_max_length 80 \
                -use_soft_prompt \
                -use_rel_prompt_emb \
                -num_beams 40 \
                -eval_tgt_max_length 30 \
                -model_path /home/dawei/projects/dawei/KG-S2S/checkpoint/FB15k-237N-2023-05-10-12:24:40.028910/FB15k-237N-epoch=035-val_mrr=0.3480.ckpt \
                -use_prefix_search 