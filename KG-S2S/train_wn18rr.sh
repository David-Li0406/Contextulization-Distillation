export CUDA_VISIBLE_DEVICES=6

# WN18RR
python3 main.py -dataset 'WN18RR' \
                -lr 0.001 \
                -epoch 100 \
                -batch_size 64 \
                -src_descrip_max_length 40 \
                -tgt_descrip_max_length 10 \
                -use_soft_prompt \
                -use_rel_prompt_emb \
                -seq_dropout 0.1 \
                -num_beams 40 \
                -eval_tgt_max_length 30 \
                -skip_n_val_epoch 30 \
                -contextualization

# evaluation commandline:
python3 main.py -dataset 'WN18RR' \
                -src_descrip_max_length 40 \
                -tgt_descrip_max_length 10 \
                -use_soft_prompt \
                -use_rel_prompt_emb \
                -num_beams 40 \
                -eval_tgt_max_length 30 \
                -model_path path/to/trained/model \
                -use_prefix_search