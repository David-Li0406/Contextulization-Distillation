export CUDA_VISIBLE_DEVICES=1

python3 main.py -dataset FB15k-237N \
                -epoch 300 \
                -batch_size 128 \
                -pretrained_model bert-base-uncased \
                -desc_max_length 40 \
                -lr 5e-4 \
                -prompt_length 10 \
                -alpha 0.1 \
                -n_lar 8 \
                -label_smoothing 0.1 \
                -embed_dim 144 \
                -k_w 12 \
                -k_h 12 \
                -alpha_step 0.00001 \
                -reconstruction true


# evaluation commandline:
python3 main.py -dataset FB15k-237N \
                -batch_size 128 \
                -pretrained_model bert-base-uncased \
                -desc_max_length 40 \
                -lr 5e-4 \
                -prompt_length 10 \
                -alpha 0.1 \
                -n_lar 8 \
                -label_smoothing 0.1 \
                -embed_dim 144 \
                -k_w 12 \
                -k_h 12 \
                -alpha_step 0.00001 \
                -model_path path-to-checkpoint