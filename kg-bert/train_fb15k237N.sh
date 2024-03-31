export CUDA_VISIBLE_DEVICES=6

python3 run_bert_link_prediction.py \
    --task_name kg  \
    --do_eval \
    --do_predict \
    --data_dir ./data/FB15k-237N\
    --bert_model bert-base-cased\
    --max_seq_length 50 \
    --train_batch_size 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --output_dir ./output_FB15k-237N/  \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 5000 \
    --reconstruction true