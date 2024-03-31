CUDA_VISIBLE_DEVICES=0 python main.py  --max_epochs=15  --num_workers=16 \
   --model_name_or_path  bert-base-uncased \
   --num_sanity_val_steps 0 \
   --model_class KGRECModel \
   --lit_model_class KGRECPretrainLitModel \
   --label_smoothing 0.1 \
   --data_class KGRECPretrainDataModule \
   --batch_size 32 \
   --dataset ml20m \
   --eval_batch_size 64 \
   --max_seq_length 256 \
   --max_entity_length 256 \
   --early_stop 0 \
   --lr 3e-4 