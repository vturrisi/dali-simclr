python3 main_pretrain.py \
    --dataset imagenet100 \
    --backbone vit_base \
    --train_data_path $1/train \
    --val_data_path $1/val \
    --data_format dali \
    --max_epochs 400 \
    --warmup_epochs 40 \
    --devices 0,1,2,3,4,5,6,7 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer adamw \
    --adamw_beta1 0.9 \
    --adamw_beta2 0.95 \
    --scheduler warmup_cosine \
    --lr 2.0e-4 \
    --classifier_lr 2.0e-4 \
    --weight_decay 0.05 \
    --batch_size 64 \
    --num_workers 8 \
    --brightness 0 \
    --contrast 0 \
    --saturation 0 \
    --hue 0 \
    --gray_scale_prob 0 \
    --gaussian_prob 0 \
    --solarization_prob 0 \
    --min_scale 0.08 \
    --num_crops_per_aug 1 \
    --name mae-vit-400ep-imagenet100 \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint \
    --method mae \
    --decoder_embed_dim 512 \
    --decoder_depth 8 \
    --decoder_num_heads 16 \
    --mask_ratio 0.75 \
    --norm_pix_loss