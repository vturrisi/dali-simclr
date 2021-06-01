python3 ../../main_contrastive.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/val \
    --max_epochs 100 \
    --gpus 0,1 \
    --distributed_backend ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 30.0 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --min_scale_crop 0.2 \
    --zero_init_residual \
    --name simsiam \
    --no_lr_scheduler_for_pred_head \
    --dali \
    --project contrastive_learning \
    --wandb \
    simsiam \
    --proj_hidden_dim 2048 \
    --pred_hidden_dim 512 \
    --output_dim 2048
