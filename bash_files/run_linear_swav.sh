python3 ../main_linear.py \
    imagenet100 \
    resnet18 \
    --data_folder /data/datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/val \
    --epochs 100 \
    --optimizer sgd \
    --scheduler step \
    --lr 3.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 10 \
    --dali \
    --name swav-linear-eval \
    --pretrained_feature_extractor trained_models/PATH \
    --project contrastive_learning \
    --wandb
