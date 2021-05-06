python3 ../main_contrastive.py \
    imagenet100 \
    resnet18 \
    --data_folder /datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/test \
    --epochs 100 \
    --optimizer sgd \
    --lars \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --classifier_lr 0.03 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --gpus 0 1 \
    --num_workers 8 \
    --hidden_dim 2048 \
    --pred_hidden_dim 4096 \
    --encoding_dim 256 \
    --queue_size 98304 \
    --name nnclr \
    --method nnclr \
    --project contrastive_learning \
    --wandb
