python3 main_linear.py \
    --dataset imagenet \
    --backbone resnet50 \
    --data_dir /data/datasets \
    --train_dir imagenet/train \
    --val_dir imagenet/val \
    --max_epochs 100 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 3.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 5 \
    --dali \
    --name mocov2plus-imagenet-linear-eval \
    --pretrained_feature_extractor PATH \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint \
    --auto_resume