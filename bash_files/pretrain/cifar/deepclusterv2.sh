python3 ../../../main_pretrain.py \
    --dataset $1 \
    --encoder resnet18 \
    --data_dir ./datasets \
    --max_epochs 1000 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --scheduler warmup_cosine \
    --lr 0.6 \
    --min_lr 0.0 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 256 \
    --num_workers 3 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --gaussian_prob 0.0 0.0 \
    --name deepclusterv2-$1 \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --method deepclusterv2 \
    --proj_hidden_dim 2048 \
    --output_dim 128 \
    --num_prototypes 3000 3000 3000
