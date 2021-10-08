python3 ../../main_pretrain.py \
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
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --classifier_lr 0.1 \
    --weight_decay 1e-6 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --crop_size 32 \
    --num_crops_per_pipeline 1 1 \
    --name dino-$1 \
    --entity unitn-mhug \
    --project solo-learn \
    --wandb \
    --save_checkpoint \
    --method dino \
    --proj_output_dim 256 \
    --proj_hidden_dim 2048 \
    --num_prototypes 4096 \
    --base_tau_momentum 0.9995 \
    --final_tau_momentum 1.0 \
    --momentum_classifier
