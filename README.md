# ViT-VAE

## Train VAE

### Train AutoencoderKL

```bash
torchrun --nnodes=$NUM_NODES --nproc-per-node=$NUM_GPUS train_vae.py \
    --pretrained_model_name_or_path /path/to/pretrained/model \
    --train_data_dir /path/to/images 
```

## Train ViT-VAE
