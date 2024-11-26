# ViT-VAE

## Train VAE

### Train AutoencoderKL

```bash
torchrun --nnodes=$NUM_NODES --nproc-per-node=$NUM_GPUS train_vae.py \
    --pretrained_model_name_or_path /path/to/pretrained/model \
    --train_data_dir /path/to/images
```

## Train ViT-VAE

```bash
torchrun --nnodes=$NUM_NODES --nproc-per-node=$NUM_GPUS train_vitvae.py \
    --pretrained_model_name_or_path /path/to/pretrained/model \
    --pretrained_vision_model_name_or_path /path/to/pretrained/vision_model \
    --train_data_dir /path/to/images
```

## Reference

[train_vae.py](https://github.com/aandyw/diffusers/blob/vae-training/examples/vae/train_vae.py)
