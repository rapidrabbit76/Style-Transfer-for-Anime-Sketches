FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
WORKDIR /training
RUN pip install \
    tqdm==4.62.3 \
    wandb==0.12.10 \
    opencv-python-headless==4.5.5.62 \
    easydict==1.9 \
    albumentations==1.1.0