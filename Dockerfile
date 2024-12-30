FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y git wget && \
    rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.24.0 \
    transformers==4.35.2 \
    accelerate \
    safetensors \
    pillow \
    torch \
    xformers \
    controlnet_aux

# 复制handler文件
COPY handler.py /app/handler.py

# 设置环境变量
ENV PYTHONPATH=/app
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# 启动命令
CMD [ "python", "-u", "handler.py" ]
