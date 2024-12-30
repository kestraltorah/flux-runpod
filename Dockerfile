FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y git wget && \
    rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip install --no-cache-dir \
    runpod \
    transformers \
    accelerate \
    safetensors \
    pillow \
    xformers \
    controlnet_aux \
    git+https://github.com/huggingface/diffusers.git

# 复制handler文件
COPY handler.py /app/handler.py

# 设置环境变量
ENV PYTHONPATH=/app

# 启动命令
CMD [ "python", "-u", "handler.py" ]
