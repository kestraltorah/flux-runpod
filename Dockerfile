FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git

# 安装依赖
RUN pip install runpod \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    pillow

# 复制handler文件
COPY handler.py /app/handler.py

# 设置环境变量
ENV PYTHONPATH=/app

# 启动命令
CMD [ "python", "-u", "handler.py" ]
