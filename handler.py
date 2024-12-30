import runpod
import torch
import logging
from diffusers import FluxControlNetModel, FluxControlNetPipeline
from diffusers.models import FluxMultiControlNetModel
from PIL import Image
import base64
import io

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_model():
    try:
        logger.info("Starting model initialization...")
        
        # 设置模型路径
        base_model_id = "black-forest-labs/FLUX.1-dev"
        controlnet_id = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
        
        # 加载控制网络
        logger.info(f"Loading controlnet from {controlnet_id}")
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # 创建多控制网络模型
        multi_controlnet = FluxMultiControlNetModel([controlnet])
        
        # 加载完整管道
        logger.info(f"Loading base model from {base_model_id}")
        pipe = FluxControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=multi_controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        
        # 移动到GPU
        pipe = pipe.to("cuda")
        logger.info("Model loaded successfully")
        
        return pipe
    
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        raise

def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise

def handler(event):
    try:
        logger.info("Received request")
        
        # 获取输入参数
        input_data = event["input"]
        prompt = input_data.get("prompt", "")
        control_image_b64 = input_data.get("control_image", "")
