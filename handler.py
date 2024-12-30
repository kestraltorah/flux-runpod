import runpod
import torch
import logging
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from PIL import Image
import base64
import io

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_model():
    try:
        logger.info("开始加载模型...")
        
        # 加载模型
        base_model = 'black-forest-labs/FLUX.1-dev'
        controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
        
        logger.info("加载控制网络...")
        controlnet_union = FluxControlNetModel.from_pretrained(
            controlnet_model_union, 
            torch_dtype=torch.bfloat16
        )
        controlnet = FluxMultiControlNetModel([controlnet_union])
        
        logger.info("加载基础模型...")
        pipe = FluxControlNetPipeline.from_pretrained(
            base_model, 
            controlnet=controlnet, 
            torch_dtype=torch.bfloat16
        )
        
        # 移动到GPU
        pipe = pipe.to("cuda")
        logger.info("模型已加载到GPU")
        
        return pipe
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

def decode_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def handler(event):
    try:
        logger.info("收到请求")
        
        # 获取输入参数
        input_data = event["input"]
        prompt = input_data.get("prompt", "")
        control_image = input_data.get("control_image", "")
        control_mode = input_data.get("control_mode", 0)
        
        if not prompt or not control_image:
            return {"error": "Missing prompt or control_image"}
        
        logger.info(f"处理提示词: {prompt}")
        logger.info(f"控制模式: {control_mode}")
        
        # 解码控制图像
        control_image = decode_base64_image(control_image)
        width, height = control_image.size
        
        # 生成图像
        output = pipe(
            prompt=prompt,
            control_image=control_image,
            control_mode=control_mode,
            width=width,
            height=height,
            controlnet_conditioning_scale=0.5,
            num_inference_steps=24,
            guidance_scale=3.5,
            generator=torch.manual_seed(42)
        ).images[0]
        
