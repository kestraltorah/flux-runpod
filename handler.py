import runpod
import torch
import logging
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import base64
import io

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_model():
    try:
        logger.info("开始加载模型...")
        
        # 加载基础模型
        base_model = "stabilityai/stable-diffusion-xl-base-1.0"
        logger.info(f"加载基础模型: {base_model}")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        # 移动到GPU
        pipe = pipe.to("cuda")
        logger.info("模型已加载到GPU")
        
        return pipe
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

def handler(event):
    try:
        logger.info("收到请求")
        logger.info(f"输入数据: {event}")
        
        # 获取输入参数
        input_data = event["input"]
        prompt = input_data.get("prompt", "")
        
        if not prompt:
            return {"error": "Missing prompt"}
        
        logger.info(f"处理提示词: {prompt}")
        
        # 生成图像
        output = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]
        
        # 将生成的图像转换为base64
        buffered = io.BytesIO()
        output.save(buffered, format="PNG")
        output_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        logger.info("图像生成完成")
        
        return {
            "generated_image": output_image
        }
        
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        return {"error": str(e)}

# 初始化模型
logger.info("正在初始化模型...")
pipe = init_model()
logger.info("模型初始化完成")

runpod.serverless.start({"handler": handler})
