import runpod
import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from diffusers.utils import load_image
from PIL import Image
import base64
import io

def init_model():
    base_model = 'black-forest-labs/FLUX.1-dev'
    controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'
    
    # 加载控制网络
    controlnet_union = FluxControlNetModel.from_pretrained(
        controlnet_model_union, 
        torch_dtype=torch.bfloat16
    )
    controlnet = FluxMultiControlNetModel([controlnet_union])
    
    # 加载管道
    pipe = FluxControlNetPipeline.from_pretrained(
        base_model, 
        controlnet=controlnet, 
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    return pipe

# 初始化模型
pipe = init_model()

def handler(event):
    try:
        # 获取输入参数
        input_data = event["input"]
        prompt = input_data.get("prompt", "")
        control_image = input_data.get("control_image", "")  # base64编码的图像
        control_mode = input_data.get("control_mode", 0)  # 控制模式
        
        if not prompt or not control_image:
            return {"error": "Missing prompt or control_image"}
        
        # 解码base64图像
        image_data = base64.b64decode(control_image)
        control_image = Image.open(io.BytesIO(image_data))
        
        # 获取图像尺寸
        width, height = control_image.size
        
        # 生成图像
        output = pipe(
            prompt,
            control_image=control_image,
            control_mode=control_mode,
            width=width,
            height=height,
            controlnet_conditioning_scale=0.5,
            num_inference_steps=24,
            guidance_scale=3.5,
            generator=torch.manual_seed(42)
        ).images[0]
        
        # 将生成的图像转换为base64
        buffered = io.BytesIO()
        output.save(buffered, format="PNG")
        output_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "generated_image": output_image
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
