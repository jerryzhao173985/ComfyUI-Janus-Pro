import os

class JanusModelLoader:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B"],),
            },
        }
    
    RETURN_TYPES = ("JANUS_MODEL", "JANUS_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "Janus-Pro"

    def load_model(self, model_name):
        print("Loading Janus model...")
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "mps"
        dtype = torch.float16

        # 获取ComfyUI根目录
        comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        # 构建模型路径
        model_dir = os.path.join(comfy_path, 
                               "models", 
                               "Janus-Pro",
                               os.path.basename(model_name))
        if not os.path.exists(model_dir):
            raise ValueError(f"Local model not found at {model_dir}. Please download the model and place it in the ComfyUI/models/Janus-Pro folder.")
            
        vl_chat_processor = VLChatProcessor.from_pretrained(model_dir)
        
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True
        ).to(dtype).to(device).eval()
        
        return (vl_gpt, vl_chat_processor) 