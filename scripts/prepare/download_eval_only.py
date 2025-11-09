from diffusers import DiffusionPipeline

repo_id = "stabilityai/stable-diffusion-2-1-base"
pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
pipeline.save_pretrained("./pretrained/stable-diffusion-2-1-base")

from diffusers import StableDiffusionPipeline
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
pipe.save_pretrained("./pretrained/stable-diffusion-v1-5")