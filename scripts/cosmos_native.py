import torch
from diffusers import CosmosTextToWorldPipeline
from diffusers.utils import export_to_video

model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"
pipe = CosmosTextToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves. The robot's metallic body gleams under the bright, even lighting, highlighting its futuristic design and intricate joints. A glowing blue light emanates from its chest, adding a touch of advanced technology. The background is dominated by rows of boxes, suggesting a highly organized storage system. The floor is lined with wooden pallets, enhancing the industrial setting. The camera remains static, capturing the robot's poised stance amidst the orderly environment, with a shallow depth of field that keeps the focus on the robot while subtly blurring the background for a cinematic effect."

output = pipe(prompt=prompt).frames[0]
export_to_video(output, "/fsx/jdelavande/benchlab/videos/data/cosmos.mp4", fps=30)
