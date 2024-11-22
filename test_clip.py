from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)
pooled_output = outputs.pooler_output  # pooled CLS states
print(pooled_output.shape)