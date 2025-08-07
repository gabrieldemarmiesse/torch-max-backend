from max_torch_backend import MaxCompiler
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

inputs = processor(images=image, return_tensors="pt")

model = torch.compile(model, backend=MaxCompiler)
inputs = inputs.to("gpu")
outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
assert model.config.id2label[predicted_class_idx] == "something???"
