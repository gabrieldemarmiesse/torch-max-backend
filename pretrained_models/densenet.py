import torch
from torchvision.models import densenet121
from torchvision import transforms
from PIL import Image
import requests
from torch_max_backend import max_backend, get_accelerators

device = "cuda" if len(list(get_accelerators())) >= 2 else "cpu"

model = densenet121(pretrained=True).to(torch.float32).to(device)
model.eval()

model = torch.compile(model, backend=max_backend, fullgraph=True)

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image_url = "https://raw.githubusercontent.com/jigsawpieces/dog-api-images/refs/heads/main/boxer/n02108089_10229.jpg"

response = requests.get(image_url)
image = Image.open(requests.get(image_url, stream=True).raw)
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_batch)

probabilities = torch.nn.functional.softmax(output[0], dim=0)

imagenet_classes_url = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)
imagenet_classes = requests.get(imagenet_classes_url).text.strip().split("\n")

top5_prob, top5_catid = torch.topk(probabilities, 5)

print("Top 5 Predictions: (boxer should come first)")
for i in range(top5_prob.size(0)):
    print(f"{i + 1}. {imagenet_classes[top5_catid[i]]}: {top5_prob[i].item():.4f}")
