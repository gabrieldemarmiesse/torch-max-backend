import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from max_torch_backend import MaxCompiler


model = models.vgg11(pretrained=True)
model.eval()
model = torch.compile(model, backend=MaxCompiler)

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_image(image_path_or_url):
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    labels = response.text.strip().split("\n")
    return labels


def predict_image(image_path_or_url, top_k=5):
    image = load_image(image_path_or_url)

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_class = torch.topk(probabilities, top_k)

    labels = load_imagenet_labels()

    for i in range(top_k):
        class_idx = top_class[i].item()
        prob = top_prob[i].item()
        label = labels[class_idx]
        print(f"{i + 1:2d}. {label:30s} ({prob:.3f})")


if __name__ == "__main__":
    image_url = "https://raw.githubusercontent.com/jigsawpieces/dog-api-images/refs/heads/main/boxer/n02108089_10229.jpg"
    predict_image(image_url, top_k=5)
