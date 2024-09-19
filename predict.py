import argparse
import torch
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import json
import numpy as np
from torchvision import transforms

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image along with the probability of that name.")
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category to name mapping file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    image = Image.open(image_path)
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = image_transforms(image)
    return image_tensor

def predict(image_path, model, topk=5, gpu=False):
    image = process_image(image_path)
    image = image.unsqueeze(0)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p.cpu().numpy().squeeze()
        top_class = top_class.cpu().numpy().squeeze()
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_class]
    return top_p, top_class

def main():
    args = get_input_args()
    model = load_checkpoint(args.checkpoint)
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    probs, classes = predict(args.input, model, args.top_k, args.gpu)
    class_names = [cat_to_name[str(cls)] for cls in classes]
    for prob, class_name in zip(probs, class_names):
        print(f"{class_name}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()