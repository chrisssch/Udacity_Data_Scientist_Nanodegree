# load required packages ----------------

import json
import torch
from torchvision import models, transforms
from PIL import Image
import argparse

# command line arguments ----------------

parser = argparse.ArgumentParser()

parser.add_argument("input",
                    action = "store",
                    type = str,
                    help = "Path to image and name of image; needs to be entered as filepath/imagename,.jpg")

parser.add_argument("checkpoint",
                    action = "store",
                    type = str,
                    help = "Location of saved checkpoints")

parser.add_argument("--gpu",
                    action = "store_const",
                    const = "cuda:0",
                    help = "Uses GPU for inference")  

parser.add_argument("--category_names",
                    action = "store",
                    default = "cat_to_name.json",
                    dest = "category_names",
                    type = str,
                    help = "JSON file with category names")

parser.add_argument("--top_k",
                    action = "store",
                    default = 3,
                    dest = "top_k",
                    type = int,
                    help = "Top K predicted categories to display")
        
args = parser.parse_args()

print("--------------------")
print("Selected arguments:")
print("Directory and name of image:", args.input)
print("JSON file with category names:", args.category_names)
print("Directory of saved checkpoint.pth:", args.checkpoint)    
print(args.gpu, "; if = cuda:0, then training on GPU")
print("Number K of predicted categories to display: K =", args.top_k)
print("--------------------")

# class names --------------------

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
# load saved model --------------------

filepathX = args.checkpoint+".pth"
print(filepathX)

def load_checkpoint(filepath = filepathX):
    model_state = torch.load(filepath)
    model = models.__dict__[model_state["arch"]](pretrained = True)
    model.classifier = model_state["classifier"]
    model.load_state_dict(model_state["state_dict"])
    model.class_to_idx = model_state["class_to_idx"]
    return model
model = load_checkpoint(filepathX)

# define device to use for inference ------------------

deviceX = args.gpu
if deviceX == "cuda:0":
    device = torch.device(deviceX)
else:
    device = torch.device("cpu")
model = model.to(device)
print("Device used for inference:", device)

model = model.to(device)

# Function for processing a PIL image for use in a PyTorch model ------------------

def process_image(image):
    image = Image.open(image).convert("RGB")
    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])
    image = image_transforms(image)
    #image = image.to("cuda")
    image = image.to(device)

    return image

# Function for predicting the category ------------------                   
                            
def predict(image_path, model, topk = args.top_k):

    #model = model.to('cuda')
    model = model.to(device)
    model.eval()
    
    image = process_image(image_path).unsqueeze(0)
    with torch.no_grad():
        output = model.forward(image)
        top_probs, top_labels = torch.topk(output, topk)
        top_probs = top_probs.exp()
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    top_classes = list()
    
    output = output.cpu()
    top_probs.cpu()
    top_labels = top_labels.cpu()
    
    for label in top_labels.numpy()[0]:
        top_classes.append(class_to_idx_inv[label])
        
    return top_probs.cpu().numpy()[0], top_classes
    #return top_probs.numpy()[0], top_classes

# Output: Display an image along with the top 5 classes ------------------

sample_image_file = args.input
top_probs, top_classes = predict(sample_image_file, model)
label = top_classes[0]

labels = []
for class_idx in top_classes:
    labels.append(cat_to_name[class_idx])

printout_list = list(zip(labels, top_probs))
print("---Prediction complete---")
print("The image is prediced to be a", labels[0])
print(f"The {args.top_k} most likely species and their probabilities are: {printout_list}")