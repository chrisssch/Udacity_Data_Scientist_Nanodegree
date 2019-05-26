# imports --------------------

import numpy as np
import json
from collections import OrderedDict
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import argparse

# command line arguments --------------------

parser = argparse.ArgumentParser()

parser.add_argument("data_directory",
                    action = "store",
                    type = str)

parser.add_argument("--category_names",
                    action = "store",
                    default = "cat_to_name.json",
                    dest = "category_names",
                    type = str,
                    help = "JSON file with category names")

parser.add_argument("--arch",
                    action = "store",
                    default = "vgg13",
                    dest = "arch",
                    type = str,
                    help = "Pretrained model; supported models: vgg11, vgg13, vgg17, vgg19, densenet121, densenet169, densent201, densenet161")

parser.add_argument("--gpu",
                    action = "store_const",
                    const = "cuda:0",
                    help = "Use GPU for training")    
                    
parser.add_argument("--learning_rate",
                    action = "store",
                    dest = "learning_rate",
                    default = 0.001,
                    type = float,
                    help = "Learning rate") # with a suggested learning rate of 0.01, all my models perform beyond dreadful

parser.add_argument("--hidden_units",
                    action = "store",
                    dest = "hidden_units",
                    default = 512,
                    type = int,
                    help = "Units in the hidden layer") 

parser.add_argument("--epochs",
                    action = "store",
                    dest = "epochs",
                    default = 20,
                    type = int,
                    help = "Epochs")

parser.add_argument("--save_dir",
                    action = "store",
                    default = ".",
                    dest = "save_dir",
                    type = str,
                    help = "Folder for saved checkpoints")

args = parser.parse_args()

print("--------------------")
print("Selected arguments:")
print("Directory of data:", args.data_directory)
print("JSON file with category names:", args.category_names)
print("Pretrained model architecture:", args.arch)
print(args.gpu, "; if = cuda:0, then training on GPU")
print("Learning rate:", args.learning_rate)
print("Units in the hidden layer:", args.hidden_units)
print("Epochs:", args.epochs)
print("Directory of saved checkpoint.pth:", args.save_dir)     
print("--------------------")
  
# data directories --------------------

data_dir = args.data_directory
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
test_dir = data_dir + "/test"

# data loaders --------------------

train_transforms = transforms.Compose([transforms.RandomRotation(25),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_dir + "/train", transform = train_transforms)
valid_data = datasets.ImageFolder(data_dir + "/valid", transform = valid_transforms)
test_data = datasets.ImageFolder(data_dir + "/test", transform = test_transforms)

batch_size = 32

trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size)

train_obs = 6552
valid_obs = 818
test_obs = 819

# class names --------------------

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
# define model and hidden units --------------------

model = models.__dict__[args.arch](pretrained=True) # ARGUMENT

for param in model.parameters():
    param.requires_grad = False
    
densenet_inputs = {"densenet121": 1024, "densenet169": 1664, "densenet201": 1920, "densenet161": 2208}

if args.arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
    input_units = model.classifier[0].in_features # ARGUMENT
if args.arch in ["densenet121", "densenet169", "densene201", "densenet161"]:
    input_units = densenet_inputs[arch] # ARGUMENT
    
hidden_units = args.hidden_units # ARGUMENT
output_units = len(cat_to_name)

classifier = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(input_units, hidden_units)), # ARGUMENT 
    ("relu", nn.ReLU()),
    ("dropout", nn.Dropout(p = 0.2)),
    ("fc2", nn.Linear(hidden_units, output_units)), # ARGUMENT 
    ("output", nn.LogSoftmax(dim = 1))
    ]))
 
model.classifier = classifier                    
                    
# set model parameters --------------------

batches = len(trainloader.batch_sampler)
print(f"Training model on {train_obs} images in {batches} batches of batch size {batch_size}")

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate) 

# define device --------------------

xdevice = args.gpu
if xdevice == "cuda:0":
    device = torch.device(xdevice)
else:
    device = torch.device("cpu")
model = model.to(device)
print("Training model on", device)

# start training --------------------

start_time = time.time()
model.zero_grad()

epochs = args.epochs

for e in range(epochs):

    print(f"Starting training for epoch {e+1} of {epochs}")
    
    total = 0
    correct = 0
    running_loss = 0

    for ii, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()     
        
        # forward and backward passes
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # running loss of epoch
        running_loss += loss.item()
    
        # accuracy of epoch
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
           
    # print after each epoch
    print(f"Epoch {e+1} of {epochs}", 
          "--- Training loss:{:.4f}".format(running_loss/batches), 
          "--- Training accuracy:{:.4f}".format(correct/total))

    # evaluate model in validation set
    
    # reset metrics for epoch
    valid_correct = 0
    valid_total = 0
    valid_running_loss = 0

    # don't calculate gradients
    with torch.no_grad():
        for ii, (images, labels) in enumerate(validloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels) #neww
            valid_running_loss += loss.item() #new
            
            # accuracy of epoch
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
            
        # print after each epoch
        print(f"Epoch {e+1} of {epochs}", 
              "--- Validation loss:{:.4f}".format(valid_running_loss/batches), 
              "--- Validation accuracy:{:.4f}".format(valid_correct/valid_total))
                
end_time = time.time()
duration = (end_time - start_time)//60
print("Training complete")
print(f"Training time: {duration} minutes")

# Save model to checkpoint --------------------

model.class_to_idx = train_data.class_to_idx
model_state = {
    "epoch": epochs,
    "state_dict": model.state_dict(),
    "optimizer_dict": optimizer.state_dict(),
    "classifier": classifier,
    "class_to_idx": model.class_to_idx,
    "arch": args.arch 
}

save_location = f"{args.save_dir}/checkpoint.pth"
torch.save(model_state, save_location)

print(f"Model saved to {save_location}/checkpoint.pth")