# Imports here
import argparse #To import data from the user
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

parser = argparse.ArgumentParser(description='Set options')
parser.add_argument('--checkpoint_pth', default="checkpoint_pth",action="store") #(-- optional) String
parser.add_argument('--image_pth', default="/10/image_07090.jpg", action="store") #String
parser.add_argument('--top_k', default=5, action="store", type=int) #Int
parser.add_argument('--dropout', default=0.2, action="store", type=int) #Int
parser.add_argument('--gpu_cpu', action="store_true", default=False) #Boolean, in case the user writes it use gpu
parser.add_argument('--JSON_pth', action="cat_to_name.json", default=False,help='Add the path to JSON file') #Boolean, in case the user writes it use gpu
parser.add_argument('--arch', default='vgg16',action="store") #String
args = parser.parse_args()

#Variables from the user
checkpoint_pth=args.checkpoint_pth
flower_image_example=args.image_pth
image_path='flowers/test' + flower_image_example
top_k=args.top_k
dropout=args.dropout
arch=args.arch
JSON_pth=args.JSON_pth
gpu_cpu='cuda' if args.gpu_cpu==True else 'cpu'

#Check models
architectures=['vgg16','vgg13']
if args.arch not in architectures:
    print('Error: Invalid architecture')
    print('Try help for more info')
    quit()  
if gpu_cpu not in ['cpu','gpu']:
    print('Error: Invalid device')
    print('Try help for more info')
    quit()

with open(JSON_pth, 'r') as f:
    cat_to_name = json.load(f)
#Examples
#'/10/image_07090.jpg'#=> "globe thistle"
#'/1/image_06743.jpg'# =>"pink primrose"
#'/102/image_08004.jpg'# => "blackberry lily"
#'/12/image_03994.jpg'# => "Colt's food"
#'/17/image_03864.jpg'# => "purple coneflower"

def load_checkpoint(filepath):
     #Other key variables
    input_size=25088 #vgg16 model input
    output_size=102 #Number of flower categories
    hidden_layers=[1500,500] #TWO
    checkpoint = torch.load(filepath,map_location=gpu_cpu)

    #Classifier
    classifier = nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                 nn.ReLU(),
                                 nn.Dropout(dropout), #Using dropout to avoid overfitting
                                 nn.Linear(hidden_layers[0], hidden_layers[1]),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_layers[1], output_size),
                                 nn.LogSoftmax(dim=1))
    
    if args.arch=="vgg13":
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    # Put the classifier on the pretrained network
    model.classifier=classifier #Same as above
  
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['classes_to_idx']    
    model.epoch = checkpoint['epoch']
    model.input_size=checkpoint['input_size']
    model.output_size=checkpoint['output_size']
    model.hidden_layers=checkpoint['hidden_layers']
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image_path)
      
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/256
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    #Transpose
    image_transform = img.transpose((2,0,1))
    return image_transform

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image_transform=process_image(image_path) #One image processed
    
    # Numpy to Tensor batch of size 1
    image_tensor = torch.from_numpy(image_transform).type(torch.FloatTensor)
    image_tensor = image_tensor.unsqueeze(0)
    
    #Use GPU if available
    if gpu_cpu=='gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device("cpu")
    
    model.to(device) 
    image_tensor=image_tensor.to(device) #Choose between GPU/CPU
    
    model.eval()
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model.forward(image_tensor) 

    # Output of the network are logits, need to take softmax for probabilities
    ps = torch.exp(logps)
    
    # Most 5 likely classes
    top_p, top_labels = ps.topk(topk, dim=1)
    top_labels=top_labels.numpy()[0] #To numpy, choose first element of the matrix
    top_p=top_p.numpy()[0]
    
    #Convert idx to class
    idx_to_class = {val: key for key, val in model.class_to_idx.items()} #Inverted dictionary
    top_class = [cat_to_name[idx_to_class[lab]] for lab in top_labels] #Using cat_to_name file find the the names that match 
                                                                      #the index from the top_label
 
    return top_p,top_class

#RUN THE MODEL
model_load = load_checkpoint(checkpoint_pth)
probs, classes = predict(image_path, model_load,topk=top_k)
print(probs)
print(classes)