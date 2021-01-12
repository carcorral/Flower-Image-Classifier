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
parser.add_argument('--checkpoint_pth', default='checkpoint_pth',action="store") #String
parser.add_argument('--arch', default='vgg16',action="store", help='vgg16 or vgg13') #String
parser.add_argument('--learning_rate', default=0.001,action="store", type=int, help='Between 0.001 and 0.003') #Int
parser.add_argument('--epochs', default=10,action="store", type=int) #Int
parser.add_argument('--batch', default=64,action="store", type=int, help='32 or 64') #Int
parser.add_argument('--hidden_layers', default=[1500,500], action="store", type=int, help='2 arguments need it') #Int
parser.add_argument('--dropout', default=0.2, action="store", type=int) #Int
parser.add_argument('--gpu_cpu', action="store_true", default=True, help='Only cpu or gpu devices') #Boolean, by default we want to use gpu

args = parser.parse_args()
#Variables from the user
checkpoint_pth=args.checkpoint_pth
learning_rate=args.learning_rate
epochs = args.epochs
batch_size=args.batch
hidden_layers=args.hidden_layers #ALWAYS TWO
dropout=args.dropout

gpu_cpu='gpu' if args.gpu_cpu==True else 'cpu'

#Check models
architectures=['vgg16','vgg13']
if args.arch not in architectures:
    print('Error: Invalid architecture')
    print('Try help for more info')
    quit()
if len(hidden_layers) is not 2:
    print('Error: Invalid number of hidden layer, only 2')
    print('Try help for more info')
    quit()  
if gpu_cpu not in ['cpu','gpu']:
    print('Error: Invalid device')
    print('Try help for more info')
    quit()

def data_loader(arch):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    #Download the model
    if arch=="vgg13":
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30), #Random rotation
                                        transforms.RandomResizedCrop(224), #Random resize to 224x224 pixels
                                        transforms.RandomHorizontalFlip(), #Random flip de image
                                        transforms.ToTensor(), #Images to tensor
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]) #Color normalization
    #TEST & VALIDATION
    test_transforms = transforms.Compose([transforms.Resize(255), #Size
                                        transforms.CenterCrop(224), #Crop
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]) #Color normalization
    valid_transforms=test_transforms

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True) #Shuffle=>To change order of chossing data
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    return model,trainloader,validloader

def model_creation(model,hidden_layers,dropout):
    #Hyperparameters
    input_size=25088 #vgg16 model input
    output_size=102 #Number of flower categories

    #Classifier
    classifier = nn.Sequential(nn.Linear(input_size, hidden_layers[0]),
                                 nn.ReLU(),
                                 nn.Dropout(dropout), #Using dropout to avoid overfitting
                                 nn.Linear(hidden_layers[0], hidden_layers[1]),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_layers[1], output_size),
                                 nn.LogSoftmax(dim=1))
    model.classifier=classifier
    #Loss fx
    criterion = nn.NLLLoss()
    #Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return output_size,input_size,model,criterion,optimizer

def model_traning(model,optimizer,criterion,gpu_cpu,trainloader,validloader,epochs):
    #Use GPU if available
    if gpu_cpu=='gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device("cpu")
    model.to(device) 

    # Only train the classifier parameters, feature parameters are frozen
    for param in model.parameters():
         param.requires_grad = False

    #TRAIN THE CLASSIFIER
    steps = 0
    running_loss = 0
    print_every = 100
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()#Clear the gradients, gradients are accumulated
        
            logps = model.forward(inputs) #Calculate log prob from inputs
            loss = criterion(logps, labels) #Calculate loss fx using log prob and labels
            loss.backward() #Calculate gradients
            optimizer.step() #Optimize classifier's wi

            running_loss += loss.item() #accumulate loss values
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval() #turns off dropout
            #Valuation mode: turns off gradient for validation,saves memory and computations
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels) 
                    
                        valid_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps) #Calc prob from logprob
                        top_p, top_class = ps.topk(1, dim=1) #get the most likely class
                        equals = top_class == labels.view(*top_class.shape) #check if the predicted classes match the labels
                                                                        #Careful with the shapes of vectors
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item() #calculate the percentage of correct predictions
                                                                        #Careful with type of equals (must be a tensor)   
                print(f"Step {steps}) "
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.1f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.1f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
                
    return model,optimizer

def save_model(checkpoint_pth,model,optimizer,train_data_class_to_idx,input_size,output_size,hidden_layers,epoch):
    #Save the network
    model.class_to_idx=train_data_class_to_idx
    model.hidden_layers=hidden_layers

    checkpoint = {'input_size': input_size,
                'output_size': output_size,
                'hidden_layers': model.hidden_layers,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'classes_to_idx':model.class_to_idx
                }
    torch.save(checkpoint, checkpoint_pth)
    print('Model has been saved')


#RUN THE MODEL
model,trainloader,validloader=data_loader(args.arch)
output_size,input_size,model,criterion,optimizer=model_creation(model,hidden_layers,dropout)
model_trained,optimizer=model_traning(model,optimizer,criterion,gpu_cpu,trainloader,validloader,epochs)    
create_savepoint(checkpoint_pth,model_trained,optimizer,train_data.class_to_idx,input_size,output_size,hidden_layers,epoch)
