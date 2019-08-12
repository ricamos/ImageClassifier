# Imports here
import time
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import logging
from tqdm import tqdm
import json
from PIL import Image

def setup_logger(name, log_file, mode="w", level=logging.DEBUG, turn_off=True ):
    """Function setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler = logging.FileHandler(log_file, mode=mode)
    handler.setFormatter(formatter)
    if turn_off:
        handler = logging.StreamHandler()

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def choose_device(device):
    # Use GPU if it's available
    if device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    return device

def load_data(data_dir, batch_size):
    """ Load Data"""
    # data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training set.
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # TODO: Define your transforms for the validation and testing sets.
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=data_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data)
    validloader = torch.utils.data.DataLoader(valid_data)

    return train_data, test_data, valid_data, trainloader, testloader, validloader

def load_model(arch, hidden_layer, output_size, lr):
    model = models.vgg19_bn(pretrained=True)
    ## Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    input_size = model.classifier[0].in_features
    
    ## Classifier config
    classifier = nn.Sequential(nn.Linear(input_size, hidden_layer),
                           nn.Dropout(0.5),
                           nn.ReLU(),
                           nn.Linear(hidden_layer, output_size),                      
                           nn.LogSoftmax(dim=1))
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    return model, criterion, optimizer, classifier

def train_model(model, trainloader, validloader, epochs, criterion, optimizer, device, print_every = 32):
    logging = setup_logger(__name__, 'train.log', "w")
    try:
        steps = 0
        running_loss = 0
        model.to(device)
        model.train()

        start = time.time()

        train_losses, valid_losses = [], []
        for epoch in tqdm(range(epochs)):
            for i, (images, labels) in tqdm(enumerate(trainloader)):
            #for images, labels in trainloader:
                steps += 1

                # Move para GPU se disponivel
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Modo Eval para Valid
                if steps % print_every == 0:
                    model.eval()
                    valid_loss = 0
                    accuracy = 0
                    with torch.no_grad():
                        for images, labels in validloader:
                            # Move para GPU se disponivel
                            images, labels = images.to(device), labels.to(device)

                            logps = model(images)
                            loss = criterion(logps, labels)
                            valid_loss += loss.item()

                            # Calculate our accuracy
                            ## Get the class probabilities
                            ps = torch.exp(logps)

                            top_ps, top_class = ps.topk(1, dim=1)
                            equality = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                    train_losses.append(running_loss / len(trainloader))
                    valid_losses.append(valid_loss / len(validloader))
                    
                    print(f" Epoch {epoch + 1}/{epochs} "
                      f"({i}/{len(trainloader)}).. "
                      f"({(i/len(trainloader))*100:.2f}%).. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Valid loss: {valid_loss / len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy / len(validloader):.3f}")
                    
                    logging.info(f" Epoch {epoch + 1}/{epochs} "
                      f"({i}/{len(trainloader)}).. "
                      f"({(i/len(trainloader))*100:.2f}%).. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Valid loss: {valid_loss / len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy / len(validloader):.3f}")
                    
                    running_loss = 0
                    model.train()
        print(f"Device = {device}; Time per batch: {(time.time() - start) / 3:.3f} seconds")
    except Exception as e:
        logging.exception("Exception occurred")

def accuracy_network(model, testloader, criterion, device):
    logging = setup_logger(__name__, 'acc.log', "w")
    try:
        # turn off gradients
        with torch.no_grad():
            # Modo Eval para test        
            model.eval()
            valid_loss = 0 
            accuracy = 0
            
            for i, (images, labels) in tqdm(enumerate(testloader)):
            #for images, labels in testloader:
                # Move para GPU se disponivel
                images, labels = images.to(device), labels.to(device)
                logps = model(images)
                loss = criterion(logps, labels)
                valid_loss += loss.item()
                # Calculate our accuracy
                ## Get the class probabilities
                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equality.type(torch.FloatTensor))
            acc_out = f'Accuracy: {accuracy.item()*100:.3f}%'   
            return acc_out
    except Exception as e:
        logging.exception("Exception occurred")

def save_checkpoint(model, train_data, optimizer, save_dir, arch, output_size, classifier, lr, batch_size, epochs ):
    # TODO: Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    try:
        checkpoint = {'input_size': model.classifier[0].in_features,
              'output_size': output_size,
              'arch':  arch,
              'learning_rate': lr,
              'batch_size': batch_size,
              'classifier' : classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}
 
        if  save_dir == None:
            torch.save(checkpoint, 'checkpoint.pth')
            return 'checkpoint.pth'
        else:
            torch.save(checkpoint, save_dir+'checkpoint.pth')
            return save_dir+'checkpoint.pth'
    except Exception as e:
        logging.exception("Exception occurred")

#Inico de predict       
def load_checkpoint(filename):
    """loads a checkpoint and rebuilds the model"""
    checkpoint = torch.load(filename, map_location='cpu')
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
            
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)  
    
    img_prepro = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = img_prepro(img_pil)

    return img_tensor

def predict(image_path, model, gpu, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file       
    # Use GPU if it's available
    device = choose_device(gpu)
    model.to(device)
    
    # Modo Eval para test        
    model.eval()

    # - The image
    image = process_image(image_path)
    image = image.unsqueeze_(0)
 
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(image.to(device))
        
    ps = F.softmax(output, dim=1)
    top_ps, top_class = ps.topk(topk)

    return top_ps, top_class

def view_classify(img, ps, classes, cat_to_name, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()
    img = Image.open(img)
    
    flowers_names = []
    for i in classes.data[0]:
        number = i.item()
        flowers_names.append(cat_to_name[str(number)])
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    ax1.set_title(flowers_names[0])
    ax1.imshow(img)  
    ax1.axis('off')
    
    ax2.barh(np.arange(len(ps)), ps, align='center')
    ax2.set_aspect(0.2)
    ax2.set_yticks(np.arange(len(ps)))
    ax2.set_yticklabels(flowers_names[0])
    ax2.invert_yaxis() # Do maior para o menor
    
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels([flowers_names[0],
                            flowers_names[1],
                            flowers_names[2],
                            flowers_names[3],
                            flowers_names[4]],
                            size= "medium");
        
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    
def cat_to_name(classes, model, category_names):
    flowers_names = []
    if category_names != None:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = model.class_to_idx
    
    for i in classes.data[0]:
        number = i.item()
        flowers_names.append(cat_to_name[str(number)])
    return flowers_names