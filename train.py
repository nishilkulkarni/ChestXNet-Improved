"""
Trains a pytorch model 
"""

import datasetup, model, engine
import torch
import os
from torchvision import transforms

#Setup hyperparameters 
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Downlaod Data 
datasetup.download_data()

# Setup directories
train_dir = "data/images/train"
test_dir = "data/images/test"

# Setup target device 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = datasetup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model 
mymodel = model.MyChestModel(out_classes = len(class_names)).to(device)

# Set loss and optimizer 
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters(),
                             lr = LEARNING_RATE)

#Start training
engine.train(model=mymodel,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)
