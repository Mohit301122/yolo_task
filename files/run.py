# DEPENDENCIES
import torch
import torch.nn as nn
import torch.optim as optim
from src.trainer import TrainCNN
from src.utils import save_model
from src.visualizer import plot_losses
from src.models.squeezenet import SqueezeNet
from src.config import model_preprocessing
from src.data_loader import get_data_loaders
from src.visualizer import visualize_misclassifications


# PATHS AND PARAMETERS
data_dir     = "data"
batch_size   = 16
epochs       = 25
model_name   = 'squeezenet'
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loaders
loaders      = get_data_loaders(data_dir   = data_dir, 
                                model_name = model_name,  
                                batch_size = batch_size)

train_loader = loaders["train"]
val_loader   = loaders["val"]
test_loader  = loaders["test"]

# Model, loss, optimizer, scheduler
model        = SqueezeNet(num_classes = 10).to(device)

# Loss function for Multiclass clasification
loss_fn      = nn.CrossEntropyLoss()

# Optimizer
optimizer    = optim.Adam(params = model.parameters(), 
                          lr     = 0.001)

# Learning-rate scheduler 
scheduler    = optim.lr_scheduler.StepLR(optimizer = optimizer, 
                                         step_size = 2, 
                                         gamma     = 0.1)

# Trainer
trainer      = TrainCNN(model      = model, 
                        model_name = model_name,
                        device     = device, 
                        loss_fn    = loss_fn, 
                        optimizer  = optimizer, 
                        scheduler  = scheduler)


# Executor
if __name__ == '__main__':

    # Train the model
    train_losses, val_losses = trainer.train(train_loader = train_loader, 
                                             val_loader   = val_loader, 
                                             epochs       = epochs) 
    
    # Test the model
    trainer.evaluate(test_loader)
    
    # Plot Training vs evaluation losses
    plot_losses(train_losses = train_losses, 
                val_losses   = val_losses,
                save_path    = f"plots/{model_name}_loss_plot.png")
    
    # Visualize top-5 missclassifications in test data 
    visualize_misclassifications(model      = model, 
                                 dataloader = test_loader , 
                                 device     = device, 
                                 save_path  = f"plots/{model_name}_misclassifications.png")

    
    save_model(model      = model, 
               model_name = model_name)

    