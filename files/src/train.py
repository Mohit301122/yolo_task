# DEPENDENCIES
import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Trains the model for one epoch

    Arguments:
    ----------
        model      {nn.Module} : the cnn model to train

        dataloader {Dataloader} : dataloader for training data 

        loss_fn    {nn.Module} : loss function 

        optimizer  {Optimizer} : optimizer 

        device     {torch.device} : device to run the model on 

    Returns:
    --------
    accuracy {float} : the accuracy of the model on the current epoch

    avg_loss {float} : the average loss over the epoch
    """
    # Set the model in training mode
    model.train()

    # Initialize epoch_loss, correct predictions and total predictions
    epoch_loss = 0.0
    correct    = 0
    total      = 0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # Feed the model the input data
        outputs = model(inputs)

        # Calculate the loss
        loss    = loss_fn(outputs, targets)

        # Backpropagate the loss
        loss.backward()

        # Optimize the loss
        optimizer.step()
        
        # Get the predictions
        epoch_loss  += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct     += (predicted == targets).sum().item()
        total       += targets.size(0)

    # Calculate the accuracy    
    accuracy = 100 * correct / total
    
    return epoch_loss / len(dataloader), accuracy
