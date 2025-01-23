# 
import torch

def evaluate_model(model, dataloader, loss_fn, device):
    """
    Evaluates the model on the validation or test dataset.

    Arguments:
    ----------


    Returns:
    --------

    """
    model.eval()
    epoch_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move the data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Get the output by model
            outputs         = model(inputs)

            # Calculate the loss
            loss            = loss_fn(outputs, targets)
            epoch_loss     += loss.item()

            # Get the predictions
            _, predicted    = torch.max(outputs, 1)
            correct        += (predicted == targets).sum().item()
            total          += targets.size(0)

    # Calculate the accuracy
    accuracy = 100 * correct / total
    
    return epoch_loss / len(dataloader), accuracy
