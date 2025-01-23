# DEPENDENCIES
import torch
import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses, save_path="plots/loss_plot.png"):
    """
    Plots training and validation losses

    Arguments:
    ----------


    Returns:
    --------
          { None } : 
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def visualize_misclassifications(model, dataloader, device, save_path="plots/misclassifications.png"):
    """
    Visualizes misclassified images

    Arguments:
    ----------


    Returns:
    --------
        { None } :
    """
    model.eval()
    misclassified = list()
    correct       = list()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs         = model(inputs)
            _, predicted    = torch.max(outputs, 1)

            for i in range(len(targets)):
                if (predicted[i] != targets[i]):
                    misclassified.append((inputs[i].cpu(), predicted[i].item(), targets[i].item()))
                
                # Collect a few correct samples for side-by-side visualization
                elif (len(correct) < 5):
                    correct.append((inputs[i].cpu(), targets[i].item()))

    fig, axes = plt.subplots(nrows   = 2, 
                             ncols   = 5, 
                             figsize = (15, 6))

    for i, (img, pred, target) in enumerate(misclassified[:5]):
        axes[0, i].imshow(img.permute(1, 2, 0))
        axes[0, i].set_title(f"Pred: {pred}, Target: {target}")
        axes[0, i].axis("off")

    for i, (img, target) in enumerate(correct):
        axes[1, i].imshow(img.permute(1, 2, 0))
        axes[1, i].set_title(f"Target: {target}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_actual_vs_generated(actual_images, generated_images, save_path="plots/actual_vs_generated.png"):
    """
    Visualizes actual images vs generated images side-by-side

    Arguments:
    ----------

    Returns:
    --------

    """
    fig, axes = plt.subplots(nrows   = len(actual_images), 
                             ncols   = 2, 
                             figsize = (10, len(actual_images) * 5))
                             
    for i in range(len(actual_images)):
        axes[i, 0].imshow(actual_images[i].permute(1, 2, 0))
        axes[i, 0].set_title("Actual Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(generated_images[i].permute(1, 2, 0))
        axes[i, 1].set_title("Generated Image")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
