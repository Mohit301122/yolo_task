# DEPENDENCIES
import torch
from .logger import setup_logger
from .train import train_one_epoch
from .visualizer import plot_losses
from .evaluate import evaluate_model
from .visualizer import visualize_misclassifications

class TrainCNN:
    def __init__(self, model, model_name, device, loss_fn, optimizer, scheduler=None, log_dir="logs"):
        self.model        = model
        self.device       = device
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.logger       = setup_logger(model_name = model_name, log_dir = 'logs')
        self.train_losses = list()
        self.val_losses   = list()

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model      = self.model, 
                                                    dataloader = train_loader, 
                                                    loss_fn    = self.loss_fn, 
                                                    optimizer  = self.optimizer, 
                                                    device     = self.device)

            val_loss, val_acc     = evaluate_model(model      = self.model, 
                                                   dataloader = val_loader, 
                                                   loss_fn    = self.loss_fn, 
                                                   device     = self.device)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

            if self.scheduler:
                self.scheduler.step()

        return self.train_losses, self.val_losses


    def evaluate(self, test_loader):
        test_loss, test_acc = evaluate_model(model      = self.model, 
                                             dataloader = test_loader, 
                                             loss_fn    = self.loss_fn, 
                                             device     = self.device)

        self.logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

