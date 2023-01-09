import os
import torch
import logging
import numpy as np
from utils import *
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# suppress MPS CPU fallback warning
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

class Trainer:
    """
    Trainer is a simple training and eval loop for PyTorch, including tdqm.

    Args
    ---------------
    model ([`PreTrainedModel`] or `torch.nn.Module`):
        The model to train, evaluate or use for predictions.

    device (`torch.device()`):
        Device on which to put model and data.

    tokenizer ([`PreTrainedTokenizerBase`]):
        The tokenizer used to preprocess the data.

    train_dataloader (`torch.utils.data.DataLoader`):
        PyTorch data loading object to feed model training input data.

    validation_dataloader (`torch.utils.data.DataLoader`):
        PyTorch data loading object to feed model training input data.

    epochs (float):
        Number of epochs the model for which the model with train.

    optimizer (`torch.optim.Optimizer`):

    val_loss_fn (`torch.nn` loss function):

    num_labels (float):
        Number of labels for classification models to predict.
        Defaults to 2.

    output_dir (str):
        Directory to which the model and model checkpoint will save

    save_freq (float):
        Frequency to save the model, by epoch. 
        EX: 1 sets saving to every epoch, 2 to every other epoch.
        If `None`, model will not be saved.
        Defaults to `None`. 

    checkpoint_freq (float):
        Frequency to save the model checkpoints, by epoch. 
        EX: 1 sets checkpointing to every epoch, 2 to every other epoch.
        If `None`, no checkpoints will be saved.
        Defaults to `None`.

    checkpoint_load (str):
        Path to a model's checkpoint.pt file.
        If `None`, no checkpoint will load.
        Defaults to `None`

    """
    def __init__(self, 
                 model, 
                 device, 
                 tokenizer, 
                 train_dataloader, 
                 validation_dataloader,
                 epochs, 
                 optimizer, 
                 val_loss_fn, 
                 num_labels=2,
                 output_dir=None, 
                 save_freq=None,
                 checkpoint_freq=None, 
                 checkpoint_load=None,):

        self.model = model 
        self.device = device 
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.epochs = epochs 
        self.optimizer = optimizer 
        self.val_loss_fn = val_loss_fn 
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.save_freq = save_freq 
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_load=checkpoint_load

    def fit(self):
        # check for existing checkpoint
        current_epoch, val_loss = self.load_checkpoint()

        for epoch in range(current_epoch, self.epochs+1):
            # ==================== Training ====================
            # Set model to training mode
            self.model.train()
            
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            # tqdm for progress bars
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for step, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    # put data on device
                    batch = tuple(t.to(self.device) for t in batch)
                    
                    # parse batch
                    b_input_ids, b_input_mask, b_labels = batch
                    
                    self.optimizer.zero_grad()

                    # Forward pass
                    train_output = self.model(b_input_ids, 
                                              token_type_ids = None, 
                                              attention_mask = b_input_mask, 
                                              labels = b_labels)
                    
                    # Backward pass
                    train_output.loss.backward()

                    self.optimizer.step()
                    
                    # Update tracking variables
                    tr_loss += train_output.loss.item()
                    nb_tr_examples += b_input_ids.size(0)
                    nb_tr_steps += 1

                # ==================== Validate ====================
                
                # check num labels for validation metrics
                if self.num_labels > 2:
                    metric_average = "micro"
                else:
                    metric_average = "binary"

                val_loss, val_acc, val_f1, val_recall, val_precision = self.validate(self.model, 
                                                                                     self.validation_dataloader, 
                                                                                     self.device, 
                                                                                     self.val_loss_fn,
                                                                                     metric_average)

                # log training information    
                logging.info('\n \t - Train loss: {:.6f}'.format(tr_loss / nb_tr_steps))
                logging.info('\t - Validation Loss: {:.6f}'.format(val_loss))
                logging.info('\t - Validation Accuracy: {:.6f}'.format(val_acc))
                logging.info('\t - Validation F1: {:.6f}'.format(val_f1))
                logging.info('\t - Validation Recall: {:.6f}'.format(val_recall))
                logging.info('\t - Validation Precision: {:.6f} \n'.format(val_precision))
                
                # ==================== Save ====================
                self.save_model(epoch, self.model, val_acc, val_f1)
                self.save_checkpoint(epoch, self.model, val_loss,  val_acc, val_f1)
                logging.info("")

    def validate(self, model, val_dl, device, loss_fn, metric_average):
        model.eval()

        val_loss = 0.0
        batch_accuracies = []
        batch_f1s = []
        batch_recalls = []
        batch_precisions = []


        with tqdm(val_dl, unit="batch") as prog:
            for step, batch in enumerate(prog):
                prog.set_description(f"\t Validation {step}")

                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                with torch.no_grad():
                    # Forward pass
                    eval_output = model(b_input_ids, 
                                        token_type_ids = None, 
                                        attention_mask = b_input_mask)
                
                    loss = loss_fn(eval_output.logits, b_labels)
                    val_loss += loss.data.item() * b_input_ids.size(0)

                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate validation metrics
                preds = np.argmax(logits, axis = 1).flatten()
                true_labels = label_ids.flatten()

                # accuracy
                batch_accuracy = accuracy_score(true_labels, preds)
                batch_accuracies.append(batch_accuracy)

                # f1
                batch_f1 = f1_score(true_labels, preds, zero_division=0, average=metric_average)
                batch_f1s.append(batch_f1)

                # recall
                batch_recall =recall_score(true_labels, preds, zero_division=0, average=metric_average)
                batch_recalls.append(batch_recall)

                # precision
                batch_precision = precision_score(true_labels, preds, zero_division=0, average=metric_average)
                batch_precisions.append(batch_precision)

        val_loss = val_loss / len(val_dl.dataset)
        validation_accuracy = sum(batch_accuracies)/len(batch_accuracies)
        validation_f1 = sum(batch_f1s)/len(batch_f1s)
        validation_recall = sum(batch_recalls)/len(batch_recalls)
        validation_precision = sum(batch_precisions)/len(batch_precisions)

        return val_loss, validation_accuracy, validation_f1, validation_recall, validation_precision

    def save_model(self, epoch, model, val_acc=0, val_f1=0):
        if self.save_freq != None and ((epoch)%self.save_freq == 0):
            
            save_name = f'E{str(epoch).zfill(2)}_A{round(val_acc, 2)}_F{round(val_f1, 2)}'

            results_path = os.path.join(self.output_dir, save_name)

            try:
                # model save
                model.save_pretrained(results_path)
                self.tokenizer.save_pretrained(results_path)
                logging.info(f'\t * Model @ epoch {epoch} saved to {results_path}')
            
            except Exception:
                logging.info(f'\t ! Model @ epoch {epoch} not saved')
                pass
        
        else:
            logging.info(f"\t ! Save Directory: {self.output_dir}, Save Frequency: {self.save_freq}, Epoch: {epoch}")

    def save_checkpoint(self, epoch, model, loss,  val_acc=0, val_f1=0):

        if self.checkpoint_freq != None and ((epoch)%self.checkpoint_freq == 0):
            checkpoint_name = f'E{str(epoch).zfill(2)}_A{round(val_acc, 2)}_F{round(val_f1, 2)}'
            results_path = os.path.join(self.output_dir, checkpoint_name, "checkpoint.pt")

            try:
                # model save
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss,
                            }, results_path)
            
                logging.info(f'\t * Model checkpoint saved to {results_path}')
            
            except Exception:
                logging.info(f'\t ! Model checkpoint not saved')
                pass
        
        else:
            logging.info(f"\t ! Checkpoint Directory: {self.output_dir}, Save Frequency: {self.checkpoint_freq}, Epoch {epoch}")

    def load_checkpoint(self):
        # load checkpoint if existing
        if self.checkpoint_load != None:
            checkpoint = torch.load(self.checkpoint_load)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            current_epoch = checkpoint['epoch']+1 # checkpoint logs saved epoch, increment
            val_loss = checkpoint['loss']

        else:
            current_epoch = 1
            val_loss = None

        return current_epoch, val_loss
