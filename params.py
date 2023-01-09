import torch
import torch.nn as nn
from transformers import RobertaTokenizer

# --------- Model Parameters --------- 
num_labels: float          = 2 # number of output labels for current model

# --------- Training Parameters --------- 
batch_size: float          = 16 # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1907.11692.pdf
learning_rate: float       = 1e-05 # Recommended Learning Rates {1e−5, 2e−5, 3e−5}. See: https://arxiv.org/pdf/1907.11692.pdf

device                     = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device                   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs: float              = 3 # Recommended number of epochs: 10. See: https://arxiv.org/pdf/1907.11692.pdf
val_loss_fn                = nn.CrossEntropyLoss() # loss function for validation loop

output_dir: str            = None
save_freq: float           = 1
checkpoint_freq: float     = 1

# --------- Tokenizer Parameters --------- 
tokenizer                  = RobertaTokenizer.from_pretrained("roberta-base")
max_length: float          = 512 # length of tokenized phrases allowed, 512 max for RoBERTa

