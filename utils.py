import torch
import random
import params
import numpy as np
from tabulate import tabulate

def preprocessing(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = params.max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

def preprocessing_trunkless(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                   )


def preprocessing_dyna(input_text, tokenizer):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = params.max_length,
                        truncation=True,
                        return_attention_mask = True,
                   )

def collate(features):
  """
  Data collator that will dynamically pad the inputs for multiple choice data via dataloader fn.
  """
  label_name = "label" if "label" in features[0].keys() else "labels"

  labels = [feature.get(label_name) for feature in features]

  labeless_feat = []

  # extract input_ids and attention_masks from features
  for feature in features:
      # label = feature.pop(label_name)
      # labeless_feat.append(feature)
      feat_dict = {}
      for k, v in feature.items():
          if k != label_name:
              feat_dict[k] = v
      labeless_feat.append(feat_dict)

  # batch_size = len(features)

  batch = params.tokenizer.pad(
      labeless_feat,
      padding=True,
      max_length=None,
      return_tensors="pt",
  )
    
  # Un-flatten
  batch = [v for k, v in batch.items()]

  # Add back labels
  batch.append(torch.tensor(labels, dtype=torch.int64))
  return batch

# helper function for displaying sequences and encodings
def print_sentence_encoding(token_id, attention_masks):
  index = random.randint(0, 1 - 1)
  tokens = params.tokenizer.tokenize(params.tokenizer.decode(token_id[index]))
  token_ids = [i.numpy() for i in token_id[index]]
  attention = [i.numpy() for i in attention_masks[index]]

  table = np.array([tokens, token_ids, attention]).T
  print(tabulate(table, 
                headers = ['Tokens', 'Token IDs', 'Attention Mask'],
                tablefmt = 'fancy_grid'))
  