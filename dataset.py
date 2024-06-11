import glob
import os
import pickle
import torch

SAVE_DIR = os.path.join('/scratch/rawhad/datasets', 'screen_ai/processed_data')


def load_pickle(file_path):
  with open(file_path, 'rb') as f: return pickle.load(f)

class CustomDataloader:
  def __init__(self, split, batch_size: int):
    self.split = split
    self.batch_size = batch_size

    self.files = glob.glob(os.path.join(SAVE_DIR, f'{split}_*.pkl'))
    self.reset()

  def next_batch(self):
    pv = self.pixel_values[self.current_idx : self.current_idx + self.batch_size]
    pm = self.pixel_mask[self.current_idx : self.current_idx + self.batch_size]
    ii = self.input_ids[self.current_idx : self.current_idx + self.batch_size]
    am = self.attention_mask[self.current_idx : self.current_idx + self.batch_size]
    tti = self.token_type_ids[self.current_idx : self.current_idx + self.batch_size]
    tl = self.target_labels[self.current_idx : self.current_idx + self.batch_size]
    bx = self.boxes[self.current_idx : self.current_idx + self.batch_size]
    self.current_idx += self.batch_size

    if self.current_idx >= self.pixel_values.shape[0]:
      self.file_ptr = (self.file_ptr + 1) % len(self.files)
      self.load_new_shard()

    return {
      'pixel_values': torch.tensor(pv, dtype=torch.float),
      'pixel_mask': torch.tensor(pm, dtype=torch.float),
      'input_ids': torch.tensor(ii, dtype=torch.long),
      'attention_mask': torch.tensor(am, dtype=torch.long),
      'token_type_ids': torch.tensor(tti, dtype=torch.long),
      'target_labels': torch.tensor(tl, dtype=torch.float),
      'boxes': torch.tensor(bx, dtype=torch.float),
    }

  def reset(self):
    self.file_ptr = 0
    self.load_new_shard()

  def load_new_shard(self):
    self.current_file = load_pickle(self.files[self.file_ptr])

    self.pixel_values = self.current_file['pixel_values']
    self.pixel_mask = self.current_file['pixel_mask']
    self.input_ids = self.current_file['input_ids']
    self.attention_mask = self.current_file['attention_mask']
    self.token_type_ids = self.current_file['token_type_ids']
    self.target_labels = self.current_file['target_labels']
    self.boxes = self.current_file['boxes']
    self.n_boxes = self.current_file['n_boxes']

    self.target_labels = self.target_labels.view(-1, self.n_boxes, self.target_labels.shape[-1])
    self.boxes = self.boxes.view(-1, self.n_boxes, self.boxes.shape[-1])
    self.current_idx = 0
