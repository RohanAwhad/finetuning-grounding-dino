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
    tl, bx = self.collate_labels(tl, bx)

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
      'target_labels': tl.to(torch.long),
      'boxes': bx.to(torch.float),
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

    target_labels = self.current_file['target_labels']
    boxes = self.current_file['boxes']
    n_boxes = self.current_file['n_boxes'].reshape(-1)


    self.target_labels = []
    self.boxes = []
    offset = 0
    for n in n_boxes:
      self.target_labels.append(target_labels[offset:offset+n])
      self.boxes.append(boxes[offset:offset+n])
      offset += n

    self.current_idx = 0


  def collate_labels(self, target_labels, boxes):
    max_boxes = 0

    for tl, b in zip(target_labels, boxes):
      n_boxes = tl.shape[0]
      if n_boxes > max_boxes: max_boxes = n_boxes

    boxes_pad = 0
    target_labels_pad = (255, 256) # (max_len, max_len+1)

    ret = {
      'target_labels': [],
      'boxes': [],
    }

    for tl, b in zip(target_labels, boxes):
      n_boxes = tl.shape[0]
      pad = torch.tensor(target_labels_pad).expand(max_boxes-n_boxes, -1)
      tl = torch.concatenate((torch.tensor(tl), pad), axis=0)

      pad = torch.tensor([boxes_pad]*4).expand(max_boxes-n_boxes, -1)
      b = torch.concatenate((torch.tensor(b), pad), axis=0)

      ret['target_labels'].append(tl.unsqueeze(0))
      ret['boxes'].append(b.unsqueeze(0))

    ret = {k: torch.vstack(v) for k, v in ret.items()}
    return ret['target_labels'], ret['boxes']
