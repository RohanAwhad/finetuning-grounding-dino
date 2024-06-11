"""
How do I want to store the encoded data?
- I have:
  - pixel values
  - pixel mask
  - input ids => set max_length to 256 and padding to "max_length"
  - attention_mask
  - token_type_ids
  - boxes <- This would be different for each image, which also influences the target labels
  - target_labels
  - n_boxes <- This would be of shape (100, 1) and will track how many boxes are there in each image

100 images per file (for now)
Different folders for each epoch only for training set
"""
import multiprocessing as mp
import os
import pandas as pd
import pickle
import random
import re
import torch

from dataclasses import dataclass
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2
from tqdm import tqdm
from transformers import AutoProcessor
from typing import Optional


# ===
# Constants
# ===
ROOT_DIR = '/home/rawhad/personal_jobs/GUI_Detection/rico'
#ROOT_DIR = '/Users/rohan/3_Resources/ai_datasets/rico'
IMAGE_DIR = os.path.join(ROOT_DIR, 'combined')
TRAIN_CSV = os.path.join(ROOT_DIR, 'screen_ai', 'train.csv')
VALID_CSV = os.path.join(ROOT_DIR, 'screen_ai', 'valid.csv')
TEST_CSV = os.path.join(ROOT_DIR, 'screen_ai', 'test.csv')

CKPT = "IDEA-Research/grounding-dino-base"
#CKPT = '/Users/rohan/3_Resources/ai_models/grounding-dino-base'
processor = AutoProcessor.from_pretrained(CKPT)
IMAGE_PROCESSOR = processor.image_processor
TOKENIZER = processor.tokenizer

IMAGE_PROCESSOR.size['shortest_edge'] = 224  # to save memory and time

AUGMENT = v2.Compose([
  v2.PILToTensor(),
  v2.RandomCrop(size=224*2, padding=0),  # 2x for compressing more information in the image
])

SHARD_SIZE = 1000
SAVE_DIR = os.path.join('/scratch/rawhad/datasets', 'screen_ai/processed_data')
os.makedirs(SAVE_DIR, exist_ok=True)

# ===
# Loading and Parsing the data
# ===
TRAIN_DF = pd.read_csv(TRAIN_CSV).sample(frac=1).reset_index(drop=True)
VALID_DF = pd.read_csv(VALID_CSV).sample(frac=1).reset_index(drop=True)
TEST_DF = pd.read_csv(TEST_CSV).sample(frac=1).reset_index(drop=True)
print("Dataframes are loaded and shuffled")


@dataclass
class StructuredAnnotation:
  class_: str
  text: str
  loc: tuple[int, int, int, int]
  children: Optional[list['StructuredAnnotation']] = None
  parent: Optional['StructuredAnnotation'] = None
    
  def add_child(self, x):
    if self.children is None: self.children = []
    self.children.append(x)

  def __repr__(self):
    return f'{self.class_} | Text: {self.text} | LOC({self.loc})'


def relabel(label):
  if label.lower() == 'pictogram': return 'icon'
  return label.lower()

def extract_info(input_str):
  pattern = r'(\w+)\s*(.*?)\s*(\d{1,3})\s+(\d+)\s+(\d+)\s+(\d+)$'
  match = re.match(pattern, input_str)
  if match:
    label = match.group(1)
    text = match.group(2)
    integers = tuple(map(int, match.group(3, 4, 5, 6)))

    label = relabel(label)
    return StructuredAnnotation(label, text, integers)
  else:
    return None

def parse_annotation(ann):
  stack = []
  start = 0
  end = start
  skip = False
  for i, ch in enumerate(ann):
    if skip: # skip for space after comma
      skip = False
      continue
    if ch == ',':
      end = i
      s = ann[start:end]
      info = extract_info(s)
      if info is None: continue
      stack.append(info)
      start = i + 2 # comma and space
      skip = True

    elif ch == '(':
      end = i-1
      s = ann[start:end]
      info = extract_info(s)
      if info is None: continue
        
      stack.append(info)
      start = i + 1 # (
      stack.append(ch)


    elif ch == ')':
      end = i-1
      if start < end:
        s = ann[start:end]
        info = extract_info(s)
        if info is None: continue
        stack.append(info)
      start = i + 3 # ), comma and space
      grp = []
      while True:
        if len(stack) <= 0: raise ValueError('should\'nt have reached here. Stack empty while searching for "("')

        ele = stack.pop()
        if isinstance(ele, str) and ele == '(': break
        else: grp.append(ele)

      grp = list(reversed(grp))
      parent = stack.pop()
      for child in grp:
        child.parent = parent
        parent.add_child(child)
      stack.append(parent)

  return stack

def flatten_annotation_tree(x):
  if x is None: return []
  ret = []
  ret.append(x)
  if x.children is not None:
    for child in x.children:
      ret.extend(flatten_annotation_tree(child))
  return ret

def get_image_id_to_flat_ann(df):
  image_id_to_flat_ann = {}
  for _, row in df.iterrows():
    try:
      pann = parse_annotation(row['screen_annotation'])
      flat_ann = []
      for ann in pann: flat_ann.extend(flatten_annotation_tree(ann))

      image_id_to_flat_ann[row['screen_id']] = flat_ann
    except Exception as e:
      print(f'Error for item {row["screen_id"]}: {e}')
  return image_id_to_flat_ann

train_image_id_to_flat_ann = get_image_id_to_flat_ann(TRAIN_DF)
valid_image_id_to_flat_ann = get_image_id_to_flat_ann(VALID_DF)
test_image_id_to_flat_ann = get_image_id_to_flat_ann(TEST_DF)
print('Annotations are parsed and flattened')



# ===
# Preprocessing the data
# ===

def process_(image_id, flat_anns):
  # load image
  img = Image.open(os.path.join(IMAGE_DIR, f'{image_id}.jpg')).convert('RGB')    

  # shuffle the annotations
  random.shuffle(flat_anns)
  shuffled_anns = flat_anns
  # get bounding boxes
  W, H = img.size
  boxes = _get_boxes(shuffled_anns, W, H)
  boxes = tv_tensors.BoundingBoxes(boxes, format='CXCYWH', canvas_size=(H, W))

  # crop the images
  transformed_img, transformed_boxes = AUGMENT(img, boxes)
  transformed_boxes = transformed_boxes.data  # convert back to torch.Tensor object
  # preprocess the image
  img_inp = IMAGE_PROCESSOR(transformed_img, return_tensors='pt')
  pixel_values = img_inp['pixel_values']
  pixel_mask = img_inp['pixel_mask']
  
  # get the text input
  labels = _get_label(shuffled_anns, transformed_boxes, canvas_size=transformed_img.shape[1:])
  input_phrase = _get_input_phrase(labels)
  inp = TOKENIZER(input_phrase, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
  input_ids = inp['input_ids']
  attention_mask = inp['attention_mask']
  token_type_ids = inp['token_type_ids']

  # get the labels and boxes
  target_labels = _get_target_labels(inp['input_ids'][0])
  # because the input phrase can have more than 256 tokens, we are truncating it early and that might change num of target labels
  # we need to account for that in out transformed boxes
  transformed_boxes = transformed_boxes[:target_labels.shape[0]]

  # normalize boxes
  H, W = transformed_img.shape[-2:]
  transformed_boxes = transformed_boxes / torch.tensor([W, H, W, H])

  assert len(transformed_boxes.shape) == 2, f"Expected 2D tensor, got {transformed_boxes.shape}"
  assert transformed_boxes.shape[1] == 4, f"Expected 4 columns, got {transformed_boxes.shape[1]}"
  assert transformed_boxes.shape[0] == target_labels.shape[0], f"Expected same number of boxes and labels, got {transformed_boxes.shape[0]} and {target_labels.shape[0]}"
  assert target_labels.shape[1] == 2, f"Expected 2 columns, got {target_labels.shape[1]}"
  
  return dict(
    input_ids = input_ids.squeeze(0),
    attention_mask = attention_mask.squeeze(0),
    token_type_ids = token_type_ids.squeeze(0),
    pixel_values = pixel_values.squeeze(0),
    pixel_mask = pixel_mask.squeeze(0),
    boxes = transformed_boxes.to(torch.float),
    target_labels = target_labels,
    n_boxes = torch.tensor([transformed_boxes.shape[0], ], dtype=torch.long),
  )
  
def _get_boxes(anns, W, H):
  new_boxes = []
  for an in anns:
    box = an.loc
    skip = False
    for a in box:
      if a < 0 or a > 999:
        new_boxes.append([0, 0, 0, 0])
        skip = True
        break
    if skip: continue

    x1, x2, y1, y2 = list(map(lambda x: x/999, box))
    x1, x2, y1, y2 = x1 * W, x2 * W, y1*H, y2*H
    if x1 > x2 or y1 > y2: box = [0, 0, 0, 0]

    else:
      cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
      w, h = x2 - x1, y2 - y1
      box = [int(cx), int(cy), int(w), int(h)] # CXCYWH
    new_boxes.append(box)
    
  return torch.tensor(new_boxes)

def _get_label(flat_anns, transformed_boxes, canvas_size):
  """
  sample input_phrases:
  - text block in the bottom right of the image with text \"COMPUTING\"`
  - text link near the top center of the image with text \"Learn more.\"
  - menu item near the top center of the image with text \"Downloads\"
  - icon of an ID card near the top-right of the image
  
  Formula: 
    - for text       : <label> (in/near) the <loc> with text ""
    - for image/icon : <label> of <desc> (in/near) the <loc>
  """
  labels = []
  loc_prefixes = ['in', 'near']
  H, W = canvas_size
  img_classes = ('image', 'icon')
  classes_that_need_labels = ('radio_button', 'switch', 'checkbox')
  
  
  for ann, box in zip(flat_anns, transformed_boxes):
    loc = _get_location(box, W, H)
    loc_prefix = random.choice(loc_prefixes)
    
    class_ = ann.class_
    text = ann.text
    if class_ in classes_that_need_labels:
      try:
        siblings = ann.parent.children
        for sib in siblings:
          if sib.class_ == 'label':
            sibling_label = sib.text
            break
        ret = f"{class_} with label \"{sibling_label}\" {loc_prefix} the {loc}"
      except Exception as e:
        ret = f"{class_} {loc_prefix} the {loc}"
      
    elif text == '':
      ret = f"{class_} {loc_prefix} the {loc}"
    
    elif class_ in img_classes:
      ret = f"{class_} of {text} {loc_prefix} the {loc}"

    else:
      ret = f"{class_} {loc_prefix} the {loc} with text \"{text}\""
      
    labels.append(ret)
  return labels

def _get_location(box, W, H):
  w_33, w_66 = W / 3, 2*W / 3
  h_33, h_66 = H / 3, 2*H / 3
  
  cx, cy = box[:2]
  
  if cx > w_66:
    if cy > h_66:  return "bottom-right"
    if cy > h_33: return "right"
    return "top-right"
  if cx > w_33:
    if cy > h_66:  return "bottom"
    if cy > h_33: return "center"
    return "top"
  
  if cy > h_66:  return "bottom-left"
  if cy > h_33: return "left"
  return "top-left"

def _get_input_phrase(labels):
  return f'{TOKENIZER.sep_token}'.join(labels)

def _get_target_labels(input_ids):
  target_labels = [] # (start_idx, end_idx)
  start_idx = 1
  for i, tok in enumerate(input_ids):
    if tok == TOKENIZER.sep_token_id:
      end_idx = i
      target_labels.append((start_idx, end_idx))
      start_idx = i+1

  return torch.tensor(target_labels).to(torch.long)

# ===
# Main
# ===
def get_new_shard():
  return {
    'pixel_values': [],
    'pixel_mask': [],
    'input_ids': [],
    'attention_mask': [],
    'token_type_ids': [],
    'boxes': [],
    'target_labels': [],
    'n_boxes': []
  }

def save_shard(fn, shard):
  fn = os.path.join(SAVE_DIR, fn)
  shard = {k: torch.vstack(v) for k, v in shard.items()}
  shard = {k: v.numpy() for k, v in shard.items()}
  with open(fn, 'wb') as f: pickle.dump(shard, f)


def mp_process_(x): return process_(*x)


def create_shards(image_id_to_flat_ann, prefix):
  n_procs = os.cpu_count() * 2  # to utiilze threading too
  with mp.Pool(n_procs) as pool:
    shard_id = 0
    shard = get_new_shard()
    pbar = tqdm(total=len(image_id_to_flat_ann), desc=f"Creating {prefix} shards")
    for data in pool.imap_unordered(mp_process_, image_id_to_flat_ann.items(), chunksize=50):
      pbar.update(1)
      for k, v in data.items(): shard[k].append(v)
      if len(shard['pixel_values']) == SHARD_SIZE:
        # save the shard
        fn = f'{prefix}_{shard_id}.pkl'
        print(f"Saving shard {fn}")
        save_shard(fn, shard)
        shard_id += 1
        shard = get_new_shard()

    if len(shard['pixel_values']) > 0:
      fn = f'{prefix}_{shard_id}.pkl'
      print(f"Saving shard {fn}")
      save_shard(fn, shard)

if __name__ == '__main__':
  create_shards(valid_image_id_to_flat_ann, 'valid')
  create_shards(test_image_id_to_flat_ann, 'test')
  create_shards(train_image_id_to_flat_ann, 'train')