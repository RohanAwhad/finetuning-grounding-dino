"""
Engine will have following functions
1. Train Step
2. Validation Step
3. Loss function

run() Another function that actually runs the training and validation steps
"""

import time
import torch
import torch.nn.functional as F

from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoLoss

# ===
# Loss function
# ===
# add to add the GroundingDinoHungarianMatcher class to deal with cost_matrix containing nan values
import torch.nn as nn
from transformers.image_transforms import center_to_corners_format
from scipy.optimize import linear_sum_assignment

# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: torch.Tensor) -> torch.Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area

class GroundingDinoHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        # requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        #print(target_ids)
        #print(target_ids.min(), target_ids.max())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()
        
        # (rohan): cost_matrix sometimes contains nan values, which crash the linear_sum_assignment function
        # setting nan values to 0
        cost_matrix[cost_matrix.isnan()] = 0

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



# updated: fixed nan loss issue, tested in Grouding_dino_v0.ipynb colab
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    # prob = prob + 1e-8  # smoothing factor trying to not get NaN | Didn't change anything

    # the following loc is the culprit behind giving NaNs
    # seems like attention mask is not being propagated
    clipped_inputs = torch.clamp(inputs, min=-10, max=10)  # this change gives a loss value and not NaNs
    ce_loss = F.binary_cross_entropy_with_logits(clipped_inputs, targets, reduction="none")

    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class ModifiedGDL(GroundingDinoLoss):
  def __init__(self, max_len, matcher, num_classes, focal_alpha, losses):
    self.max_len = max_len
    super().__init__(matcher, num_classes, focal_alpha, losses)

  def loss_labels(self, outputs, targets, indices, num_boxes):
      """
      Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
      of dim [nb_target_boxes, 2]
      """
      if "logits" not in outputs:
          raise KeyError("No logits were found in the outputs")
      source_logits = outputs["logits"]
      batch_size, num_boxes = source_logits.shape[:2]

      idx = self._get_source_permutation_idx(indices)
      target_classes_o = torch.cat([t["target_labels"][J] for t, (_, J) in zip(targets, indices)])

      target_classes = torch.full((batch_size, num_boxes, self.num_classes), self.max_len, dtype=torch.int64, device=source_logits.device)
      target_classes[:, :, 1] += 1
      target_classes[idx] = target_classes_o

      target_classes_onehot = torch.zeros(
          [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
          dtype=source_logits.dtype,
          layout=source_logits.layout,
          device=source_logits.device,
      )

      for b in range(batch_size):
        for q in range(num_boxes):  # or num_boxes which is 900
          start_idx, end_idx = target_classes[b, q]
          target_classes_onehot[b, q, start_idx: end_idx] = 1

      target_classes_onehot = target_classes_onehot[:, :, :-1]
      loss_ce = (
          sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
          * source_logits.shape[1]
      )
      losses = {"loss_ce": loss_ce}

      return losses


# ===
# Training and Validation Steps
# ===
def common_step(model, batch, device):
  pixel_values = batch["pixel_values"].to(device)
  pixel_mask = batch["pixel_mask"].to(device)
  input_ids = batch["input_ids"].to(device)
  attention_mask = batch["attention_mask"].to(device)
  token_type_ids = batch["token_type_ids"].to(device)

  # loss function
  matcher = GroundingDinoHungarianMatcher(
    class_cost=model.config.class_cost, bbox_cost=model.config.bbox_cost, giou_cost=model.config.giou_cost
  )
  losses = ["labels", "boxes", "cardinality"]
  criterion = ModifiedGDL(
    max_len=256,
    matcher=matcher,
    num_classes=model.config.num_labels,
    focal_alpha=model.config.focal_alpha,
    losses=losses,
  )
    
  with torch.autocast(device_type=device, dtype=torch.bfloat16):
    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    outputs_loss = {}
    outputs_loss["logits"] = outputs.logits
    pred_boxes = outputs.pred_boxes
    # clamp boxes values to 0-1
    pred_boxes = pred_boxes.clamp_(0, 1)
    outputs_loss["pred_boxes"] = pred_boxes


    target_labels = batch["target_labels"].to(device)
    boxes = batch["boxes"].to(device)

    labels = [{'class_labels': tl[:, 0], 'boxes': bx, 'target_labels': tl} for tl, bx in zip(target_labels, boxes)]
    loss_dict = criterion(outputs_loss, labels)
    # compute total loss, as a weighted sum of the various losses
    weight_dict = {"loss_ce": 1, "loss_bbox": model.config.bbox_loss_coefficient, "loss_giou": model.config.giou_loss_coefficient}
    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

  return loss, loss_dict


def run(
  model,
  train_dataloader,
  val_dataloader,
  optimizer,
  get_lr,
  num_steps,
  val_every_n_steps,
  val_steps,
  grad_accum_steps,
  device,
  logger,
  model_path,
  overfit_batch: bool = False
):

  model.train()

  for step in range(num_steps):
    # validation step
    if (step % val_every_n_steps == 0) and (val_steps > 0):
      val_loss = 0
      val_sublosses = {}
      model.eval()
      val_dataloader.reset()
      for i in range(val_steps):
        val_batch = val_dataloader.next_batch()
        with torch.no_grad():
          loss, loss_dict = common_step(model, val_batch, device)

        val_loss += (loss / val_steps).item()
        for k, v in loss_dict.items():
          if k not in val_sublosses: val_sublosses[k] = 0
          val_sublosses[k] += v.item() / val_steps

      val_sublosses['loss'] = val_loss
      logger.log({'validation': val_sublosses}, step=step)
      model.train()
      model.save_pretrained(model_path)
      print(f"Step: {step:4d}, Val Loss: {val_loss:.6f}")

    # training step
    start = time.monotonic()
    train_loss = 0
    train_sublosses = {}
    for micro_step in range(grad_accum_steps):
      train_batch = train_dataloader.next_batch()
      loss, loss_dict = common_step(model, train_batch, device)
      loss /= grad_accum_steps
      loss.backward()
      train_loss += loss.item()
      for k, v in loss_dict.items():
        if k not in train_sublosses: train_sublosses[k] = 0
        train_sublosses[k] += v.item() / grad_accum_steps

    lr = get_lr(step)
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    optimizer.step()

    train_sublosses['loss'] = train_loss
    logger.log({'train': train_sublosses, 'lr': lr}, step=step)

    #torch.cuda.synchronize()
    end = time.monotonic()
    if overfit_batch: train_dataloader.reset()

    print(f"Step: {step:4d}, Train Loss: {train_loss:.6f}, LR: {lr:.2e} Time Taken: {end - start:.2f} secs")

