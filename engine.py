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

from transformers.models.grounding_dino.modeling_grounding_dino import (
  GroundingDinoHungarianMatcher,
  GroundingDinoLoss,
)

# ===
# Loss function
# ===
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


def training_step(model, batch, device, logger):
  loss, loss_dict = common_step(model, batch, device)
  # logs metrics for each training_step, and the average across the epoch
  logger.log({"training_loss": loss})
  for k,v in loss_dict.items(): logger.log({"train_" + k: v.item()})
  return loss

@torch.no_grad()
def validation_step(model, batch, device, logger):
  loss, loss_dict = common_step(model, batch, device)
  logger.log({"validation_loss": loss})
  for k, v in loss_dict.items(): logger.log({"validation_" + k: v.item()})
  return loss

def run(model, train_dataloader, val_dataloader, optimizer, get_lr, num_steps, val_every_n_steps, val_steps, grad_accum_steps, device, logger):
  model.train()
  # TODO (rohan): add logging

  for step in range(num_steps):

    # validation step
    if step % val_every_n_steps == 0 and val_steps:
      val_loss = 0
      model.eval()
      val_dataloader.reset()
      for i in range(val_steps):
        val_batch = val_dataloader.next_batch()
        loss = validation_step(model, val_batch, device, logger)
        val_loss += (loss / val_steps).item()
      model.train()
      print(f"Step: {step:4d}, Val Loss: {val_loss:.6f}, LR: {lr:.2e} Time Taken: {end - start:.2f} secs")

    # training step
    start = time.monotonic()
    train_loss = 0
    for micro_step in range(grad_accum_steps):
      train_batch = train_dataloader.next_batch()
      loss = training_step(model, train_batch, device, logger)
      loss /= grad_accum_steps
      loss.backward()
      train_loss += loss.item()

    lr = get_lr(step)
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    optimizer.step()

    torch.cuda.synchronize()
    end = time.monotonic()

    print(f"Step: {step:4d}, Train Loss: {train_loss:.6f}, LR: {lr:.2e} Time Taken: {end - start:.2f} secs")
