# %%
# %matplotlib inline
import matplotlib.pyplot as plt

import sys
import numpy as np
import torch

sys.path.insert(0, '..')
from isegm.utils import vis, exp

from isegm.inference import utils
from isegm.inference.evaluation import evaluate_dataset, evaluate_sample

device = torch.device('cuda:1')
cfg = exp.load_config_file('./config.yml', return_edict=True)

# %% [markdown]
# ### Init dataset

# %%
DATASET = 'BraTS'
dataset = utils.get_dataset(DATASET, cfg)
print(cfg.BRATS_PATH)
dataset.__len__()

# %% [markdown]
# ### Init model

# %%
from isegm.inference.predictors import get_predictor
import os

EVAL_MAX_CLICKS = 20
MODEL_THRESH = 0.49

checkpoint_path = utils.find_checkpoint(os.path.join('./experiments/hrnet32_brats_itermask/','000_brats/checkpoints/'), '080.pth')
model = utils.load_is_model(checkpoint_path, device)

# Possible choices: 'NoBRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C', 'RGB-BRS', 'DistMap-BRS'
brs_mode = 'f-BRS-B'
predictor = get_predictor(model, brs_mode, device, prob_thresh=MODEL_THRESH)

# %% [markdown]
# ### Dataset evaluation

# %%
TARGET_IOU = 0.9

all_ious, elapsed_time = evaluate_dataset(dataset, predictor, pred_thr=MODEL_THRESH, 
                                          max_iou_thr=TARGET_IOU, max_clicks=EVAL_MAX_CLICKS)
mean_spc, mean_spi = utils.get_time_metrics(all_ious, elapsed_time)
noc_list, over_max_list = utils.compute_noc_metric(all_ious,
                                                   iou_thrs=[0.8, 0.85, 0.9],
                                                   max_clicks=EVAL_MAX_CLICKS)

header, table_row = utils.get_results_table(noc_list, over_max_list, brs_mode, DATASET,
                                            mean_spc, elapsed_time, EVAL_MAX_CLICKS)
print(header)
print(table_row)

# %% [markdown]
# ### Single sample eval

# %%
sample_id = 20
TARGET_IOU = 0.

sample = dataset.get_sample(sample_id)
gt_mask = sample.gt_mask

clicks_list, ious_arr, pred = evaluate_sample(sample.image, gt_mask, predictor, 
                                              pred_thr=MODEL_THRESH, 
                                              max_iou_thr=TARGET_IOU, max_clicks=EVAL_MAX_CLICKS)

pred_mask = pred > MODEL_THRESH
draw = vis.draw_with_blend_and_clicks(sample.image, mask=pred_mask, clicks_list=clicks_list)
draw = np.concatenate((draw,
    255 * pred_mask[:, :, np.newaxis].repeat(3, axis=2),
    255 * (gt_mask > 0)[:, :, np.newaxis].repeat(3, axis=2)
), axis=1)

# print(ious_arr)

plt.figure(figsize=(20, 30))
plt.imshow(draw)
plt.show()

pred_mask = pred_mask.astype(bool)
gt_mask = gt_mask.astype(bool)

true_positive = np.sum(pred_mask & gt_mask)

pred_sum = np.sum(pred_mask)
gt_sum = np.sum(gt_mask)

if (pred_sum + gt_sum) == 0:
    dice = 1.0
else:
    dice_coeff = 2 * true_positive / (pred_sum + gt_sum)
    dice = dice_coeff

print(f"Dice Score: {dice:.4f}")
print(f"Required Clicks: {len(clicks_list)}")

# %%
plt.imshow(sample.image)


