import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from isegm.utils.exp_imports.default import *
from easydict import EasyDict as edict


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (240, 240)
    model_cfg.num_max_points = 20

    model = HRNetModel(width=32, ocr_width=128, with_aux_output=True, use_leaky_relu=True,
                       use_rgb_conv=True, use_disks=True, norm_radius=5, with_prev_mask=True)
    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W32)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    # loss 계산
    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.4

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = BraTSDataset(cfg.BRATS_PATH, split='train', min_object_area=1000, keep_background_prob=0.05,
                            points_sampler=points_sampler, epoch_len=30000, stuff_prob=0.30,
                            temp=cfg.temp)

    valset = BraTSDataset(cfg.BRATS_PATH, split='val', min_object_area=1000, keep_background_prob=0.05,
                          points_sampler=points_sampler, epoch_len=30000, stuff_prob=0.30,
                          temp=cfg.temp)

    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }
    # epoch이 milestones에 도달하면 학습률을 gamma만큼 곱해서 줄임
    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[200, 220], gamma=0.1)

    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 5), (200, 1)],
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)

    trainer.run(num_epochs=200, validation=False)
