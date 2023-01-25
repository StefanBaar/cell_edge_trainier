import numpy as np

import torch

import pytorch_lightning as pl

import segmentation_models_pytorch as smp

class CellTrainer(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, weights,**kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes,encoder_weights=weights, **kwargs
        )

        params = smp.encoders.get_preprocessing_params(encoder_name,pretrained='imagenet')
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))


        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE,
                                           from_logits=True)

    def forward(self, image):

        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image = batch[0]
        mask  = batch[1]

        logits_mask = self.forward(image)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        loss = self.loss_fn(logits_mask, mask)

        new_mask = torch.unsqueeze(pred_mask.argmax(1),1)

        tp, fp, fn, tn = smp.metrics.get_stats(new_mask.long(), mask.long(),
                                               mode        = "multiclass",
                                               num_classes = pred_mask.shape[1])

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):

        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        #dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        per_image_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
        per_image_pre = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")

        metrics = {
            f"{stage}_loss": [i["loss"] for i in outputs][0],
            f"{stage}_IOU": per_image_iou,
            f"{stage}_acc": per_image_acc,
            f"{stage}_pre": per_image_pre,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        mets = self.shared_step(batch, "train")
        return mets

    def training_epoch_end(self, outputs):
        #print(2)
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        #print(3)
        mets = self.shared_step(batch, "valid")
        return mets

    def validation_epoch_end(self, outputs):
        #print(4)
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        #print(5)
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        #print(6)
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        #print(7)
        return torch.optim.Adam(self.parameters(), lr=0.0001)
