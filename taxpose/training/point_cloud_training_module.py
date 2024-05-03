import pytorch_lightning as pl
import torch
import wandb
from torchvision.transforms import ToTensor

to_tensor = ToTensor()


class PointCloudTrainingModule(pl.LightningModule):
    def __init__(self, model=None, lr=1e-3, image_log_period=500):
        super().__init__()
        self.model = model
        self.lr = lr
        self.image_log_period = image_log_period
        self.global_val_step = 0

    def module_step(self, batch, batch_idx):
        raise NotImplementedError("module_step must be implemented by child class")
        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        return {}

    def training_step(self, batch, batch_idx):
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val)

        if (self.global_step % self.image_log_period) == 0:
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if isinstance(val, wandb.Object3D):
                    wandb.log(
                        {
                            key: val,
                            "trainer/global_step": self.global_step,
                        }
                    )
                else:
                    self.logger.log_image(
                        key,
                        images=[val],  # self.global_step
                    )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log("val_" + key, val)

        if (self.global_val_step % self.image_log_period) == 0:
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if isinstance(val, wandb.Object3D):
                    wandb.log(
                        {
                            "val_" + key: val,
                            "trainer/global_step": self.global_val_step,
                        }
                    )
                else:
                    self.logger.log_image(
                        "val_" + key,
                        images=[val],  # self.global_val_step
                    )
        self.global_val_step += 1

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val)

        if (self.global_step % self.image_log_period) == 0:
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if isinstance(val, wandb.Object3D):
                    wandb.log(
                        {
                            "test_" + key: val,
                        }
                    )
                else:
                    self.logger.log_image("test_" + key, val)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer
