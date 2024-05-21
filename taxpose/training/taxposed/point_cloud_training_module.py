import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from torchvision.transforms import ToTensor

to_tensor = ToTensor()
torch.set_printoptions(threshold=5, linewidth=1000)

class PointCloudTrainingModule(pl.LightningModule):

    def __init__(self,
                 model=None,
                 lr=1e-3,
                 image_log_period=500):
        super().__init__()
        self.model = model
        self.lr = lr
        self.image_log_period = image_log_period
        self.global_val_step = 0
        self.automatic_optimization = True

    def module_step(self, batch, batch_idx):
        raise NotImplementedError(
            'module_step must be implemented by child class')
        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        return {}

    def manual_training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        batch_size = batch[list(batch.keys())[0]].shape[0]
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val, batch_size=batch_size)

        if((self.global_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D) or isinstance(val, wandb.Html)):
                    wandb.log(
                        {key: val, "trainer/global_step": self.global_step, })
                else:
                    self.logger.log_image(
                        key, images=[val],  # self.global_step
                    )
        self.log('train_loss', loss, batch_size=batch_size, prog_bar=True)
        
        # opt.zero_grad()
        # self.manual_backward(loss)
        # opt.step()

        # # Print all the gradients
        # for name, param in self.named_parameters():
        #     print(f'param: {name} | grad: {param.grad}')
        
        # print(f'loss: {loss}')
        
        # breakpoint()
        
        return loss

    def training_step(self, batch, batch_idx):
        if not self.automatic_optimization:
            return self.manual_training_step(batch, batch_idx)
        
        batch_size = batch[list(batch.keys())[0]].shape[0]
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val, batch_size=batch_size)

        if((self.global_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D) or isinstance(val, wandb.Html)):
                    wandb.log(
                        {key: val, "trainer/global_step": self.global_step, })
                else:
                    self.logger.log_image(
                        key, images=[val],  # self.global_step
                    )
        self.log('train_loss', loss, batch_size=batch_size)
        
        # Temp way to disable training
        # loss = torch.zeros(loss.shape)
        # loss.requires_grad = True
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch[list(batch.keys())[0]].shape[0]
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(f'val_{key}/val_{dataloader_idx}', val, add_dataloader_idx=False, batch_size=batch_size)

        if((self.global_val_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D), isinstance(val, wandb.Html)):
                    wandb.log(
                        {f'val_{key}/val_{dataloader_idx}': val, "trainer/global_step": self.global_val_step, })
                else:
                    self.logger.log_image(
                        f'val_{key}/val_{dataloader_idx}', images=[val],  # self.global_val_step
                    )
        self.global_val_step += 1

        self.log(f'val_loss/val_{dataloader_idx}', loss, add_dataloader_idx=False, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        batch_size = batch[list(batch.keys())[0]].shape[0]
        loss, log_values = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val, batch_size=batch_size)

        if((self.global_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D), isinstance(val, wandb.Html)):
                    wandb.log({'test_' + key: val, })
                else:
                    self.logger.log_image(
                        'test_' + key, val)

        self.log('test_loss', loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer


class AdversarialPointCloudTrainingModule(pl.LightningModule):

    def __init__(self,
                 generator=None,
                 discriminator=None,
                 generator_loss_weight=1.0,
                 discriminator_loss_weight=1.0,
                 gradient_clipping=None,
                 lr=1e-3,
                 image_log_period=500):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss_weight = generator_loss_weight
        self.discriminator_loss_weight = discriminator_loss_weight
        self.real_label_val = 1.0
        self.fake_label_val = 0.0
        
        self.lr = lr
        self.image_log_period = image_log_period
        self.global_val_step = 0
        self.gradient_clipping = gradient_clipping
        
        self.automatic_optimization = False

    def maybe_freeze_parameters(self):
        raise NotImplementedError(
            'maybe_freeze_parameters must be implemented by child class')
        
    def maybe_unfreeze_parameters(self):
        raise NotImplementedError(
            'maybe_unfreeze_parameters must be implemented by child class')

    def module_step(self, batch, batch_idx):
        raise NotImplementedError(
            'module_step must be implemented by child class')
        return loss, log_values

    def adversarial_module_step(self, batch, batch_idx):
        raise NotImplementedError(
            'adversarial_module_step must be implemented by child class')
        return loss, log_values

    def visualize_results(self, batch, batch_idx):
        return {}

    def training_step(self, batch, batch_idx):
        gen_opt, disc_opt = self.optimizers()
        
        batch_size = batch[list(batch.keys())[0]].shape[0]
        points_action = batch['points_action']
        points_anchor = batch['points_anchor']
        points_trans_action = batch['points_action_trans']
        
        pzX_log_values = {}
        pzX_loss = 0
        
        # Do standard p(z|X) training step
        pzX_loss, pzX_log_values, pred_points_action_orig_frame = self.adversarial_module_step(batch, batch_idx)
        
        gen_opt.zero_grad()
        disc_opt.zero_grad()
        self.manual_backward(pzX_loss, retain_graph=True)
        self.clip_gradients(gen_opt, self.gradient_clipping, gradient_clip_algorithm='norm')
        gen_opt.step()
        
        adv_log_values = {}
        discriminator_loss = 0
        
        # Run the discriminator
        real_logits, _ = self.discriminator(points_action.permute(0, 2, 1), points_anchor.permute(0, 2, 1))
        fake_logits, _ = self.discriminator(pred_points_action_orig_frame.permute(0, 2, 1).detach(), points_anchor.permute(0, 2, 1))
        # fake_logits, _ = self.discriminator(points_trans_action.permute(0, 2, 1).detach(), points_anchor.permute(0, 2, 1))
        adv_log_values['disc_real_predictions'] = F.sigmoid(real_logits).mean()
        adv_log_values['disc_fake_predictions'] = F.sigmoid(fake_logits).mean()
        
        real_pred = F.binary_cross_entropy_with_logits(real_logits, torch.full_like(real_logits, self.real_label_val), reduction='mean')
        fake_pred = F.binary_cross_entropy_with_logits(fake_logits, torch.full_like(fake_logits, self.fake_label_val), reduction='mean')
        
        # Compute the discriminator loss
        discriminator_loss = self.discriminator_loss_weight * (real_pred + fake_pred)
        adv_log_values['discriminator_loss'] = discriminator_loss
        
        gen_opt.zero_grad()
        disc_opt.zero_grad()
        self.manual_backward(discriminator_loss)
        self.clip_gradients(disc_opt, self.gradient_clipping, gradient_clip_algorithm='norm')
        disc_opt.step()

        generator_loss = 0
        
        # Compute the generator loss, maximize log(D(G(z)))
        # Get new classsifications from discriminator since we just updated it
        fake_logits, _ = self.discriminator(pred_points_action_orig_frame.permute(0, 2, 1), points_anchor.permute(0, 2, 1))
        adv_log_values['gen_fake_predictions'] = F.sigmoid(fake_logits).mean()
        
        generator_loss = self.generator_loss_weight * \
            F.binary_cross_entropy_with_logits(fake_logits, torch.full_like(fake_logits, self.real_label_val), reduction='mean')
        adv_log_values['generator_loss'] = generator_loss
        
        gen_opt.zero_grad()
        disc_opt.zero_grad()
        self.manual_backward(generator_loss)
        self.clip_gradients(gen_opt, self.gradient_clipping, gradient_clip_algorithm='norm')
        gen_opt.step()
        
        for key, val in {**pzX_log_values, **adv_log_values}.items():
            self.log(key, val, batch_size=batch_size)

        if((self.global_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D) or isinstance(val, wandb.Html)):
                    wandb.log(
                        {key: val, "trainer/global_step": self.global_step, })
                else:
                    self.logger.log_image(
                        key, images=[val],  # self.global_step
                    )
                
        train_loss = pzX_loss + discriminator_loss + generator_loss    
        self.log_dict({'train_loss': train_loss, 'pzX_loss': pzX_loss, 'gen_loss': generator_loss, 'disc_loss': discriminator_loss}, batch_size=batch_size, prog_bar=True)        
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch[list(batch.keys())[0]].shape[0]
        loss, log_values, _ = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(f'val_{key}/val_{dataloader_idx}', val, add_dataloader_idx=False, batch_size=batch_size)

        if((self.global_val_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D), isinstance(val, wandb.Html)):
                    wandb.log(
                        {f'val_{key}/val_{dataloader_idx}': val, "trainer/global_step": self.global_val_step, })
                else:
                    self.logger.log_image(
                        f'val_{key}/val_{dataloader_idx}', images=[val],  # self.global_val_step
                    )
        self.global_val_step += 1

        self.log(f'val_loss/val_{dataloader_idx}', loss, add_dataloader_idx=False, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        batch_size = batch[list(batch.keys())[0]].shape[0]
        loss, log_values, _ = self.module_step(batch, batch_idx)

        for key, val in log_values.items():
            self.log(key, val, batch_size=batch_size)

        if((self.global_step % self.image_log_period) == 0):
            results_images = self.visualize_results(batch, batch_idx)

            for key, val in results_images.items():
                if(isinstance(val, wandb.Object3D), isinstance(val, wandb.Html)):
                    wandb.log({'test_' + key: val, })
                else:
                    self.logger.log_image(
                        'test_' + key, val)

        self.log('test_loss', loss, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        # disc_opt = torch.optim.SGD(self.discriminator.parameters(), lr=self.lr, momentum=0.9)
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return gen_opt, disc_opt
