from tqdm.auto import tqdm
import os
import torch


def train_step(vae, unet, text_encoder, noise_scheduler, dataloader, criterion, optimizer, device):
    unet.train()

    epoch_loss = 0.0

    for batch_data in dataloader:
        text, images = batch_data
        images = images.to(device)
        with torch.no_grad():
            text_embeddings = text_encoder(text["input_ids"].squeeze(1).to(device))[0]
        batch_size = images.shape[0]

        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()     
            latents = latents * vae.config.scaling_factor
        latents = latents.to(device)

        # create noise for latents
        noise = torch.randn_like(latents).to(device)
        # Sample a random timestep for each image
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        
        noisy_images = noise_scheduler.add_noise(latents, noise, t)
        noise_pred = unet(noisy_images, t, encoder_hidden_states=text_embeddings).sample

        loss = criterion(noise_pred.float(), noise.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
    
    return loss / len(dataloader)

def eval_step(vae, unet, text_encoder, noise_scheduler, dataloader, device, height, width):
    unet.eval()
    vae.eval()
    text_encoder.eval()
    
    metric = 0.0
    for batch_data in dataloader:
        text, images = batch_data
        images = images.to(device)
        with torch.no_grad():
            text_embeddings = text_encoder(text["input_ids"].squeeze(1).to(device))[0]
        batch_size = images.shape[0]

        latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8))
        latents = latents * vae.config.scaling_factor
        latents = latents.to(device)
        
        for t in noise_scheduler.timesteps:
            latent_model_input = noise_scheduler.scale_model_input(latents, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        latents = 1 / vae.config.scaling_factor * latents

        with torch.no_grad():
            images = vae.decode(latents).sample

        # compute metrics
        # todo
        metric += images.mean()

    return metric / len(dataloader)

def train(vae, unet, text_encoder, noise_scheduler, num_epochs, train_loader, 
          val_loader, criterion, optimizer, save_path, logger, device, inf_freq=None):
    vae.eval()
    text_encoder.eval()
    im_height, im_width = val_loader.dataset[0][1].shape[1:3]

    best_metric = 0.0
    for epoch in range(num_epochs):
        train_loss = train_step(vae, unet, text_encoder, noise_scheduler, 
                                train_loader, criterion, optimizer, device)
        
        # log train loss to wandb
        logger.log({"train_loss":train_loss}, step=epoch)

        if epoch % inf_freq == 0:
            val_metric = eval_step(vae, unet, text_encoder, noise_scheduler,
                                   val_loader, device, im_height, im_width)
            logger.log({"val_metric":val_metric}, step=epoch)

            if val_metric > best_metric:
                # save best model
                torch.save({
                    'epoch': epoch,
                    'unet_state_dict': unet.state_dict(),
                    'vae_state_dict': vae.state_dict(),
                    'text_enc_state_dict': text_encoder.state_dict()
                    }, os.path.join(save_path, f"diffusion_model_{round(val_metric, 2)}.pt"))
                
                prev_file = os.path.join(save_path, f"diffusion_model_{round(best_metric)}.pt")
                if os.path.exists(prev_file):
                    os.remove(prev_file)
                best_metric = val_metric

    # load best model weights
    best_checkpoint = torch.load(os.path.join(save_path, f"diffusion_model_{round(best_metric, 2)}.pt"))
    vae.load_state_dict(best_checkpoint["vae_state_dict"])
    unet.load_state_dict(best_checkpoint["unet_state_dict"])
    text_encoder.load_state_dict(best_checkpoint["text_enc_state_dict"])

    model = {'vae': vae, 'unet': unet, "text_encoder": text_encoder}
    return model

def generate_images(text_prompts):
    # todo
    pass
