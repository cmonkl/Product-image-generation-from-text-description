from tqdm.auto import tqdm
import os
import torch
import math
from PIL import Image

def train_step(vae, unet, text_encoder, noise_scheduler, dataloader, criterion,
               optimizer, device, accelerator, scaler):
    unet.train()

    epoch_loss = 0.0
    NUM_ACCUMULATION_STEPS = 2
    for idx, batch_data in tqdm(enumerate(dataloader)):
        text, images = batch_data
        optimizer.zero_grad()
        
        text_embeddings = text_encoder(text["input_ids"].to(device).squeeze(1))[0]
        batch_size = images.shape[0]

        #with torch.no_grad():
        latents = vae.encode(images.to(device, dtype=torch.float16)).latent_dist.sample()     
        latents = latents * vae.config.scaling_factor

        # create noise for latents
        noise = torch.randn_like(latents).to(latents.device)
        # Sample a random timestep for each image
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device).long()

        noisy_images = noise_scheduler.add_noise(latents, noise, t)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            noise_pred = unet(noisy_images, t, encoder_hidden_states=text_embeddings).sample
            loss = criterion(noise_pred.float(), noise.float(), reduction="mean") / NUM_ACCUMULATION_STEPS
        
        scaler.scale(loss).backward() #loss.backward()
        
        if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (idx + 1 == len(dataloader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        epoch_loss += loss.item()
        
    return loss / len(dataloader)

def eval_step(vae, unet, text_encoder, noise_scheduler, dataloader, 
              device, height, width, num_inference_steps, logger):
    unet.eval()
    vae.eval()
    text_encoder.eval()
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    
    num_images_to_log = 10
    num_iters = (num_images_to_log / dataloader.batch_size) + 1
    
    metric = 0.0
    for idx, batch_data in enumerate(dataloader):
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
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        latents = 1 / vae.config.scaling_factor * latents

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                pred_images = vae.decode(latents).sample
        
        if idx < num_iters:
            pred_images = (pred_images / 2 + 0.5).clamp(0, 1)
            pred_images = pred_images.cpu().permute(0, 2, 3, 1).float().numpy()
            true_images = (images / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
            image_array = [(true_images[i] * 255).astype(np.uint8) for i in range(true_images.shape[0])]
            pred_images = [(pred_images[i] * 255).astype(np.uint8) for i in range(pred_images.shape[0])]
            
            lbl_idx = idx * dataloader.batch_size
            indices = dataloader.dataset.indices
            labels = [dataloader.dataset.dataset.descriptions.iloc[indices[lbl_idx + i]]['description'] for i in range(true_images.shape[0])]
            
            true_images = [Image.fromarray(image) for image in image_array]
            pred_images = [Image.fromarray(image) for image in pred_images]
            logger.log({"true_images": [wandb.Image(image, caption=labels[i]) for i, image in enumerate(images)],
                      "pred_images": [wandb.Image(image, caption=labels[i]) for i, image in enumerate(pred_images)]})
        else:
            break
            
        # compute metrics
        # todo
        metric += images.mean().item()
        

    return metric / len(dataloader)

def train(vae, unet, text_encoder, noise_scheduler, num_epochs, train_loader, 
          val_loader, criterion, optimizer, save_path, logger, device, args, inf_freq=None):
    #vae.eval()
    #text_encoder.eval()
    im_height, im_width = val_loader.dataset[0][1].shape[1:3]
    
    accelerator = args.accelerator

    #unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    #text_encoder.to(accelerator.device,  dtype=torch.float16)
    #vae.to(accelerator.device,  dtype=torch.float16)
    text_encoder.to(device,  dtype=torch.float16)
    vae.to(device,  dtype=torch.float16)
    unet.to(device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    best_metric = 0.0
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_step(vae, unet, text_encoder, noise_scheduler, 
                                train_loader, criterion, optimizer, device, accelerator, scaler)
        
        # log train loss to wandb
        logger.log({"train_loss":train_loss}, step=epoch)

        if (epoch + 1) % inf_freq == 0:
            val_metric = eval_step(vae, unet, text_encoder, noise_scheduler,
                                   val_loader, device, im_height, im_width, 50, logger)
            logger.log({"val_metric": val_metric}, step=epoch)

            if val_metric > best_metric:
                # save best model
                torch.save({
                    'epoch': epoch,
                    'unet_state_dict': unet.state_dict(),
                    'vae_state_dict': vae.state_dict(),
                    'text_enc_state_dict': text_encoder.state_dict()
                    }, os.path.join(save_path, f"diffusion_model_{round(val_metric, 2)}.pt"))
                
                #prev_file = os.path.join(save_path, f"diffusion_model_{round(best_metric)}.pt")
                #if os.path.exists(prev_file):
                #    os.remove(prev_file)
                best_metric = val_metric

    # load best model weights
    #best_checkpoint = torch.load(os.path.join(save_path, f"diffusion_model_{round(best_metric, 2)}.pt"))
    #vae.load_state_dict(best_checkpoint["vae_state_dict"])
    #unet.load_state_dict(best_checkpoint["unet_state_dict"])
    #text_encoder.load_state_dict(best_checkpoint["text_enc_state_dict"])

    #model = {'vae': vae, 'unet': unet, "text_encoder": text_encoder}
    #return model

def generate_images(text_prompts, vae, unet, noise_scheduler, text_encoder, tokenizer, im_height=512, im_width=512):
    noise_scheduler.set_timesteps(70)
    text = [tokenizer(text_prmt, padding="max_length", 
                                max_length=tokenizer.model_max_length, truncation=True,
                                return_tensors="pt")["input_ids"] for text_prmt in text_prompts]
    text = torch.cat(text)#.half()
    batch_size = text.shape[0]

    with torch.no_grad():
        text_embeddings = text_encoder(text.to(device))[0].half()

    latents = torch.randn((batch_size, unet.in_channels, im_height // 8, im_width // 8))
    latents = latents.half().to(device) #* noise_scheduler.init_noise_sigma
    latents = latents * vae.config.scaling_factor 

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

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    return images
