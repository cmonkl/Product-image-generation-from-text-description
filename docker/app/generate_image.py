def generate_image(text_prompt, vae, unet, noise_scheduler, text_encoder, tokenizer, im_height=512, im_width=512):
        noise_scheduler.set_timesteps(70)
        text = tokenizer(text_promt, padding="max_length",max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"]
        text = torch.cat(text)   #.half()
        batch_size = text.shape[0]

        with torch.no_grad():
            text_embeddings = text_encoder(text.to(device))[0] #.half()
        print('Text embedding gone')
        latents = torch.randn((batch_size, unet.in_channels, im_height // 8, im_width // 8))
        latents = latents.half().to(device) #* noise_scheduler.init_noise_sigma
        latents = latents * vae.config.scaling_factor 
        i = 0
        for t in noise_scheduler.timesteps:
            print('Timestamp:', i)
            latent_model_input = noise_scheduler.scale_model_input(latents, t)
            
            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            i += 1
           
        latents = 1 / vae.config.scaling_factor * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        return image
                                                                                                                                        
