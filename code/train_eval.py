from diffusers import (
    UNet2DConditionModel, 
    LMSDiscreteScheduler, 
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import bitsandbytes as bnb
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

def eval_step(unet, text_encoder, tokenizer, vae, accelerator, dataloader, logger, epoch, args, weight_dtype):
    indices = dataloader.dataset.indices
    n = 10
    labels_to_log = [dataloader.dataset.dataset.descriptions.iloc[indices[i]]['description'] for i in range(n)]
    true_images = [dataloader.dataset[i][1].float().permute(1, 2, 0) for i in range(n)]
    true_images = [(image / 2 + 0.5).clamp(0, 1).numpy() for image in true_images]
    image_array = [(true_images[i] * 255).astype(np.uint8) for i in range(len(true_images))]
    true_images_to_log = [Image.fromarray(image) for image in image_array]

    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.safety_checker = lambda images, clip_input: (images, False)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
   
    metric_fid = 0.0
    metric_inception = 0.0
    to_tens = transforms.ToTensor()
    num_iters = 10
    
    for idx, batch_data in enumerate(dataloader):
        if idx < num_iters:
            text, images = batch_data
            lbl_idx = idx * len(text)
            indices = dataloader.dataset.indices
            labels = [dataloader.dataset.dataset.descriptions.iloc[indices[lbl_idx + i]]['description'] for i in range(images.shape[0])]

            with torch.autocast("cuda"):
                pred_images = pipeline(labels, 
                                 num_inference_steps=args.num_inference_steps, 
                                 generator=generator, width=args.width, 
                                 height=args.height).images

            if idx == 0:
                imgs_to_log = pred_images[:n]

            fid = FrechetInceptionDistance(feature=2048, normalize=True)
            inception_score = InceptionScore(feature=2048, normalize=True)    

            pred_images = torch.cat([to_tens(im).unsqueeze(0) for im in pred_images], 0)
            true_images = (images / 2 + 0.5).clamp(0, 1) #torch.cat([to_tens(im) for im in true_images], 0)
            fid.update(true_images.cpu(), real=True)
            fid.update(pred_images.cpu(), real=False)
            
            inception_score.update(pred_images.cpu())
            metric_fid += fid.compute().item() #(pred_images, true_images)
            metric_inception += inception_score.compute()[0].item()

    logger.log({"pred_images": [wandb.Image(image, caption=labels_to_log[i]) for i, image in enumerate(imgs_to_log)],
                "true_images": [wandb.Image(image, caption=labels_to_log[i]) for i, image in enumerate(true_images_to_log)]},
                      step=epoch)
    metric_fid /= num_iters
    metric_inception /= num_iters
    
    del pipeline
    torch.cuda.empty_cache()
    
    return metric_fid, metric_inception

def train(args, train_dataloader, val_dataloader):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", 
                                    revision=args.revision)#,  torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",
                                            revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, 
                                                subfolder="text_encoder",
                                                revision=args.revision)#,  torch_dtype=torch.float16)

    noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    params_to_optimize = unet.parameters()
    optimizer = bnb.optim.AdamW8bit(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.seed is not None:
    set_seed(args.seed)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )


    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, test_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, test_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = os.path.basename(args.checkpoint_path)

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.checkpoint_path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
                    
            with accelerator.accumulate(unet):
                text, images = batch
                # Convert images to latent space
                latents = vae.encode(images.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                    
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(text["input_ids"].squeeze(1))[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    if global_step % (args.checkpointing_steps * num_update_steps_per_epoch) == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

                    if global_step % (args.validation_steps * num_update_steps_per_epoch) == 0:
                        eval_step(unet, text_encoder, tokenizer, vae, accelerator, test_dataloader, 
                                args.logger, epoch, args,weight_dtype) 
                        
            logs = {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            epoch_loss += loss.detach().item()
            if global_step >= args.max_train_steps:
                break

        args.logger.log({"train_loss": epoch_loss / num_update_steps_per_epoch}, step=epoch)
        args.logger.log({"lr":lr_scheduler.get_last_lr()[0]}, step=epoch)
        print(f"Epoch: {epoch}, loss: {epoch_loss / num_update_steps_per_epoch}")

    accelerator.wait_for_everyone()
    
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    print(f"Saved state to {save_path}")
    return unet, text_encoder