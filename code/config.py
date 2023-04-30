from easydict import EasyDict as edict
from accelerate import Accelerator
from tqdm import tqdm 

args = edict()

args.gradient_accumulation_steps = 2
args.mixed_precision = "fp16" 
args.gradient_checkpointing = True

accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
args.revision = "fp16"
args.pretrained_model_name_or_path = 'CompVis/stable-diffusion-v1-4'
args.use_8bit_adam = True
args.train_batch_size = 8
args.max_train_steps = None
args.num_train_epochs = 10
args.train_text_encoder = False
args.set_grads_to_none = False
args.seed = None
args.scale_lr = False #???????????
args.learning_rate = 1e-6
args.adam_beta1 = 0.9
args.adam_beta2 = 0.999
args.adam_weight_decay = 1e-2
args.adam_epsilon = 1e-08
args.output_dir = '/kaggle/working/'
args.height, args.width = test_dataloader.dataset[0][1].shape[1:3]
args.num_inference_steps = 50
args.enable_xformers_memory_efficient_attention = False
args.max_grad_norm = 1.0
args.validation_steps = 1
args.checkpointing_steps = 7 #args.num_train_epochs // 2 + 1
args.lr_scheduler = 'constant'
args.lr_warmup_steps = 500
args.lr_num_cycles = 1
args.lr_power = 1
args.revision = "fp16"
args.resume_from_checkpoint = True
args.checkpoint_path = '/kaggle/input/fashion-data/checkpoint-10545/checkpoint-10545'
