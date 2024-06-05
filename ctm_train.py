import argparse
import json
import logging
import math
import os
import random
import time

import datasets
import diffusers
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import transformers
import wandb
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from ctm.script_util import (
    add_dict_to_argparser,
    create_ema_and_scales_fn,
    create_model_and_diffusion,
    ctm_train_defaults,
)
from ctm.train_util import CTMTrainLoop
from datasets import load_dataset
# from diffusers.utils.import_utils import is_xformers_available
# from packaging import version
from tango_edm.models_edm import build_pretrained_models
from torch.utils.data import DataLoader, Dataset
# from transformers import SchedulerType

logger = get_logger(__name__)
def rand_fix(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    

def create_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--seed", type=int, default=5031,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--train_file", type=str, default="data/train_audiocaps.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--num_examples", type=int, default=-1,
        help="How many audio samples to use for training and validation from entire training dataset.",
    )
    parser.add_argument(
        "--text_encoder_name", type=str, default="google/flan-t5-large",
        help="Text encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_model_config", type=str, default="configs/diffusion_model_config.json",
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--ctm_unet_model_config", type=str, default="configs/diffusion_model_config.json",
        help="CTM's UNet model config json path.",
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", default=False,
        help="Freeze the text encoder model.",
    )
    parser.add_argument(
        "--text_column", type=str, default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column", type=str, default="location",
        help="The name of the column in the datasets containing the audio paths.",
    )
    parser.add_argument(
        "--tango_data_augment", action="store_true", default=False,
        help="Augment training data by tango's data augmentation.",
    )
    parser.add_argument(
        "--augment_num", type=int, default=2,
        help="number of augment training data.",
    )
    parser.add_argument(
        "--uncond_prob", type=float, default=0.1,
        help="Dropout rate of conditon text.",
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Add prefix in text prompts.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=6,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=40,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="input audio duration"
    )
    parser.add_argument(
        "--checkpointing_steps", type=str, default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )

    parser.add_argument(
        "--model_grad_clip_value", type=float, default=1000.,
        help="Clipping value for gradient of model"
    )
    parser.add_argument(
        "--sigma_data", type=float, default=0.25,
        help="sigma_data of the teacher model"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="If the training should continue from a local checkpoint folder.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='bf16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--teacher_model_path", type=str, default="ckpt/teacher/pytorch_model_2_sigma_025.bin",
        help="Path to teacher model ckpt",
    )
    parser.add_argument(
        "--stage1_path", type=str, default="ckpt/audioldm-s-full.ckpt",
        help="Path to stage1 model ckpt",
    )
    
    defaults = dict()
    defaults.update(ctm_train_defaults())
    defaults.update()
    
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    return args



class Text2AudioDataset(Dataset):
    def __init__(self, dataset, prefix, text_column, audio_column, uncond_prob=0.1 ,num_examples=-1):

        inputs = list(dataset[text_column])
        self.inputs = [prefix + inp for inp in inputs]
        self.audios = list(dataset[audio_column])
        self.indices = list(range(len(self.inputs)))
        self.uncond_prob = uncond_prob

        self.mapper = {}
        for index, audio, text in zip(self.indices, self.audios, inputs):
            self.mapper[index] = [audio, text]

        if num_examples != -1:
            self.inputs, self.audios = self.inputs[:num_examples], self.audios[:num_examples]
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.inputs)

    def get_num_instances(self):
        return len(self.inputs)

    def __getitem__(self, index):
        text = self.inputs[index]
        text = "" if random.random() < self.uncond_prob else text
        s1, s2, s3 = text, self.audios[index], self.indices[index]
        return s1, s2, s3

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


def main():
    args = create_argparser() 
    
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
        
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    # deepspeed_plugin = DeepSpeedPlugin(zero_stage=2)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        mixed_precision=args.mixed_precision, 
    #   deepspeed_plugin=deepspeed_plugin,
    #   kwargs_handlers=[ddp_kwargs],
        **accelerator_log_kwargs
    )
    

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        datasets.utils.logging.set_verbosity_error()


    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rand_fix(args.seed)

    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = "saved/" + str(int(time.time()))
            
            if not os.path.exists("saved"):
                os.makedirs("saved")
                
            os.makedirs(args.output_dir, exist_ok=True)
            
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(args.output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        wandb.init(project="Crusoe Latent CTM for TANGO-based T2S")

    accelerator.wait_for_everyone()

    # Get the datasets
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file   

    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    text_column, audio_column = args.text_column, args.audio_column

    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode, 
        start_ema=args.start_ema, 
        scale_mode=args.scale_mode, 
        start_scales=args.start_scales, 
        end_scales=args.end_scales, 
        total_steps=args.total_training_steps, 
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    

    # Load stage1 models, vocoder, and mel preprocess function
    pretrained_model_name = "audioldm-s-full"
    vae, stft = build_pretrained_models(pretrained_model_name, stage1_ckpt=args.stage1_path)
    # vae, stft = build_pretrained_models(pretrained_model_name)
    vae.requires_grad_(False)
    stft.requires_grad_(False)
    vae.eval()
    stft.eval()

    # Load Model
    logger.info("creating the student model")
    model, diffusion = create_model_and_diffusion(args)
    model.train()


    # Load teacher model
    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.info(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model, _ = create_model_and_diffusion(args, teacher=True)
        
        if os.path.exists(args.teacher_model_path):
            model_ckpt = th.load(args.teacher_model_path, map_location=accelerator.device)
            teacher_model.load_state_dict(model_ckpt, strict=False)
        teacher_model.eval()
        
        # Initialize model parameters with teacher model
        for dst_name, dst in model.ctm_unet.named_parameters():
            for src_name, src in teacher_model.unet.named_parameters():
                if dst_name in ['.'.join(src_name.split('.')[1:]), src_name]:
                    dst.data.copy_(src.data)
                    break

        for dst_name, dst in model.text_encoder.named_parameters():
            for src_name, src in teacher_model.text_encoder.named_parameters():
                if dst_name in ['.'.join(src_name.split('.')[1:]), src_name]:
                    dst.data.copy_(src.data)
                    break
        
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        logger.info(f"Initialized parameters of student (online) model synced with the teacher model from {args.teacher_model_path}")
        
    else:
        teacher_model = None
    # Load the target model for distillation, if path specified.
    logger.info("creating the target model")
    target_model, _ = create_model_and_diffusion(args)
    logger.info(f"Copy parameters of student model with the target_model model")
    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)
        
    target_model.requires_grad_(False)
    target_model.train()
    
    vae.to(accelerator.device)
    stft.to(accelerator.device)

    target_model.to(accelerator.device)
    teacher_model.to(accelerator.device)

    # Define dataloader
    logger.info("creating data loader...")
    if args.prefix:
            prefix = args.prefix
    else:
        prefix = ""

    with accelerator.main_process_first():
        train_dataset = Text2AudioDataset(raw_datasets["train"], prefix, text_column, audio_column, args.uncond_prob, args.num_examples)
        accelerator.print("Num instances in train: {}".format(train_dataset.get_num_instances()))
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=train_dataset.collate_fn)
    
    # Optimizer 
    if args.freeze_text_encoder:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
            model.text_encoder.eval()
        
        if args.ctm_unet_model_config:
            optimizer_parameters = model.ctm_unet.parameters()
            accelerator.print("Optimizing CTM UNet parameters.")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print("Num CTM UNet trainable parameters: {}".format(num_trainable_parameters))
    
    optimizer = th.optim.RAdam(
        optimizer_parameters, lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    overrode_total_training_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.total_training_steps is None:
        args.total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_total_training_steps = True
    
    
    model, optimizer,  train_dataloader,  = accelerator.prepare(
        model, optimizer,  train_dataloader
    )


    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if overrode_total_training_steps:
        args.total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.total_training_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        accelerator.init_trackers("text_to_audio_diffusion", experiment_config)


    # Train
    total_batch_size = (args.per_device_train_batch_size + args.augment_num) * args.gradient_accumulation_steps * accelerator.num_processes
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device + augment_num = {args.per_device_train_batch_size} + {args.augment_num}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation & data augmentation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.total_training_steps}")
    # progress_bar = tqdm(range(args.total_training_steps), disable=not accelerator.is_local_main_process)
    resume_epoch = 0
    resume_step = 0
    resume_global_step = 0
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            progress_state = th.load(os.path.join(args.resume_from_checkpoint, "progress_state.pth"), map_location=accelerator.device)
            resume_step = progress_state['completed_steps']
            resume_global_step = progress_state['completed_global_steps']
            resume_epoch = progress_state['completed_epochs']
            accelerator.load_state(args.resume_from_checkpoint)
            
            state_dict = th.load(os.path.join(args.resume_from_checkpoint, f"ema_{args.ema_rate}_{resume_step:06d}.pt"), map_location=accelerator.device)
            target_model.load_state_dict(state_dict, strict=False)
            target_model.requires_grad_(False)
            target_model.train()
            target_model.to(accelerator.device)
            accelerator.print(f"Resumed from local checkpoint: {args.resume_from_checkpoint}")
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            
    CTMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        latent_decoder=vae,
        stft=stft,
        ema_scale_fn=ema_scale_fn,
        diffusion=diffusion,
        data=train_dataloader,
        args=args,
        accelerator=accelerator, 
        opt=optimizer, 
        resume_step=resume_step,
        resume_global_step=resume_global_step,
        resume_epoch=resume_epoch,
    ).run_loop()


if __name__ == "__main__":
    main()