import argparse
import csv
import json
import os
import random
import time

import numpy as np
import soundfile as sf
import torch
from accelerate.utils import set_seed
from ctm.inference_sampling import karras_sample
from ctm.script_util import (
    create_model_and_diffusion,
)
from tango_edm.models_edm import build_pretrained_models
from tqdm import tqdm


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def rand_fix(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--training_args", type=str, default=None,
        help="Path for 'summary.jsonl' file saved during training."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the output."
    )
    parser.add_argument(
        "--seed", type=int, default=5031,
        help="A seed for reproducible inference."
    )
    parser.add_argument(
        "--text_encoder_name", type=str, default="google/flan-t5-large",
        help="Text encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--ctm_unet_model_config", type=str, default="configs/diffusion_model_config.json",
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--sampling_rate", type=float, default=16000,
        help="Sampling rate of training data",
    )
    parser.add_argument(
        "--target_length", type=float, default=10,
        help="Audio length of training data",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--ema_model", type=str, default=None,
        help="Path for saved EMA model bin file."
    )
    parser.add_argument(
        "--sampler", type=str, default='determinisitc',
        help="Inference sampling methods. You can choose ['determinisitc' (gamma=0), 'cm_multistep' (gamma=1), 'gamma_multistep']."
    )
    parser.add_argument(
        "--sampling_gamma", type=float, default=0.9,
        help="\gamma for gamma-sampling if we use 'gamma_multistep'."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="json file containing the test prompts for generation."
    )
    parser.add_argument(
        "--test_references", type=str, default="data/audiocaps_test_references/subset",
        help="Folder containing the test reference wav files."
    )
    parser.add_argument(
        "--num_steps", type=int, default=1,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--nu", type=float, default=1.,
        help="Guidance scale for \nu interpolation."
    )
    parser.add_argument(
        "--omega", type=float, default=3.5,
        help="Omega for student model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="How many samples per prompt.",
    )
    parser.add_argument(
        "--sigma_data", type=float, default=0.25,
        help="Sigma data",
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Add prefix in text prompts.",
    )
    parser.add_argument(
        "--stage1_ckpt", type=str, default='ckpt/audioldm-s-full.ckpt',
        help="Path for stage1 model (VAE part)'s checkpoint",
    )
    
    args = parser.parse_args()

    return args

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU...")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU...")
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
        rand_fix(args.seed)
    
    train_args = dotdict(json.loads(open(args.training_args).readlines()[0]))
    
    # Load decoder and vocoder
    name = "audioldm-s-full"
    vae, stft = build_pretrained_models(name, args.stage1_ckpt)
    vae, stft = vae.to(device), stft.to(device)
    
    # Load Main network
    model, diffusion = create_model_and_diffusion(train_args, teacher=False)
    model.to(device)
    model.eval()
    
    model.load_state_dict(torch.load(args.model, map_location=device))
    ema_ckpt = torch.load(args.ema_model, map_location=device)
    state_dict = model.ctm_unet.state_dict()
    for i, (name, _value) in enumerate(model.ctm_unet.named_parameters()):
        assert name in state_dict
        state_dict[name] = ema_ckpt[name]
    del state_dict
    
    # Load Data #
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = ""
    with open(args.test_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        text_prompts = [row['caption'] for row in reader]
    text_prompts = [prefix + inp for inp in text_prompts]
    with open(args.test_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        file_names = [row['file_name'] for row in reader]
    # Generate #
    num_steps, nu, batch_size, num_samples = args.num_steps, args.nu, args.batch_size, args.num_samples
    all_outputs = []
        
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k: k+batch_size]
        
        with torch.no_grad():
            latents = karras_sample(
                diffusion=diffusion,
                model=model,
                shape=(batch_size, train_args.latent_channels, train_args.latent_t_size, train_args.latent_f_size),
                steps=num_steps,
                cond=text,
                nu=args.nu,
                model_kwargs={},
                device=device,
                omega=args.omega,
                sampler=args.sampler,
                gamma=args.sampling_gamma,
                x_T=None,
                sigma_min=train_args.sigma_min,
                sigma_max=train_args.sigma_max,
            )
            mel = vae.decode_first_stage(latents)
            wave = vae.decode_to_waveform(mel)
            wave = (wave.cpu().numpy() * 32768).astype("int16") # This is fixed by pretrained vocoder
            wave = wave[:, :int(args.sampling_rate * args.target_length)] 
            all_outputs += [item for item in wave]
            
    # Save #
    exp_id = str(int(time.time()))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if num_samples == 1:
        if args.omega is not None:
            output_dir = "outputs/{}_steps_{}_nu_{}_omega_{}_seed_{}".format(exp_id, num_steps, nu, args.omega, args.seed)
        else:
            output_dir = "outputs/{}_steps_{}_nu_{}_seed_{}".format(exp_id, num_steps, nu, args.seed)
        output_dir = os.path.join(args.output_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            filename = os.path.splitext(os.path.basename(file_names[j]))[0]
            sf.write("{}/{}.wav".format(output_dir, filename), wav, samplerate=args.sampling_rate)
        
if __name__ == "__main__":
    main()