import argparse
import csv
import os
import time

import soundfile as sf
import torch
from accelerate.utils import set_seed
from tqdm import tqdm

from tango_edm.models_edm import AudioDiffusionEDM, build_pretrained_models


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    # parser.add_argument(
    #     "--original_args", type=str, default=None,
    #     help="Path for summary jsonl file saved during training."
    # )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the output."
    )
    parser.add_argument(
        "--seed", type=int, default=5031, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--text_encoder_name", type=str, default="google/flan-t5-large",
        help="Text encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_model_config", type=str, default=None,
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path for saved model bin file."
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
        "--num_steps", type=int, default=200,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3,
        help="Guidance scale for classifier free guidance."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
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
        "--stocastic", type=bool, default=False,
        help="Enable stocastic sampling of EDM Heun sampler",
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Add prefix in text prompts.",
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
    
    name = "audioldm-s-full"
    vae, stft = build_pretrained_models(name)
    vae, stft = vae.to(device), stft.to(device)
    
    model = AudioDiffusionEDM(
        text_encoder_name=args.text_encoder_name, 
        unet_model_config_path=args.unet_model_config,
        sigma_data=args.sigma_data,
        teacher=True,
    ).to(device)
    model.eval()
    
    # Load Trained Weight #
    model.load_state_dict(torch.load(args.model))
    
    # Load Data #
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = ""
        
    # text_prompts = [json.loads(line)[args.text_key] for line in open(args.test_file).readlines()]
    with open(args.test_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        text_prompts = [row['caption'] for row in reader]
    text_prompts = [prefix + inp for inp in text_prompts]
    with open(args.test_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        file_names = [row['file_name'] for row in reader]
    # Generate #
    num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
    all_outputs = []
        
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k: k+batch_size]
        
        with torch.no_grad():
            latents = model.inference(text, num_steps, guidance, num_samples, args.stocastic)
            mel = vae.decode_first_stage(latents)
            with torch.no_grad():
                wave = vae.decode_to_waveform(mel)
                wave = (wave.cpu().numpy() * 32768).astype("int16")
                wave = wave[:, :160000]
            all_outputs += [item for item in wave]
            
    # Save #
    exp_id = str(int(time.time()))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if num_samples == 1:
        output_dir = "outputs/{}_steps_{}_guidance_{}_seed_{}".format(exp_id, num_steps, guidance, args.seed)
        output_dir = os.path.join(args.output_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            filename = os.path.splitext(os.path.basename(file_names[j]))[0]
            sf.write("{}/{}.wav".format(output_dir, filename), wav, samplerate=16000)
        
if __name__ == "__main__":
    main()