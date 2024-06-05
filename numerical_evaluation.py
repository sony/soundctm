import argparse
import json

import torch
from audioldm_eval import EvaluationHelper
# from audioldm_eval import EvaluationHelper_CLAP

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
    parser.add_argument(
        "--reference_dir", type=str, default="/data/audiocaps/test/audio",
        help="Folder containing the test reference wav files."
    )
    parser.add_argument(
        "--generated_dir", type=str, default="/path/to/generated_dir",
        help="Folder containing the generated wav files."
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Using GPU...")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU...")
    evaluator = EvaluationHelper(32000, device)
    # clap_evaluator = EvaluationHelper_CLAP(32000, device)
    result = evaluator.main(args.generated_dir, args.reference_dir)
    # result_clap = clap_evaluator.main(args.generated_dir, args.reference_dir)
    with open(f"{args.generated_dir}/summary.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n\n")
    
    # with open(f"{args.generated_dir}/summary_clap.jsonl", "a") as f:
    #     f.write(json.dumps(result_clap) + "\n\n")
        
            
if __name__ == "__main__":
    main()