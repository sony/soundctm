# [ICLR'25] SoundCTM: Unifying Score-based and Consistency Models for Full-band Text-to-Sound Generation

This repository is the official implementation of "SoundCTM: Uniting Score-based and Consistency Models for Text-to-Sound Generation"

- Paper (ICLR'25 version): [Openreview](https://openreview.net/forum?id=KrK6zXbjfO)
- Paper (NeurIPS'25 audio imagenation workshop version): [arxiv](https://arxiv.org/abs/2405.18503)
- Demo page of SoundCTM UNet 16kHz (NeurIPS'25 audio imagenation workshop): [Audio Samples](https://koichi-saito-sony.github.io/soundctm/)
- Chekpoints of SoundCTM UNet 16kHz (NeurIPS'25 audio imagenation workshop): [Hugging Face](https://huggingface.co/Sony/soundctm)

- GitHub repository of [SoundCTM-DiT (ICLR'25)](https://github.com/koichi-saito-sony/soundctm_dit_iclr/)
- Checkpoints of [SoundCTM-DiT (ICLR'25)](https://huggingface.co/koichisaito/soundctm_dit)


Contact:
- Koichi SAITO: koichi.saito@sony.com

## Info 
- [2025/03/30] SoundCTM-DiT is uploaded.
- [2024/12/04] We're plainig to open-source codebase/checkpoints of DiT backbone with full-band text-to-sound setting and downstream tasks, as well.
- [2024/02/10] Our paper, updated version [openreview](https://openreview.net/forum?id=KrK6zXbjfO) from [previous version](https://arxiv.org/abs/2405.18503), is accepted at ICLR'25!!

## Checkpoints (Current checkpoint is based on previous version.)

- Download and put the [teacher model's checkpoints](https://huggingface.co/Sony/soundctm/tree/main/ckpt/teacher) and [AudioLDM-s-full checkpoints for VAE+Vocoder part](https://huggingface.co/Sony/soundctm/blob/main/ckpt/audioldm-s-full.ckpt) to `soundctm/ckpt`
- [SoundCTM checkpoint](https://huggingface.co/Sony/soundctm/tree/main/soundctm_ckpt) on AudioCaps (ema=0.999, 30K training iterations)

For inference, both [AudioLDM-s-full (for VAE's decoder+Vocoder)](https://huggingface.co/Sony/soundctm/blob/main/ckpt/audioldm-s-full.ckpt) and [SoundCTM](https://huggingface.co/Sony/soundctm/tree/main/soundctm_ckpt) checkpoints will be used.

## Prerequisites

Install docker to your own server and build docker container:

```bash
docker build -t soundctm .
```

Then run scripts in the container.

## Training
Please see `ctm_train.sh` and `ctm_train.py` and modify folder path dependeing on your environment.

Then run `bash ctm_train.sh`

## Inference
Please see `ctm_inference.sh` and `ctm_inference.py` and modify folder path dependeing on your environment.

Then run `bash ctm_inference.sh`

## Numerical evaluation
Please see `numerical_evaluation.sh` and `numerical_evaluation.py` and modify folder path dependeing on your environment.

Then run `bash numerical_evaluation.sh`


## Dataset
Follow the instructions given in the [AudioCaps repository](https://github.com/cdjkim/audiocaps) for downloading the data. 
Data locations are needed to be spesificied in `ctm_train.sh`. 
You can also see some examples at `data/train.csv`.


## WandB for logging
The training code also requires a [Weights & Biases](https://wandb.ai/site) account to log the training outputs and demos. Create an account and log in with:
```bash
$ wandb login
```
Or you can also pass an API key as an environment variable `WANDB_API_KEY`.
(You can obtain the API key from https://wandb.ai/authorize after logging in to your account.)
```bash
$ WANDB_API_KEY="12345x6789y..."
```


## Citation
```
@inproceedings{
  saito2025soundctm,
  title={Sound{CTM}: Unifying Score-based and Consistency Models for Full-band Text-to-Sound Generation},
  author={Koichi Saito and Dongjun Kim and Takashi Shibuya and Chieh-Hsin Lai and Zhi Zhong and Yuhta Takida and Yuki Mitsufuji},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=KrK6zXbjfO}
}
```

## Reference
Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution. 
> https://github.com/sony/ctm

> https://github.com/declare-lab/tango

> https://github.com/haoheliu/AudioLDM

> https://github.com/haoheliu/audioldm_eval


