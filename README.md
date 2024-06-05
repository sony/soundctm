# SoundCTM: Uniting Score-based and Consistency Models for Text-to-Sound Generation

This repository is the official implementation of "SoundCTM: Uniting Score-based and Consistency Models for Text-to-Sound Generation"

- [arxiv](https://arxiv.org/abs/2405.18503)
- [Audio Samples](https://koichi-saito-sony.github.io/soundctm/)
- [Hugging Face (Now only checkpoints are avaiable.](https://huggingface.co/Sony/soundctm)

Contact:
- Koichi SAITO: koichi.saito@sony.com

## Checkpoints

- Download and put the [teacher model's checkpoints](https://huggingface.co/Sony/soundctm/tree/main/ckpt/teacher) and [AudioLDM-s-full checkpoints for VAE+Vocoder part](https://huggingface.co/Sony/soundctm/blob/main/ckpt/audioldm-s-full.ckpt) to `soundctm/ckpt`
- [SoundCTM checkpoint](https://huggingface.co/Sony/soundctm/tree/main/soundctm_ckpt) on AudioCaps (ema=0.999, 30K training iterations)

For inference, both [AudioLDM-s-full (for VAE's decoder+Vocoder)](https://huggingface.co/Sony/soundctm/blob/main/ckpt/audioldm-s-full.ckpt) and [SoundCTM](https://huggingface.co/Sony/soundctm/tree/main/soundctm_ckpt) checkpoints will be used.

## Prerequisites

Install docker to your own server and biuld docker container:

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
@article{saito2024soundctm,
  title={SoundCTM: Uniting Score-based and Consistency Models for Text-to-Sound Generation}, 
  author={Koichi Saito and Dongjun Kim and Takashi Shibuya and Chieh-Hsin Lai and Zhi Zhong and Yuhta Takida and Yuki Mitsufuji},
  journal={arXiv preprint arXiv:2405.18503},
  year={2024}
}
```

## Reference
Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution. 
> https://github.com/sony/ctm

> https://github.com/declare-lab/tango

> https://github.com/haoheliu/AudioLDM

> https://github.com/haoheliu/audioldm_eval


