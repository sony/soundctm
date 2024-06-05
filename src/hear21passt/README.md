# PaSST package for HEAR 2021 NeurIPS Challenge Holistic Evaluation of Audio Representations

This is an implementation for [Efficient Training of Audio Transformers with Patchout](https://arxiv.org/abs/2110.05069) for HEAR 2021 NeurIPS Challenge
Holistic Evaluation of Audio Representations

# CUDA version

This is an implementation is tested with CUDA version 11.1, and torch installed:

```shell
pip3 install torch==1.8.1+cu111  torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

but should work on newer versions of CUDA and torch.

# Installation

Install the latest version of this repo:

```shell
pip install hear21passt
```

The models follow the [common API](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api) of HEAR 21
:

```shell
hear-validator --model hear21passt.base.pt hear21passt.base
hear-validator --model noweights.txt hear21passt.base2levelF
hear-validator --model noweights.txt hear21passt.base2levelmel
 ```

There are three modules available `hear21passt.base`,`hear21passt.base2level`, `hear21passt.base2levelmel` :

```python
import torch

from hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings

model = load_model().cuda()
seconds = 15
audio = torch.ones((3, 32000 * seconds))*0.5
embed, time_stamps = get_timestamp_embeddings(audio, model)
print(embed.shape)
embed = get_scene_embeddings(audio, model)
print(embed.shape)
```

# Getting the Logits/Class Labels

You can get the logits (before the sigmoid activation) for the 527 classes of audioset:

```python
from hear21passt.base import load_model

model = load_model(mode="logits").cuda()
logits = model(wave_signal)
```

The class labels indices can be found [here](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/metadata/class_labels_indices.csv)

You can also use different pre-trained models, for example, the model trained with KD `passt_s_kd_p16_128_ap486`:

```python
from hear21passt.base import get_basic_model

model = get_basic_model(mode="logits", arch="passt_s_kd_p16_128_ap486")
logits = model(wave_signal)

```

# Supporting longer clips

In case of an input longer than 10 seconds, the `get_scene_embeddings` method compute the average of the embedding of a 10-second overlapping windows.
Depending on the application, it may be useful to use a pre-trained that can extract embeddings from 20 or 30 seconds without averaging. These variant has pre-trained time positional encoding or 20/30 seconds:

```python
# from version 0.0.18, it's possible to use:
from hear21passt.base20sec import load_model # up to 20 seconds of audio.
# or 
from hear21passt.base30sec import load_model # up to 30 seconds of audio.

model = load_model(mode="logits").cuda()
logits = model(wave_signal)
```

# Loading other pre-trained models for logits or fine-tuning

Each pre-trained model has a specific frequency/time positional encoding, it's necessary to select the correct input shape to be able to load the models. The important variables for loading are `input_tdim`, `fstride` and `tstride` to specify the spectrograms time frames, the patches stride over frequency, and patches stride over time, respectively.

```python
import torch

from hear21passt.base import get_basic_model, get_model_passt

model = get_basic_model(mode="logits")

logits = model(some_wave_signal)

# Examples of other pre-trained models using the same spectrograms

# pre-traind on openMIC-18
model.net = get_model_passt(arch="openmic",  n_classes=20)
# pre-traind on FSD-50k
model.net = get_model_passt(arch="fsd50k",  n_classes=200)
# pre-traind on FSD-50k without patch-overlap (faster)
model.net = get_model_passt(arch="fsd50k-n",  n_classes=200, fstride=16, tstride=16)

# models are trained on 10 seconds audios from Audioset, but accept longer audios (20s, or 30s)
# These models are trained by sampling a 10-second time-pos-encodings sequence 
model.net = get_model_passt("passt_20sec", input_tdim=2000)
model.net = get_model_passt("passt_30sec", input_tdim=3000)
```

If you provide the wrong spectrograms, the model may fail silently, by generating low-quality embeddings and logits. Make sure you have the correct spectrograms' config for the selected pre-trained models.
Models with higher spectrogram resolutions, need to specify the correct spectrogram config:

```python
from hear21passt.models.preprocess import AugmentMelSTFT

# high-res pre-trained on Audioset
model.net = get_model_passt("stfthop160", input_tdim=2000)

# hopsize=160 for this pretrained model
model.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=160, n_fft=1024, freqm=48,
                         timem=192,
                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000)



# higher-res pre-trained on Audioset
model.net = get_model_passt("stfthop100", input_tdim=3200)

# hopsize=100 for this pretrained model
model.mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=100, n_fft=1024, freqm=48,
                         timem=192,
                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000)



```
