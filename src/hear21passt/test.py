import torch

from hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings
from hear21passt.base20sec import load_model, get_scene_embeddings, get_timestamp_embeddings
from hear21passt.base30sec import load_model, get_scene_embeddings, get_timestamp_embeddings

if __name__ == '__main__':
    model = load_model(mode="logits").cuda()
    seconds = 15
    audio = torch.ones((3, 32000 * seconds))*0.5
    embed, time_stamps = get_timestamp_embeddings(audio, model)
    print(embed.shape)
    # print(time_stamps)
    embed = get_scene_embeddings(audio, model)
    print(embed.shape)
    print(embed[0, 10])

    # test pretrained models
    from hear21passt.base import get_basic_model, get_model_passt
    import torch
    # get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
    model = get_basic_model(mode="logits")
    print(model.mel)  # Extracts mel spectrogram from raw waveforms.

    # optional replace the transformer with one that has the required number of classes i.e. 50
    model.net = get_model_passt(arch="openmic",  n_classes=20)
    model.net = get_model_passt(arch="fsd50k",  n_classes=200)
    model.net = get_model_passt(
        arch="fsd50k-n",  n_classes=200, fstride=16, tstride=16)
    model.net = get_model_passt("stfthop100", input_tdim=3200)
    model.net = get_model_passt("stfthop160", input_tdim=2000)
    model.net = get_model_passt("passt_20sec", input_tdim=2000)
    model.net = get_model_passt("passt_30sec", input_tdim=3000)
    
    model.net = get_model_passt("passt_l_kd_p16_128_ap47")
