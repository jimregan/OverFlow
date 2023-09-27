
import os
import json
import sys
sys.path.append('src/model')
sys.path.insert(0, './hifigan')
import torch

from src.hparams import create_hparams
from src.training_module import TrainingModule
from src.utilities.text import text_to_sequence, phonetise_text, feat_to_sequence
from hifigan.env import AttrDict
from hifigan.models import Generator
from nltk import word_tokenize
from hifigandenoiser import Denoiser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_checkpoint(filepath, device):
    print(filepath)
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def load_hifigan(filepath='hifigan/',config='config_v1.json',generator='generator_v1',device='cuda'):
    # load the hifi-gan model
    hifigan_loc = filepath
    config_file = hifigan_loc + config
    hifi_checkpoint_file = hifigan_loc + generator
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(hifi_checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.remove_weight_norm()
    denoiser = Denoiser(generator, mode='zeros')
    return generator,denoiser

checkpoint_path = sys.argv[1]
filelist = sys.argv[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hparams = create_hparams()

# load OverFlow
model = TrainingModule.load_from_checkpoint(checkpoint_path)
_ = model.to(device).eval()
print(f'The model has {count_parameters(model):,} trainable parameters')
model.model.hmm.hparams.max_sampling_time = 1800
model.model.hmm.hparams.duration_quantile_threshold=0.55
model.model.hmm.hparams.deterministic_transition=True
model.model.hmm.hparams.predict_means=False
model.model.hmm.hparams.prenet_dropout_while_eval=True

# load HiFiGan
generator,denoiser = load_hifigan()


data = open(filelist).readlines()

sequences = []
for item in data[0:10]:
    if '|' in item:
        featstring = item.split('|')[1].strip()
    else:
        featstring = item.strip()

    seq = torch.FloatTensor(feat_to_sequence(featstring)).to(device)
   
    weight = torch.ones(hparams.n_features).to(device)
    bias = torch.zeros(hparams.n_features).to(device)

    # x low vowels 
    # weight[22] = 0.3
    # x high vowels 
    # weight[22] = 0.7
    # bias[22] = 0.3

    # x back vowels
    # weight[23] = 0.3
    # x front vowels
    # weight[23] = 0.5
    # bias[23] = 0.5

    # extra nasal
    # weight[6] = 0.3
    # bias[6] = 0.7

    # no fricatives + approx
    # weight[4] = 0
    # weight[9] = 0.3
    # bias[9] = 0.7

    # cold: non nasals
    # weight[6] = 0

    # replace dentals with bilabials
    # seq[(seq[:,13]==1).nonzero(),11]=1
    # weight[13]=0

    # make dentals retroflex
    # seq[(seq[:,13]==1).nonzero(),16]=1

    # make bilabials fricative
    # seq[(seq[:,11]==1).nonzero(),3]=0
    # seq[(seq[:,11]==1).nonzero(),4]=1

    # make approximants palatal 
    # seq[(seq[:,9]==1).nonzero(),17]=1
    # seq[(seq[:,9]==1).nonzero(),13]=0

    # reduce bilabials
    # weight[11] = 0.0
    # weight[12] = 0.0

    # 'finska': make stops unvoiced    
    # seq[(seq[:,3]==1).nonzero(),1]=0
    # weight[24] =3
    # weight[25] =0
    # weight[26] =0

    seq = seq * weight + bias

    sequences.append(seq)

t = 0.667
from tqdm.auto import tqdm
with torch.no_grad():
    mel_outputs, hidden_state_travelled_all = [], []
    for sequence in tqdm(sequences, leave=False):
        mel_output, hidden_state_travelled, _, _ = model.sample(sequence.squeeze(0), sampling_temp=t)
        mel_outputs.append(mel_output)
        hidden_state_travelled_all.append(hidden_state_travelled)


with torch.no_grad():
    audios = []
    for i, mel_output in enumerate(mel_outputs):
        mel_output = mel_output.transpose(1, 2)
        audio = generator(mel_output)
        audio = denoiser(audio[:, 0], strength=0.004)[:, 0]
        audios.append(audio)

import soundfile as sf
from pathlib import Path

basepath = Path('synthesised_wavs')
basepath.mkdir(parents=True, exist_ok=True)

for i, audio in enumerate(audios):
        filename = basepath / f'OverFlow_{i + 1}.wav'
        sr = 22500
        sf.write(filename, audio.data.squeeze().cpu().numpy(), 
                 22500, 'PCM_24')
        print(f'Successfully written: {filename}')
