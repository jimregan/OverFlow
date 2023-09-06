import matplotlib.pylab as plt

# Download tokenizer if not present
import nltk
#nltk.download('punkt')

import IPython.display as ipd 
import os
import json
import sys
sys.path.append('src/model')
sys.path.insert(0, './hifigan')
import numpy as np
import torch

from src.hparams import create_hparams
from src.training_module import TrainingModule
from src.utilities.text import text_to_sequence, phonetise_text
from hifigan.env import AttrDict
from hifigan.models import Generator
from nltk import word_tokenize
from hifigandenoiser import Denoiser
import soundfile as sf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = sys.argv[1] #"checkpoint_100000.ckpt"
basename = sys.argv[2]
def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.title("Synthesised Mel-Spectrogram")
    fig.canvas.draw()
    plt.show()
    
def plot_hidden_states(hidden_states):
    plt.plot(hidden_states)
    plt.xlabel("Time steps")
    plt.ylabel("HMM states")
    plt.title("Hidden states vs Time")
    plt.show()


hparams = create_hparams()

model = TrainingModule.load_from_checkpoint(checkpoint_path)
_ = model.to(device).eval().half()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def load_checkpoint(filepath, device):
    print(filepath)
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

# load the hifi-gan model
hifigan_loc = 'hifigan/'
config_file = hifigan_loc + 'config_v1.json'
hifi_checkpoint_file = 'hifigan/generator_v1'

with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
torch.manual_seed(h.seed)
generator = Generator(h).to(device)
state_dict_g = load_checkpoint(hifi_checkpoint_file, device)
generator.load_state_dict(state_dict_g['generator'])
generator.eval().half()
generator.remove_weight_norm()

model.model.hmm.hparams.max_sampling_time = 1800
model.model.hmm.hparams.duration_quantile_threshold=0.55
model.model.hmm.hparams.deterministic_transition=True
model.model.hmm.hparams.predict_means=False
model.model.hmm.hparams.prenet_dropout_while_eval=False


denoiser = Denoiser(generator, mode='zeros')

def tts(text=None,phones=None,fname='out.wav',temp=0.667):
    if phones == None:
        phones = phonetise_text(hparams.cmu_phonetiser, text, word_tokenize)
    print(phones)
    sequence =  np.array(text_to_sequence(phones, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device).long()
    with torch.no_grad():
        mel_output, hidden_state_travelled, _, _ = model.sample(sequence.squeeze(0), sampling_temp=temp)
        mel_output = mel_output.transpose(1, 2)
        audio = generator(mel_output)
        audio = denoiser(audio[:, 0], strength=0.004)[:, 0]

    sr = 22500
    sf.write(fname, audio.data.squeeze().cpu().numpy(), sr, 'PCM_24')
    print('wrote:',fname)

   
def interp_tts(items,fname='out.wav',temp=0.667):

    data = []
    for phones,weight in items:
        
        sequence =  np.array(text_to_sequence(phones, ['english_cleaners']))[None, :]
        sequence = torch.from_numpy(sequence).to(device).long().squeeze(0)
        data.append((sequence,weight))
        
    with torch.no_grad():
        mel_output, hidden_state_travelled, _, _ = model.sample2(data, sampling_temp=temp)
        mel_output = mel_output.transpose(1, 2)
        audio = generator(mel_output)
        audio = denoiser(audio[:, 0], strength=0.004)[:, 0]

    sr = 22500
    sf.write(fname, audio.data.squeeze().cpu().numpy(), sr, 'PCM_24')
    print('wrote:',fname)


#tts('a bit, again')
    
txt = [
    ' {AH0} {B IH1 T}, {AH0 G EH1 N} .',
    ' {AH0} {B IY1 T}, {AH0 G EH1 N} .',
    ' {AH0} {B EH1 T}, {AH0 G EH1 N} .',
    ' {AH0} {B AE1 T}, {AH0 G EH1 N} .',
    ' {AH0} {B AO1 T}, {AH0 G EH1 N} .'
]
'''
txt = [
    ' {AH0} {B AE1 G} .',
    ' {AH0} {D AE1 G} .',
    ' {AH0} {G AE1 G} .'
]
'''
#txt = ['{R AE1 K}, {AH0 G EH1 N} .','{L AE1 K}, {AH0 G EH1 N} .']

#tts(phones=texts[0])

interpdata = []


nsteps = 8

for i in range(0,(len(txt)-1)*nsteps+1):

    k = int(i/nsteps)
    m = i%nsteps
    w = float(m/nsteps)
    idata = [(txt[k],(1-w))]
    try:
        idata.append((txt[k+1],w))
    except:
        pass
             
    fn = '{}{:02d}.wav'.format(basename,i)
    print(i,k,m,w,idata,fn)
    interp_tts(idata,fname = fn,temp=0.0)


'''
for i in range(0,11):
    w = i*0.1    
    idata= [(txt[2],w),(txt[1],1-w)]
    fn = 'interp{}.wav'.format(i)
    print(idata,fn)
    interp_tts(idata,fname = fn,temp=0.0)
'''
exit()

sequences = []

for i, text in enumerate(texts):
    print(f"\n{''.join(['*'] * 20)}\n{i + 1} - Input text: \n{''.join(['*'] * 20)}\n{text}")
    text = phonetise_text(hparams.cmu_phonetiser, text, word_tokenize)
    print(f"\n{''.join(['*'] * 20)}\n{i + 1} - Phonetised text: \n{''.join(['*'] * 20)}\n{text}")
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device).long()
    sequences.append(sequence)
    
    print(''.join(['='] * 100))


t = 0.667
from tqdm.auto import tqdm
with torch.no_grad():
    mel_outputs, hidden_state_travelled_all = [], []
    for sequence in tqdm(sequences, leave=False):
        mel_output, hidden_state_travelled, _, _ = model.sample(sequence.squeeze(0), sampling_temp=t)
        mel_outputs.append(mel_output)
        hidden_state_travelled_all.append(hidden_state_travelled)


for i, mel_output in enumerate(mel_outputs):
    print(i, texts[i])
    plot_spectrogram_to_numpy(np.array(mel_output.float().cpu()).T)

with torch.no_grad():
    audios = []
    for i, mel_output in enumerate(mel_outputs):
        mel_output = mel_output.transpose(1, 2)
        audio = generator(mel_output)
        audio = denoiser(audio[:, 0], strength=0.004)[:, 0]
        audios.append(audio)
        print(f"{''.join(['*'] * 10)} \t{i + 1}\t {''.join(['*'] * 10)}")
        print(f"Text: {texts[i]}")
        ipd.display(ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate))
        print(f"{''.join(['*'] * 35)}\n")

from pathlib import Path

basepath = Path('synthesised_wavs')
basepath.mkdir(parents=True, exist_ok=True)

for i, audio in enumerate(audios):
        filename = basepath / f'OverFlow_{i + 1}.wav'
        sr = 22500
        sf.write(filename, audio.data.squeeze().cpu().numpy(), 
                 22500, 'PCM_24')
        print(f'Successfully written: {filename}')
