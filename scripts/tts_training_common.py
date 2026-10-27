"""Shared preprocessing, adapted from SpeechBrain v1.0.3 LJSpeech/TTS/tacotron2.
Official reference: https://github.com/speechbrain/speechbrain/tree/v1.0.3/recipes/LJSpeech/TTS/tacotron2
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / 'models' / 'huggingface')
import json, hashlib, random
import numpy as np
import torch, torchaudio, soundfile as sf
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.utils.text_to_sequence import text_to_sequence
from speechbrain.lobes.models.Tacotron2 import TextMelCollate, mel_spectogram, Loss

MEL = dict(sample_rate=22050, hop_length=256, win_length=1024, n_fft=1024,
           n_mels=80, f_min=0.0, f_max=8000.0, power=1, normalized=False,
           norm='slaney', mel_scale='slaney', compression=True)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_tts(trainable=False):
    # Use existing local files; no new model download or random initialization.
    folder = ROOT/'models'/'tacotron2'
    wrapper = Tacotron2.from_hparams(source=str(folder), savedir=str(folder),
        run_opts={'device':'cuda'}, local_strategy=LocalStrategy.COPY,
        freeze_params=not trainable)
    model = wrapper.mods.model
    model.requires_grad_(trainable)
    return wrapper, model

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, manifest):
        self.records = json.loads(Path(manifest).read_text(encoding='utf-8'))
        self.cache = {}
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        if index not in self.cache:
            r=self.records[index]
            wav=ROOT/r['wav']
            audio,sr=sf.read(str(wav),dtype='float32')
            assert sr==MEL['sample_rate'] and audio.ndim==1
            # soundfile float32 PCM conversion matches normalized float audio reading.
            mel=mel_spectogram(audio=torch.from_numpy(audio), **MEL)
            tokens=torch.IntTensor(text_to_sequence(r['label'], ['english_cleaners']))
            assert tokens.numel()>0 and tokens.max()<148 and torch.isfinite(mel).all()
            self.cache[index]={'mel_text_pair':(tokens,mel,len(tokens)),
                               'label':r['label'],'wav':str(wav)}
        return dict(self.cache[index])

def batch(dataset, indices):
    # Official collator sorts text lengths, pads mels and builds stop-gate targets.
    b=TextMelCollate(n_frames_per_step=1)([dataset[i] for i in indices])
    text,lengths,mel,gate,out_lengths=b[:5]
    text,lengths,mel,gate,out_lengths=[x.cuda() for x in (text,lengths,mel,gate,out_lengths)]
    return (text.long(),lengths.long(),mel.float(),int(lengths.max()),out_lengths.long()), (mel.float(),gate.float())

def criterion():
    # Pretrained model already has an alignment. Disable scratch-training auxiliary term.
    return Loss(guided_attention_sigma=.2, gate_loss_weight=1., guided_attention_weight=0.).cuda()

def loss_stats(model, loss_fn, packed):
    inputs,targets=packed
    prediction=model(inputs,alignments_dim=int(inputs[1].max()))
    return loss_fn(prediction,targets,inputs[1],inputs[4],1)

def synthesize(wrapper, output, text):
    folder=ROOT/'models'/'hifigan'
    vocoder=HIFIGAN.from_hparams(source=str(folder),savedir=str(folder),
        run_opts={'device':'cuda'},local_strategy=LocalStrategy.COPY)
    vocoder.eval(); vocoder.requires_grad_(False); wrapper.eval()
    with torch.inference_mode():
        mel,lengths,_=wrapper.encode_text(text)
        assert 0<int(lengths[0])<wrapper.mods.model.decoder.max_decoder_steps, 'Decoder failed to stop'
        waveform=vocoder.decode_batch(mel)
    audio=waveform.squeeze().cpu().numpy()
    assert audio.ndim==1 and np.isfinite(audio).all() and np.abs(audio).max()>0
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    sf.write(str(output),audio,22050,subtype='PCM_16')
    decoded,sr=sf.read(str(output)); assert sr==22050 and len(decoded)==len(audio)
    return {'wav':str(output),'seconds':len(audio)/sr,'sample_rate':sr,
            'peak':float(np.abs(audio).max()),'finite_nonzero_audio':True,
            'tts_device':str(next(wrapper.mods.parameters()).device),
            'vocoder_device':str(next(vocoder.mods.parameters()).device)}
