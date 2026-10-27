from pathlib import Path
import os, time, json, platform, importlib.metadata
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / 'models' / 'huggingface')
import torch, torchaudio, soundfile as sf
# SpeechBrain 1.x may query the legacy backend API removed in torchaudio 2.9+.
# Audio writing below uses soundfile, not a torchaudio backend.
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.utils.fetching import LocalStrategy

def main():
    assert torch.cuda.is_available(), 'CUDA GPU unavailable'
    torch.manual_seed(42)
    text = 'Hello. This is our first speech synthesis test for Assignment Three.'
    options = dict(run_opts={'device': 'cuda'}, local_strategy=LocalStrategy.COPY)
    print('Loading Tacotron2 on GPU...', flush=True)
    tts = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech', savedir=str(ROOT/'models'/'tacotron2'), **options)
    print('Loading HiFi-GAN on GPU...', flush=True)
    vocoder = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech', savedir=str(ROOT/'models'/'hifigan'), **options)
    tts_device = str(next(tts.mods.parameters()).device)
    vocoder_device = str(next(vocoder.mods.parameters()).device)
    assert tts_device.startswith('cuda') and vocoder_device.startswith('cuda')
    torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        mel, lengths, alignment = tts.encode_text(text)
        waveform = vocoder.decode_batch(mel)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    audio=waveform.squeeze().detach().cpu().numpy()
    import numpy as np
    assert audio.ndim==1 and audio.size>0 and np.isfinite(audio).all()
    assert np.max(np.abs(audio))>0
    output=ROOT/'wav_examples'/'first_gpu_test.wav'
    sf.write(str(output),audio,22050,subtype='PCM_16')
    decoded,sr=sf.read(str(output))
    assert sr==22050 and len(decoded)==len(audio)
    report=dict(text=text,device=torch.cuda.get_device_name(0),tts_device=tts_device,vocoder_device=vocoder_device,python=platform.python_version(),torch=torch.__version__,torchaudio=torchaudio.__version__,speechbrain=importlib.metadata.version('speechbrain'),cuda=torch.version.cuda,sample_rate=sr,duration_seconds=len(audio)/sr,inference_seconds=elapsed,rtf=elapsed/(len(audio)/sr),peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20,output=str(output),note='First-run functional check, not a warmed-up benchmark. Pretrained models; no fine-tuning.')
    (ROOT/'logs'/'first_gpu_test.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
