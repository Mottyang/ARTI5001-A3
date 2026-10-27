"""Create deterministic pilot manifests from existing fixed splits; never touch test."""
from tts_training_common import ROOT, MEL, sha, write_json
import json, wave
import yaml

def main():
    out=ROOT/'data_manifests'/'pilot'
    metadata=ROOT/'LJSpeech-1.1'/'metadata.csv'
    rows={r[0]:r for r in (line.split('|') for line in metadata.read_text(encoding='utf-8').splitlines())}
    selected={}
    for split,count in [('train',32),('val',8)]:
        ids=(ROOT/'splits'/f'{split}_ids.txt').read_text(encoding='utf-8').splitlines()
        records=[]
        for uid in ids:
            wav=ROOT/'LJSpeech-1.1'/'wavs'/f'{uid}.wav'
            with wave.open(str(wav),'rb') as f:
                duration=f.getnframes()/f.getframerate()
                assert f.getframerate()==22050 and f.getnchannels()==1
            # Small, short-utterance subset is ONLY a functionality test.
            if 1.0<=duration<=5.0:
                records.append({'id':uid,'wav':wav.relative_to(ROOT).as_posix(),
                    'label':rows[uid][2],'duration':duration})
            if len(records)==count: break
        assert len(records)==count
        write_json(out/f'{split}.json',records)
        (out/f'{split}_ids.txt').write_text('\n'.join(r['id'] for r in records)+'\n',encoding='utf-8')
        selected[split]={r['id'] for r in records}
    assert selected['train'].isdisjoint(selected['val'])
    reference=ROOT/'reference'/'speechbrain-v1.0.3'/'train.yaml'
    official=yaml.load(reference.read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    pretrained=yaml.load((ROOT/'models'/'tacotron2'/'hyperparams.yaml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    architecture_keys=['n_mel_channels','n_symbols','symbols_embedding_dim','encoder_kernel_size',
        'encoder_n_convolutions','encoder_embedding_dim','attention_rnn_dim','attention_dim',
        'attention_location_n_filters','attention_location_kernel_size','n_frames_per_step',
        'decoder_rnn_dim','prenet_dim','max_decoder_steps','gate_threshold','p_attention_dropout',
        'p_decoder_dropout','postnet_embedding_dim','postnet_kernel_size','postnet_n_convolutions',
        'decoder_no_early_stopping','mask_padding']
    assert all(official[k].lower()==pretrained[k].lower() for k in architecture_keys)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [metadata,reference,
        ROOT/'models'/'tacotron2'/'hyperparams.yaml',ROOT/'models'/'tacotron2'/'model.ckpt',
        ROOT/'splits'/'train_ids.txt',ROOT/'splits'/'val_ids.txt',out/'train.json',out/'val.json']}
    write_json(out/'provenance.json',{'counts':{'train':32,'val':8},'selection':'first eligible IDs in fixed split order; duration 1-5 seconds',
        'text_column':'normalized text (third pipe-delimited column), then english_cleaners',
        'audio':'float32 PCM; no resampling, trimming or peak normalization',
        'mel_parameters':MEL,'architecture_keys_verified':architecture_keys,'all_architecture_values_match':True,
        'source_url':'https://github.com/speechbrain/speechbrain/blob/v1.0.3/recipes/LJSpeech/TTS/tacotron2/hparams/train.yaml',
        'limitation':'Version-matched official recipe and model architecture verified; historical pretraining settings cannot be proven solely from inference config. LJSpeech pretrained/test overlap exists.',
        'test_used':False,'sha256':hashes})
    print('Prepared 32 train / 8 validation; no test data used. Architecture matches official v1.0.3 recipe.',flush=True)

if __name__=='__main__': main()
