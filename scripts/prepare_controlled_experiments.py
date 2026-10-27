"""Freeze shared formal data/configs. Does not start expA or expB training."""
from tts_training_common import ROOT, MEL, sha, write_json
import json, wave, statistics

def main():
    folder=ROOT/'data_manifests'/'formal'
    if (folder/'protocol.json').exists():
        raise RuntimeError('Formal protocol exists; refusing to redraw or overwrite.')
    metadata=ROOT/'LJSpeech-1.1'/'metadata.csv'
    rows={r[0]:r for r in (line.split('|') for line in metadata.read_text(encoding='utf-8').splitlines())}
    counts={'train':2000,'val':1310,'test':1310}
    groups={}; summaries={}
    for split,count in counts.items():
        ids=(ROOT/'splits'/f'{split}_ids.txt').read_text(encoding='utf-8').splitlines()
        ids=ids[:count] if split=='train' else ids
        assert len(ids)==count and len(set(ids))==count
        records=[]
        for uid in ids:
            path=ROOT/'LJSpeech-1.1'/'wavs'/f'{uid}.wav'
            with wave.open(str(path),'rb') as audio:
                assert audio.getframerate()==22050 and audio.getnchannels()==1
                seconds=audio.getnframes()/audio.getframerate()
            records.append({'id':uid,'wav':path.relative_to(ROOT).as_posix(),
                            'label':rows[uid][2],'duration':seconds})
        groups[split]=set(ids)
        write_json(folder/f'{split}.json',records)
        (folder/f'{split}_ids.txt').write_text('\n'.join(ids)+'\n',encoding='utf-8')
        times=[r['duration'] for r in records]
        summaries[split]={'count':count,'hours':sum(times)/3600,
                         'mean_seconds':statistics.mean(times),'min_seconds':min(times),'max_seconds':max(times)}
    assert all(groups[a].isdisjoint(groups[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    base={'seed':42,'train_manifest':'data_manifests/formal/train.json',
          'val_manifest':'data_manifests/formal/val.json','batch_size':4,
          'weight_decay':0.000006,'gradient_clip':1.0,'max_steps':1000,
          'validation_interval':250,'guided_attention_weight':0.0,'precision':'float32',
          'prompt':'Hello. This is our first speech synthesis test for Assignment Three.'}
    for name,lr in [('expA',1e-5),('expB',5e-5)]:
        cfg={'name':name,'learning_rate':lr,**base}
        write_json(ROOT/'configs'/f'{name}.json',cfg)
    a=json.loads((ROOT/'configs'/'expA.json').read_text(encoding='utf-8'))
    b=json.loads((ROOT/'configs'/'expB.json').read_text(encoding='utf-8'))
    differences=[k for k in a if a[k]!=b[k]]
    assert set(differences)=={'name','learning_rate'}
    sources=[metadata,ROOT/'models'/'tacotron2'/'model.ckpt',
        ROOT/'models'/'hifigan'/'generator.ckpt',ROOT/'models'/'tacotron2'/'hyperparams.yaml',
        ROOT/'scripts'/'train_tacotron2.py',ROOT/'scripts'/'tts_training_common.py']
    sources += [ROOT/'splits'/f'{s}_ids.txt' for s in counts]
    sources += [folder/f'{s}.json' for s in counts]
    sources += [ROOT/'configs'/f'{e}.json' for e in ['expA','expB']]
    protocol={'selection':'First 2000 IDs in existing seed-42 shuffled train list; no duration filtering. All original validation/test IDs retained.',
        'summaries':summaries,'shared_config':base,'learning_rates':{'expA':1e-5,'expB':5e-5},
        'only_config_differences':differences,'epochs':2,'training_updates':1000,
        'validation_steps':[0,250,500,750,1000],'checkpoint_selection':'lowest validation loss among saved checkpoints at steps 250/500/750/1000',
        'initialization':'Each run independently loads original models/tacotron2/model.ckpt; never pilot or benchmark weights.',
        'hifigan':'Frozen and shared; not used in training loss.',
        'mel':MEL,'test_policy':'Prepared only; not used by training, validation or timing benchmark.',
        'limitations':['One training seed; no claim of multi-seed significance.','LJSpeech pretrained checkpoint may have seen test recordings.','Checkpoint selection uses stochastic-prenet teacher-forced validation loss; quality needs separate fixed-prompt evaluation.'],
        'sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources}}
    write_json(folder/'protocol.json',protocol)
    print(json.dumps(protocol,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
