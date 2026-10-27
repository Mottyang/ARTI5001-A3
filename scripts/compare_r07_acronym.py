"""R07 before/after pronunciation override, with real recording as reference."""
from tts_training_common import *
from tts_acronym_preprocessing import prepare_acronym_text
import re, shutil, html, csv, gc

def main():
    source=ROOT/'comparisons/listening_diagnostics_20'
    data=json.loads((source/'results.json').read_text(encoding='utf-8'))
    old=[r for r in data['reference_records'] if r['id']=='R07']
    text=old[0]['text'];fixed=prepare_acronym_text(text, expand_initialisms=True)
    assert fixed=='to a level of sustained profits within one year from the inauguration of en, ar, ay.'
    assert prepare_acronym_text('Dr. Smith has 12 books.')=='Dr. Smith has twelve books.'
    assert prepare_acronym_text('banana and N.R.A. policy', expand_initialisms=True)=='banana and en, ar, ay. policy'
    assert prepare_acronym_text(fixed, expand_initialisms=True)==fixed
    out=ROOT/'comparisons/R07_acronym_fix'
    if (out/'results.json').exists():raise RuntimeError('Result exists')
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in old:
        name=row['variant'];target=out/'before'/f'{name}.wav';target.parent.mkdir(exist_ok=True)
        shutil.copy2(source/row['wav'],target);assert sha(target)==sha(source/row['wav'])
        records.append({**row,'condition':'reference' if name in ('real','reconstructed') else 'before',
            'wav':target.relative_to(out).as_posix()})
    wrapper,model=load_tts(trainable=False)
    folder=ROOT/'models/hifigan'
    vocoder=HIFIGAN.from_hparams(source=str(folder),savedir=str(folder),run_opts={'device':'cuda'},local_strategy=LocalStrategy.COPY)
    vocoder.eval();vocoder.requires_grad_(False);best_steps={}
    for variant in ('pretrained','expA','expB'):
        if variant!='pretrained':
            checkpoint=ROOT/'checkpoints'/variant/'best.pt'
            assert sha(checkpoint)==data['hashes'][variant]
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True);best_steps[variant]=state['step'];del state
        model.eval();wrapper.eval();seed_all(42)
        with torch.inference_mode():
            mel,lengths,_=wrapper.encode_text(fixed)
            waveform=vocoder.decode_batch(mel)
        audio=waveform.squeeze().cpu().numpy()
        assert audio.ndim==1 and np.isfinite(audio).all() and np.abs(audio).max()>0
        path=out/'after'/f'{variant}.wav';path.parent.mkdir(exist_ok=True)
        sf.write(str(path),audio,22050,subtype='PCM_16')
        readback,sr=sf.read(str(path));assert sr==22050 and len(readback)==len(audio)
        records.append({'id':'R07','variant':variant,'condition':'after','text':text,'normalized_text':fixed,
            'wav':path.relative_to(out).as_posix(),'duration_seconds':len(audio)/sr,
            'decoder_limit_reached':int(lengths[0])>=model.decoder.max_decoder_steps,
            'clipped_fraction':float(np.mean(np.abs(audio)>=1)),'readable_nonzero':True})
    del model,wrapper,vocoder;gc.collect();torch.cuda.empty_cache()
    os.environ['TORCH_HOME']=str(ROOT/'models/torch')
    bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    recognizer=bundle.get_model().cuda().eval();labels=bundle.get_labels()
    for row in records:
        if row['condition']!='after':continue
        audio,sr=sf.read(str(out/row['wav']),dtype='float32')
        signal=torchaudio.functional.resample(torch.from_numpy(audio),sr,bundle.sample_rate).unsqueeze(0).cuda()
        with torch.inference_mode():emissions,_=recognizer(signal)
        tokens=torch.unique_consecutive(emissions[0].argmax(-1)).tolist()
        row['asr_transcript']=''.join(labels[i] for i in tokens if i!=0).replace('|',' ').strip()
        print(row['variant'],row['asr_transcript'],flush=True)
    result={'original_text':text,'fixed_model_input':fixed,'pronunciation_target':'English letter names N / R / A (en / ar / ay)',
        'seed':42,'best_steps':best_steps,'records':records,'source_hashes':data['hashes'],
        'preprocessor_sha256':sha(ROOT/'scripts/tts_acronym_preprocessing.py'),
        'limits':'Local ASR is supplementary. Letter names can be transcribed as letters or ordinary words; listen to verify. No training or checkpoint changes.'}
    write_json(out/'results.json',result)
    labels_cn={'real':'真实录音','reconstructed':'真实频谱重建','pretrained':'原始模型','expA':'expA','expB':'expB'}
    refs=''.join(f'<section><h3>{labels_cn[r["variant"]]}</h3><audio controls preload="none" src="{r["wav"]}"></audio><p>{html.escape(r["asr_transcript"])}</p></section>' for r in records if r['condition']=='reference')
    rows=[]
    for v in ('pretrained','expA','expB'):
        before=next(r for r in records if r['variant']==v and r['condition']=='before')
        after=next(r for r in records if r['variant']==v and r['condition']=='after')
        files=html.escape(json.dumps(['before/real.wav',before['wav'],after['wav']]),quote=True)
        rows.append(f'<article><h2>{labels_cn[v]}</h2><div class="grid"><section><h3>修复前</h3><audio controls preload="none" src="{before["wav"]}"></audio><p>ASR：{html.escape(before["asr_transcript"])}</p></section><section><h3>逐字母修复后</h3><audio controls preload="none" src="{after["wav"]}"></audio><p>ASR：{html.escape(after["asr_transcript"])}</p></section></div><button data-playlist="{files}">连续试听：真实 → 修复前 → 修复后</button></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>R07 缩写逐字母修复对照</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;background:#f4f6fa;color:#192535}article{background:white;border:1px solid #dbe2ea;border-radius:14px;padding:22px;margin:20px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}audio{width:100%}button{padding:10px}p{overflow-wrap:anywhere}@media(max-width:650px){.grid{grid-template-columns:1fr}}</style><h1>R07：N.R.A. 逐字母发音修复</h1>'''+f'<p>原句：{html.escape(text)}</p><p>模型实际输入：{html.escape(fixed)}</p><p>使用明确字母名称 en / ar / ay；只改变推理前端，同一 seed=42、同一声码器与原权重。没有增加训练。ASR 可能将字母名转写为普通词，请以试听核实。</p><h2>参照</h2><div class="grid">'+refs+'</div>'+''.join(rows)+r'''<script>let token=0,active=null;const all=[...document.querySelectorAll('audio')];function stop(){token++;active=null;all.forEach(a=>a.pause())}all.forEach(a=>a.addEventListener('play',()=>{if(a!==active){token++;active=null}all.forEach(b=>{if(a!==b)b.pause()})}));document.querySelectorAll('[data-playlist]').forEach(button=>button.onclick=async()=>{stop();const mine=token;for(const file of JSON.parse(button.dataset.playlist)){if(mine!==token)return;const a=all.find(x=>x.getAttribute('src')===file);active=a;a.currentTime=0;try{await a.play()}catch(e){alert('请手动点击播放器');return}await new Promise(resolve=>{const done=()=>{a.removeEventListener('ended',done);a.removeEventListener('pause',done);resolve()};a.addEventListener('ended',done);a.addEventListener('pause',done)});if(mine!==token)return;await new Promise(r=>setTimeout(r,800))}});</script></html>'''
    (out/'listen.html').write_text(page,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
