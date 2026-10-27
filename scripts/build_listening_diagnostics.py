"""Fixed validation diagnostic references plus existing custom prompts.
Uses official SpeechBrain mel extraction and frozen HiFi-GAN; no training.
"""
from tts_training_common import *
from tts_text_preprocessing import expand_numbers_and_money
import shutil, csv, html, re, gc, statistics

VARIANTS=('pretrained','expA','expB')
LABELS={'real':'真实录音','reconstructed':'真实频谱重建','pretrained':'原始模型','expA':'expA','expB':'expB'}

def words(text):return re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())

def edits(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def save_audio(path,audio):
    assert audio.ndim==1 and len(audio)>0 and np.isfinite(audio).all() and np.abs(audio).max()>0
    path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(str(path),audio,22050,subtype='PCM_16')
    decoded,sr=sf.read(str(path));assert sr==22050 and len(decoded)==len(audio)
    return {'duration_seconds':len(audio)/22050,'raw_peak':float(np.abs(audio).max()),
            'clipped_fraction':float(np.mean(np.abs(audio)>=1)),'readable_nonzero':True}

def main():
    out=ROOT/'comparisons'/'listening_diagnostics_20'
    if (out/'results.json').exists():raise RuntimeError('Diagnostic result already exists')
    out.mkdir(parents=True,exist_ok=True)
    old=ROOT/'comparisons/pretrained_expA_expB_numbers_fixed'
    if (old/'results.json').exists():
        old_results=json.loads((old/'results.json').read_text(encoding='utf-8'))
    else:
        # After cleanup, the migrated custom audio and metadata are self-contained.
        old=out
        old_results=json.loads((out/'custom_results.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'data_manifests/formal/protocol.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in protocol['sha256'].items())
    validation=json.loads((ROOT/'data_manifests/formal/val.json').read_text(encoding='utf-8'))
    training={r['id'] for r in json.loads((ROOT/'data_manifests/formal/train.json').read_text(encoding='utf-8'))}
    bins=[('short',3,lambda d:d<4),('medium',4,lambda d:4<=d<7),('long',3,lambda d:d>=7)]
    rng=random.Random(42);selection=[];strata={}
    for label,count,predicate in bins:
        pool=sorted([r for r in validation if predicate(r['duration'])],key=lambda r:r['id'])
        strata[label]={'available':len(pool),'selected':count}
        chosen=sorted(rng.sample(pool,count),key=lambda r:r['duration'])
        for row in chosen:selection.append({**row,'length_group':label})
    assert len({r['id'] for r in selection})==10 and not ({r['id'] for r in selection}&training)
    for i,row in enumerate(selection,1):row['display_id']=f'R{i:02d}'
    hashes={**old_results['hashes'],'validation_manifest':sha(ROOT/'data_manifests/formal/val.json')}
    for variant in ('expA','expB'):assert sha(ROOT/'checkpoints'/variant/'best.pt')==hashes[variant]
    assert sha(ROOT/'scripts/tts_text_preprocessing.py')==hashes['preprocessor']
    write_json(out/'reference_selection.json',{'seed':42,'selection':'Random sample from ID-sorted duration strata: short<4s (3), medium 4-7s (4), long>=7s (3); chosen before synthesis, no output-based replacements.',
        'strata':strata,'records':selection,'hashes':hashes,'purpose':'Validation diagnosis, not unseen final test. Pretrained LJSpeech overlap possible.'})
    (out/'reference_ids.txt').write_text('\n'.join(r['id'] for r in selection)+'\n',encoding='utf-8')
    # Migrate custom-prompt audio, measurements and transcripts without regeneration.
    custom_records=[]
    for old_row in old_results['records']:
        row=dict(old_row);destination=out/'custom'/row['variant']/f'{row["id"]}.wav'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if (old/row['wav']).resolve()!=destination.resolve():
            shutil.copy2(old/row['wav'],destination)
        assert sha(old/row['wav'])==sha(destination)
        row['wav']=destination.relative_to(out).as_posix();row['copied_sha256']=sha(destination)
        custom_records.append(row)
    write_json(out/'custom_results.json',{**old_results,'records':custom_records})
    folder=ROOT/'models/hifigan'
    vocoder=HIFIGAN.from_hparams(source=str(folder),savedir=str(folder),run_opts={'device':'cuda'},local_strategy=LocalStrategy.COPY)
    vocoder.eval();vocoder.requires_grad_(False);records=[]
    for row in selection:
        uid=row['display_id'];source=ROOT/row['wav']
        audio,sr=sf.read(str(source),dtype='float32');assert sr==22050 and audio.ndim==1
        real=out/'reference'/uid/'real.wav';real.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,real)
        assert sha(source)==sha(real)
        records.append({'id':uid,'dataset_id':row['id'],'length_group':row['length_group'],'variant':'real',
            'text':row['label'],'wav':real.relative_to(out).as_posix(),'duration_seconds':len(audio)/sr,
            'readable_nonzero':True,'clipped_fraction':float(np.mean(np.abs(audio)>=1)),
            'source_sha256':sha(source),'decoder_limit_reached':False})
        mel=mel_spectogram(audio=torch.from_numpy(audio),**MEL).unsqueeze(0).cuda()
        with torch.inference_mode():waveform=vocoder.decode_batch(mel)
        reconstructed=out/'reference'/uid/'reconstructed.wav'
        checks=save_audio(reconstructed,waveform.squeeze().cpu().numpy())
        records.append({'id':uid,'dataset_id':row['id'],'length_group':row['length_group'],'variant':'reconstructed',
            'text':row['label'],'wav':reconstructed.relative_to(out).as_posix(),'decoder_limit_reached':False,**checks})
        print(f'{uid} {row["id"]} {row["length_group"]}: real {len(audio)/sr:.2f}s / reconstruction {checks["duration_seconds"]:.2f}s',flush=True)
    wrapper,model=load_tts(trainable=False);steps={}
    for variant in VARIANTS:
        if variant!='pretrained':
            state=torch.load(ROOT/'checkpoints'/variant/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True);steps[variant]=state['step'];del state
        model.eval();wrapper.eval()
        for row in selection:
            uid=row['display_id'];text=expand_numbers_and_money(row['label']);seed_all(42)
            with torch.inference_mode():
                mel,lengths,_=wrapper.encode_text(text)
                waveform=vocoder.decode_batch(mel)
            path=out/'reference'/uid/f'{variant}.wav'
            checks=save_audio(path,waveform.squeeze().cpu().numpy())
            records.append({'id':uid,'dataset_id':row['id'],'length_group':row['length_group'],'variant':variant,
                'text':row['label'],'normalized_text':text,'wav':path.relative_to(out).as_posix(),
                'decoder_frames':int(lengths[0]),'decoder_limit_reached':int(lengths[0])>=model.decoder.max_decoder_steps,**checks})
            print(f'{uid} {variant}: {checks["duration_seconds"]:.2f}s; limit={records[-1]["decoder_limit_reached"]}',flush=True)
    del model,wrapper,vocoder;gc.collect();torch.cuda.empty_cache()
    write_json(out/'reference_synthesis.json',{'selection':selection,'records':records,'hashes':hashes,'best_steps':steps,'mel':MEL})
    os.environ['TORCH_HOME']=str(ROOT/'models/torch')
    bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    recognizer=bundle.get_model().cuda().eval();labels=bundle.get_labels()
    from speechbrain.utils.text_to_sequence import english_cleaners
    for row in records:
        audio,sr=sf.read(str(out/row['wav']),dtype='float32')
        signal=torchaudio.functional.resample(torch.from_numpy(audio),sr,bundle.sample_rate).unsqueeze(0).cuda()
        with torch.inference_mode():emissions,_=recognizer(signal)
        token_ids=torch.unique_consecutive(emissions[0].argmax(-1)).tolist()
        transcript=''.join(labels[i] for i in token_ids if i!=0).replace('|',' ').strip()
        reference=words(english_cleaners(expand_numbers_and_money(row['text'])))
        errors=edits(reference,words(transcript))
        row.update(asr_transcript=transcript,word_errors=errors,reference_words=len(reference),wer=errors/len(reference))
    del recognizer;gc.collect();torch.cuda.empty_cache()
    summary={}
    for variant in ('real','reconstructed',*VARIANTS):
        part=[r for r in records if r['variant']==variant]
        summary[variant]={'micro_wer':sum(r['word_errors'] for r in part)/sum(r['reference_words'] for r in part),
            'word_errors':sum(r['word_errors'] for r in part),'reference_words':sum(r['reference_words'] for r in part),
            'decoder_limit_count':sum(r['decoder_limit_reached'] for r in part),'count':len(part)}
    result={'reference_records':records,'reference_summary':summary,'selection':selection,
        'custom_records':custom_records,'custom_summary':old_results['summary'],'hashes':hashes,'best_steps':steps,
        'limitations':['Validation diagnostic set, not a held-out final test; LJSpeech pretraining overlap possible.',
        'ASR measures recognition, not naturalness. Human ratings are blank.',
        'True-mel reconstruction is lossy; no claim of waveform identity.',
        'Reference and custom prompts must be summarized separately.',
        'No output-based sample replacement; any decoder limit flags are retained.']}
    write_json(out/'results.json',result)
    with (out/'reference_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['id','dataset_id','length_group','variant','text','wav','duration_seconds','asr_transcript','wer','decoder_limit_reached']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    with (out/'listening_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['section','id','variant','clarity_1to5','naturalness_1to5','noise_or_metallic','missing_or_repeated_words','notes'])
        for row in records:w.writerow(['reference',row['id'],row['variant'],'','','','',''])
        for row in custom_records:w.writerow(['custom',row['id'],row['variant'],'','','','',''])
    build_page(out,result)
    print(json.dumps({'reference_summary':summary,'audio_files':len(list(out.rglob('*.wav')))},indent=2),flush=True)

def build_page(out,result):
    cards=[]
    for section,group in [('reference',result['selection']),('custom',[r for r in result['custom_records'] if r['variant']=='pretrained'])]:
        cards.append('<h2 id="'+section+'">'+('第一部分：10 条真实录音五路诊断' if section=='reference' else '第二部分：原有 10 条自造句子三组比较')+'</h2>')
        for item in group:
            uid=item['display_id'] if section=='reference' else item['id']
            variants=('real','reconstructed',*VARIANTS) if section=='reference' else VARIANTS
            rows=[next(r for r in result['reference_records' if section=='reference' else 'custom_records'] if r['id']==uid and r['variant']==v) for v in variants]
            text=item['label'] if section=='reference' else item['text']
            extra=f"{item['id']} · {item['length_group']} · 真实时长 {item['duration']:.2f}s" if section=='reference' else item['category']
            players=''.join(f'<section><h4>{LABELS[r["variant"]]}</h4><audio controls preload="none" src="{r["wav"]}"></audio><small>{r["duration_seconds"]:.2f}s'+(' · ⚠ 解码到达上限' if r['decoder_limit_reached'] else '')+'</small></section>' for r in rows)
            playlist=html.escape(json.dumps([r['wav'] for r in rows]),quote=True)
            transcripts=''.join(f'<p><b>{LABELS[r["variant"]]}</b>：{html.escape(r["asr_transcript"])} · WER {r["wer"]:.1%}</p>' for r in rows)
            cards.append(f'<article><div class="tag">{uid} · {html.escape(extra)}</div><h3>{html.escape(text)}</h3><div class="players {section}">{players}</div><button data-playlist="{playlist}">按列顺序连续试听（间隔 0.8 秒）</button><p><label>诊断 / 偏好 <select data-key="{section}-{uid}"><option value="">尚未记录</option><option>原始录音本身不理想</option><option>频谱重建明显变差</option><option>文字合成明显变差</option><option>原始模型更好</option><option>expA 更好</option><option>expB 更好</option><option>差异不明显</option><option>各有优劣</option></select></label></p><input data-note="{section}-{uid}" placeholder="描述金属感、杂音、语调、漏词、重复或停顿"><details><summary>辅助识别转写（不代表人工评分）</summary>{transcripts}</details></article>')
    table=''.join(f'<tr><td>{LABELS[v]}</td><td>{result["reference_summary"][v]["micro_wer"]:.2%}</td><td>{result["reference_summary"][v]["decoder_limit_count"]}</td></tr>' for v in ('real','reconstructed',*VARIANTS))
    header='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>20 条试听：真实参照与模型比较</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1450px;margin:35px auto;padding:0 24px;background:#f3f5f8;color:#192535}article{background:white;border:1px solid #dbe2ea;border-radius:14px;padding:22px;margin:20px 0}.tag,small{color:#526b7d}h3{font-size:19px}h4{font-size:15px}.players{display:grid;gap:12px}.reference{grid-template-columns:repeat(5,minmax(0,1fr))}.custom{grid-template-columns:repeat(3,minmax(0,1fr))}audio{width:100%;min-width:0}input{display:block;width:95%;padding:10px;margin:12px 0}button,select{padding:8px}a{color:#215e91}td,th{padding:7px 20px;text-align:left}nav{position:sticky;top:0;background:#f3f5f8;padding:12px;border-bottom:1px solid #dbe2ea}nav a{margin-right:20px}@media(max-width:1050px){.reference{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.players{grid-template-columns:1fr}}</style><h1>先听真实参照，再定位合成瓶颈</h1><p>验证集固定 3 条短句、4 条中句、3 条长句；另保留原有 10 条自造句子。相同前端、HiFi-GAN、seed=42，使用 A/B 各自最佳检查点。没有重新训练。</p><nav><a href="#reference">真实录音五路对照</a><a href="#custom">自造句子三组对照</a><button id="export">导出我的试听记录</button><button id="stop">停止播放</button></nav><p>先比较“真实录音 → 真实频谱重建”，再比较“重建 → 文字合成”。重建差，优先检查频谱/声码器；重建好但合成差，关注声学预测、对齐和韵律。不能仅凭 ASR 判断自然度。保持相同设备和音量，未做音量归一化。记录不会自动保存，刷新前请导出。</p><details><summary>查看真实参照组辅助指标</summary><table><tr><th>版本</th><th>辅助 WER</th><th>解码超限</th></tr>'''+table+'</table><p>真实参照组与自造句组分开分析，不能混为一个平均分。</p></details>'
    script=r'''<script>let token=0;let activeAuto=null;const all=[...document.querySelectorAll('audio')];function stop(){token++;activeAuto=null;all.forEach(a=>a.pause())}document.getElementById('stop').onclick=stop;all.forEach(a=>a.addEventListener('play',()=>{if(a!==activeAuto){token++;activeAuto=null}all.forEach(b=>{if(a!==b)b.pause()})}));document.querySelectorAll('[data-playlist]').forEach(button=>button.onclick=async()=>{stop();const mine=token;const files=JSON.parse(button.dataset.playlist);for(const path of files){if(mine!==token)return;const audio=all.find(a=>a.getAttribute('src')===path);audio.currentTime=0;activeAuto=audio;try{await audio.play()}catch(e){alert('浏览器阻止了连续播放，请手动点击各播放器。');return}await new Promise(resolve=>{const done=()=>{audio.removeEventListener('ended',done);audio.removeEventListener('pause',done);resolve()};audio.addEventListener('ended',done);audio.addEventListener('pause',done)});if(mine!==token)return;await new Promise(r=>setTimeout(r,800))}});document.getElementById('export').onclick=()=>{const rows=[['sample','judgment','notes']];document.querySelectorAll('select[data-key]').forEach(s=>rows.push([s.dataset.key,s.value,document.querySelector('input[data-note="'+s.dataset.key+'"]').value]));const content='\ufeff'+rows.map(row=>row.map(v=>'"'+v.replaceAll('"','""')+'"').join(',')).join(String.fromCharCode(13,10));const url=URL.createObjectURL(new Blob([content],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='diagnostic_listening_notes.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};</script></html>'''
    (out/'listen.html').write_text(header+''.join(cards)+script,encoding='utf-8')
    (out/'README.md').write_text('''# 20 条统一试听

listen.html 是统一入口。reference_selection.json / reference_ids.txt 保存生成前固定的验证集名单。

reference/ 包含 10 条 x 5 路：原始录音、真实频谱经同一 HiFi-GAN 重建、文字经原始模型、expA、expB 合成。
custom/ 包含原有 10 条 x 3 组语音，逐个校验后复制，未重新生成。custom_results.json 保留原有 ASR 和重复计时记录。

连续试听通过页面依次播放现有 WAV，没有另存重复的拼接音频。listening_scores.csv 是空白人工评分表，页面也可导出诊断备注。

真实频谱使用与训练相同的 MEL 参数和官方 mel_spectogram 函数；soundfile float32 读取，无音量归一化、静音裁剪或采样率修改。重建不要求与原录音逐采样点相同。

这 10 条从验证集按时长分层随机选取，不属于本次微调训练子集；可能仍被预训练模型见过。用于开发诊断，不当作未见最终测试。没有根据输出好坏替换样本。ASR 只辅助检查可懂度，不能评定自然度；两类样本必须分开汇总。

historical_records/（如存在）保留旧试听的结果、转写、指标和说明。清理旧目录前会验证迁移文件，原数据、环境、模型、正式检查点和日志保留。
''',encoding='utf-8')

if __name__=='__main__':main()
