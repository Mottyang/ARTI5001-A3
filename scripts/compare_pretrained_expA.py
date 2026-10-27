"""Paired diagnostic synthesis; pretrained vs expA best, same prompts/seed/vocoder.
ASR is an imperfect intelligibility proxy, not a human naturalness rating.
"""
from tts_training_common import *
import time, re, csv, html, statistics
from speechbrain.utils.text_to_sequence import english_cleaners

PROMPTS=[
('01','普通短句','The quick brown fox jumps over the lazy dog.','The quick brown fox jumps over the lazy dog.'),
('02','日常陈述','Please close the window before you leave the room.','Please close the window before you leave the room.'),
('03','项目介绍','Hello. This is our first speech synthesis test for Assignment Three.','Hello. This is our first speech synthesis test for Assignment Three.'),
('04','标点停顿','After the rain stopped, we opened the windows and listened to the birds.','After the rain stopped, we opened the windows and listened to the birds.'),
('05','长句','Although the road was narrow and the weather was cold, the travelers continued their journey until they reached the village before sunset.','Although the road was narrow and the weather was cold, the travelers continued their journey until they reached the village before sunset.'),
('06','数字','There are 12 books on the shelf and 3 pencils on the desk.','There are twelve books on the shelf and three pencils on the desk.'),
('07','金额','The ticket costs $12.50, including the booking fee.','The ticket costs twelve dollars and fifty cents, including the booking fee.'),
('08','缩写','Dr. Smith and Mrs. Brown arrived before the meeting began.','Doctor Smith and Mrs Brown arrived before the meeting began.'),
('09','问句','Could you explain why this experiment produced a different result?','Could you explain why this experiment produced a different result?'),
('10','较难词语','Accurate pronunciation requires careful attention to rhythm, emphasis, and unfamiliar vocabulary.','Accurate pronunciation requires careful attention to rhythm, emphasis, and unfamiliar vocabulary.')]

def normalize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())

def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def main():
    out=ROOT/'comparisons'/'pretrained_vs_expA'
    if (out/'results.json').exists():raise RuntimeError('Comparison exists; refusing to overwrite')
    out.mkdir(parents=True,exist_ok=True)
    best_path=ROOT/'checkpoints'/'expA'/'best.pt'
    source_hashes={'pretrained':sha(ROOT/'models/tacotron2/model.ckpt'),'expA_best':sha(best_path),
                   'vocoder':sha(ROOT/'models/hifigan/generator.ckpt')}
    saved=torch.load(best_path,map_location='cpu',weights_only=False)
    assert saved['config']['name']=='expA'
    wrapper,model=load_tts(trainable=False)
    folder=ROOT/'models/hifigan'
    vocoder=HIFIGAN.from_hparams(source=str(folder),savedir=str(folder),run_opts={'device':'cuda'},local_strategy=LocalStrategy.COPY)
    vocoder.eval();vocoder.requires_grad_(False)
    records=[]
    for variant in ('pretrained','expA'):
        if variant=='expA':
            model.load_state_dict(saved['model'],strict=True)
            assert all(torch.equal(v.cpu(),saved['model'][k]) for k,v in model.state_dict().items())
        model.eval();wrapper.eval()
        (out/variant).mkdir(exist_ok=True)
        for uid,category,text,reference in PROMPTS:
            # Reset equal RNG for each pair; keep all inference settings unchanged.
            seed_all(42)
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                mel,lengths,alignment=wrapper.encode_text(text)
                waveform=vocoder.decode_batch(mel)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            audio=waveform.squeeze().cpu().numpy()
            assert audio.ndim==1 and audio.size and np.isfinite(audio).all() and np.abs(audio).max()>0
            path=out/variant/f'{uid}.wav'
            sf.write(str(path),audio,22050,subtype='PCM_16')
            reread,sr=sf.read(str(path));assert sr==22050 and len(reread)==len(audio)
            # Token round-trip exposes any characters silently dropped by the front end.
            actual_text=wrapper.hparams.text_to_sequence(text,wrapper.text_cleaners)
            row={'id':uid,'category':category,'variant':variant,'text':text,'reference_spoken':reference,
                 'cleaner_output':english_cleaners(text),'text_token_count':len(actual_text),
                 'wav':path.relative_to(out).as_posix(),'duration_seconds':len(audio)/sr,
                 'inference_seconds':elapsed,'rtf_single_run':elapsed/(len(audio)/sr),
                 'decoder_frames':int(lengths[0]),'decoder_limit_reached':int(lengths[0])>=model.decoder.max_decoder_steps,
                 'raw_peak':float(np.abs(audio).max()),'clipped_fraction':float(np.mean(np.abs(audio)>=1)),
                 'rms':float(np.sqrt(np.mean(audio**2))),'finite_nonzero_readable':True}
            records.append(row)
            print(f'{variant} {uid}: {row["duration_seconds"]:.2f}s, decoder limit={row["decoder_limit_reached"]}',flush=True)
    del saved,wrapper,model,vocoder
    import gc;gc.collect();torch.cuda.empty_cache()
    # Persist synthesis before optional ASR download, so generated audio is preserved.
    write_json(out/'synthesis_results.json',{'source_hashes':source_hashes,'records':records})

    # Local recognition: audio is not uploaded. Official torchaudio pretrained bundle.
    os.environ['TORCH_HOME']=str(ROOT/'models'/'torch')
    bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    print('Loading local wav2vec2 ASR for supplementary intelligibility comparison...',flush=True)
    asr=bundle.get_model().cuda().eval();labels=bundle.get_labels()
    for row in records:
        a,sr=sf.read(str(out/row['wav']),dtype='float32')
        signal=torchaudio.functional.resample(torch.from_numpy(a),sr,bundle.sample_rate).unsqueeze(0).cuda()
        with torch.inference_mode():emissions,_=asr(signal)
        tokens=torch.unique_consecutive(emissions[0].argmax(-1)).tolist()
        transcription=''.join(labels[i] for i in tokens if i!=0).replace('|',' ').strip()
        reference=normalize(row['reference_spoken']);hypothesis=normalize(transcription)
        errors=distance(reference,hypothesis)
        row.update(asr_transcript=transcription,word_errors=errors,reference_words=len(reference),wer=errors/len(reference))
        print(f'ASR {row["variant"]} {row["id"]}: WER={row["wer"]:.3f}; {transcription}',flush=True)
    del asr;gc.collect();torch.cuda.empty_cache()

    summary={}
    for variant in ('pretrained','expA'):
        subset=[r for r in records if r['variant']==variant]
        summary[variant]={'micro_wer':sum(r['word_errors'] for r in subset)/sum(r['reference_words'] for r in subset),
            'decoder_limit_count':sum(r['decoder_limit_reached'] for r in subset),
            'all_wavs_readable':all(r['finite_nonzero_readable'] for r in subset),
            'total_audio_seconds':sum(r['duration_seconds'] for r in subset),
            'clipped_samples_fraction_max':max(r['clipped_fraction'] for r in subset)}
    paired=[]
    pairdir=out/'pairs';pairdir.mkdir(exist_ok=True)
    for uid,category,text,reference in PROMPTS:
        a=next(r for r in records if r['id']==uid and r['variant']=='pretrained')
        b=next(r for r in records if r['id']==uid and r['variant']=='expA')
        x,sr=sf.read(str(out/a['wav']),dtype='float32');y,_=sf.read(str(out/b['wav']),dtype='float32')
        # Preserve original amplitude. A 0.8s silence separates pretrained then expA.
        sf.write(str(pairdir/f'{uid}_pretrained_then_expA.wav'),np.concatenate([x,np.zeros(int(.8*sr),dtype=np.float32),y]),sr,subtype='PCM_16')
        paired.append({'id':uid,'category':category,'text':text,'pretrained_wer':a['wer'],'expA_wer':b['wer'],
                       'pretrained_transcript':a['asr_transcript'],'expA_transcript':b['asr_transcript'],
                       'pretrained_duration':a['duration_seconds'],'expA_duration':b['duration_seconds']})
    result={'source_hashes':source_hashes,'best_step':1000,'seed_per_utterance':42,'summary':summary,'pairs':paired,'records':records,
        'recognizer':'torchaudio WAV2VEC2_ASR_BASE_960H, local GPU, greedy CTC',
        'wer_normalization':'lowercase word tokens; punctuation ignored; expected spoken number/money references explicitly stored; lexical Mrs variation retained',
        'limitations':['ASR errors are not verified speech errors; no automatic naturalness score.','10 diagnostic prompts, single seed; not statistically conclusive.','Raw numeric prompts deliberately retain current frontend, which does not expand digits in this installed version; shared frontend issue, not necessarily model quality.','Single-run synthesis times are not a warmed-up repeated benchmark.','No human listening scores have been entered.']}
    write_json(out/'results.json',result)
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    with (out/'listening_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','category','text','preference_pretrained_expA_tie','pretrained_clarity_1to5','expA_clarity_1to5','pretrained_naturalness_1to5','expA_naturalness_1to5','notes'])
        for uid,cat,text,_ in PROMPTS:writer.writerow([uid,cat,text,'','','','','',''])
    cards=[]
    for row in paired:
        uid=row['id']
        cards.append(f'''<article><div class="tag">{uid} · {html.escape(row['category'])}</div><h2>{html.escape(row['text'])}</h2>
<div class="players"><section><h3>原始预训练模型</h3><audio controls preload="none" src="pretrained/{uid}.wav"></audio></section><section><h3>expA 最佳模型</h3><audio controls preload="none" src="expA/{uid}.wav"></audio></section></div>
<p><a href="pairs/{uid}_pretrained_then_expA.wav">连续试听：原始 → 0.8 秒间隔 → expA</a></p>
<label>你的判断 <select data-id="{uid}"><option value="">尚未评分</option><option>原始更好</option><option>expA 更好</option><option>接近</option><option>各有优劣</option></select></label>
<input data-note="{uid}" placeholder="记录漏词、重复、发音或停顿问题">
<details><summary>辅助 ASR 转写（可能有错，不代表人工评分）</summary><p>原始：{html.escape(row['pretrained_transcript'])} · WER {row['pretrained_wer']:.1%}</p><p>expA：{html.escape(row['expA_transcript'])} · WER {row['expA_wer']:.1%}</p></details></article>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原始模型与 expA · 成对试听</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;background:#f4f6fa;color:#192535}h1{font-size:30px}article{background:white;border:1px solid #dbe2ea;border-radius:14px;padding:24px;margin:20px 0}.tag{color:#456a87;font-weight:600}h2{font-size:20px}.players{display:grid;grid-template-columns:1fr 1fr;gap:20px}audio{width:100%}h3{font-size:15px}input{display:block;width:95%;padding:10px;margin:12px 0}select,button{padding:8px}summary{cursor:pointer}a{color:#215e91}@media(max-width:650px){.players{grid-template-columns:1fr}}</style>
<h1>原始预训练模型 vs expA</h1><p>相同 10 句 · 相同随机种子 42 · 同一 HiFi-GAN · expA 第 1000 步最佳检查点</p>
<p>先听清晰度、漏词、重复、停顿和自然度，再展开辅助转写。保持同一音量。未做音量归一化；更响不代表更好。评分可导出，刷新页面前请保存。</p><p>数字和金额句用于检查当前文字处理的弱点；本轮保持两组输入流程一致，没有修正数字展开。</p><button id="export">导出我的试听记录</button>'''+''.join(cards)+r'''
<script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));document.getElementById('export').onclick=()=>{let rows=[['id','preference','notes']];document.querySelectorAll('select[data-id]').forEach(s=>rows.push([s.dataset.id,s.value,document.querySelector('input[data-note="'+s.dataset.id+'"]').value]));let csv='\ufeff'+rows.map(r=>r.map(v=>'"'+v.replaceAll('"','""')+'"').join(',')).join('\r\n');let u=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));let a=document.createElement('a');a.href=u;a.download='my_listening_scores.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};</script></html>'''
    (out/'listen.html').write_text(page,encoding='utf-8')
    lines=['# 成对试听比较','', '本轮使用同一 10 条诊断句、每句相同 seed=42、同一冻结声码器。未修改训练代码、权重或正式实验配置。','',
           '| 模型 | 辅助 ASR 总 WER | 达到解码上限 |','|---|---:|---:|']
    for v,s in summary.items():lines.append(f'| {v} | {s["micro_wer"]:.2%} | {s["decoder_limit_count"]}/10 |')
    lines += ['', 'WER 是自动识别辅助指标，低一些只支持当前识别器下的可懂度线索，不证明自然度改善。请对照转写逐条试听，尤其是数字、金额、缩写和长句。',
              '当前安装版本的 english_cleaners 并不展开数字，字符表还可能丢弃数字；两组共享这一弱点。不要把这类失败直接归因为学习率。',
              'listen.html：并排播放器与可导出的偏好记录；pairs/：每句原始→间隔→expA；comparison.csv：逐句转写及指标；listening_scores.csv：空白人工评分表；results.json：完整设置、来源哈希和结果。',
              '本轮没有人工听感评分。后续若修正前端，应两组同步修正并重新生成，不覆盖本轮证据。']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
