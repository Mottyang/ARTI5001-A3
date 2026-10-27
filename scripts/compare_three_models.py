"""Three-way diagnostic comparison with identical numeric preprocessing and prompts.
Local ASR is a supplementary intelligibility proxy; no invented listening scores.
"""
from tts_training_common import *
from tts_text_preprocessing import expand_numbers_and_money
from compare_pretrained_expA_numbers_fixed import PROMPTS, normalize, distance
import time, csv, html, statistics, gc

VARIANTS=('pretrained','expA','expB')

def infer(wrapper,vocoder,text):
    seed_all(42)
    torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():
        mel,lengths,_=wrapper.encode_text(text)
        waveform=vocoder.decode_batch(mel)
    torch.cuda.synchronize()
    return waveform,lengths,time.perf_counter()-start

def main():
    out=ROOT/'comparisons'/'pretrained_expA_expB_numbers_fixed'
    if (out/'results.json').exists():raise RuntimeError('Comparison already exists; refusing to overwrite')
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'data_manifests/formal/protocol.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in protocol['sha256'].items())
    previous=json.loads((ROOT/'comparisons/pretrained_vs_expA_numbers_fixed/results.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'scripts/tts_text_preprocessing.py')==previous['source_hashes']['preprocessing']
    assert sha(ROOT/'checkpoints/expA/best.pt')==previous['source_hashes']['expA_best']
    hashes={'pretrained':sha(ROOT/'models/tacotron2/model.ckpt'),
        'expA':sha(ROOT/'checkpoints/expA/best.pt'),'expB':sha(ROOT/'checkpoints/expB/best.pt'),
        'vocoder':sha(ROOT/'models/hifigan/generator.ckpt'),
        'preprocessor':sha(ROOT/'scripts/tts_text_preprocessing.py')}
    wrapper,model=load_tts(trainable=False)
    folder=ROOT/'models/hifigan'
    vocoder=HIFIGAN.from_hparams(source=str(folder),savedir=str(folder),
        run_opts={'device':'cuda'},local_strategy=LocalStrategy.COPY)
    vocoder.eval();vocoder.requires_grad_(False)
    best_steps={'pretrained':None};records=[]
    for variant in VARIANTS:
        if variant!='pretrained':
            state=torch.load(ROOT/'checkpoints'/variant/'best.pt',map_location='cpu',weights_only=False)
            assert state['config']['name']==variant
            model.load_state_dict(state['model'],strict=True)
            assert all(torch.equal(v.cpu(),state['model'][k]) for k,v in model.state_dict().items())
            best_steps[variant]=state['step'];del state
        model.eval();wrapper.eval();(out/variant).mkdir(exist_ok=True)
        # Excluded model-specific warm-up; model loading is outside timing.
        infer(wrapper,vocoder,'This is a short warm up sentence.')
        for uid,category,text,reference in PROMPTS:
            processed=expand_numbers_and_money(text)
            timings=[]
            for repeat in range(3):
                waveform,lengths,elapsed=infer(wrapper,vocoder,processed)
                timings.append(elapsed)
                if repeat==0:
                    audio=waveform.squeeze().cpu().numpy()
                    frames=int(lengths[0])
            assert audio.ndim==1 and audio.size and np.isfinite(audio).all() and np.abs(audio).max()>0
            path=out/variant/f'{uid}.wav';sf.write(str(path),audio,22050,subtype='PCM_16')
            readback,sr=sf.read(str(path));assert sr==22050 and len(readback)==len(audio)
            seconds=len(audio)/sr
            row={'id':uid,'category':category,'variant':variant,'text':text,'normalized_text':processed,
                'reference_spoken':reference,'wav':path.relative_to(out).as_posix(),
                'duration_seconds':seconds,'decoder_frames':frames,
                'decoder_limit_reached':frames>=model.decoder.max_decoder_steps,
                'raw_peak':float(np.abs(audio).max()),'clipped_fraction':float(np.mean(np.abs(audio)>=1)),
                'finite_nonzero_readable':True,'latency_runs_seconds':timings,
                'latency_mean_seconds':statistics.mean(timings),
                'latency_stdev_seconds':statistics.stdev(timings),
                'rtf_mean':statistics.mean(timings)/seconds}
            if variant in ('pretrained','expA'):
                old=next(r for r in previous['records'] if r['variant']==variant and r['id']==uid)
                row['byte_identical_to_previous_fixed_comparison']=sha(path)==sha(ROOT/'comparisons/pretrained_vs_expA_numbers_fixed'/old['wav'])
            records.append(row)
            print(f'{variant} {uid}: duration {seconds:.2f}s; RTF {row["rtf_mean"]:.3f}; limit {row["decoder_limit_reached"]}',flush=True)
    del wrapper,model,vocoder;gc.collect();torch.cuda.empty_cache()
    write_json(out/'synthesis_results.json',{'records':records,'hashes':hashes,'best_steps':best_steps})
    os.environ['TORCH_HOME']=str(ROOT/'models/torch')
    bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    recognizer=bundle.get_model().cuda().eval();labels=bundle.get_labels()
    for row in records:
        audio,sr=sf.read(str(out/row['wav']),dtype='float32')
        signal=torchaudio.functional.resample(torch.from_numpy(audio),sr,bundle.sample_rate).unsqueeze(0).cuda()
        with torch.inference_mode():emissions,_=recognizer(signal)
        tokens=torch.unique_consecutive(emissions[0].argmax(-1)).tolist()
        transcript=''.join(labels[i] for i in tokens if i!=0).replace('|',' ').strip()
        ref=normalize(row['reference_spoken']);hyp=normalize(transcript)
        errors=distance(ref,hyp)
        row.update(asr_transcript=transcript,word_errors=errors,reference_words=len(ref),wer=errors/len(ref))
        print(f'ASR {row["variant"]} {row["id"]}: WER={row["wer"]:.3f}; {transcript}',flush=True)
    del recognizer;gc.collect();torch.cuda.empty_cache()
    summaries={}
    for variant in VARIANTS:
        subset=[r for r in records if r['variant']==variant]
        summaries[variant]={'micro_wer':sum(r['word_errors'] for r in subset)/sum(r['reference_words'] for r in subset),
            'mean_latency_seconds':statistics.mean(r['latency_mean_seconds'] for r in subset),
            'aggregate_rtf':sum(r['latency_mean_seconds'] for r in subset)/sum(r['duration_seconds'] for r in subset),
            'decoder_limit_count':sum(r['decoder_limit_reached'] for r in subset),
            'all_wavs_readable':all(r['finite_nonzero_readable'] for r in subset),
            'max_clipped_fraction':max(r['clipped_fraction'] for r in subset)}
    triples=out/'triples';triples.mkdir(exist_ok=True)
    for uid,_,_,_ in PROMPTS:
        parts=[]
        for variant in VARIANTS:
            audio,sr=sf.read(str(out/variant/f'{uid}.wav'),dtype='float32')
            if parts:parts.append(np.zeros(int(.8*sr),dtype=np.float32))
            parts.append(audio)
        sf.write(str(triples/f'{uid}_pretrained_expA_expB.wav'),np.concatenate(parts),22050,subtype='PCM_16')
    audits={v:json.loads((ROOT/'logs'/v/'completion_audit.json').read_text(encoding='utf-8')) for v in ('expA','expB')}
    result={'hashes':hashes,'best_steps':best_steps,'seed_per_utterance':42,'summary':summaries,'records':records,
        'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'sample_rate':22050,
        'protocol':'Same 10 prompts, repaired numeric frontend, seed=42 per synthesis, same frozen HiFi-GAN. One excluded warmup/model; 3 timed runs/prompt/model. Inference only; no loading or WAV writing included.',
        'recognizer':'Local torchaudio WAV2VEC2_ASR_BASE_960H greedy CTC',
        'training_validation':{v:{'best_step':a['best_step'],'best_validation_loss':a['best_validation_loss'],'validation_results':a['validation_results']} for v,a in audits.items()},
        'limitations':['No human listening scores; ASR WER is an imperfect intelligibility proxy, not naturalness.','Known spelling/Mrs variants retained in simple WER for consistency with previous comparison.','10 prompts, single seed; small speed differences are not conclusive.','No full test-set evaluation. LJSpeech pretraining overlap remains.']}
    write_json(out/'results.json',result)
    flat=[{k:r[k] for k in ('id','category','variant','text','normalized_text','asr_transcript','wer','duration_seconds','latency_mean_seconds','latency_stdev_seconds','rtf_mean','decoder_limit_reached')} for r in records]
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    with (out/'listening_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['id','variant','clarity_1to5','naturalness_1to5','omissions_repetitions','notes'])
        for uid,_,_,_ in PROMPTS:
            for v in VARIANTS:w.writerow([uid,v,'','','',''])
    cards=[]
    for uid,category,text,_ in PROMPTS:
        items=[next(r for r in records if r['id']==uid and r['variant']==v) for v in VARIANTS]
        players=''.join(f'<section><h3>{html.escape(v)}</h3><audio controls preload="none" src="{v}/{uid}.wav"></audio><small>时长 {r["duration_seconds"]:.2f}s · RTF {r["rtf_mean"]:.3f}</small></section>' for v,r in zip(VARIANTS,items))
        transcripts=''.join(f'<p><b>{r["variant"]}</b>：{html.escape(r["asr_transcript"])} · WER {r["wer"]:.1%}</p>' for r in items)
        cards.append(f'<article><div class="tag">{uid} · {html.escape(category)}</div><h2>{html.escape(text)}</h2><p>实际输入：{html.escape(items[0]["normalized_text"])}</p><div class="players">{players}</div><p><a href="triples/{uid}_pretrained_expA_expB.wav">连续试听：原始 → expA → expB（间隔 0.8 秒）</a></p><label>偏好 <select data-id="{uid}"><option value="">尚未评分</option><option>pretrained</option><option>expA</option><option>expB</option><option>接近</option><option>各有优劣</option></select></label><input data-note="{uid}" placeholder="记录漏词、重复、发音、停顿等"><details><summary>查看辅助 ASR 转写（可能有误）</summary>{transcripts}</details></article>')
    summary_rows=''.join(f'<tr><td>{v}</td><td>{summaries[v]["micro_wer"]:.2%}</td><td>{summaries[v]["aggregate_rtf"]:.3f}</td></tr>' for v in VARIANTS)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原始模型、expA、expB 三组试听</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;background:#f4f6fa;color:#192535}article{background:white;border:1px solid #dbe2ea;border-radius:14px;padding:24px;margin:20px 0}.tag{color:#456a87}h2{font-size:20px}.players{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}audio{width:100%}input{display:block;width:95%;padding:10px;margin:12px 0}select,button{padding:8px}summary{cursor:pointer}td,th{padding:8px 18px;text-align:left}a{color:#215e91}@media(max-width:800px){.players{grid-template-columns:1fr}}</style><h1>原始模型 · expA · expB</h1><p>相同 10 句 · 修复后的数字/美元前端 · 每句相同 seed=42 · 同一冻结 HiFi-GAN · 各自最佳检查点</p><p>先试听再看 ASR；更响不代表更好。音频未做音量归一化。ASR 不能评定自然度。评分不会自动保存，刷新前请导出。</p><details><summary>查看辅助指标</summary><table><tr><th>模型</th><th>辅助 WER</th><th>RTF</th></tr>'''+summary_rows+'''</table><p>每句计时重复 3 次，排除模型加载；小差异不可过度解读。</p></details><button id="export">导出我的试听记录</button>'''+''.join(cards)+r'''<script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));document.getElementById('export').onclick=()=>{const rows=[['id','preference','notes']];document.querySelectorAll('select[data-id]').forEach(s=>rows.push([s.dataset.id,s.value,document.querySelector('input[data-note="'+s.dataset.id+'"]').value]));const csv='\ufeff'+rows.map(r=>r.map(v=>'"'+v.replaceAll('"','""')+'"').join(',')).join(String.fromCharCode(13,10));const u=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='three_model_listening_scores.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};</script></html>'''
    (out/'listen.html').write_text(page,encoding='utf-8')
    lines=['# 三组对照结果','',result['protocol'],'','| 模型 | 最佳步数 | 验证损失 | 辅助 WER | RTF |','|---|---:|---:|---:|---:|']
    for v in VARIANTS:
        validation=f'{audits[v]["best_validation_loss"]:.6f}' if v in audits else '—'
        lines.append(f'| {v} | {best_steps[v] or "—"} | {validation} | {summaries[v]["micro_wer"]:.2%} | {summaries[v]["aggregate_rtf"]:.4f} |')
    lines+=['','listen.html 提供三组播放器和导出偏好；triples/ 每句按原始→A→B连续播放。comparison.csv 和 results.json 记录转写、时长和计时。',
        '这里没有人工听感评分；验证损失、ASR 和速度不能替代对自然度的试听。数字、金额正确与否可参考 06、07 句的转写。旧对照文件保留。',
        '原始及 A 的新音频是否与前次修复版逐字节一致，已逐条记录。测试集尚未整体评估；模型预训练使用过 LJSpeech。']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':summaries,'best_steps':best_steps},indent=2),flush=True)

if __name__=='__main__':main()
