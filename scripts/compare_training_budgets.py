"""Audit preserved 1k checkpoints and 2k continuation; compare exact endpoints.
Same existing 10 validation diagnostics + 10 custom prompts. No test-set use.
"""
from tts_training_common import *
from tts_acronym_preprocessing import prepare_acronym_text
import csv,math,time,statistics,shutil,html,re,gc

VARIANTS=('expA_1k','expA_2k','expB_1k','expB_2k')
LABELS={'real':'真实录音','reconstructed':'真实频谱重建','pretrained':'原始模型',
        'expA_1k':'A · 1k','expA_2k':'A · 2k','expB_1k':'B · 1k','expB_2k':'B · 2k'}

def audit():
    continuation=json.loads((ROOT/'data_manifests/formal/continuation_2k.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/x['path'])==x['sha256'] for x in continuation['preserved_1k_files'])
    protocol=json.loads((ROOT/'data_manifests/formal/protocol.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in protocol['sha256'].items())
    reports={}
    for name in ('expA','expB'):
        original=list(csv.DictReader((ROOT/'logs'/name/'metrics.csv').open(encoding='utf-8')))
        extra=list(csv.DictReader((ROOT/'logs'/name/'stage_2k/metrics.csv').open(encoding='utf-8')))
        assert [int(x['step']) for x in original]==list(range(1,1001))
        assert [int(x['step']) for x in extra]==list(range(1001,2001))
        for row in extra:
            assert all(math.isfinite(float(row[k])) for k in ('train_loss','mel_loss','gate_loss','gradient_norm'))
            assert float(row['gradient_norm'])>0
        vals=[{'step':int(x['step']),'loss':float(x['validation_loss'])} for x in original+extra if x['validation_loss']]
        assert [x['step'] for x in vals]==list(range(250,2001,250))
        lastpath=ROOT/'checkpoints'/name/'stage_2k/last.pt'
        bestpath=ROOT/'checkpoints'/name/'stage_2k/best.pt'
        last=torch.load(lastpath,map_location='cpu',weights_only=False)
        best=torch.load(bestpath,map_location='cpu',weights_only=False)
        config=json.loads((ROOT/'configs'/f'{name}.json').read_text(encoding='utf-8'))
        assert last['step']==2000 and last['config']==config and best['config']==config
        assert best['step']==min(vals,key=lambda x:x['loss'])['step']
        assert math.isclose(best['best_validation'],min(x['loss'] for x in vals),abs_tol=1e-10)
        for state in (last,best):
            assert state['optimizer']['state']
            assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point())
        wrapper,model=load_tts(trainable=False)
        model.load_state_dict(last['model'],strict=True)
        assert all(torch.equal(v.cpu(),last['model'][k]) for k,v in model.state_dict().items())
        model.load_state_dict(best['model'],strict=True)
        assert all(torch.equal(v.cpu(),best['model'][k]) for k,v in model.state_dict().items())
        reports[name]={'completed_additional_updates':1000,'end_step':2000,'all_metrics_finite':True,
            'validation':vals,'best_step_within_2k':best['step'],'best_loss_within_2k':best['best_validation'],
            'loss_at_exact_2k':vals[-1]['loss'],'last_sha256':sha(lastpath),'best_sha256':sha(bestpath),
            'last_and_best_strict_reload_tensor_equality':True,
            'continuation_report':json.loads((ROOT/'logs'/name/'stage_2k/resume_training_report.json').read_text(encoding='utf-8'))}
        write_json(ROOT/'logs'/name/'stage_2k/completion_audit.json',reports[name])
        del wrapper,model,last,best;gc.collect();torch.cuda.empty_cache()
    return reports

def synth(wrapper,vocoder,text):
    seed_all(42);torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():
        mel,lengths,_=wrapper.encode_text(text);wav=vocoder.decode_batch(mel)
    torch.cuda.synchronize()
    return wav,lengths,time.perf_counter()-start

def norm(text):return re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())

def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def main():
    out=ROOT/'comparisons/training_1k_vs_2k'
    if (out/'results.json').exists():raise RuntimeError('Comparison already exists')
    out.mkdir(parents=True,exist_ok=True)
    audits=audit()
    olddir=ROOT/'comparisons/listening_diagnostics_20'
    old=json.loads((olddir/'results.json').read_text(encoding='utf-8'))
    reference=[{'section':'reference','id':x['display_id'],'dataset_id':x['id'],'text':x['label'],
        'length_group':x['length_group']} for x in old['selection']]
    custom=[{'section':'custom','id':x['id'],'text':x['text'],'reference_spoken':x['reference_spoken'],
        'category':x['category']} for x in old['custom_records'] if x['variant']=='pretrained']
    cases=reference+custom;records=[]
    write_json(out/'fixed_cases.json',{'cases':cases,'source':'existing listening_diagnostics_20; no reselection',
        'preprocessing':'Numeric/money enabled, dotted initials disabled, as in previous budget evaluation.'})
    # Keep original reference/reconstruction/baseline as context, unchanged.
    for section,key in [('reference','reference_records'),('custom','custom_records')]:
        for row in old[key]:
            if row['variant'] not in ('real','reconstructed','pretrained'):continue
            dest=out/section/row['id']/f'{row["variant"]}.wav';dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(olddir/row['wav'],dest);assert sha(dest)==sha(olddir/row['wav'])
            records.append({**row,'section':section,'variant':row['variant'],'wav':dest.relative_to(out).as_posix()})
    wrapper,model=load_tts(trainable=False)
    folder=ROOT/'models/hifigan'
    vocoder=HIFIGAN.from_hparams(source=str(folder),savedir=str(folder),run_opts={'device':'cuda'},local_strategy=LocalStrategy.COPY)
    vocoder.eval();vocoder.requires_grad_(False)
    for variant in VARIANTS:
        name=variant.split('_')[0];is2k=variant.endswith('2k')
        path=ROOT/'checkpoints'/name/('stage_2k/last.pt' if is2k else 'last.pt')
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['step']==(2000 if is2k else 1000)
        model.load_state_dict(state['model'],strict=True)
        assert all(torch.equal(v.cpu(),state['model'][k]) for k,v in model.state_dict().items())
        del state;model.eval();wrapper.eval()
        synth(wrapper,vocoder,'This is a short warm up sentence.')
        for case in cases:
            processed=prepare_acronym_text(case['text'],expand_initialisms=False)
            timings=[]
            for repeat in range(3):
                wave,lengths,elapsed=synth(wrapper,vocoder,processed);timings.append(elapsed)
                if repeat==0:audio=wave.squeeze().cpu().numpy();frames=int(lengths[0])
            assert audio.ndim==1 and audio.size and np.isfinite(audio).all() and np.abs(audio).max()>0
            target=out/case['section']/case['id']/f'{variant}.wav';target.parent.mkdir(parents=True,exist_ok=True)
            sf.write(str(target),audio,22050,subtype='PCM_16')
            back,sr=sf.read(str(target));assert sr==22050 and len(back)==len(audio)
            row={**case,'variant':variant,'normalized_text':processed,'wav':target.relative_to(out).as_posix(),
                'checkpoint':str(path.relative_to(ROOT)),'checkpoint_step':2000 if is2k else 1000,
                'duration_seconds':len(audio)/sr,'decoder_limit_reached':frames>=model.decoder.max_decoder_steps,
                'clipped_fraction':float(np.mean(np.abs(audio)>=1)),'readable_nonzero':True,
                'latency_runs_seconds':timings,'latency_mean_seconds':statistics.mean(timings),
                'latency_stdev_seconds':statistics.stdev(timings),'rtf_mean':statistics.mean(timings)/(len(audio)/sr)}
            if not is2k:
                key='reference_records' if case['section']=='reference' else 'custom_records'
                prior=next(x for x in old[key] if x['id']==case['id'] and x['variant']==name)
                row['matches_original_1k_audio']=sha(target)==sha(olddir/prior['wav'])
                row['asr_transcript']=prior['asr_transcript'];row['word_errors']=prior['word_errors']
                row['reference_words']=prior['reference_words'];row['wer']=prior['wer']
            records.append(row)
            print(variant,case['section'],case['id'],f'{row["duration_seconds"]:.2f}s',flush=True)
    del model,wrapper,vocoder;gc.collect();torch.cuda.empty_cache()
    write_json(out/'synthesis_results.json',{'records':records,'audits':audits})
    os.environ['TORCH_HOME']=str(ROOT/'models/torch')
    bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H;asr=bundle.get_model().cuda().eval();labels=bundle.get_labels()
    from speechbrain.utils.text_to_sequence import english_cleaners
    for row in records:
        if row['variant'] not in ('expA_2k','expB_2k'):continue
        audio,sr=sf.read(str(out/row['wav']),dtype='float32')
        signal=torchaudio.functional.resample(torch.from_numpy(audio),sr,bundle.sample_rate).unsqueeze(0).cuda()
        with torch.inference_mode():emissions,_=asr(signal)
        ids=torch.unique_consecutive(emissions[0].argmax(-1)).tolist()
        transcript=''.join(labels[i] for i in ids if i!=0).replace('|',' ').strip()
        reference_text=row.get('reference_spoken') if row['section']=='custom' else english_cleaners(prepare_acronym_text(row['text'],expand_initialisms=False))
        reference_words=norm(reference_text);errors=distance(reference_words,norm(transcript))
        row.update(asr_transcript=transcript,word_errors=errors,reference_words=len(reference_words),wer=errors/len(reference_words))
    summaries={}
    for section in ('reference','custom'):
        summaries[section]={}
        for variant in ('pretrained',*VARIANTS):
            subset=[r for r in records if r['section']==section and r['variant']==variant]
            s={'micro_wer':sum(x['word_errors'] for x in subset)/sum(x['reference_words'] for x in subset),
               'word_errors':sum(x['word_errors'] for x in subset),'reference_words':sum(x['reference_words'] for x in subset),
               'decoder_limit_count':sum(x['decoder_limit_reached'] for x in subset)}
            if variant!='pretrained':
                s['mean_latency_seconds']=statistics.mean(x['latency_mean_seconds'] for x in subset)
                s['aggregate_rtf']=sum(x['latency_mean_seconds'] for x in subset)/sum(x['duration_seconds'] for x in subset)
            summaries[section][variant]=s
    result={'audits':audits,'cases':cases,'records':records,'summary':summaries,
        'comparison':'Exact endpoints 1000 vs 2000, not cherry-picked best checkpoints. Best-so-far within 2k reported separately.',
        'evaluation_frontend':'Numeric/money enabled; dotted initialisms disabled for every model. Same existing 20 cases.',
        'timing':'One excluded warmup/model, 3 repeats/case/model, CUDA synchronization, no file writing or model load in measured interval.',
        'limits':['No new human scores; ASR is not naturalness.','Validation diagnostics already used for development; not independent final test.','LJSpeech pretraining overlap and one training seed remain.']}
    write_json(out/'results.json',result)
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['section','id','variant','text','wav','duration_seconds','asr_transcript','wer','decoder_limit_reached','latency_mean_seconds','rtf_mean']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(records)
    build_page(out,result)
    print(json.dumps({'audits':audits,'summary':summaries},indent=2),flush=True)

def build_page(out,result):
    cards=[]
    for section in ('reference','custom'):
        cards.append('<h2 id="'+section+'">'+('真实参照10条：1k 与 2k' if section=='reference' else '自造10句：1k 与 2k')+'</h2>')
        for case in [x for x in result['cases'] if x['section']==section]:
            rows=[next(r for r in result['records'] if r['section']==section and r['id']==case['id'] and r['variant']==v) for v in ('pretrained',*VARIANTS)]
            references=''
            if section=='reference':
                refs=[next(r for r in result['records'] if r['section']==section and r['id']==case['id'] and r['variant']==v) for v in ('real','reconstructed')]
                references='<details><summary>听真实录音与频谱重建参照</summary><div class="refs">'+''.join(f'<section><h4>{LABELS[r["variant"]]}</h4><audio controls preload="none" src="{r["wav"]}"></audio></section>' for r in refs)+'</div></details>'
            players=''.join(f'<section><h4>{LABELS[r["variant"]]}</h4><audio controls preload="none" src="{r["wav"]}"></audio><small>{r["duration_seconds"]:.2f}s'+(' · ⚠ 解码超限' if r['decoder_limit_reached'] else '')+'</small></section>' for r in rows)
            transcripts=''.join(f'<p>{LABELS[r["variant"]]}：{html.escape(r["asr_transcript"])} · WER {r["wer"]:.1%}</p>' for r in rows)
            cards.append(f'<article><div>{section} · {case["id"]}</div><h3>{html.escape(case["text"])}</h3>{references}<div class="players">{players}</div><label>判断 <select data-key="{section}-{case["id"]}"><option value="">尚未记录</option><option>A 2k 比 A 1k 好</option><option>A 2k 比 A 1k 差</option><option>B 2k 比 B 1k 好</option><option>B 2k 比 B 1k 差</option><option>各有优劣</option><option>差别不明显</option></select></label><input data-note="{section}-{case["id"]}" placeholder="分别记录A/B变化：发音、漏词、重复、停顿、自然度"><details><summary>辅助ASR</summary>{transcripts}</details></article>')
    tables=[]
    for section in ('reference','custom'):
        rows=''.join(f'<tr><td>{LABELS[v]}</td><td>{s["micro_wer"]:.2%}</td><td>{s["word_errors"]}/{s["reference_words"]}</td><td>'+ (f'{s["aggregate_rtf"]:.4f}' if 'aggregate_rtf' in s else '—')+'</td></tr>' for v,s in result['summary'][section].items())
        tables.append(f'<h3>{section}</h3><table><tr><th>版本</th><th>WER</th><th>词错误</th><th>RTF</th></tr>{rows}</table>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>训练预算：1k 对比 2k</title><style>body{font:16px/1.6 system-ui;max-width:1450px;margin:32px auto;padding:0 24px;background:#f3f5f8;color:#192535}article{background:white;border:1px solid #dbe2ea;border-radius:14px;padding:22px;margin:20px 0}.players{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}.refs{display:grid;grid-template-columns:1fr 1fr;gap:20px}audio{width:100%}input{display:block;width:95%;padding:10px;margin:12px 0}select,button{padding:8px}h3{font-size:19px}th,td{padding:6px 18px;text-align:left}@media(max-width:1100px){.players{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.players,.refs{grid-template-columns:1fr}}</style><h1>A/B：同一句的 1k 与 2k 训练结果</h1><p>保留原1k，从各自Adam和随机状态续训到2k；同一数据、学习率与训练逻辑。这里比较准确的第1000和第2000步，预算内最佳检查点另在结果记录中说明。</p><p>相同10条验证录音＋10条自造句；数字/金额修复启用，缩写展开关闭。先听再看ASR；没有预填人工分数。刷新前请导出。</p><p><a href="#reference">真实参照组</a> · <a href="#custom">自造句组</a> <button id="export">导出试听记录</button></p><details><summary>辅助指标</summary>'''+''.join(tables)+'</details>'+''.join(cards)+r'''<script>const all=[...document.querySelectorAll('audio')];all.forEach(a=>a.addEventListener('play',()=>all.forEach(b=>{if(a!==b)b.pause()})));document.getElementById('export').onclick=()=>{const rows=[['sample','judgment','notes']];document.querySelectorAll('select[data-key]').forEach(s=>rows.push([s.dataset.key,s.value,document.querySelector('input[data-note="'+s.dataset.key+'"]').value]));const csv='\ufeff'+rows.map(r=>r.map(v=>'"'+v.replaceAll('"','""')+'"').join(',')).join(String.fromCharCode(13,10));const u=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='training_budget_listening.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};</script></html>'''
    (out/'listen.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
