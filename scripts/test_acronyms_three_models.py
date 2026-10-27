"""Same generic A-Z rule tested on distinct dotted initialisms; no per-word overrides."""
from tts_training_common import *
from tts_acronym_preprocessing import prepare_acronym_text, LETTER_NAMES
from tts_text_preprocessing import expand_numbers_and_money
import re,html,gc,csv

CASES=[('01','U.S.A.','The report mentions U.S.A.'),('02','F.B.I.','The report mentions F.B.I.'),
       ('03','C.P.U.','The report mentions C.P.U.'),('04','U.N.','The report mentions U.N.'),
       ('05','B.B.C.','The report mentions B.B.C.'),('06','E.U.','The report mentions E.U.'),
       ('07','W.H.O.','The report mentions W.H.O.'),
       ('08','N.R.A.','to a level of sustained profits within one year from the inauguration of N.R.A.')]
VARIANTS=('pretrained','expA','expB')

def recognized_suffix(transcript,acronym):
    """Transparent recognition proxy, accepting letter glyphs/names, suffix only."""
    target=acronym.replace('.','')
    tokens=re.findall('[A-Z]+',transcript.upper())
    if tokens and tokens[-1]==target:return True
    aliases={v.upper():k for k,v in LETTER_NAMES.items() if ' ' not in v}
    aliases.update({k:k for k in LETTER_NAMES})
    # ASR often writes the spoken letter name R as ARE, A as A, U as YOU.
    aliases.update({'ARE':'R','A':'A','B':'B','C':'C','E':'E','G':'G','P':'P','T':'T','U':'U','V':'V','Z':'Z'})
    normalized=[];i=0
    while i<len(tokens):
        if tokens[i:i+2] in (['DOUBLE','YOU'],['DOUBLE','U']):normalized.append('W');i+=2
        else:normalized.append(aliases.get(tokens[i],tokens[i]));i+=1
    return len(normalized)>=len(target) and normalized[-len(target):]==list(target)

def main():
    out=ROOT/'comparisons/dotted_initialisms_generic'
    if (out/'results.json').exists():raise RuntimeError('Results already exist')
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'data_manifests/formal/protocol.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in protocol['sha256'].items())
    reference=json.loads((ROOT/'comparisons/listening_diagnostics_20/results.json').read_text(encoding='utf-8'))
    write_json(out/'test_cases.json',{'cases':CASES,'letter_names':LETTER_NAMES,
        'rule':'At least two uppercase letters separated by dots; optional final dot. One A-Z mapping, comma-separated letter names. Lowercase/undotted forms not guessed.'})
    wrapper,model=load_tts(trainable=False)
    folder=ROOT/'models/hifigan'
    vocoder=HIFIGAN.from_hparams(source=str(folder),savedir=str(folder),run_opts={'device':'cuda'},local_strategy=LocalStrategy.COPY)
    vocoder.eval();vocoder.requires_grad_(False);records=[]
    for variant in VARIANTS:
        if variant!='pretrained':
            checkpoint=ROOT/'checkpoints'/variant/'best.pt';assert sha(checkpoint)==reference['hashes'][variant]
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True);del state
        model.eval();wrapper.eval()
        for uid,acronym,text in CASES:
            for condition in ('before','after'):
                processed=expand_numbers_and_money(text) if condition=='before' else prepare_acronym_text(text, expand_initialisms=True)
                seed_all(42)
                with torch.inference_mode():
                    mel,lengths,_=wrapper.encode_text(processed);waveform=vocoder.decode_batch(mel)
                audio=waveform.squeeze().cpu().numpy()
                assert audio.ndim==1 and np.isfinite(audio).all() and np.abs(audio).max()>0
                path=out/variant/condition/f'{uid}.wav';path.parent.mkdir(parents=True,exist_ok=True)
                sf.write(str(path),audio,22050,subtype='PCM_16')
                reread,sr=sf.read(str(path));assert sr==22050 and len(reread)==len(audio)
                records.append({'id':uid,'acronym':acronym,'text':text,'normalized_text':processed,
                    'variant':variant,'condition':condition,'wav':path.relative_to(out).as_posix(),
                    'seconds':len(audio)/sr,'decoder_limit_reached':int(lengths[0])>=model.decoder.max_decoder_steps,
                    'clipped_fraction':float(np.mean(np.abs(audio)>=1))})
            print(variant,acronym,'generated before/after',flush=True)
    del model,wrapper,vocoder;gc.collect();torch.cuda.empty_cache()
    write_json(out/'synthesis.json',records)
    os.environ['TORCH_HOME']=str(ROOT/'models/torch')
    bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H;asr=bundle.get_model().cuda().eval();labels=bundle.get_labels()
    for row in records:
        audio,sr=sf.read(str(out/row['wav']),dtype='float32')
        signal=torchaudio.functional.resample(torch.from_numpy(audio),sr,bundle.sample_rate).unsqueeze(0).cuda()
        with torch.inference_mode():emissions,_=asr(signal)
        ids=torch.unique_consecutive(emissions[0].argmax(-1)).tolist()
        row['asr_transcript']=''.join(labels[i] for i in ids if i!=0).replace('|',' ').strip()
        row['asr_letter_suffix_matches']=recognized_suffix(row['asr_transcript'],row['acronym'])
        print(row['variant'],row['condition'],row['acronym'],row['asr_letter_suffix_matches'],row['asr_transcript'],flush=True)
    summary={v:{c:sum(r['asr_letter_suffix_matches'] for r in records if r['variant']==v and r['condition']==c) for c in ('before','after')} for v in VARIANTS}
    result={'cases':CASES,'records':records,'summary_matches_out_of_8':summary,'letter_names':LETTER_NAMES,
        'seed':42,'rule_has_per_acronym_overrides':False,'expansion_explicitly_enabled_for_after':True,'preprocessor_sha256':sha(ROOT/'scripts/tts_acronym_preprocessing.py'),
        'checkpoint_hashes':{v:sha(ROOT/'checkpoints'/v/'best.pt') for v in ('expA','expB')},
        'recognizer':'Local torchaudio WAV2VEC2_ASR_BASE_960H, greedy CTC',
        'recognition_policy':'Suffix must be exact collapsed acronym, or last N tokens match A-Z letter symbols/names using the saved table; ARE accepted as spoken R. No approximate spelling correction.',
        'limits':'ASR suffix match is a screening proxy, not proof of natural pronunciation. Only uppercase dotted initialisms tested; full English acronym disambiguation is not solved. No training.'}
    write_json(out/'results.json',result)
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    cards=[]
    for uid,acronym,text in CASES:
        blocks=[]
        for v in VARIANTS:
            a=next(r for r in records if r['id']==uid and r['variant']==v and r['condition']=='before')
            b=next(r for r in records if r['id']==uid and r['variant']==v and r['condition']=='after')
            blocks.append(f'<section><h3>{v}</h3><p>处理前</p><audio controls preload="none" src="{a["wav"]}"></audio><p>处理后</p><audio controls preload="none" src="{b["wav"]}"></audio><details><summary>辅助识别</summary><p>前：{html.escape(a["asr_transcript"])}</p><p>后：{html.escape(b["asr_transcript"])}</p><p>字母序列识别匹配：{b["asr_letter_suffix_matches"]}</p></details></section>')
        fixed=prepare_acronym_text(text, expand_initialisms=True)
        cards.append(f'<article><h2>{uid} · {acronym}</h2><p>{html.escape(text)}</p><p>通用规则输出：{html.escape(fixed)}</p><div class="grid">'+''.join(blocks)+'</div></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>通用带点字母缩写测试</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:35px auto;padding:0 24px;background:#f4f6fa;color:#192535}article{background:white;border:1px solid #dbe2ea;border-radius:14px;padding:22px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px}audio{width:100%}p{overflow-wrap:anywhere}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style><h1>通用带点字母缩写：8 个测试 × 3 个模型</h1><p>同一份 A-Z 字母名称表，自动识别至少两个大写字母组成的带点缩写；不为测试词写例外。每个样本提供处理前后试听。seed=42，同一声码器及权重，没有增加训练。</p><p>不自动拆分 NASA/CPU 等无点形式，也不把 e.g./i.e./Dr. 或网址当成字母缩写。识别匹配只是辅助，仍需听字母发音和停顿。</p>'''+''.join(cards)+'''<script>const all=[...document.querySelectorAll('audio')];all.forEach(a=>a.addEventListener('play',()=>all.forEach(b=>{if(a!==b)b.pause()})));</script></html>'''
    (out/'listen.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
