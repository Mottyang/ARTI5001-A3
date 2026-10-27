"""Assemble existing evidence into a portable, executable, training-disabled notebook."""
from pathlib import Path
import json, csv, shutil, hashlib, ast, os, re
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
SUB=ROOT/'submission'/'62_A3'
SUB.mkdir(parents=True,exist_ok=True)
WAV=SUB/'wav_examples';WAV.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads((ROOT/p).read_text(encoding='utf-8'))
original_results=read('comparisons/training_1k_vs_2k/results.json')
formal=read('data_manifests/formal/protocol.json')
assert all(sha(ROOT/p)==h for p,h in formal['sha256'].items())
checkpoint_inventory=[]
for exp in ('expA','expB'):
    for stage in ('','stage_2k'):
        for name in ('best.pt','last.pt'):
            path=ROOT/'checkpoints'/exp/stage/name
            checkpoint_inventory.append({'relative_path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size})

# Correct execution paths only. Preserve immutable historical logs/configs/hashes.
migration=[]
for name in ('prepare_continuation_2k.py','finalize_training_results.py'):
    p=ROOT/'scripts'/name;s=p.read_text(encoding='utf-8')
    old="ROOT=Path(r'C:\\Users\\Admin\\Desktop\\Assignment\\5001\\a3')"
    if old in s:
        backup=ROOT/'migration_records'/'scripts_before_path_fix'/name
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
        p.write_text(s.replace(old,'ROOT=Path(__file__).resolve().parents[1]'),encoding='utf-8')
        migration.append({'file':p.relative_to(ROOT).as_posix(),'change':'hard-coded C root -> script-relative root','backup':backup.relative_to(ROOT).as_posix()})
for name in ('activate','activate.bat'):
    p=ROOT/'.venv/Scripts'/name
    s=p.read_text(encoding='utf-8')
    fixed=s.replace('C:\\Users\\Admin\\Desktop\\Assignment\\5001\\a3\\.venv',str(ROOT/'.venv')).replace('C:/Users/Admin/Desktop/Assignment/5001/a3/.venv',(ROOT/'.venv').as_posix())
    if fixed!=s:
        backup=ROOT/'migration_records'/'activation_before_path_fix'/name
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
        p.write_text(fixed,encoding='utf-8')
        migration.append({'file':p.relative_to(ROOT).as_posix(),'change':'copied environment activation path updated to D drive'})
kernel=ROOT/'.venv/share/jupyter/kernels/a3-62'
kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[str(ROOT/'.venv/Scripts/python.exe'),'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Assignment 3 (Group 62)','language':'python'},indent=2),encoding='utf-8')

# Generated WAVs only. Do not put original LJSpeech recordings in the submission.
audio_manifest=[]
for row in original_results['records']:
    if row['variant']=='real':continue
    source=ROOT/'comparisons/training_1k_vs_2k'/row['wav']
    name=f'{row["section"]}_{row["id"]}_{row["variant"]}.wav'
    destination=WAV/name;shutil.copy2(source,destination);assert sha(source)==sha(destination)
    audio_manifest.append({**row,'submission_wav':'wav_examples/'+name,'sha256':sha(destination)})
assert len(audio_manifest)==110
with (WAV/'audio_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['section','id','variant','text','submission_wav','duration_seconds','sha256']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(audio_manifest)

metrics={}
for exp in ('expA','expB'):
    rows=[]
    for relative in (f'logs/{exp}/metrics.csv',f'logs/{exp}/stage_2k/metrics.csv'):
        for row in csv.DictReader((ROOT/relative).open(encoding='utf-8')):
            rows.append({k:(int(v) if k=='step' else float(v) if v else None) for k,v in row.items()})
    assert [r['step'] for r in rows]==list(range(1,2001));metrics[exp]=rows
human=read('evaluations/human_listening_observations.json')
for observation in human['observations']:
    if observation['id']=='human_1k_vs_2k_no_clear_gain_20261028':
        # Restore user quotation from the conversation where an earlier export was garbled.
        observation['user_statement']='人工试听后，发现并没有明显进步，有些地方反而有退化。'
source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'scripts/tts_training_common.py',ROOT/'scripts/train_tacotron2.py',ROOT/'scripts/train_tacotron2_stage2k.py',ROOT/'scripts/tts_text_preprocessing.py',ROOT/'scripts/tts_acronym_preprocessing.py']}
E={'group':62,'protocol':formal,'configs':{x:read(f'configs/{x}.json') for x in ('expA','expB')},
   'split_ids':{x:(ROOT/f'splits/{x}_ids.txt').read_text(encoding='utf-8').splitlines() for x in ('train','val','test')},
   'manifests':{x:read(f'data_manifests/formal/{x}.json') for x in ('train','val','test')},
   'metrics':metrics,'comparison':original_results,'audio_manifest':audio_manifest,
   'human':human,'checkpoints':checkpoint_inventory,'source_hashes':source_hashes,
   'diagnostic':read('comparisons/listening_diagnostics_20/results.json'),
   'numeric_before_after':read('comparisons/listening_diagnostics_20/historical_records/pretrained_vs_expA_numbers_fixed/before_after_audit.json'),
   'acronyms':read('comparisons/dotted_initialisms_generic/results.json'),
   'r07':read('comparisons/R07_acronym_fix/results.json')}
nb=nbf.v4.new_notebook()
cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text))
def code(text):cells.append(nbf.v4.new_code_cell(text))
md('''# ARTI5001 — Assignment 3: Single-speaker Text-to-Speech
**Group 62 | Tacotron2 + frozen HiFi-GAN | Existing runs: 1,000 and 2,000 updates**

This notebook contains the implementation, fixed data protocol, saved experimental evidence, evaluation methods and playable generated examples. Its default **Run All** path reviews existing evidence; it does **not** train, download models or generate new audio. Full training and inference functions are included behind explicit switches.

Team members / student IDs / contributions: **complete with the actual three members before submission**. Do not infer contribution allocations from this notebook.

The ZIP folder is designed to contain this notebook and `wav_examples/`. Original LJSpeech data, virtual environments and download caches are excluded. Existing fine-tuned checkpoints remain in the project and are optional external inputs for generating new speech; configure `A3_CHECKPOINT_ROOT` on another computer. Cached examples and embedded results can be reviewed without them.

Sections: setup; data and splits; preprocessing; model/loss implementation; controlled experiments and checkpointing; results/curves; objective and human evaluation; audio diagnosis; arbitrary-text inference; conclusions and provenance.''')
md('''## 1. Portable paths and safe execution
Find the notebook/submission folder from the current directory, then locate the optional full project by walking upward or setting `A3_PROJECT_ROOT`. No machine-specific drive is required. The copied Windows environment was tested at the new location; environment activation and helper-script roots are recorded separately in the migration report.''')
code('''from pathlib import Path
import os, sys, json, csv, random, hashlib, statistics, platform, importlib.metadata
import numpy as np
from IPython.display import display, Markdown, HTML, Audio

def find_submission(start):
    start = Path(start).resolve()
    candidates = [start, *start.parents]
    for parent in candidates:
        for candidate in [parent, parent / 'submission' / '62_A3']:
            if (candidate / '62_A3.ipynb').exists() and (candidate / 'wav_examples' / 'audio_manifest.csv').is_file():
                return candidate
    raise FileNotFoundError('Open this notebook from the folder containing wav_examples/.')

SUBMISSION_ROOT = find_submission(Path.cwd())
if os.environ.get('A3_PROJECT_ROOT'):
    PROJECT_ROOT = Path(os.environ['A3_PROJECT_ROOT']).expanduser().resolve()
else:
    PROJECT_ROOT = next((p for p in [SUBMISSION_ROOT, *SUBMISSION_ROOT.parents]
                         if (p / 'LJSpeech-1.1' / 'metadata.csv').exists()), SUBMISSION_ROOT)
DATA_ROOT = Path(os.environ.get('LJSPEECH_ROOT', str(PROJECT_ROOT / 'LJSpeech-1.1')))
CHECKPOINT_ROOT = Path(os.environ.get('A3_CHECKPOINT_ROOT', str(PROJECT_ROOT / 'checkpoints')))
OUTPUT_ROOT = SUBMISSION_ROOT / 'optional_new_outputs'
RUN_TRAINING = False
RUN_NEW_INFERENCE = False
RUN_NEW_ASR = False
print('Submission:', SUBMISSION_ROOT)
print('Optional full project:', PROJECT_ROOT)
print('Training:', RUN_TRAINING, '| New inference:', RUN_NEW_INFERENCE)
# These switches are False in the distributed review notebook.
''')
md('''## 2. Embedded evidence and environment
The following data are copied from saved logs/configurations; they are not newly simulated measurements. Historical absolute paths inside evidence are provenance, not execution paths. SHA-256 values identify the original sources and checkpoints.''')
code('EVIDENCE = json.loads('+repr(json.dumps(E,ensure_ascii=True,separators=(',',':')))+')\nprint("Group:", EVIDENCE["group"], "| Logged updates:", {k: len(v) for k,v in EVIDENCE["metrics"].items()})')
code('''versions = {}
for name in ['torch', 'torchaudio', 'speechbrain', 'numpy', 'soundfile', 'matplotlib']:
    try: versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed'
print('Review Python:', platform.python_version())
print(versions)
print('Recorded training: Python 3.10.11, torch/torchaudio 2.11.0+cu128, SpeechBrain 1.0.3; RTX 5070 Ti 16GB; RAM 32GB.')
print('Install matching torch/torchaudio from https://download.pytorch.org/whl/cu128; other recorded versions are in the project requirements files.')
''')
md('''## 3. Data handling and fixed splits
LJSpeech has 13,100 paired utterances. Prior full-file verification found no missing, extra, empty or truncated WAVs. Audio is mono 22,050 Hz / 16-bit PCM, approximately 23.92 hours. `metadata.csv` has three literal pipe-separated fields: ID, original transcript and normalized transcript. Treat quotes as text, not CSV quoting.

Protocol: lexicographically sort IDs, shuffle with Python `random.Random(42)`, slice 80/10/10. The saved lists are authoritative. Fine-tuning uses the **first 2,000 IDs of the saved shuffled training list**, without duration filtering; both experiments share all 1,310 validation items. The 1,310-item test split is reserved and was not evaluated in full. Validation diagnostics have been used for development and are not an independent final test.''')
code('''def reproduce_split_ids(all_ids, seed=42):
    ids = sorted(all_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids); a = int(.8*n); b = int(.1*n)
    return {'train': ids[:a], 'val': ids[a:a+b], 'test': ids[a+b:]}

def read_ljspeech_metadata(root):
    rows = [line.split('|') for line in (Path(root)/'metadata.csv').read_text(encoding='utf-8').splitlines()]
    assert all(len(row)==3 and all(v.strip() for v in row) for row in rows)
    assert len({row[0] for row in rows}) == len(rows)
    return {row[0]: {'original':row[1], 'normalized':row[2]} for row in rows}

splits = EVIDENCE['split_ids']
assert [len(splits[k]) for k in ['train','val','test']] == [10480,1310,1310]
assert all(set(splits[a]).isdisjoint(splits[b]) for a,b in [('train','val'),('train','test'),('val','test')])
assert reproduce_split_ids(set().union(*map(set,splits.values()))) == splits
assert [r['id'] for r in EVIDENCE['manifests']['train']] == splits['train'][:2000]
assert [r['id'] for r in EVIDENCE['manifests']['val']] == splits['val']
print('Saved split reproduction and non-overlap: PASS')
for split, rows in EVIDENCE['manifests'].items():
    print(split, len(rows), 'utterances;', round(sum(r['duration'] for r in rows)/3600,3),'hours')
if DATA_ROOT.exists():
    metadata = read_ljspeech_metadata(DATA_ROOT)
    missing = [uid for uid in metadata if not (DATA_ROOT/'wavs'/f'{uid}.wav').exists()]
    assert not missing
    print('Local metadata:',len(metadata),'| all audio paths exist')
else:
    print('Dataset not bundled. Download from https://keithito.com/LJ-Speech-Dataset/ and set LJSPEECH_ROOT to perform local checks/training.')
''')
md('''## 4. Official preprocessing, model and loss
The version-matched SpeechBrain v1.0.3 LJSpeech/Tacotron2 recipe was inspected; 22 architecture fields matched the downloaded inference configuration. This establishes implementation/architecture compatibility, not proof of every historical pretraining setting.

Mel: 22,050 Hz, FFT/window 1024, hop 256, 80 bands, 0–8,000 Hz, magnitude power 1, no STFT magnitude normalization, Slaney normalization/scale, log compression with floor 1e-5. Use the normalized transcript (column 3), then official `english_cleaners`. No trimming or peak normalization was used in training.

The code below is the project implementation using official model, collation, mel and loss components. Its path root is adapted for this notebook; no training is invoked by defining it. The soundfile backend compatibility shim is local to this Python process. Source: [SpeechBrain v1.0.3 recipe](https://github.com/speechbrain/speechbrain/tree/v1.0.3/recipes/LJSpeech/TTS/tacotron2).''')
common=(ROOT/'scripts/tts_training_common.py').read_text(encoding='utf-8')
common=common.replace('ROOT = Path(__file__).resolve().parents[1]','ROOT = PROJECT_ROOT')
code(common)
md('''Training is teacher-forced: text IDs and the preceding target mel frame enter Tacotron2. The target mel is extracted from real audio. Total loss = decoder mel MSE + postnet mel MSE + stop-gate BCE. Guided-attention weight is zero for this fine-tuning study. HiFi-GAN is frozen and absent from the training loss. `TextMelCollate` sorts lengths, pads sequences and constructs stop targets; `batch` moves tensors to CUDA. The dataset cache is suitable for this subset but should be memory-budgeted when scaling up.''')
code('''# A lightweight real-sample sanity check: preprocessing only, no model update.
if DATA_ROOT.exists():
    sample = EVIDENCE['manifests']['train'][0]
    audio, sr = sf.read(str(DATA_ROOT/'wavs'/f"{sample['id']}.wav"), dtype='float32')
    target_mel = mel_spectogram(audio=torch.from_numpy(audio), **MEL)
    tokens = text_to_sequence(sample['label'], ['english_cleaners'])
    assert sr == 22050 and target_mel.shape[0] == 80 and np.isfinite(target_mel.numpy()).all()
    print(sample['id'], '| tokens:',len(tokens),'| mel:',tuple(target_mel.shape))
else:
    print('Optional local audio sanity check skipped: dataset not supplied.')
''')
md('''## 5. Inference front-end repairs and regression
The installed cleaner does not expand numbers; digits and `$` are absent from its character inventory. Explicit numeric/money expansion fixed lost quantities. Dotted-initialism expansion was then tested using one A–Z table, but user listening found **U.N. correct before expansion and wrong after**. Therefore numbers/money remain enabled while initialism expansion is **opt-in, off by default**. Neither repair was inserted into the frozen training procedure.

The following implementation is included in full. It handles supported integers, decimals and US dollar amounts, and fails explicitly for certain ambiguous formats rather than claiming universal English normalization. Dotted expansion is restricted to uppercase letter sequences; titles, lowercase Latin abbreviations, domains and undotted acronyms are not guessed.''')
code((ROOT/'scripts/tts_text_preprocessing.py').read_text(encoding='utf-8'))
acronym=(ROOT/'scripts/tts_acronym_preprocessing.py').read_text(encoding='utf-8').replace('from tts_text_preprocessing import expand_numbers_and_money','')
code(acronym)
code('''assert prepare_acronym_text('U.N. has 12 reports costing $12.50.') == 'U.N. has twelve reports costing twelve dollars and fifty cents.'
assert prepare_acronym_text('U.N.', expand_initialisms=True) == 'you, en.'
assert expand_numbers_and_money('$1.01') == 'one dollar and one cent'
print('Default policy: original abbreviations preserved; numbers/money repaired. Regression checks: PASS')
''')
md('''## 6. Controlled configurations and reproducible training implementation
Only learning rate differs: A = 1e-5, B = 5e-5. Both use batch 4, Adam weight decay 6e-6, gradient clipping 1, FP32, seed 42 and full validation every 250 updates. 1,000 updates = two passes through 2,000 training examples; 2,000 = four passes. The 2k runs continue each experiment's own 1k `last.pt`, restoring Adam and Python/NumPy/CPU/GPU random states. These are two extended training trajectories, not four independent replicates.

The implementation below follows the executed trainer. Notebook-only adaptations are: configuration comes from embedded evidence and outputs go to `optional_new_outputs/`, protecting submitted evidence. The loss, optimizer, validation, ordering and checkpoint logic are retained. Review mode does not call the training function.''')
code('''EXPERIMENT_CONFIGS = EVIDENCE['configs']
for name, config in EXPERIMENT_CONFIGS.items(): print(name, config)
a, b = EXPERIMENT_CONFIGS['expA'], EXPERIMENT_CONFIGS['expB']
assert {k for k in a if a[k] != b[k]} == {'name','learning_rate'}
print('Only experimental difference: learning_rate (name only routes outputs).')
''')
training=(ROOT/'scripts/train_tacotron2.py').read_text(encoding='utf-8')
training=training.replace('from tts_training_common import *','')
training=training.replace('def main():','def training_entrypoint():')
training=training.replace("if __name__=='__main__':main()",'')
training=training.replace("p.add_argument('--config',default='configs/pilot.json')","p.add_argument('--experiment',choices=['expA','expB'],required=True)")
training=training.replace("c=json.loads((ROOT/args.config).read_text(encoding='utf-8'))","c=dict(EXPERIMENT_CONFIGS[args.experiment])")
training=training.replace("directory=ROOT/'checkpoints'/c['name']","directory=OUTPUT_ROOT/c['name']/'checkpoints'")
training=training.replace("logdir=ROOT/'logs'/c['name']","logdir=OUTPUT_ROOT/c['name']/'logs'")
training=training.replace("Pilot functional test only; short-utterance subset and very few updates. No claim of quality improvement.","Optional reproduction run; no quality claim based solely on training loss.")
code(training)
code('''def materialize_embedded_manifests_for_reproduction():
    """Only called by the explicit training branch; does not overwrite project manifests."""
    destination = OUTPUT_ROOT/'data_manifests'
    destination.mkdir(parents=True, exist_ok=True)
    for split, records in EVIDENCE['manifests'].items():
        target = destination/f'{split}.json'
        text = json.dumps(records, ensure_ascii=False, indent=2)+'\\n'
        if target.exists():
            assert json.loads(target.read_text(encoding='utf-8')) == records
        else: target.write_text(text, encoding='utf-8')
    for config in EXPERIMENT_CONFIGS.values():
        config['train_manifest'] = str(destination/'train.json')
        config['val_manifest'] = str(destination/'val.json')

def reproduce_training(experiment, total_steps=1000, resume=False):
    if not torch.cuda.is_available(): raise RuntimeError('The recorded trainer requires CUDA.')
    if DATA_ROOT.resolve() != (PROJECT_ROOT/'LJSpeech-1.1').resolve():
        raise RuntimeError('For exact trainer reproduction, place dataset under PROJECT_ROOT/LJSpeech-1.1.')
    materialize_embedded_manifests_for_reproduction()
    old_argv = sys.argv[:]
    try:
        sys.argv = ['notebook-training','--experiment',experiment,'--steps',str(total_steps)] + (['--resume'] if resume else [])
        training_entrypoint()
    finally: sys.argv = old_argv

if RUN_TRAINING:
    reproduce_training('expA',1000)
    reproduce_training('expB',1000)
    reproduce_training('expA',2000,resume=True)
    reproduce_training('expB',2000,resume=True)
else:
    print('Training intentionally disabled. Existing checkpoints/results are not overwritten.')
''')
md('''Checkpoint payloads contain model state, Adam state, step, best validation value, configuration, data hashes and random states. `last.pt` represents the endpoint; `best.pt` is selected by lowest validation loss among measured saving points. Resume validates configuration and manifest hashes. Baseline validation preserves training RNG around its measurement. Training order is deterministically reconstructed from seed + epoch and step offset.

The original source hashes and checkpoint inventory below refer to the **executed project**, not the notebook's output-routing adaptations. The historical pilot-warning string in some logs is an inherited label; the actual formal runs are 2,000 train / 1,310 validation items, 1k then 2k updates.''')
code('''print('Executed source hashes:')
display(EVIDENCE['source_hashes'])
print('Recorded checkpoint inventory (existence optional for review):')
for item in EVIDENCE['checkpoints']:
    current = PROJECT_ROOT/item['relative_path']
    print(item['relative_path'], '| present:', current.exists(), '| SHA256:', item['sha256'])
''')
md('''## 7. Recorded 1k / 2k results
The following table compares **exact endpoints**, not a cherry-picked checkpoint mislabeled as 2k. True-mel reference clips are a diagnostic validation subset (148 words); custom prompts have 120 words. Do not merge these into one headline accuracy.''')
code('''def html_table(headers, rows):
    import html
    return HTML('<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>')

table_rows=[]
for name in ['expA','expB']:
    audit=EVIDENCE['comparison']['audits'][name]
    for step in [1000,2000]:
        variant=f'{name}_{step//1000}k'
        loss=next(v['loss'] for v in audit['validation'] if v['step']==step)
        ref=EVIDENCE['comparison']['summary']['reference'][variant]
        custom=EVIDENCE['comparison']['summary']['custom'][variant]
        table_rows.append([name,EXPERIMENT_CONFIGS[name]['learning_rate'],step,f'{loss:.6f}',
                           f"{ref['micro_wer']:.2%} ({ref['word_errors']}/148)",
                           f"{custom['micro_wer']:.2%} ({custom['word_errors']}/120)",
                           f"{ref['aggregate_rtf']:.5f}",f"{custom['aggregate_rtf']:.5f}"])
display(html_table(['Experiment','LR','Step','Validation loss','Reference WER','Custom WER','Reference RTF','Custom RTF'],table_rows))
display(html_table(['Experiment','Best step within 2k','Best validation loss'],
    [[name,a['best_step_within_2k'],f"{a['best_loss_within_2k']:.6f}"] for name,a in EVIDENCE['comparison']['audits'].items()]))
''')
md('''## 8. Training and validation curves
Raw batch losses and a trailing 50-step mean are both shown. Different batch difficulty causes substantial raw variation. The vertical line marks the continuation boundary. Validation uses all 1,310 items; stars identify the lowest measured value within 2k. The validation axis is magnified and is not a percentage.''')
code('''import matplotlib.pyplot as plt
colors={'expA':'#166d9c','expB':'#b45521'}
fig,axes=plt.subplots(1,2,figsize=(13,4.5))
for name,rows in EVIDENCE['metrics'].items():
    steps=np.array([r['step'] for r in rows]);loss=np.array([r['train_loss'] for r in rows])
    axes[0].plot(steps,loss,color=colors[name],alpha=.13,lw=.6)
    axes[0].plot(steps[49:],np.convolve(loss,np.ones(50)/50,mode='valid'),color=colors[name],label=name)
    validation=EVIDENCE['comparison']['audits'][name]['validation']
    axes[1].plot([0]+[v['step'] for v in validation],[.3928682383235174]+[v['loss'] for v in validation],'-o',color=colors[name],label=name)
    best=EVIDENCE['comparison']['audits'][name]
    axes[1].scatter(best['best_step_within_2k'],best['best_loss_within_2k'],marker='*',s=160,color=colors[name],edgecolor='white',zorder=5)
for ax in axes:
    ax.axvline(1000,color='grey',ls='--',lw=1);ax.set_xlabel('Optimizer updates');ax.grid(alpha=.2);ax.legend()
axes[0].set(title='Raw training loss and trailing 50-step mean',ylabel='Total training loss')
axes[1].set(title='Teacher-forced validation loss',ylabel='Validation loss')
fig.tight_layout();plt.show()
''')
md('''## 9. Objective evaluation implementation and measurement protocol
Latency measures text-to-waveform inference with CUDA synchronization; RTF = synthesis seconds / output seconds. One warm-up per model is excluded; each prompt is repeated three times. Model loading and file saving are outside the timed interval. Output length changes affect RTF, and sequential system load can affect timing; small differences are not conclusive.

Local wav2vec2 ASR is a supplementary intelligibility proxy, not a naturalness score. The word-level edit distance below is used with lowercase, punctuation-stripped reference tokens. Word-boundary, spelling and recognizer errors can inflate WER; human feedback is stored separately.''')
code('''import re, time
def word_tokens(text): return re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
def word_edit_distance(reference,hypothesis):
    previous=list(range(len(hypothesis)+1))
    for i,x in enumerate(reference,1):
        current=[i]
        for j,y in enumerate(hypothesis,1):
            current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def timed_synthesis(tts,vocoder,text,repeats=3):
    times=[];audio=None
    for _ in range(repeats):
        seed_all(42)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            mel,lengths,_=tts.encode_text(text)
            waveform=vocoder.decode_batch(mel)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        if audio is None: audio=waveform.squeeze().detach().cpu().numpy()
    duration=len(audio)/22050
    return audio,{'latencies':times,'mean_latency':statistics.mean(times),
                  'stdev_latency':statistics.stdev(times),'rtf':statistics.mean(times)/duration}

def transcribe_local_wav(path,recognizer,bundle,labels,device):
    audio,sr=sf.read(str(path),dtype='float32')
    signal=torchaudio.functional.resample(torch.from_numpy(audio),sr,bundle.sample_rate).unsqueeze(0).to(device)
    with torch.inference_mode(): emission,_=recognizer(signal)
    ids=torch.unique_consecutive(emission[0].argmax(-1)).tolist()
    return ''.join(labels[i] for i in ids if i!=0).replace('|',' ').strip()

assert word_edit_distance(['a','b'],['a','c'])==1
print('Evaluation functions defined. Existing measurements below are loaded, not recomputed.')
for section in ['reference','custom']:
    print(section)
    display(html_table(['Version','Word errors','WER'],[[v,f"{s['word_errors']}/{s['reference_words']}",f"{s['micro_wer']:.2%}"]
        for v,s in EVIDENCE['comparison']['summary'][section].items()]))
''')
md('''## 10. Fixed audio examples and five-way diagnosis
Ten validation items were selected before synthesis using seed 42: three <4 seconds, four 4–7 seconds and three ≥7 seconds. The five paths are real WAV, real-mel reconstruction, pretrained Tacotron2, A and B. No samples were replaced after hearing outputs.

Original dataset WAVs are **not included** in this submission. Their IDs/text and diagnostic results are retained; listen to raw references from a locally downloaded LJSpeech copy. All bundled WAVs are generated/reconstructed outputs. The code resolves local source recordings when requested, while default outputs play generated examples only. No output-level loudness normalization is applied.''')
code('''display(html_table(['ID','LJSpeech ID','Length group','Seconds','Text'],
    [[r['display_id'],r['id'],r['length_group'],f"{r['duration']:.2f}",r['label']] for r in EVIDENCE['diagnostic']['selection']]))

def play_generated(section,sample_id,variant):
    row=next(r for r in EVIDENCE['audio_manifest'] if r['section']==section and r['id']==sample_id and r['variant']==variant)
    path=SUBMISSION_ROOT/row['submission_wav']
    assert path.exists()
    display(Markdown(f"**{section} {sample_id} — {variant}**: {row['text']}"))
    display(Audio(filename=str(path),normalize=False))

def play_local_reference(sample_id):
    row=next(r for r in EVIDENCE['diagnostic']['selection'] if r['display_id']==sample_id)
    path=DATA_ROOT/'wavs'/f"{row['id']}.wav"
    if not path.exists(): raise FileNotFoundError('Download LJSpeech locally; original recordings are not packaged.')
    display(Audio(filename=str(path),normalize=False))

# Ten cached playable generated examples; all 110 files are indexed below.
for sample_id in ['06','07']:
    for variant in ['pretrained','expA_1k','expA_2k','expB_1k','expB_2k']:
        play_generated('custom',sample_id,variant)
''')
code('''# Compact index of every bundled example (real reference recordings deliberately excluded).
display(html_table(['Section','ID','Version','Generated WAV'],
    [[r['section'],r['id'],r['variant'],r['submission_wav']] for r in EVIDENCE['audio_manifest']]))
print('Five-way diagnostic ASR (148 reference words):')
display(html_table(['Path','Errors','WER'],[[v,s['word_errors'],f"{s['micro_wer']:.2%}"] for v,s in EVIDENCE['diagnostic']['reference_summary'].items()]))
''')
md('''## 11. Error analysis and actual human observations
- Numeric/money content was silently lost by the unmodified frontend. Explicit expansion repaired `12`, `3`, `$12.50` for all models, separating a frontend repair from learning-rate effects.
- R07 `N.R.A.` was recognizable in real/reconstructed audio but unstable in text-generated output. A letter-name repair with pauses helped that case; a no-pause trial was not stable.
- General dotted expansion had successes and failures. **U.N. was manually judged correct before processing and wrong after**; ASR non-match therefore cannot be equated with a true pronunciation error. Default expansion was disabled.
- A/B gave mixed local changes: improved recognition on some phrases, regressions on others. Human listening found real-mel reconstruction close to real recordings, supporting investigation of text/acoustic prediction before assuming the vocoder is the principal bottleneck. This is a diagnostic inference, not proof of a single cause.
- After 2k, the user reported no obvious improvement and some regression. No model/sample locations or numerical ratings were supplied, so none are invented.

These are one user's qualitative observations, not a three-rater MOS study. Final contributor review should retain this distinction.''')
code('''for observation in EVIDENCE['human']['observations']:
    display(Markdown('**'+observation['id']+'**'))
    print(observation.get('user_statement',''))
    print('Scope:',observation.get('scope',observation.get('model_scope','')))
    print('Numeric score:',observation.get('numeric_score'))
print('Historical dotted-expansion ASR screening, not human pass rates:')
display(EVIDENCE['acronyms']['summary_matches_out_of_8'])
''')
md('''## 12. Arbitrary-text inference implementation (optional; off by default)
The original pretrained model can be obtained from the official repositories if explicitly requested. Existing fine-tuned inference requires the corresponding project checkpoint, or `A3_CHECKPOINT_ROOT` pointing to it. The notebook does not claim that the fine-tuned weights are embedded. Missing weights never trigger training.

Model keys: pretrained; expA_1k; expA_2k (exact 2k endpoint); expB_1k; expB_2k; expA_best (1,750 steps); expB_best (1,000 steps). Numeric repair is on and dotted expansion off unless explicitly enabled.''')
code('''MODEL_CHECKPOINTS = {
    'expA_1k':('expA/last.pt',1000), 'expA_2k':('expA/stage_2k/last.pt',2000),
    'expB_1k':('expB/last.pt',1000), 'expB_2k':('expB/stage_2k/last.pt',2000),
    'expA_best':('expA/stage_2k/best.pt',1750), 'expB_best':('expB/stage_2k/best.pt',1000)}

def load_inference_system(model_key='pretrained',device=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tts_folder=PROJECT_ROOT/'models/tacotron2'
    voc_folder=PROJECT_ROOT/'models/hifigan'
    tts_source=str(tts_folder) if (tts_folder/'model.ckpt').exists() else 'speechbrain/tts-tacotron2-ljspeech'
    voc_source=str(voc_folder) if (voc_folder/'generator.ckpt').exists() else 'speechbrain/tts-hifigan-ljspeech'
    if model_key!='pretrained':
        if model_key not in MODEL_CHECKPOINTS: raise ValueError(model_key)
        path=CHECKPOINT_ROOT/MODEL_CHECKPOINTS[model_key][0]
        if not path.exists(): raise FileNotFoundError(f'Set A3_CHECKPOINT_ROOT to preserved project checkpoints: {path}')
    tts=Tacotron2.from_hparams(source=tts_source,savedir=str(tts_folder),run_opts={'device':device},local_strategy=LocalStrategy.COPY)
    voc=HIFIGAN.from_hparams(source=voc_source,savedir=str(voc_folder),run_opts={'device':device},local_strategy=LocalStrategy.COPY)
    if model_key!='pretrained':
        state=torch.load(path,map_location='cpu',weights_only=False)  # only locally produced, trusted checkpoints
        assert state['step']==MODEL_CHECKPOINTS[model_key][1]
        tts.mods.model.load_state_dict(state['model'],strict=True)
    tts.eval();voc.eval();tts.requires_grad_(False);voc.requires_grad_(False)
    return tts,voc

def generate_arbitrary_text(text,model_key='pretrained',expand_initialisms=False):
    processed=prepare_acronym_text(text,expand_initialisms=expand_initialisms)
    tts,voc=load_inference_system(model_key)
    audio,measurement=timed_synthesis(tts,voc,processed,repeats=3)
    assert np.isfinite(audio).all() and np.abs(audio).max()>0
    destination=OUTPUT_ROOT/'wavs'/f'{model_key}_new.wav'
    destination.parent.mkdir(parents=True,exist_ok=True)
    sf.write(str(destination),audio,22050,subtype='PCM_16')
    print('Input:',processed,'| measurements:',measurement)
    display(Audio(filename=str(destination),normalize=False))
    return destination

if RUN_NEW_INFERENCE:
    generate_arbitrary_text('There are 12 books, and the ticket costs $12.50.',model_key='expA_1k')
else:
    print('New model loading/download/synthesis disabled. Cached generated examples above remain playable.')
''')
md('''## 13. Conclusions and limitations
Within this specific dataset subset, pretrained system and single-seed budget, extending from 1k to 2k did not yield a consistent perceptible improvement. A's exact endpoint validation loss changed only slightly (0.373902 → 0.373427); its lowest value was at 1,750. B worsened (0.383728 → 0.387289), and its validation-selected checkpoint stayed at 1k. Small ASR differences do not establish a naturalness advantage. The user chose to stop additional training.

Original pretrained speech and A1k remain reasonable practical candidates; A1750 is the validation-loss optimum, not a verified listening winner. Scaling/deployment considerations include cached weights, hardware-dependent latency, bounded sentence length, explicit text normalization and failure reporting. Additional work could include larger predeclared evaluation sets, independent raters, reliable pronunciation modeling and multiple training seeds; these were not performed here.

The primary dataset was also used for pretraining, so present splits do not establish unseen pretraining generalization. Validation diagnostics informed development. The full reserved test split was not evaluated. Original data/outputs are preserved in the project; this notebook distinguishes measured evidence from optional reproducibility code.

## 14. Attribution, team review and AI disclosure
External components: [LJSpeech](https://keithito.com/LJ-Speech-Dataset/); [SpeechBrain v1.0.3 recipe](https://github.com/speechbrain/speechbrain/tree/v1.0.3/recipes/LJSpeech/TTS/tacotron2); [Tacotron2 checkpoint](https://huggingface.co/speechbrain/tts-tacotron2-ljspeech); [HiFi-GAN checkpoint](https://huggingface.co/speechbrain/tts-hifigan-ljspeech); local torchaudio WAV2VEC2_ASR_BASE_960H. Source code retains relevant in-code links. Add the course-required Harvard bibliography in the report using sources actually consulted.

**AI assistance used in this project:** requirement clarification, planning, environment setup, code scaffolding/debugging, experiment execution support, result organization, visualization and notebook assembly. Human listening observations were supplied by the user. The team must review and understand all code, verify results, complete actual member contributions and ensure its disclosure/submission follows the course policy. No human scores, citations consulted by the team, or member contributions are invented here.

This technical notebook does not replace the separately required report PDF and presentation PPTX.''')
code('''# Read-only submission checks. No optimizer is created or stepped by this check.
assert EVIDENCE['group']==62
assert len(EVIDENCE['audio_manifest'])==110
missing=[r['submission_wav'] for r in EVIDENCE['audio_manifest'] if not (SUBMISSION_ROOT/r['submission_wav']).exists()]
assert not missing
assert all(hashlib.sha256((SUBMISSION_ROOT/r['submission_wav']).read_bytes()).hexdigest()==r['sha256'] for r in EVIDENCE['audio_manifest'])
assert not RUN_TRAINING and not RUN_NEW_INFERENCE and not RUN_NEW_ASR
print('PASS: 110 generated WAVs, saved evidence, group 62 and review-only switches verified.')
print('Training was not run. Original LJSpeech recordings and environments are not bundled.')
''')
# Keep evidence external for a readable notebook.
evidence_folder=SUB/'data'
evidence_folder.mkdir(parents=True,exist_ok=True)
(evidence_folder/'experiment_evidence.json').write_text(json.dumps(E,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for evidence_cell in cells:
    if evidence_cell.cell_type=='code' and evidence_cell.source.startswith('EVIDENCE = json.loads('):
        evidence_cell.source = '''EVIDENCE_PATH = SUBMISSION_ROOT / "data" / "experiment_evidence.json"
with EVIDENCE_PATH.open(encoding="utf-8") as handle:
    EVIDENCE = json.load(handle)
assert EVIDENCE["group"] == 62
print("Recorded evidence loaded from", EVIDENCE_PATH)
'''

nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Assignment 3 (Group 62)','language':'python','name':'python3'},
             'language_info':{'name':'python','version':'3.10.11'},
             'a3':{'group':62,'training_disabled_by_default':True,'project_path_policy':'relative / environment override',
                   'evidence_sha256':hashlib.sha256(json.dumps(E,sort_keys=True).encode()).hexdigest()}}
nbf.validate(nb)
for cell in nb.cells:
    if cell.cell_type=='code':ast.parse(cell.source)
nbf.write(nb,SUB/'62_A3.ipynb')
nbf.write(nb,ROOT/'62_A3.ipynb')
(ROOT/'migration_records').mkdir(exist_ok=True)
(ROOT/'migration_records/path_migration.json').write_text(json.dumps({'new_project_root':str(ROOT),'changes':migration,
    'historical_paths':'Preserved in immutable original logs; notebook uses relative paths and embedded evidence.',
    'checkpoint_inventory':checkpoint_inventory,'no_training':True},indent=2),encoding='utf-8')
(SUB/'SUBMISSION_NOTES.md').write_text('''# Group 62 notebook review

62_A3.ipynb and wav_examples/ contain the reviewable notebook and 110 generated/reconstructed WAV examples. No raw dataset, virtual environment or download cache is included.

Run All is review-only: no training, new downloads or synthesis. Full implementations are included. Optional new fine-tuned inference requires the external preserved checkpoints; set A3_CHECKPOINT_ROOT. Dataset-dependent checks/training require a downloaded LJSpeech copy. Use A3_PROJECT_ROOT and LJSPEECH_ROOT if needed. Checkpoint weights are not embedded or silently available on another machine.

Before final upload, complete actual member/student-ID/contribution details, review the AI disclosure and code, and prepare the separate PDF/PPTX. A final ZIP has not been automatically submitted. Course-required root names: 62_A3.ipynb and wav_examples/.
''',encoding='utf-8')
print(json.dumps({'notebook':str(SUB/'62_A3.ipynb'),'root_copy':str(ROOT/'62_A3.ipynb'),'cells':len(cells),
    'generated_wavs':len(audio_manifest),'code_cells':sum(c.cell_type=='code' for c in cells),'migration_changes':migration},indent=2))

