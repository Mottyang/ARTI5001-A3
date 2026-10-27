"""Stage 2k outputs separately; retain frozen 1k files byte-for-byte."""
from pathlib import Path
import json,hashlib,shutil,ast

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((ROOT/'data_manifests/formal/protocol.json').read_text(encoding='utf-8'))
assert all(sha(ROOT/p)==h for p,h in protocol['sha256'].items())
source_path=ROOT/'scripts/train_tacotron2.py'
source=source_path.read_text(encoding='utf-8')
changes={
 "directory=ROOT/'checkpoints'/c['name'];":"directory=ROOT/'checkpoints'/c['name']/'stage_2k';",
 "logdir=ROOT/'logs'/c['name'];":"logdir=ROOT/'logs'/c['name']/'stage_2k';",
}
modified=source
for old,new in changes.items():
    assert modified.count(old)==1;modified=modified.replace(old,new)
restored=modified
for old,new in changes.items():restored=restored.replace(new,old)
assert restored==source
ast.parse(modified)
target=ROOT/'scripts/train_tacotron2_stage2k.py'
target.write_text(modified,encoding='utf-8')
inputs=[]
for name in ['expA','expB']:
    old=ROOT/'checkpoints'/name;new=old/'stage_2k'
    if new.exists():raise RuntimeError(f'Continuation directory exists: {new}; refusing to overwrite')
    new.mkdir()
    for file in ['last.pt','best.pt']:
        shutil.copy2(old/file,new/file)
        assert sha(old/file)==sha(new/file)
        inputs.append({'path':str((old/file).relative_to(ROOT)),'sha256':sha(old/file)})
    logs=ROOT/'logs'/name/'stage_2k';logs.mkdir(parents=True)
    (logs/'metrics.csv').write_text('step,train_loss,mel_loss,gate_loss,gradient_norm,validation_loss,elapsed_seconds\n',encoding='utf-8')
    inputs += [{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in (ROOT/'logs'/name).iterdir() if p.is_file()]
out=ROOT/'data_manifests/formal/continuation_2k.json'
record={'start_step':1000,'end_step':2000,'additional_updates_per_model':1000,
 'source_training_script_sha256':sha(source_path),'continuation_script_sha256':sha(target),
 'code_change_scope':'Exactly two output-path substitutions; all training, validation, optimizer, data order and RNG restoration logic unchanged.',
 'output_routing_changes':changes,'config_policy':'Original config kept unchanged to restore checkpoint exactly; --steps 2000 overrides effective total budget.',
 'same_training_settings':True,'training_frontend_unchanged':True,
 'resume_source':'Each experiment own 1k last.pt, including Adam and RNG states. best.pt seeded from its 1k best.',
 'selection_policy':'Keep best-so-far validation checkpoint; last.pt at 2000 is the endpoint for the literal 1k-vs-2k comparison. Best-so-far may be earlier than 2000.',
 'evaluation_policy':'Same 10 fixed validation references and same 10 custom prompts; numeric/money repair enabled; optional dotted expansion disabled.',
 'preserved_1k_files':inputs}
out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Prepared separate stage_2k checkpoints/logs for both experiments; 1k sources verified and unchanged.')
