"""Audit the completed expA logs/checkpoints, then synthesize from best.pt."""
from tts_training_common import *
import csv, math

def main():
    config=json.loads((ROOT/'configs'/'expA.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'data_manifests'/'formal'/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in protocol['sha256'].items()), 'Frozen experiment inputs changed'
    folder=ROOT/'logs'/'expA'
    rows=list(csv.DictReader((folder/'metrics.csv').open(encoding='utf-8',newline='')))
    assert [int(r['step']) for r in rows]==list(range(1,config['max_steps']+1))
    for r in rows:
        assert all(math.isfinite(float(r[k])) for k in ('train_loss','mel_loss','gate_loss','gradient_norm','elapsed_seconds'))
        assert float(r['gradient_norm'])>0
    validation=[r for r in rows if r['validation_loss']]
    assert [int(r['step']) for r in validation]==[250,500,750,1000]
    assert all(math.isfinite(float(r['validation_loss'])) for r in validation)
    chosen=min(validation,key=lambda r:float(r['validation_loss']))
    last_path=ROOT/'checkpoints'/'expA'/'last.pt'
    best_path=ROOT/'checkpoints'/'expA'/'best.pt'
    last=torch.load(last_path,map_location='cpu',weights_only=False)
    best=torch.load(best_path,map_location='cpu',weights_only=False)
    assert last['step']==1000 and best['step']==int(chosen['step'])
    assert math.isclose(best['best_validation'],float(chosen['validation_loss']),abs_tol=1e-10)
    for state in (last,best):
        assert state['config']==config
        assert state['manifest_hashes']=={k:sha(ROOT/config[k]) for k in ('train_manifest','val_manifest')}
        assert state['optimizer']['state'], 'Adam state missing'
        assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point())
    del last
    seed_all(42)
    wrapper,model=load_tts(trainable=False)
    original_embedding=model.embedding.weight.detach().cpu().clone()
    model.load_state_dict(best['model'],strict=True)
    exact=all(torch.equal(v.detach().cpu(),best['model'][k]) for k,v in model.state_dict().items())
    assert exact
    delta=float((model.embedding.weight.detach().cpu()-original_embedding).abs().max())
    assert delta>0
    model.eval()
    best_step=best['step']; del best
    prompts=[('01',config['prompt']),('02','The quick brown fox jumps over the lazy dog.')]
    outputs=[]
    for label,text in prompts:
        seed_all(42)
        output=ROOT/'wav_examples'/f'expA_best_step{best_step}_{label}.wav'
        detail=synthesize(wrapper,output,text)
        outputs.append({'text':text,**detail})
    training_report=json.loads((folder/'training_report.json').read_text(encoding='utf-8'))
    report={'status':'PASS','completed_updates':len(rows),'all_training_metrics_finite':True,
            'all_gradient_norms_positive':True,'validation_results':[
                {'step':int(r['step']),'loss':float(r['validation_loss'])} for r in validation],
            'initial_validation_loss':training_report['initial_validation_loss'],
            'best_step':best_step,'best_validation_loss':float(chosen['validation_loss']),
            'first_50_train_loss_mean':sum(float(r['train_loss']) for r in rows[:50])/50,
            'last_50_train_loss_mean':sum(float(r['train_loss']) for r in rows[-50:])/50,
            'checkpoint_tensor_equality_after_reload':exact,'embedding_change_vs_pretrained':delta,
            'last_checkpoint_step':1000,'checkpoint_sha256':{'best':sha(best_path),'last':sha(last_path)},
            'training_report':training_report,'audio_checks':outputs,
            'limits':'Audio decoded, finite/nonzero, decoder stopped before limit. No human listening score or quality improvement claim. Validation loss uses teacher forcing and stochastic prenet. Test set not evaluated.'}
    write_json(folder/'completion_audit.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
