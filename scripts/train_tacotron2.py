"""Small supervised fine-tuning scaffold; official SB model, collator, mel and loss.
Use --verify-only to reload the saved checkpoint in a fresh process and synthesize.
Use --resume --steps 5 to prove optimizer/RNG restoration and run one more update.
This pilot is not a complete assignment or evidence of improved speech quality.
"""
from tts_training_common import *
import argparse, time, csv, sys, platform

def validate(model, fn, dataset, size):
    model.eval(); totals=[]
    with torch.no_grad():
        for start in range(0,len(dataset),size):
            ids=list(range(start,min(start+size,len(dataset))))
            value=loss_stats(model,fn,batch(dataset,ids)).loss
            assert torch.isfinite(value)
            totals.append((float(value),len(ids)))
    return sum(v*n for v,n in totals)/sum(n for _,n in totals)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/pilot.json')
    p.add_argument('--steps',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    c=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    assert c['precision']=='float32' and c['guided_attention_weight']==0
    assert torch.cuda.is_available()
    seed_all(c['seed']); torch.cuda.reset_peak_memory_stats()
    directory=ROOT/'checkpoints'/c['name']; directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/'last.pt'
    manifests={key:sha(ROOT/c[key]) for key in ('train_manifest','val_manifest')}
    wrapper,model=load_tts(trainable=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    fn=criterion()
    start_step=0; best=float('inf'); restored=False
    if args.resume or args.verify_only:
        # This is a locally produced trusted checkpoint containing optimizer/RNG state.
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['manifest_hashes']==manifests, 'Data manifests changed'
        assert state['config']==c, 'Config changed; start a new experiment instead'
        model.load_state_dict(state['model'],strict=True)
        assert all(torch.equal(v.detach().cpu(),state['model'][k]) for k,v in model.state_dict().items())
        optimizer.load_state_dict(state['optimizer'])
        start_step=state['step']; best=state['best_validation']; restored=True
        random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch']); torch.cuda.set_rng_state_all(state['rng_cuda'])
        print(f'Restored all model tensors and Adam state at step {start_step}',flush=True)
        del state
    elif checkpoint.exists():
        raise RuntimeError('Pilot checkpoint exists. Use --resume or choose a new config/name.')

    if args.verify_only:
        model.eval()
        audio=synthesize(wrapper,ROOT/'wav_examples'/f'{c["name"]}_reloaded_step{start_step}.wav',c['prompt'])
        write_json(ROOT/'logs'/c['name']/'reload_verification.json',
            {'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'step':start_step,
             'all_model_tensors_match_checkpoint':True,'optimizer_state_loaded':restored,
             'fresh_process':True,**audio})
        print(json.dumps(audio,indent=2),flush=True)
        return

    training=SpeechDataset(ROOT/c['train_manifest']); validation=SpeechDataset(ROOT/c['val_manifest'])
    assert len(training)>=c['batch_size']
    before=model.embedding.weight.detach().clone()
    rng_cpu=torch.get_rng_state(); rng_gpu=torch.cuda.get_rng_state_all()
    initial_validation=validate(model,fn,validation,c['batch_size'])
    # Validation uses stochastic prenet dropout. Preserve training RNG around baseline
    # measurement so a resumed run continues from the saved random state.
    torch.set_rng_state(rng_cpu); torch.cuda.set_rng_state_all(rng_gpu)
    max_steps=args.steps if args.steps is not None else c['max_steps']
    assert max_steps>start_step
    logdir=ROOT/'logs'/c['name'];logdir.mkdir(parents=True,exist_ok=True)
    write_json(logdir/'config.json',c)
    metrics=logdir/'metrics.csv'
    if start_step==0:
        metrics.write_text('step,train_loss,mel_loss,gate_loss,gradient_norm,validation_loss,elapsed_seconds\n',encoding='utf-8')
    batches_per_epoch=len(training)//c['batch_size']
    start=time.perf_counter(); final_val=initial_validation
    for step in range(start_step,max_steps):
        epoch,offset=divmod(step,batches_per_epoch)
        order=list(range(len(training)));random.Random(c['seed']+epoch).shuffle(order)
        ids=order[offset*c['batch_size']:(offset+1)*c['batch_size']]
        model.train();optimizer.zero_grad(set_to_none=True)
        stats=loss_stats(model,fn,batch(training,ids))
        assert torch.isfinite(stats.loss), 'Nonfinite training loss'
        stats.loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
        assert float(grad)>0
        optimizer.step()
        values=[step+1,float(stats.loss.detach()),float(stats.mel_loss.detach()),float(stats.gate_loss.detach()),float(grad),'',time.perf_counter()-start]
        if (step+1)%c['validation_interval']==0 or step+1==max_steps:
            final_val=validate(model,fn,validation,c['batch_size']);values[5]=final_val
            is_best=final_val<best; best=min(best,final_val)
            payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,
                     'best_validation':best,'config':c,'manifest_hashes':manifests,
                     'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
                     'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()}
            temporary=directory/'last.tmp';torch.save(payload,temporary);temporary.replace(checkpoint)
            if is_best:
                import shutil
                shutil.copy2(checkpoint,directory/'best.pt')
        with metrics.open('a',encoding='utf-8',newline='') as f:csv.writer(f).writerow(values)
        print(f'Step {step+1}/{max_steps}: loss={values[1]:.4f}, grad={values[4]:.4f}, val={values[5]}',flush=True)
    change=float((model.embedding.weight.detach()-before).abs().max())
    assert change>0, 'Weights did not update'
    report={'status':'training_checkpoint_saved','start_step':start_step,'end_step':max_steps,
        'resumed_optimizer':restored,'embedding_max_abs_change':change,
        'initial_validation_loss':initial_validation,'final_validation_loss':final_val,
        'peak_gpu_memory_mib':torch.cuda.max_memory_allocated()/2**20,'seconds':time.perf_counter()-start,
        'device':torch.cuda.get_device_name(0),'python':platform.python_version(),'torch':torch.__version__,
        'checkpoint':str(checkpoint),'manifest_hashes':manifests,
        'warning':'Pilot functional test only; short-utterance subset and very few updates. No claim of quality improvement.'}
    write_json(logdir/('resume_training_report.json' if restored else 'training_report.json'),report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
