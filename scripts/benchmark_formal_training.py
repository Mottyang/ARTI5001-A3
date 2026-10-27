"""Short representative timing run. Updates in RAM only; no formal checkpoint saved."""
from tts_training_common import *
import time, math, statistics, gc

def sync_time():
    torch.cuda.synchronize();return time.perf_counter()

def main():
    c=json.loads((ROOT/'configs'/'expA.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'data_manifests'/'formal'/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in protocol['sha256'].items()), 'Frozen inputs changed'
    seed_all(c['seed']); torch.cuda.reset_peak_memory_stats()
    start=sync_time();wrapper,model=load_tts(trainable=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    fn=criterion();load_seconds=sync_time()-start
    train=SpeechDataset(ROOT/c['train_manifest']);val=SpeechDataset(ROOT/c['val_manifest'])
    order=list(range(len(train)));random.Random(c['seed']).shuffle(order)
    bs=c['batch_size'];measurements=[]
    for step in range(14):
        ids=order[step*bs:(step+1)*bs]
        start=sync_time();packed=batch(train,ids);prepared=sync_time()
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=loss_stats(model,fn,packed).loss
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();end=sync_time()
        row={'step':step+1,'data_seconds':prepared-start,'compute_seconds':end-prepared,'total_seconds':end-start,
             'max_audio_seconds':max(train.records[i]['duration'] for i in ids)}
        measurements.append(row)
        print(f'Benchmark step {step+1}/14: {row["total_seconds"]:.3f}s',flush=True)
    # Widely spaced validation batches cover the fixed full list without evaluating test.
    n_batches=math.ceil(len(val)/bs)
    starts=sorted(set(round(i*(n_batches-1)/11)*bs for i in range(12)))
    model.eval();validation=[]
    with torch.no_grad():
        for position in starts:
            ids=list(range(position,min(position+bs,len(val))))
            start=sync_time();packed=batch(val,ids);prepared=sync_time()
            loss=loss_stats(model,fn,packed).loss
            assert torch.isfinite(loss)
            end=sync_time();validation.append({'count':len(ids),'data_seconds':prepared-start,
                'compute_seconds':end-prepared,'total_seconds':end-start})
    warm=measurements[2:]
    train_compute=statistics.mean(r['compute_seconds'] for r in warm)
    train_data=statistics.mean(r['data_seconds'] for r in warm)
    val_compute=statistics.mean(r['compute_seconds'] for r in validation)
    val_data=statistics.mean(r['data_seconds'] for r in validation)
    passes=1+math.ceil(c['max_steps']/c['validation_interval'])
    # Cached feature extraction happens on first access; budget one pass over train/val.
    seconds=load_seconds+c['max_steps']*train_compute+len(train)/bs*train_data+passes*n_batches*val_compute+n_batches*val_data
    report={'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'batch_size':bs,
        'model_load_seconds':load_seconds,'training_updates_measured':14,'warmup_updates_excluded':2,
        'validation_batches_measured':len(validation),'train_mean_compute_seconds':train_compute,
        'train_mean_cold_data_seconds':train_data,'val_mean_compute_seconds':val_compute,
        'val_mean_cold_data_seconds':val_data,'validation_batches_full_pass':n_batches,'validation_passes_per_run':passes,
        'estimated_per_experiment_minutes':seconds/60,
        'planning_per_experiment_minutes_range':[seconds/60*.75,seconds/60*1.6+2],
        'planning_both_experiments_minutes_range':[seconds/60*1.5,seconds/60*3.2+4],
        'peak_allocated_gpu_mib':torch.cuda.max_memory_allocated()/2**20,
        'peak_reserved_gpu_mib':torch.cuda.max_memory_reserved()/2**20,
        'training_batches':measurements,'validation_batches':validation,
        'scope':'Timing only; original weights unchanged on disk; no expA/expB checkpoints. No test inference. Estimate includes five validation passes and approximate feature caching cost; allowance for checkpoint I/O, duration mix, system load.',
        'caveat':'Small timing sample; does not guarantee throughput or final model quality.'}
    write_json(ROOT/'logs'/'formal_timing_benchmark.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('training_batches','validation_batches')},indent=2),flush=True)

if __name__=='__main__':main()
