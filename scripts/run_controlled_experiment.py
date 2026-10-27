"""Verify the frozen protocol before starting one formal experiment."""
from tts_training_common import ROOT, sha
import argparse, json, subprocess, sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('experiment',choices=['expA','expB'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    protocol=json.loads((ROOT/'data_manifests/formal/protocol.json').read_text(encoding='utf-8'))
    for relative,expected in protocol['sha256'].items():
        if sha(ROOT/relative)!=expected:
            raise RuntimeError(f'Frozen experiment input changed: {relative}')
    configs=[json.loads((ROOT/'configs'/f'{name}.json').read_text(encoding='utf-8')) for name in ['expA','expB']]
    assert {key for key in configs[0] if configs[0][key]!=configs[1][key]}=={'name','learning_rate'}
    print('PASS: shared data, code, original weights and configs match frozen hashes.',flush=True)
    if args.check_only:return
    command=[sys.executable,str(ROOT/'scripts/train_tacotron2.py'),'--config',f'configs/{args.experiment}.json']
    if args.resume:command.append('--resume')
    # Each invocation is a new process; default startup loads original pretrained weights.
    raise SystemExit(subprocess.call(command,cwd=ROOT))

if __name__=='__main__':main()
