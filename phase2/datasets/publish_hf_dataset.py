import argparse, os
from datasets import DatasetDict, load_dataset

p=argparse.ArgumentParser()
p.add_argument('--train', required=True)
p.add_argument('--eval')
p.add_argument('--repo', required=True)
p.add_argument('--private', action='store_true')
a=p.parse_args()

files={'train': a.train}
if a.eval: files['validation']=a.eval
ds=DatasetDict({split: load_dataset('json', data_files=path, split='train') for split, path in files.items()})
token=os.getenv('HF_TOKEN')
if not token: raise SystemExit('Set HF_TOKEN first')
ds.push_to_hub(a.repo, private=a.private, token=token)
print({'repo': a.repo, 'splits': list(ds.keys()), 'private': a.private})
