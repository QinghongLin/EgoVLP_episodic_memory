import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root',type=str)
args = parser.parse_args()
folders = sorted(os.listdir(args.root))

results = {}
for f in folders:
    results_f = f'{args.root}{f}/results.txt'
    if os.path.exists(results_f):
        content = open(results_f).readlines()
        recalls = []
        for row in content[-5:-2]:
            recalls.append(row.replace(' [','').replace('[','').replace(']','').replace('\n',''))
        recalls = ' '.join(recalls).replace('  ',' ')
        results[f] = recalls

with open('./parsed_results.txt','w') as f:
    for k,v in results.items():
        f.write(f'{k} {v}\n')