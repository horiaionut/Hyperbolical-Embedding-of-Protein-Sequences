import pandas as pd
import re
import json

root = {
    'idx' : 0
}

labels = {
    'id': [],
    'number': [],
    'name': [],
    'parent_idx' : [],
    'connected': []
}

with open('res/cath/raw/cath-names.txt', 'r') as f:
    for line in f:
        if line[0] == '#':
            continue
            
        words = list(filter(None, re.split(' |:|\n', line)))

        name = ''
        for word in words[2:]:
            name += word + ' '

        labels['number'].append(words[0])
        labels['id'].append(words[1])
        labels['name'].append(name)
        labels['parent_idx'].append(-1)
        labels['connected'].append(set())

        steps = words[0].split('.')

        level = root
        for s in steps:
            if 'subclasses' not in level:
                level['subclasses'] = {}

            if s not in level['subclasses']:
                level['subclasses'][s] = {}

            level = level['subclasses'][s]

        level['idx'] = len(labels['number'])

def dfs(node):
    idx = node['idx']

    if 'subclasses' in node:
        for child in node['subclasses'].values():
            labels['connected'][idx - 1].add(child['idx'])
            labels['connected'][child['idx'] - 1].add(idx)
            labels['parent_idx'][child['idx'] - 1] = idx

            dfs(child)

for child in root['subclasses'].values():
    labels['connected'][child['idx'] - 1].add(0)
    labels['parent_idx'][child['idx'] - 1] = 0
    dfs(child)

df = pd.DataFrame(labels) 
df.index += 1
df.to_pickle('res/cath/labels2.dat')
