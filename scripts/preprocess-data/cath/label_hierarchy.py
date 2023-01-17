import pandas as pd
import re

labels = {
    # 'id': [],
    'number': [],
    'name': []
}

with open('data/cath/raw/cath-names.txt', 'r') as f:
    for line in f:
        if line[0] == '#':
            continue
            
        words = list(filter(None, re.split(' |:|\n', line)))

        name = ''
        for word in words[2:]:
            name += word + ' '

        labels['number'].append(words[0])
        # labels['id'].append(words[1])
        labels['name'].append(name)

df = pd.DataFrame(labels) 
df.index += 1
df.to_csv('data/cath/labels.csv', sep=' ')
