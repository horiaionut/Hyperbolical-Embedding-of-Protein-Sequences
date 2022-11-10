import json

annotations = {}

with open('res/cath/raw/cath-domain-list.txt', 'r') as f:
    for line in f:
        if line[0] != '#':
            words = line.split()
            annotations[words[0]] = [int(words[1]), int(words[2]), int(words[3]), int(words[4])]

for src, dst in [('cath_v430_trainS95_nrTopoBetween.fa', 'train/'),
                 ('cath_v430_finalVal_nrTopoBetween_nrHomoWithin.fa', 'val/'),
                 ('cath_v430_finalTest_nrTopoBetween_nrHomoWithin.fa', 'test/')]:

    seqs_with_anotation = []
    seqs_without_anotation = []
    
    with open('res/cath/raw/' + src, 'r') as f:
        lines = f.readlines()

        for i in range(0, len(lines), 2):
            id = lines[i][1:-1]
            
            if id in annotations:
                seqs_with_anotation.append({
                    'id' : id,
                    'annotation' : annotations[id],
                    'seq' : lines[i + 1][:-1]
                })
            else:
                seqs_without_anotation.append({
                    'id' : id,
                    'seq' : lines[i + 1][:-1]
                })

        with open('res/cath/' + dst + 'seqs_with_annotation.json', 'w') as f2:
            json.dump(seqs_with_anotation, f2)

        with open('res/cath/' + dst + 'seqs_without_annotation.json', 'w') as f2:
            json.dump(seqs_without_anotation, f2)