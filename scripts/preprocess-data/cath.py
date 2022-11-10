import pandas as pd

annotations = {}

with open('res/cath/raw/cath-domain-list.txt', 'r') as f:
    for line in f:
        if line[0] != '#':
            words = line.split()
            annotations[words[0]] = [int(words[1]), int(words[2]), int(words[3]), int(words[4])]

# TRAIN

# all sequences in this file are annotated
with open('res/cath/raw/cath_v430_trainS95_nrTopoBetween.fa', 'r') as f:
    annotated_seqs = {
        "id": [],
        "annotation": [],
        "seq": []
    }

    lines = f.readlines()

    for i in range(0, len(lines), 2):
        id = lines[i][1:-1]

        annotated_seqs['id'].append(id)
        annotated_seqs['annotation'].append(annotations[id])
        annotated_seqs['seq'].append(lines[i + 1][:-1])

    df = pd.DataFrame(annotated_seqs) 
    df.to_csv('res/cath/train/annotated_seqs.csv')


# VALIDATION, TEST

# all sequences in these files are annotated            
for src, dst in [('cath_v430_finalVal_nrTopoBetween_nrHomoWithin.fa', 'val.csv'),
                 ('cath_v430_finalTest_nrTopoBetween_nrHomoWithin.fa', 'test.csv')]:
    
    with open('res/cath/raw/' + src, 'r') as f:
        lines = f.readlines()

        annotated_seqs = {
            "id": [],
            "annotation": []
        }

        for i in range(0, len(lines), 2):
            id = lines[i][1:-1]

            annotated_seqs['id'].append(id)
            annotated_seqs['annotation'].append(annotations[id])
            
        df = pd.DataFrame(annotated_seqs) 
        df.to_csv('res/cath/' + dst)