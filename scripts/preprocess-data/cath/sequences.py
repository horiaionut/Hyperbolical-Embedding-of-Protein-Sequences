import pandas as pd

protein_id_to_number = {}

with open('data/cath/raw/cath-domain-list.txt', 'r') as f:
    for line in f:
        if line[0] != '#':
            words = line.split()
            protein_id_to_number[words[0]] = words[1] + '.' + words[2] + '.' + words[3] + '.' + words[4]

df = pd.read_csv('data/cath/labels.csv', sep=' ')

number_to_label = dict(zip(df.number, df.index))

for src, dst in [
        ('cath_v430_trainS95_nrTopoBetween.fa', 'train.csv'),
        ('cath_v430_finalVal_nrTopoBetween_nrHomoWithin.fa', 'val.csv'),
        ('cath_v430_finalTest_nrTopoBetween_nrHomoWithin.fa', 'test.csv')]:

    with open('data/cath/raw/' + src, 'r') as f:
        seq = {
            "label": [],
            "seq": []
        }    

        lines = f.readlines()

        for i in range(0, len(lines), 2):
            id = lines[i][1:-1]
            number = protein_id_to_number[id]
            protein_seq = lines[i + 1][:-1]

            # reference: https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing#scrollTo=6OC1toF1EM9n
            # repl. all non-standard AAs and map them to unknown/X
            protein_seq = protein_seq.replace('U','X').replace('Z','X').replace('O','X')

            seq['seq'].append(protein_seq)
            seq['label'].append(number_to_label[number])

        df = pd.DataFrame(seq).astype('object')
        df.to_csv('data/cath/' + dst, sep=' ', index=False)