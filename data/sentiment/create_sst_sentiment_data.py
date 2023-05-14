"""
construct sentiment data from SST-5 (script borrowed from: https://github.com/alisawuffles/DExperts/blob/main/scripts/data/create_sst_sentiment_data.py)
"""

import pytreebank
from sacremoses import MosesDetokenizer
import sys 

datadir = sys.argv[1]
md = MosesDetokenizer(lang="en")

dataset = pytreebank.load_sst()

examples = dataset['train'] 
trainf = [open(f"{datadir}/train_{i}.txt", "w") for i in range(2)]
label_count = [0 for _ in range(2)]
for ex in examples:
    for label, sentence in ex.to_labeled_lines()[:1]:    
        # sentence = md.detokenize(sentence.strip().split())
        if label in [3,4]: #positive
            trainf[1].write(sentence+"\n")
            label_count[1] += 1
        elif label in [0, 1]: #negative or neutral
            trainf[0].write(sentence+"\n")
            label_count[0] += 1
for f in trainf:
    f.close()
print(label_count)

examples = dataset['dev'] 
devf = [open(f"{datadir}/dev_{i}.txt", "w") for i in range(2)]
label_count = [0 for _ in range(2)]
for ex in examples:
    for label, sentence in ex.to_labeled_lines()[:1]:    
        # sentence = md.detokenize(sentence.strip().split())
        if label in [3,4]: #positive
            devf[1].write(sentence+"\n")
            label_count[1] += 1
        elif label in [0, 1]: #negative
            devf[0].write(sentence+"\n")
            label_count[0] += 1
        
for f in devf:
    f.close()
print(label_count)

examples = dataset['test'] 
testf = [open(f"{datadir}/test_{i}.txt", "w") for i in range(2)]
label_count = [0 for _ in range(2)]
for ex in examples:
    for label, sentence in ex.to_labeled_lines()[:1]:    
        # sentence = md.detokenize(sentence.strip().split())
        if label in [3,4]: #positive
            testf[1].write(sentence+"\n")
            label_count[1] += 1
        elif label in [0, 1]: #negative
            testf[0].write(sentence+"\n")
            label_count[0] += 1
for f in testf:
    f.close()
print(label_count)