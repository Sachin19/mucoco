from datasets import load_dataset
import sys

datadir = sys.argv[1]
dataset = load_dataset("yelp_polarity")


ftrain = [open(f"{datadir}/train_0.txt", "w"), open(f"{datadir}/train_1.txt", "w")]
#fdev = [open("dev_0.txt", "w"), open("dev_1.txt", "w")]
ftest = [open(f"{datadir}/test_0.txt", "w"), open(f"{datadir}/test_1.txt", "w")]

for text, label in zip(dataset['train']['text'], dataset['train']['label']):
    ftrain[label].write(text+"\n")

#for text, label in zip(dataset['dev']['text'], dataset['dev']['label']):
#    fdev[label].write(text+"\n")

for text, label in zip(dataset['test']['text'], dataset['test']['label']):
    ftest[label].write(text+"\n")

ftrain[0].close()
ftrain[1].close()

#fdev[0].close()
#fdev[1].close()

ftest[0].close()
ftest[1].close()
