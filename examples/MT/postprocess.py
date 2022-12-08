import click
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.pipeline = [('tagger', nlp.tagger)]

def tokenize_sentence(sentence):
    a = ""
    for token in nlp(sentence):
        a += token.text
        a += ' '
    return a.rstrip()

@click.command()
@click.option('--input_file', required=True, type=str, help='a jsonl file with generations and attribute scores')
@click.option('--output_file', required=True, type=str, help='filename to write outputs')
@click.option('--tokenize', is_flag=True, help='')
def main(input_file, output_file, tokenize):
    with open(input_file) as fin, open(output_file, "w") as fout:
        for l in fin:
            line = l.strip()
            if "<pad>" in line:
                print(line)
                line = line.replace("<pad>", "")
                print(line)

            if tokenize:
                line = tokenize_sentence(line)
                print(line)
            
            fout.write(line+"\n")

if __name__ == '__main__':
    main()  