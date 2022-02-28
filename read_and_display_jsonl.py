import json
import sys
from pprint import pprint 

flag = sys.argv[4] == "only_negative"
flag2 = sys.argv[4] == "only_positive"

with open(sys.argv[1]) as f, open(sys.argv[2]) as f2, open(sys.argv[3]) as f3:
    for l in f:
        output = json.loads(l)
        print(output['prompt']['text'])
        for i, gens in enumerate(output['generations']):
            l2 = eval(f2.readline().strip())
            l3 = eval(f3.readline().strip())
            if flag and l2['label'] == "NEGATIVE":
                print(i, gens['text'].replace("\n", "\\n"), l2['label'], l3['label'])
            if flag2 and l2['label'] == "POSITIVE":
                print(i, gens['text'].replace("\n", "\\n"), l2['label'], l3['label'])
        #        pprint(output)
        input("press any key for next output")

