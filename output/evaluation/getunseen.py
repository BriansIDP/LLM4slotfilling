import json


testset = []
with open("testcleanslot.jsonl") as fin:
    for line in fin:
        testset.append(json.loads(line))

traindict = set()
with open("/home/gs534/rds/hpc-work/work/slurp/lm/SLURP/wlist.txt") as fin:
    for line in fin:
        traindict.add(line.strip().lower())

newlines = []
for line in testset:
    unseen = False
    for word in line["sentence"].split():
        if word not in traindict:
            unseen = True
    if unseen:
        newlines.append(line)

with open("testunseen.jsonl", "w") as fout:
    for line in newlines:
        json.dump(line, fout)
        fout.write('\n')
