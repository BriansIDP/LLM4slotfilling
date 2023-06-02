import json
import re
import sys, os


infile = sys.argv[1]

def norm(val):
    if normalise:
        return normaliser(val)
    else:
        return val

with open(infile) as fin:
    output = json.load(fin)

id2files = {}
with open("testcleanslot.jsonl") as fin:
    for line in fin:
        data = json.loads(line)
        id2files[data["slurp_id"]] = [rec["file"] for rec in data["recordings"]]

with open("../dataset/slotlist.json") as fin:
    slotlist = json.load(fin).keys()

outdict = []
unsolved = []
for slurpid, result in output.items():
    print(slurpid)
    text = result.split("</s>")[0]
    parsed_responses = re.findall("\{[^\{\}]*\}", text)
    slotvalues = {}
    entities = []
    for each_slot in parsed_responses:
        each_slot = each_slot.replace("\\", "")
        if len(each_slot) > 3:
            if each_slot[-2] == ",":
                each_slot = each_slot[:-2] + each_slot[-1]
            elif each_slot[-3] == "," and each_slot[-2] == "\n":
                each_slot = each_slot[:-3] + each_slot[-2:]
        try:
            each_slot_json = json.loads(each_slot)
        except:
            print(each_slot)
            unsolved.append(each_slot)
            each_slot_json = {}
        for key, value in each_slot_json.items():
            if isinstance(value, list):
                if key not in slotvalues and key in slotlist:
                    slotvalues[key] = value
                    for val in value:
                        entities.append({"type": key, "filler": norm(val)})
                elif key in slotvalues and key in slotlist:
                    for val in value:
                        if val not in slotvalues[key]:
                            slotvalues[key].append(val)
                            entities.append({"type": key, "filler": norm(val)})
            elif isinstance(value, str):
                value = value.strip()
                # if value != "" and value != "none" and value is not None:
                if key not in slotvalues and key in slotlist and value != "":
                    if "&" in value:
                        slotvalues[key] = []
                        for val in value.split("&"):
                            entities.append({"type": key, "filler": norm(val)})
                    else:
                        entities.append({"type": key, "filler": norm(value)})
                    slotvalues[key] = [value]
                elif key in slotvalues and key in slotlist and value not in slotvalues[key]:
                    if "&" in value:
                        slotvalues[key] = []
                        for val in value.split("&"):
                            entities.append({"type": key, "filler": norm(val)})
                    else:
                        entities.append({"type": key, "filler": norm(value)})
                    slotvalues[key].append(value)
    for filename in id2files[int(slurpid)]:
        outdict.append({"file": filename, "scenario": "takeaway", "action": "query", "entities": entities})

with open(infile + 'l', "w") as fout:
    for item in outdict:
        json.dump(item, fout)
        fout.write("\n")

with open("unsolved.json", "w") as fout:
    json.dump(unsolved, fout, indent=4)
