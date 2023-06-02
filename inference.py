import os, sys
import time
import pickle
import json
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import StoppingCriteriaList, StoppingCriteria
from peft import PeftModel, PeftConfig

device = 'cuda:0' if torch.cuda.is_available() else "cpu"
reranking = False
asrname = "gt"

parser = argparse.ArgumentParser(description="LLM finetuning")
parser.add_argument(
    "--peftmodelpath",
    type=str,
    default="",
    help="Path to the model file",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="llama13b",
    help="model name",
)
parser.add_argument(
    "--recogfile",
    type=str,
    default="dataset/gt_nbest_sel.json",
    help="Path to the model file",
)
parser.add_argument(
    "--logfile",
    type=str,
    default="log.output",
    help="Path to the model file",
)
parser.add_argument(
    "--topn",
    type=int,
    default=1,
    help="model name",
)
parser.add_argument(
    "--samples",
    type=int,
    default=2000,
    help="model name",
)
parser.add_argument(
    "--asrname",
    type=str,
    default="gt",
    help="model name",
)
parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="model name",
)
args = parser.parse_args()

finetuned = False
if args.peftmodelpath != "" and os.path.exists(args.peftmodelpath):
    finetuned = True

MODEL_PATHS = {
    'llama': 'hf_models/',
    'alpaca': '/scratch/LLM/LLM.ckpts/alpaca',
    'stablelm': '/home/shutong/LLM.ckpts/stablelm-base-alpha-7b',
    'gpt4all-j': '/home/shutong/LLM.ckpts/gpt4all-j',
    'vicuna13b': './vicuna.13b',
    'llama13b': 'hf_models_13b/',
    'vicuna': './vicuna',
    'stablevicuna': './stable_vicuna',
    'gpt2': 'exp/slurp_gpt2_baseline_{}_samples_r16/checkpoint.best'.format(args.samples),
}


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = '</s>'):
        self.stops = stops
        StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
        return tokenizer.decode(input_ids[0, -5:]).endswith(self.stops)


def logging(s, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

model_path = MODEL_PATHS[args.model_name]

stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops='</s>')])

if args.model_name == "gpt2":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
if finetuned and args.model_name != "gpt2":
    # peft_model_id = PEFT_MODEL_PATHS[model_name]
    config = PeftConfig.from_pretrained(args.peftmodelpath)
    model = PeftModel.from_pretrained(model, args.peftmodelpath)
model = model.to(device)
model.eval()

with open("dataset/slotlist.txt") as fin:
    slots = "".join(fin.readlines())

with open("dataset/slotlist.json") as fin:
    slotlist = json.load(fin)
    slotlist_str = ",".join(list(slotlist.keys()))

with open(args.recogfile) as fin:
    utterances = json.load(fin)

if "noschema" in args.tag:
    system = "You will be presented with sentences. Please extract slot and values in JSON format."
    user1 = ""
elif "listschema" in args.tag:
    system = "You will be presented with a JSON list of slot types, and you will be presented with sentences. Please extract slot and values in JSON format."
    user1 = f"Consider the following list of slot types provided to you as a json list:\n{slotlist_str}\n"
else:
    system = "You will be presented with a JSON list of slot types, and you will be presented with sentences. Please extract slot and values in JSON format."
    user1 = f"Consider the following list of slot types provided to you as a json list:\n{slots}\n"

if finetuned:
    if "noschema" in args.tag:
        user2 = "Consider the following sentence(s) containing one or more slot types. Can you extract slots belonging to that slot list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot.\n"
    elif "listschema" in args.tag:
        user2 = "Consider the following sentence(s) containing one or more of the above slot types. Can you extract slots belonging to that slot list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot.\n"
    else:
        user2 = "Now consider the following sentence(s) containing one or more of the above slot types. Can you extract slots belonging to that slot list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot.\n"
else:
    examples = ""
    if "fewshot" in args.tag:
        with open("dataset/trainlabel_norm_100.json") as fin:
            dataset = json.load(fin)
        examples = []
        for utterance in dataset["data"][:10]:
            label = {}
            for ent in utterance["entities"]:
                if ent["type"] in label:
                    label[ent["type"]] += " & " + ent["value"]
                else:
                    label[ent["type"]] = ent["value"]
            labelstr = json.dumps(label)
            examples.append("given \"{}\", you should extract {}".format(utterance["text"], label))
        examples = "\n".join(examples)
    if "fewshot" in args.tag:
        # user2 = "For example, {}\n".format(examples)
        user2 = ""
        system += "For example, {}\n".format(examples)
    else:
        user2 = "For example, given \"order me chinese food\", you should extract {\"food_type\": \"chinese\"}\n"
    if asrname == "gt":
        user2 += "Now consider the following sentence(s) containing one or more of the above slot types. Can you extract slots belonging to that list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot."
    else:
        user2 += "Now consider the following transcription from an ASR system containing one or more of the above slot types. Can you extract slots belonging to that slot list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot."

if finetuned:
    systemnbest = "You will be presented with a list of hypotheses from an ASR system for one utterance. Please extract slot and values of that utterance in JSON format."
    user2nbest = "Now consider the following candidate hypotheses containing one or more of the above slot types. Can you extract slots belonging to that list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot.\n"
else:
    systemnbest = "You will be presented with a list of hypotheses from an ASR system for one utterance. Please extract slot and values of that utterance in JSON format."
    user2nbest = "Now consider the following candidate hypotheses containing one or more of the above slot types. Can you extract slots belonging to that list and their values in json format i.e. \{\"slot type\": \"value\"\}? For example, given \"order me chinese food\"\n\"order chinese foods\", you should extract {\"food_type\": \"chinese\"}. ONLY print out the json, or only print \{\} if no slot."

# Knowledge injection
if "DKI" in args.tag:
    with open("dataset/ontology_perutt_medium.json") as fin:
        ontology = json.load(fin)
        user1 += "The task is to extract slot from a list of ASR hypothesis. Let's do it step by step."
        user1 += "\n###Assistant: OK###Human: "

print("Start Inference")
start = time.time()
outputdict = {}
count = 0
with torch.no_grad():
    for slurpid, utt in utterances.items():
        if "DKI" in args.tag and ontology[slurpid] != "{}":
            user1comb = "First, possible values for some slots are provided in the following dynamic KB:\n{}\nThe KB is a strong prior, i.e. values not in the KB are very not likely to be the correct value\n".format(ontology[slurpid])
            user2comb = "If, for a slot, you can not find a value in the provided KB, please select the most likely value from KB instead."
        if isinstance(utt, list):
            if args.model_name == "gpt2":
                user3 = f"\"{utt[0]}\""
                prompt = f"###Human: {user3}\n###Assistant: "
            elif args.topn == 1:
                user3 = f"\"{utt[0]}\""
                if "DKI" in args.tag and ontology[slurpid] != "{}":
                    user = f"{user1}{user1comb}{user2}{user2comb}{user3}"
                else:
                    user = f"{user1}{user2}{user3}"
                prompt = f"{system}\n###Human: {user}\n###Assistant: "
            else:
                user3 = "\n".join(utt[:args.topn])
                user = f"{user1}{user2nbest}{user3}"
                prompt = f"{systemnbest}\n###Human: {user}\n###Assistant: "
        else:
            user3 = f"\"{utt}\""
            if args.model_name == "gpt2":
                prompt = f"###Human: {user3}\n###Assistant: "
            else:
                if "DKI" in args.tag and ontology[slurpid] != "{}":
                    user = f"{user1}{user1comb}{user2}{user2comb}{user3}"
                else:
                    user = f"{user1}{user2}{user3}"
                prompt = f"{system}\n###Human: {user}\n###Assistant: "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
        if args.model_name in ['alpaca', 'vicuna', 'llama', 'stablevicuna', 'vicuna13b', 'llama13b']:
            if finetuned:
                generate_ids = model.generate(input_ids=inputs.input_ids, max_new_tokens=100, stopping_criteria=stopping_criteria)
            else:
                generate_ids = model.generate(inputs.input_ids, max_new_tokens=150)
            output = tokenizer.batch_decode(generate_ids[:,inputs.input_ids.size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            output = output.split("###")[0]
        else:
            tokens = model.generate(**inputs, max_new_tokens=100, stopping_criteria=stopping_criteria, pad_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(tokens[0, inputs.input_ids.size(1):], skip_special_tokens=True)
            output = output.split("###")[0]
        outputdict[slurpid] = output
        count += 1
        logging("Finished {}, Elapsed time {:.2f}".format(count, time.time()-start))

with open("outputs/output_{}{}{}{}{}.json".format(args.model_name,
        args.asrname,
        "nbest" if args.topn > 1 else "",
        "_tuned_{}samples".format(args.samples) if finetuned else "",
        args.tag
    ), "w") as fout:
    json.dump(outputdict, fout, indent=4)
