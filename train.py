import os
import random
import argparse
import math
import pickle
import time
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import SchedulerType, AdamW, get_scheduler
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel
from torch.nn.utils.rnn import pad_sequence

from TCPGen.utils import BiasingProcessor
from TCPGen.TCPGen import LlamaBiasing

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

random.seed(1)
torch.manual_seed(1)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

## Parameter groups
parser = argparse.ArgumentParser(description="LLM finetuning")
parser.add_argument(
    "--model_path",
    type=str,
    default="./hf_models",
    help="Path to the model file",
)
parser.add_argument(
    "--train_data_path",
    type=str,
    default="./hf_models",
    help="Path to the train data file",
)
parser.add_argument(
    "--val_data_path",
    type=str,
    default="./hf_models",
    help="Path to the val data file",
)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="Path to the saved checkpoint",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=1,
    help="Batch size (per device) for the evaluation dataloader.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-5,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--lr_scheduler_type",
    type=SchedulerType,
    default="linear",
    help="The scheduler type to use.",
    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
)
parser.add_argument(
    "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
)
parser.add_argument(
    "--logfile",
    type=str,
    default='./log.txt',
    help="Path to the log file",
)
parser.add_argument(
    "--outputdir",
    type=str,
    default='./exp/clip_vlm',
    help="Path to the output dir",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="log interval",
)
parser.add_argument(
    "--topn",
    type=int,
    default=1,
    help="Top n from the list to use",
)
parser.add_argument(
    "--ontology",
    type=str,
    default="",
    help="KB for biasing",
)
parser.add_argument(
    "--maxKBsize",
    type=int,
    default=10,
    help="Size of the biasing list to use",
)
parser.add_argument(
    "--KBdrop",
    type=float,
    default=0.5,
    help="Drop ratio for true biasing entities",
)
parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="Schema config",
)
args = parser.parse_args()


def logging(s, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(args.logfile, 'a+') as f_log:
            f_log.write(s + '\n')

## Meta data
with open("dataset/slotlist.txt") as fin:
    slots = "".join(fin.readlines())

with open("dataset/slotlist.json") as fin:
    slotlist = json.load(fin)
    slotlist_str = ",".join(list(slotlist.keys()))

## Initialise dataloaders
trainloader = []
validloader = []
with open(args.train_data_path) as fin:
    utterances = json.load(fin)
    for utterance in utterances["data"]:
        label = {}
        values = []
        for ent in utterance["entities"]:
            if ent["type"] in label:
                label[ent["type"]] += " & " + ent["value"]
            else:
                label[ent["type"]] = ent["value"]
            if ent["value"] not in values:
                values.append(ent["value"])
        labelstr = json.dumps(label)
        if "nbest" in utterance:
            utterance["nbest"].append(utterance["text"])
            trainloader.append((utterance["nbest"], labelstr, values))
        else:
            trainloader.append((utterance["text"], labelstr, values))

with open(args.val_data_path) as fin:
    utterances = json.load(fin)
    for utterance in utterances["data"]:
        label = {}
        values = []
        for ent in utterance["entities"]:
            if ent["type"] in label:
                label[ent["type"]] += " & " + ent["value"]
            else:
                label[ent["type"]] = ent["value"]
            if ent["value"] not in values:
                values.append(ent["value"])
        labelstr = json.dumps(label)
        if "nbest" in utterance:
            validloader.append((utterance["nbest"], labelstr, values))
        else:
            validloader.append((utterance["text"], labelstr, values))
trainsize = len(trainloader)

## Initialise models
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
model = model.to(device)
if args.model_path != "gpt2":
    # Use LoRA
    if args.resume != "":
        print("Resuming from {}".format(args.resume))
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

## Initialise DKI
if args.ontology != "":
    with open(args.ontology) as fin:
        ontology = json.load(fin)
# usebias = True if args.ontology != "" else False
# if usebias:
#     biasingproc = BiasingProcessor(tokenizer, args.ontology, args.maxKBlen, args.KBdrop)
    # tcpgen = LlamaBiasing(tokenizer, model.config.embdim, model.config.hiddim, 256)

## Initialise criterion and optimiser
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

##########################
# Optimiser
##########################
# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

# Scheduler and math around the number of training steps.
num_update_steps_per_epoch = math.ceil(len(trainloader) / args.gradient_accumulation_steps)
max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=max_train_steps,
)

## Initialise base prompts
if "noschema" in args.tag:
    # system = "You will be presented with sentences. Please extract slot and values in JSON format."
    system = "You will be presented with a JSON list of slot types, and you will be presented with sentences. Please extract slot and values in JSON format."
    user1 = f"Consider the following list of slot types provided to you as a json list:\n{slotlist_str}\n"
    user2 = "Consider the following sentence(s) containing one or more of the above slot types. Can you extract slots belonging to that slot list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot.\n"
else:
    system = "You will be presented with a JSON list of slot types, and you will be presented with sentences. Please extract slot and values in JSON format."
    user1 = f"Consider the following list of slot types provided to you as a json list:\n{slots}\n"
    user2 = "Now consider the following sentence(s) containing one or more of the above slot types. Can you extract slots belonging to that slot list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot.\n"
user2hyp = "Now consider the following transcription from an ASR system containing one or more of the above slot types. Can you extract slots belonging to that slot list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot.\n"
systemnbest = "You will be presented with a list of hypotheses from an ASR system for one utterance. Please extract slot and values of that utterance in JSON format."
user2nbest = "Now consider the following candidate hypotheses containing one or more of the above slot types. Can you extract slots belonging to that list and their values in json format i.e. \{\"slot type\": \"value\"\}? ONLY print out the json, or only print \{\} if no slot.\n"

bestvalloss = 100000

for epoch in range(args.num_train_epochs):
    model.train()
    random.shuffle(trainloader)
    optimizer.zero_grad()
    start = time.time()
    for batch, i in enumerate(range(0, len(trainloader), args.batch_size)):
        samples = trainloader[i:i+args.batch_size]
        totalinput = []
        totallabel = []
        totalvalues = []
        for sample in samples:
            content, label, values = sample
            totalvalues.extend(values)
            if isinstance(content, list):
                random.shuffle(content)
                if args.topn == 1:
                    user3 = f"\"{content[0]}\""
                    user = f"{user1}{user2}{user3}"
                    prompt = f"{system}\n###Human: {user}\n###Assistant: "
                else:
                    user3 = "\n".join([utt for utt in content[:args.topn]])
                    user = f"{user1}{user2nbest}{user3}"
                    prompt = f"{systemnbest}\n###Human: {user}\n###Assistant: "
            else:
                user3 = f"\"{content}\""
                if args.model_path == "gpt2":
                    prompt = f"###Human: {user3}\n###Assistant: "
                else:
                    user = f"{user1}{user2}{user3}"
                    prompt = f"{system}\n###Human: {user}\n###Assistant: "
            prompt_inputs = tokenizer(prompt, return_tensors="pt")
            label_ids = tokenizer(label + "</s>", return_tensors="pt")
            label_ids = label_ids["input_ids"][0] if args.model_path == "gpt2" else label_ids["input_ids"][0, 1:]
            totalinput.append(torch.cat([prompt_inputs["input_ids"][0], label_ids]))
            totallabel.append(torch.cat([prompt_inputs["input_ids"][0]*0-1, label_ids]))
        totalinput = pad_sequence(totalinput, batch_first=True, padding_value=0).to(device)
        totallabel = pad_sequence(totallabel, batch_first=True, padding_value=-1).to(device)
        # if usebias:
        #     lextree = biasingproc.get_lextree(totalvalues)
        attnmask = totalinput != 0
        inputs = {"input_ids": totalinput[:, :-1], "attention_mask": attnmask[:, :-1]}
        output = model(**inputs, return_dict=True)
        logits = output.logits
        loss = criterion(logits.view(-1, logits.size(-1)), totallabel[:, 1:].reshape(-1))
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        if (batch+1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if (batch+1) % args.log_interval == 0:
            elasped_time = time.time() - start
            PPL = math.exp(loss.item() * args.gradient_accumulation_steps)
            logging(f"Epoch {epoch} | Batch {i}/{trainsize} | PPL: {PPL} | time {elasped_time}")

    # Evaluation starts
    model.eval()
    with torch.no_grad():
        total_tokens = 0
        total_loss = 0.
        for batch, i in enumerate(range(0, len(validloader), args.eval_batch_size)):
            samples = validloader[i:i+args.batch_size]
            totalinput = []
            totallabel = []
            for sample in samples:
                content, label, values = sample
                if isinstance(content, list):
                    if args.topn == 1:
                        user3 = f"\"{content[0]}\""
                        user = f"{user1}{user2}{user3}"
                        prompt = f"{system}\n###Human: {user}\n###Assistant: "
                    else:
                        user3 = "\n".join([utt for utt in content[:args.topn]])
                        user = f"{user1}{user2nbest}{user3}"
                        prompt = f"{systemnbest}\n###Human: {user}\n###Assistant: "
                else:
                    user3 = f"\"{content}\""
                    if args.model_path == "gpt2":
                        prompt = f"###Human: {user3}\n###Assistant: "
                    else:
                        user = f"{user1}{user2}{user3}"
                        prompt = f"{system}\n###Human: {user}\n###Assistant: "
                prompt_inputs = tokenizer(prompt, return_tensors="pt")
                label_ids = tokenizer(label + "</s>", return_tensors="pt")
                label_ids = label_ids["input_ids"][0] if args.model_path == "gpt2" else label_ids["input_ids"][0, 1:]
                totalinput.append(torch.cat([prompt_inputs["input_ids"][0], label_ids]))
                totallabel.append(torch.cat([prompt_inputs["input_ids"][0]*0-1, label_ids]))
            totalinput = pad_sequence(totalinput, batch_first=True, padding_value=0).to(device)
            totallabel = pad_sequence(totallabel, batch_first=True, padding_value=-1).to(device)
            attnmask = totalinput != 0
            inputs = {"input_ids": totalinput[:, :-1].to(device), "attention_mask": attnmask[:, :-1].to(device)}
            output = model(**inputs, return_dict=True)
            logits = output.logits
            loss = criterion(logits.view(-1, logits.size(-1)), totallabel[:, 1:].reshape(-1))
            tokens = (totallabel != -1).sum()
            total_tokens += tokens
            total_loss += loss.item() * tokens
        val_loss = total_loss / total_tokens
        val_ppl = math.exp(val_loss)
        logging(f"Epoch {epoch} | Validation PPL: {val_ppl}")
    if val_loss < bestvalloss:
        logging(f"Saving best model at Epoch {epoch}")
        # torch.save(model.state_dict(), os.path.join(args.outputdir, f"snapshot.ep.{epoch}"))
        if args.model_path != "gpt2":
            model.save_pretrained(args.outputdir)
        else:
            torch.save(model.state_dict(), os.path.join(args.outputdir, "checkpoint.best".format(epoch)))
            torch.save(model.state_dict(), os.path.join(args.outputdir, "checkpoint.ep.{}".format(epoch)))
        bestvalloss = val_loss
    logging("Current learning rate {}".format(optimizer.param_groups[0]["lr"]))
