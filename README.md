# LLM4slotfilling

## Training
Use `train.sh`, specific arguments:
```
--model_path path to your huggingface version of large LM, e.g. llama/

--train_data_path dataset/trainlabel_norm_${nsample}.json, nsample is the number of data samples used for finetuning. 
30000 is the full data set, and I used 100, 500, 2000 etc. 
"norm" means this is text-normalised version to be compatible with ASR

--resume resume from the current saved model. Give the model dir, e.g. exp/slurp_llama13b_baseline_30000_samples/

--tag determine whether you use the task description or not.
currently I kept using task description which is slightly better than not using
Use "noschema" to disable the task description.
```

## Inference
Use `eval.sh`, specific arguments:
```
--model_name llama13b or vicuna13b, and it looks up the table in inference.py and find the pretrained weights
--peftmodelpath the dir of trained LoRA weights, e.g. exp/slurp_llama13b_baseline_30000_samples/
--topn Set it to 1 for the moment, if not 1, it uses nbest lists
--asrname "gt" or "medium", "gt" is groundtruth, "medium" is ASR output from Whisper Medium.en model
--recogfile determined by the asrname
```

## Scoring
The output from `eval.sh` is saved to, e.g. `outputs/output_llama13bgt_tuned_30000samples.json`. See the last block of code in `inference.py`

Then use `python process.py outputs/output_llama13bgt_tuned_30000samples.json` under `outputs/` dir to generate `outputs/output_llama13bgt_tuned_30000samples.jsonl`.

`cd evaluation/`
Modify the second row in `eval.sh` to `python evaluate.py -g ../testlabel_norm.jsonl -p ../<your_output_jsonl_file>`

Then you can read the score for SLU-F1

GPT2 has SLU-F1 of 81
