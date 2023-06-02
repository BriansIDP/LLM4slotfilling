. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate espnet
python evaluate.py -g ../testlabel_norm.jsonl -p ../output_llama13bgt_tuned_2000samplesnoschema.jsonl
