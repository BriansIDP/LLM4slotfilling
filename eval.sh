. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate llama

asrname="gt"
asrfile="dataset/${asrname}_nbest_sel.json"
nsamples=30000
peftpath="exp/slurp_llama13b_baseline_${nsamples}_samples"
# peftpath=""
logfile="$peftpath/log.txt"
# logfile=log.txt
python inference.py \
    --model_name llama13b \
    --peftmodelpath $peftpath \
    --recogfile $asrfile \
    --topn 1 \
    --samples ${nsamples} \
    --asrname ${asrname} \
    --logfile $logfile \
    # --tag noschema \
