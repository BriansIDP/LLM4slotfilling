. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate llama

nsample=30000
expdir="exp/slurp_llama13b_baseline_${nsample}_samples"

trainfile=dataset/trainlabel_norm_${nsample}.json
valfile=dataset/validlabel_norm.json

# expdir="exp/debug"
mkdir -p $expdir
python train.py \
    --model_path hf_models_13b \
    --batch_size 2 \
    --eval_batch_size 1 \
    --learning_rate 10e-6 \
    --gradient_accumulation_steps 5 \
    --num_train_epochs 10 \
    --outputdir $expdir \
    --logfile $expdir/log.txt \
    --log_interval 2000 \
    --train_data_path $trainfile \
    --topn 1 \
    --val_data_path $valfile \
    # --resume exp/slurp_llama13b_baseline_2000_samples \
    # --tag noschema \
    # --ontology dataset/KB.json \
    # --maxKBsize 10 \
    # --KBdrop 0.5 \
