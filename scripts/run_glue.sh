TASK=sst2

OUTPUT_ROOT_DIR=pruned_models
DATA_ROOT_DIR=/path/to/glue/datasets

teacher_model_path=teacher_models/${TASK}/pytorch_model.bin

IS_alpha_head=3e-1
accu=1
ngpu=1
batch_size=32
length=128
ep=20
lr=3
seed=1337
weights_ratio=05

taskname=${TASK}
DATA_DIR=${DATA_ROOT_DIR}/${taskname}
end_at=0.4
pf=1
IS_beta=0.998
embsize=192
NAME=lr${lr}e${ep}_s${i}_bs${batch_size}_${end_at}_pf${pf}_IS${IS_beta}_Reg${IS_alpha_head}_E${embsize}
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/bert/${taskname}-${weights_ratio}/${NAME}

mkdir -p $OUTPUT_DIR
model_config_json_file=bert_base_uncased.json

python -u glue/main.py \
    --data_dir  $DATA_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length ${length} \
    --train_batch_size ${batch_size} \
    --seed $seed \
    --num_train_epochs ${ep} \
    --learning_rate ${lr}e-5 \
    --ckpt_frequency 2 \
    --output_dir $OUTPUT_DIR \
    --gradient_accumulation_steps ${accu} \
    --taskname ${taskname} \
    --model_spec_file ${model_config_json_file} \
    --teacher_model_path $teacher_model_path \
    --max_grad_norm 1 \
    --fp16 \
    --end_pruning_at ${end_at} \
    --end_weights_ratio 0.${weights_ratio} \
    --pruning_frequency ${pf} \
    --IS_beta ${IS_beta} \
    --is_global \
    --output_hidden_states \
    --matching_layers_S 0,2,4,6,8,10,12 \
    --matching_layers_T 0,2,4,6,8,10,12 \
    --IS_alpha_head ${IS_alpha_head} \
    --pruner_type FineISPruner \
    --transform_embed ${embsize} \
