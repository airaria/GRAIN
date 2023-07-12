import logging
# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO,)
logger = logging.getLogger(__name__)

import os,random
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import divide_parameters
from pruners_and_distiller.distiller import TrainingConfig, DistillationConfig, PruningConfig
from pruners_and_distiller.distiller import PruneDistiller as PruneDistillerHidden
from pruners_and_distiller.utils import show_masks, transform_embed
from torch.utils.data import DataLoader, RandomSampler
from functools import partial
from predict_function import predict
from utils_glue import output_modes, get_glue_dataset
from config_prunedistiller import parse_specs, parse_args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def args_check(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.warning(f"Output directory ({args.output_dir}) already exists and is not empty.")
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        if not args.no_cuda and not torch.cuda.is_available():
            raise ValueError("No CUDA available!")
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    else:
    # Initializes the distributed backend which sychronizes nodes/GPUs
        #torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        #torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                     args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    return device, args.n_gpu

def main():
    args = parse_args()
    logger.setLevel(logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    for k,v in vars(args).items():
        logger.info(f"{k}:{v}")

    device, args.n_gpu = args_check(args)
    set_seed(args)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    #Build Model and load checkpoint
    speclist = parse_specs(args.model_spec_file)
    spec = speclist[0]
    config, tokenizer, model_class = spec['config'], spec['tokenizer'], spec['model_class']
    ckpt_file = spec['ckpt_file']
    prefix = spec['prefix']
    model_type = spec['model_type']
    return_token_type_ids = (model_type=='bert')
    print("debug:",return_token_type_ids)



    if args.local_rank == 0:
        torch.distributed.barrier()

    #read data
    train_dataset = None
    eval_datasets  = None
    num_train_steps = None

    eval_langs=train_langs=['en']
    train_dataset = get_glue_dataset(args.taskname, args.data_dir,'train', tokenizer,args.max_seq_length,return_token_type_ids)
    num_labels = train_dataset.features['label'].num_classes
    eval_datasets = []
    split = 'test' if args.do_test else 'validation' 
    if args.taskname=='mnli':
        eval_langs = ['m','mm']
        eval_datasets = [get_glue_dataset(args.taskname, args.data_dir,split+'_matched', tokenizer,args.max_seq_length,return_token_type_ids),
                            get_glue_dataset(args.taskname, args.data_dir,split+'_mismatched', tokenizer,args.max_seq_length,return_token_type_ids)]
    else:
        eval_dataset = get_glue_dataset(args.taskname, args.data_dir,split, tokenizer,args.max_seq_length,return_token_type_ids)
        eval_datasets = [eval_dataset]


    logger.info("Data loaded")

    config.num_labels = num_labels
    if args.output_hidden_states is True:
        config.output_hidden_states=True
    model = model_class.from_pretrained(ckpt_file,config=config)
    state_dict = torch.load(args.teacher_model_path,map_location='cpu')
    if args.transform_embed>0:
        transform_embed(model,args.transform_embed)
    model_T = model_class.from_pretrained(None,config=config,state_dict=state_dict)
    callback_func = None
    if args.do_predict:
        callback_func = partial(predict, eval_datasets=eval_datasets,eval_lang=eval_langs, args=args,taskname=args.taskname)
    if args.do_train:
        forward_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
        args.forward_batch_size = forward_batch_size
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.forward_batch_size,drop_last=True)



        def AdaptorTrain(batch, model_outputs):
            return {'losses':(model_outputs[0],)}
        def AdaptorLogits(batch, model_outputs):
            return {'logits': (model_outputs.logits,)}
        def AdaptorLogitsHidden(batch, model_outputs):
            return {'logits': (model_outputs.logits,),
                    'hidden': (model_outputs.hidden_states),
                    'inputs_mask': batch['attention_mask'],}

        if args.output_hidden_states is True:
            print("use hidden")
            Adaptor = AdaptorLogitsHidden
        else:
            Adaptor = AdaptorLogits
        PruneDistiller = PruneDistillerHidden

        #parameters
        params = list(model.named_parameters())
        frozen_params = []
        if args.freeze_embeddings:
            frozen_params = ['word_embeddings']
        #all_trainable_params = divide_parameters(params, lr=args.learning_rate)
        no_decay = ['bias','LayerNorm.weight']
        large_lr = ['attention_head_scale']
        all_trainable_params = [
            {
                "params":[p for n,p in params if not any(nd in n for nd in no_decay+frozen_params)],
                "weight_decay": args.weight_decay_rate,
            },
            {   
                'params': [p for n,p in params if any(nd in n for nd in no_decay)],
                'weight_decay':0.0
            },
            {
                'params': [p for n,p in params if any(nd in n for nd in frozen_params)], 
                'lr':0.0
            }
        ]
        logger.info("Length of all_trainable_params: %d", len(all_trainable_params))

        ########## PruneDistiller ###########
        train_config = TrainingConfig(
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            ckpt_frequency = args.ckpt_frequency,
            #ckpt_steps = int(num_train_steps//args.num_train_epochs//2),
            log_dir = args.output_dir,
            output_dir = args.output_dir,
            fp16 = args.fp16,
            device = args.device)
        if args.matching_layers_S is not None:
            matching_layers = list(zip(map(int,args.matching_layers_S.split(',')),map(int,args.matching_layers_T.split(','))))
        else:
            matching_layers = None
        distill_config = DistillationConfig(temperature=8,
                                            matching_layers=matching_layers)
        prune_config = PruningConfig(end_pruning_at=args.end_pruning_at, start_pruning_at=args.start_pruning_at,
                                     end_weights_ratio=args.end_weights_ratio,
                                     pruning_frequency=args.pruning_frequency, 
                                     IS_beta=args.IS_beta, is_global=args.is_global, is_reweight=args.is_reweight,
                                     is_two_ratios=args.is_two_ratios,FFN_weights_ratio=args.FFN_weights_ratio,MHA_weights_ratio=args.MHA_weights_ratio,
                                     score_type=args.score_type,
                                     pruner_type=args.pruner_type,
                                     dynamic_head_size=args.dynamic_head_size,
                                     IS_gamma=args.IS_gamma,
                                     IS_alpha=args.IS_alpha,
                                     IS_alpha_head=args.IS_alpha_head,
                                     IS_alpha_ffn=args.IS_alpha_ffn,
                                     IS_alpha_mha=args.IS_alpha_mha,
                                     dbw=(not args.no_dbw)
                                     )
        distiller = PruneDistiller(train_config = train_config, distill_config=distill_config,
                                    prune_config=prune_config, model_T = model_T, model_S = model,
                                 adaptor = Adaptor)
        num_train_steps = int(len(train_dataloader)//args.gradient_accumulation_steps * args.num_train_epochs)
        optimizer = AdamW(all_trainable_params,lr=args.learning_rate,eps=args.adam_epsilon)
        scheduler_args = {'num_warmup_steps': int(args.warmup_proportion*num_train_steps), 
                          'num_training_steps': num_train_steps}
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,**scheduler_args)

        logger.info("***** Running Prune Distiller *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Forward batch size = %d", forward_batch_size)
        logger.info("  Num backward steps = %d", num_train_steps)


        def batch_postprocessor(batch):
            if 'token_type_ids' in batch:
               return {'input_ids':batch['input_ids'],'attention_mask':batch['attention_mask'],'token_type_ids':batch['token_type_ids'],
               'labels':batch['label']}
            else:
               return {'input_ids':batch['input_ids'],'attention_mask':batch['attention_mask'],
               'labels':batch['label']}
        with distiller:
            distiller.train(train_dataloader, optimizer, scheduler, args.num_train_epochs, 
                            max_grad_norm=args.max_grad_norm, callback=callback_func, batch_postprocessor=batch_postprocessor)
        del optimizer
        logger.info("*********************Prune Distiller Finished*****************")

    if not args.do_train and args.do_predict:
        model.to(device)
        res = predict(model,eval_datasets,step=0,eval_lang=eval_langs, args=args,taskname=args.taskname)
        print (res)

    show_masks(model.state_dict())

if __name__ == "__main__":
    main()
