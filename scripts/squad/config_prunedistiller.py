import argparse
import json

from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering
MODEL_CLASSES = {
  "bert": (BertConfig, BertTokenizer, BertForQuestionAnswering ),
}

def parse_specs(speclist): # list of specifications
    if isinstance(speclist,str):
        with open(speclist,'r') as f:
            speclist = json.load(f)
    else:
        assert isinstance(speclist,dict)
    for item in speclist:
        model_type = item['model_type']
        config_class, tokenizer_class, model_class = MODEL_CLASSES[model_type]

        item['model_class'] = model_class

        if item['config_file'] is not None:
            config = config_class.from_json_file(item['config_file'])
        else:
            config = None
        item['config'] = config

        if item['vocab_file'] is not None:
            kwargs = item.get('tokenizer_kwargs',{})
            tokenizer = tokenizer_class(vocab_file=item['vocab_file'],**kwargs)
        else:
            tokenizer= None
        item['tokenizer'] = tokenizer

    return speclist

def parse_args(opt=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--max_seq_length", default=384, type=int)
    parser.add_argument("--max_query_length",default=64, type=int)
    parser.add_argument("--max_answer_length",default=30,type=int)
    parser.add_argument("--do_train", default=False, action='store_true')
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precisoin instead of 32-bit")

    parser.add_argument('--seed',type=int,default=10236797)
    parser.add_argument('--weight_decay_rate',type=float,default=0.01)
    parser.add_argument('--do_eval',action='store_true')
    parser.add_argument('--do_test',action='store_true')
    parser.add_argument('--PRINT_EVERY',type=int,default=200)
    parser.add_argument('--ckpt_frequency',type=int,default=2)

    parser.add_argument('--model_spec_file',type=str)
    parser.add_argument('--teacher_model_path',type=str)
    parser.add_argument('--max_grad_norm',type=float,default=1.0)

    parser.add_argument('--n_best_size',default=20,type=int)
    parser.add_argument('--doc_stride',default=128,type=int)
    parser.add_argument("--null_score_diff_threshold",type=float,default=0.0)
    parser.add_argument("--version_2_with_negative",action='store_true')
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--do_lower_case",action='store_true') #used in decoding?
    parser.add_argument("--adam_epsilon",default=1e-6,type=float)
    parser.add_argument("--is_save_logits",action='store_true')

    parser.add_argument("--end_pruning_at",default=0.7,type=float)
    parser.add_argument("--start_pruning_at",default=0.2,type=float)

    parser.add_argument("--end_weights_ratio",default=0.33,type=float)
    parser.add_argument("--pruning_frequency",default=50,type=int)
    parser.add_argument("--pruner_type",default="Pruner",type=str)
    parser.add_argument("--IS_beta",default=0.99,type=float)
    parser.add_argument("--is_global",action='store_true')
    parser.add_argument("--is_reweight",type=float,default=1)
    parser.add_argument("--is_two_ratios",action='store_true')
    parser.add_argument("--FFN_weights_ratio",default=None,type=float)
    parser.add_argument("--MHA_weights_ratio",default=None,type=float)
    parser.add_argument("--score_type",default='grad',type=str,choices=['grad','magnitude-sumabs','magnitude-Linf','magnitude-L1','random'])
    parser.add_argument("--output_hidden_states",action='store_true')
    parser.add_argument("--dynamic_head_size",action='store_true')

    parser.add_argument("--matching_layers_S",type=str,default=None)
    parser.add_argument("--matching_layers_T",type=str,default=None)

    parser.add_argument("--IS_gamma",default=0,type=float)
    parser.add_argument("--IS_alpha",default=0.0001,type=float)
    parser.add_argument("--IS_alpha_head",default=None,type=float)
    parser.add_argument("--IS_alpha_ffn",default=None,type=float)
    parser.add_argument("--IS_alpha_mha",default=None,type=float)
    parser.add_argument("--no_dbw",action='store_true')
    parser.add_argument("--transform_embed",default=0,type=int)

    global args
    if opt is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(opt)
    return args

if __name__ == '__main__':
    print (args)
    parse_args(['--SAVE_DIR','test'])
    print(args)
