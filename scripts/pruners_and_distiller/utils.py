import torch
import torch.nn.functional as F
import logging
import os, json
logger = logging.getLogger(__name__)

def select_logits_with_mask(logits_list, masks_list):
    output_logits = []
    if len(masks_list)==len(logits_list):
        for logits,mask in zip(logits_list,masks_list):
            if len(logits.shape)==3:
                mask = mask.unsqueeze(-1).expand_as(logits).to(torch.bool)
                logits_select = torch.masked_select(logits,mask).view(-1,logits.size(-1))
            else:
                logits_select = logits #Logits_mask has no effect on logits of shape (batch_size, logits_to_be_softmaxed)
            output_logits.append(logits_select)
    elif len(masks_list)==1:
        mask = masks_list[0]
        for logits in logits_list:
            if len(logits.shape)==3:
                mask = mask.unsqueeze(-1).expand_as(logits).to(torch.bool)
                logits_select = torch.masked_select(logits,mask).view(-1,logits.size(-1))
            else:
                logits_select = logits #Logits_mask has no effect on logits of shape (batch_size, logits_to_be_softmaxed)
            output_logits.append(logits_select)
    else:
        raise AssertionError("lengths of logits list and masks list mismatch")
    return output_logits

def kd_ce_loss(logits_S, logits_T, temperature=1):
    '''
    Calculate the cross entropy between logits_S and logits_T

    :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
    :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
    '''
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss

def hid_mse_loss(state_S, state_T, mask=None):
    '''
    * Calculates the mse loss between `state_S` and `state_T`, which are the hidden state of the models.
    * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
    * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.

    :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*)
    :param torch.Tensor mask:    tensor of shape  (*batch_size*, *length*)
    '''
    if mask is None:
        loss = F.mse_loss(state_S, state_T)
    else:
        mask = mask.to(state_S)
        valid_count = mask.sum() * state_S.size(-1)
        loss = (F.mse_loss(state_S, state_T, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count
    return loss

class DistillationContext:
    def __init__(self):
        self.model = self.model_S = None
        self.model_T = None
    def __enter__(self):
        if isinstance(self.model_T,(list,tuple)):
            self.model_T_is_training = [model_t.training for model_t in self.model_T]
            for model_t in self.model_T:
                model_t.eval()
        elif isinstance(self.model_T,dict):
            self.model_T_is_training = {name:model.training for name,model in self.model_T.items()}
            for name in self.model_T:
                self.model_T[name].eval()
        else:
            self.model_T_is_training = self.model_T.training
            self.model_T.eval()

        if isinstance(self.model_S,(list,tuple)):
            self.model_S_is_training = [model_s.training for model_s in self.model_S]
            for model_s in self.model_S:
                model_s.train()
        elif isinstance(self.model_S,dict):
            self.model_S_is_training = {name:model.training for name,model in self.model_S.items()}
            for name in self.model_S:
                self.model_S[name].train()
        else:
            self.model_S_is_training = self.model_S.training
            self.model_S.train()

    def __exit__(self, exc_type, exc_val, exc_tb):
        #Restore model status
        if isinstance(self.model_T,(list,tuple)):
            for i in range(len(self.model_T_is_training)):
                self.model_T[i].train(self.model_T_is_training[i])
        elif isinstance(self.model_T,dict):
            for name,is_training  in self.model_T_is_training.items():
                self.model_T[name].train(is_training)
        else:
            self.model_T.train(self.model_T_is_training)

        if isinstance(self.model_S,(list,tuple)):
            for i in range(len(self.model_S_is_training)):
                self.model_S[i].train(self.model_S_is_training[i])
        elif isinstance(self.model_S,dict):
            for name,is_training  in self.model_S_is_training.items():
                self.model_S[name].train(is_training)
        else:
            self.model_S.train(self.model_S_is_training)


class Config:
    """Base class for TrainingConfig and DistillationConfig."""
    def __init__(self,**kwargs):
        pass

    @classmethod
    def from_json_file(cls, json_filename):
        """Construct configurations from a json file."""
        with open(json_filename,'r') as f:
            json_data = json.load(f)
        return cls.from_dict(json_data)

    @classmethod
    def from_dict(cls, dict_object):
        """Construct configurations from a dict."""
        config = cls(**dict_object)
        return config

    def __str__(self):
        str = ""
        for k,v in self.__dict__.items():
            str += f"{k} : {v}\n"
        return str

    def __repr__(self):
        classname = self.__class__.__name__
        return classname +":\n"+self.__str__()


class TrainingConfig(Config):
    def __init__(self,gradient_accumulation_steps = 1,
                 ckpt_frequency = 1,
                 ckpt_epoch_frequency = 1,
                 ckpt_steps = None,
                 log_dir = None,
                 output_dir = './saved_models',
                 device = 'cuda',
                 fp16 = False,
                 fp16_opt_level = 'O1',
                 data_parallel = False,
                 local_rank = -1
                 ):
        super(TrainingConfig, self).__init__()

        self.gradient_accumulation_steps =gradient_accumulation_steps
        self.ckpt_frequency = ckpt_frequency
        self.ckpt_epoch_frequency = ckpt_epoch_frequency
        self.ckpt_steps = ckpt_steps
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.device = device
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.data_parallel = data_parallel

        self.local_rank = local_rank
        if self.local_rank == -1 or torch.distributed.get_rank() == 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)


class DistillationConfig(Config):

    def __init__(self,temperature=4,
                 hard_label_weight=0,
                 kd_loss_weight=1,
                 matching_layers = None):
        super(DistillationConfig, self).__init__()

        self.temperature = temperature
        self.hard_label_weight = hard_label_weight
        self.kd_loss_weight = kd_loss_weight
        self.matching_layers = matching_layers


#------pruning related---------#

class PruningConfig(Config):
    def __init__(self, 
                start_pruning_at = 0.2,
                start_weights_ratio = 1.0,
                end_pruning_at = 0.7,
                end_weights_ratio = 0.33,
                pruning_frequency = 50,
                IS_alpha = 0.0001, # the dgree that biases towards pruning whole head
                IS_alpha_head = None,
                IS_alpha_ffn = None,
                IS_alpha_mha = None,
                IS_gamma = 1,
                IS_beta = None,
                is_global = False,
                is_reweight = 0,
                is_two_ratios = False,
                FFN_weights_ratio = None,
                MHA_weights_ratio = None,
                score_type = 'grad',
                dbw = True,
                pruner_type = "Pruner",
                dynamic_head_size = False
                ):
        super(PruningConfig, self).__init__()

        self.start_pruning_at =start_pruning_at
        self.start_weights_ratio = start_weights_ratio
        self.end_pruning_at = end_pruning_at
        self.end_weights_ratio = end_weights_ratio
        self.pruning_frequency = pruning_frequency
        self.IS_alpha = IS_alpha 
        self.IS_alpha_head = IS_alpha_head
        self.IS_alpha_ffn = IS_alpha_ffn
        self.IS_alpha_mha = IS_alpha_mha
        self.IS_beta = IS_beta
        self.IS_gamma = IS_gamma
        self.is_global = is_global
        self.is_reweight = is_reweight
        self.pruner_type = pruner_type
        self.is_two_ratios = is_two_ratios
        self.FFN_weights_ratio = FFN_weights_ratio
        self.MHA_weights_ratio = MHA_weights_ratio
        self.score_type = score_type
        self.dbw = dbw
        self.dynamic_head_size = dynamic_head_size


def schedule_threshold(
    step: int,
    total_step: int,
    p_config : PruningConfig,
    overwrite_end_ratio : float = None 
):
    start_pruning_steps = int(p_config.start_pruning_at * total_step)
    end_pruning_steps = int(p_config.end_pruning_at * total_step)
    start_weights_ratio = p_config.start_weights_ratio
    end_weights_ratio = p_config.end_weights_ratio if overwrite_end_ratio is None else overwrite_end_ratio
    if step <= start_pruning_steps:
        weights_ratio = p_config.start_weights_ratio
    elif step > end_pruning_steps:
        weights_ratio = p_config.end_weights_ratio if overwrite_end_ratio is None else overwrite_end_ratio
    else:
        mul_coeff = 1 - (step - start_pruning_steps) / (end_pruning_steps - start_pruning_steps)
        weights_ratio = end_weights_ratio + (start_weights_ratio - end_weights_ratio) * (mul_coeff**3)
    return weights_ratio


def show_masks(state_dict):

    if 'bert.encoder.layer.0.attention.self.query.bias_mask' in state_dict:
        print("=====VO======")
        qk_mask_list = torch.stack([state_dict[f'bert.encoder.layer.{i}.attention.self.query.bias_mask'] for i in range(12)]).int()
        vo_mask_list = torch.stack([state_dict[f'bert.encoder.layer.{i}.attention.self.value.bias_mask'] for i in range(12)]).int()
        qk_head_size_list = [t.reshape(12,64).sum(-1) for t in qk_mask_list]
        vo_head_size_list = vo_mask_list.reshape(12,12,64).sum(-1)
        for i in range(12):
            print(f"{i}: {[i for i in vo_head_size_list[i].tolist() if i >0]}, {vo_head_size_list[i].sum().item()}, {(vo_head_size_list[i]>0).sum().item()}")
        print(f"avg head size: {(vo_head_size_list).sum().item()/(vo_head_size_list>0).sum().item():.2f}")
        print("Total number of heads:",(vo_head_size_list>0).sum().item())
        print("Total number of MHA layer:",(vo_head_size_list.sum(-1)>0).sum().item())

    elif 'bert.encoder.layer.0.attention.output.dense.weight_mask' in state_dict:
        print("=====HEAD=====")
        head_mask_list = torch.stack([state_dict[f'bert.encoder.layer.{i}.attention.output.dense.weight_mask'] for i in range(12)]).int()
        number_heads_per_layer = (head_mask_list[:,0,:].view(12,12,64).sum(-1)==64).sum(-1)
        print("heads per layer:",number_heads_per_layer.tolist())
        print("Total number of heads:",(number_heads_per_layer).sum().item())
        print("Total number of MHA layer:",(number_heads_per_layer>0).sum().item())

    if 'bert.encoder.layer.0.output.dense.weight_mask' in state_dict:
        print("=====FFN======")
        ffn_mask_list = torch.stack([state_dict[f'bert.encoder.layer.{i}.output.dense.weight_mask'][0] for i in range(12)]).int()
        print(f"FFN size/12: {ffn_mask_list.sum(-1).tolist()} {(ffn_mask_list).sum().item()/12:.1f}")
        print("Total number of FFN layers:",(ffn_mask_list.sum(-1)>0).sum().item())

from torch import nn
import types

def feed_forward_chunk_for_empty_ffn(self, attention_output):
        layer_output = self.output(attention_output)
        return layer_output

def output_forward(self, input_tensor):
        #dropped_bias = self.dropout(self.dense.bias)
        #return self.LayerNorm(dropped_bias + input_tensor)
        return self.LayerNorm(self.dense.bias + input_tensor)

def attetion_forward_for_empty_attention(self,
                                        hidden_states,
                                        attention_mask=None,
                                        head_mask=None,
                                        encoder_hidden_states=None,
                                        encoder_attention_mask=None,
                                        past_key_value=None,
                                        output_attentions=False):
    #dropped_bias = self.output.dropout(self.output.dense.bias)
    hidden_states = self.output.LayerNorm(self.output.dense.bias + hidden_states)
    return (hidden_states,)

def transform(model: nn.Module, always_ffn=False, always_mha=False):
    base_model = model.base_model
    bert_layers = base_model.encoder.layer
    for layer in bert_layers:
        output = layer.output
        if always_ffn or output.dense.weight.numel()==0: #empty ffn
            print("replace ffn")
            layer.feed_forward_chunk = types.MethodType(feed_forward_chunk_for_empty_ffn,layer)
            layer.output.forward = types.MethodType(output_forward,layer.output)
        attention_output = layer.attention.output
        if always_mha or attention_output.dense.weight.numel()==0: #empty attention
            print("replace mha")
            layer.attention.forward = types.MethodType(attetion_forward_for_empty_attention,layer.attention)




def fact_embedding_forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]

    seq_length = input_shape[1]

    if position_ids is None:
        position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

    if token_type_ids is None:
        if hasattr(self, "token_type_ids"):
            buffered_token_type_ids = self.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
        inputs_embeds = self.proj(inputs_embeds)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = inputs_embeds + token_type_embeddings
    if self.position_embedding_type == "absolute":
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


def transform_embed(model: nn.Module,dim=128):
    if dim==0:
        return
    print('Word embedding reduced dim:',dim)
    base_model = model.base_model
    embedding_layer = base_model.embeddings
    u,s,v = torch.linalg.svd(embedding_layer.word_embeddings.weight)
    sm = torch.vstack([torch.diag(s),torch.zeros(u.size(0)-s.size(0),s.size(0))])
    sm128=sm[:,:dim]
    v128 = v[:dim]
    reduced_embeddings =u@sm128
    print(reduced_embeddings.shape, v128.shape)

    embedding_layer.proj = torch.nn.Linear(in_features=dim,out_features=embedding_layer.word_embeddings.weight.size(1),bias=None)

    vocab_size = embedding_layer.word_embeddings.num_embeddings
    pad_token_id = embedding_layer.word_embeddings.padding_idx
    embedding_layer.word_embeddings = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
    embedding_layer.word_embeddings.weight.data = reduced_embeddings
    embedding_layer.proj.weight.data = v128.t()
    embedding_layer.forward = types.MethodType(fact_embedding_forward,embedding_layer)
