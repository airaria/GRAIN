import torch
from torch import nn
import os

from torch.nn.functional import softmax, log_softmax
from ..pruners.utils import move_to_device, generate_mask, infer_model_type
from ..pruners.utils import infer_logits, infer_loss
from ..configurations import GeneralConfig

from ..model_map import MODEL_MAP
import logging
from tqdm import tqdm
from collections import abc
from typing import Mapping, Optional, List
from copy import deepcopy

logger = logging.getLogger(__name__)

from .configurations import  FineGrainedPruningConfig




class FineGrainedPruner:
    def __init__(self, model : nn.Module, 
                       finegrained_pruning_config : Optional[FineGrainedPruningConfig] = None,
                       general_config : Optional[GeneralConfig] = None,
                       base_model_prefix : Optional[str] = None):
        self.model = model
        base_model, model_type = infer_model_type(model, base_model_prefix)
        assert model_type in MODEL_MAP, \
            f"Model type {self.model_type} is not supported, or not understood. Model type must be one of {list(MODEL_MAP.keys())}"
        self.base_model = base_model
        self.model_type = model_type
        self.model_structure = MODEL_MAP[self.model_type]['structure']

        self.general_config = GeneralConfig() if general_config is None else general_config
        self.finegrained_pruning_config = FineGrainedPruningConfig() if finegrained_pruning_config is None else finegrained_pruning_config

        self.model.to(self.general_config.device)

        self.output_dir : str = self.general_config.output_dir

        # None before pruning
        self.QK_mask_list : Optional[List[torch.Tensor]] = None # n_layers * (head_size,)
        self.VO_mask_list : Optional[List[torch.Tensor]] = None # n_layers * (head_size,)
        self.keep_shape : Optional[bool] = None
        os.makedirs(self.output_dir, exist_ok=True)

        self.shoule_cache_logits = True
        self.soft_labels = []
        if self.finegrained_pruning_config.use_logits is True:
            self.model_rep = deepcopy(model)
            self.model_rep.half().to(model.device)
        self.save_dir = None

    def prune(self, dataloader=None, adaptor=None, batch_postprocessor=None, 
                QK_mask_list: Optional[List[torch.Tensor]] =None, VO_mask_list: Optional[List[torch.Tensor]]=None, 
                keep_shape=False, save_model=True, rewrite_cache=True):

        pruning_method = self.finegrained_pruning_config.pruning_method
        if pruning_method == 'masks':
            if QK_mask_list is not None or VO_mask_list is not None:
                save_dir = self.prune_with_masks(QK_mask_list=QK_mask_list, VO_mask_list=VO_mask_list, set_masks=True, save_model=save_model)
            else:
                raise TypeError("Pruning method is 'masks', but no masks are given.")
        elif pruning_method == 'iterative':
            assert (dataloader is not None ), "Pruning method is 'iterative', but dataloader is not given."
            save_dir = self.iterative_pruning(dataloader, adaptor, batch_postprocessor, keep_shape, save_model=save_model, rewrite_cache=rewrite_cache)
        else:
            raise NotImplementedError(f"Unknow pruning method {pruning_method}.")
        self.save_dir = save_dir
        return save_dir

    def prune_with_masks(self,QK_mask_list: Optional[List[torch.Tensor]] = None, 
                                VO_mask_list: Optional[List[torch.Tensor]] = None, 
                                keep_shape : bool = False, 
                                set_masks = False, 
                                save_model = False) -> Optional[str]:
        if QK_mask_list is None:
            QK_mask_list = self.QK_mask_list
        if VO_mask_list is None:
            VO_mask_list = self.VO_mask_list
        if set_masks is True:
            if QK_mask_list is not None:
                self.QK_mask_list = QK_mask_list
            if VO_mask_list is not None:
                self.VO_mask_list = VO_mask_list

        QK_mask_tensor_list = [mask.clone().detach().to(dtype=torch.float32, device=self.general_config.device) \
            for mask in QK_mask_list] if QK_mask_list is not None else None

        VO_mask_tensor_list = [mask.clone().detach().to(dtype=torch.float32, device=self.general_config.device) \
            for mask in VO_mask_list] if VO_mask_list is not None else None
        self.reorder_attention_weights(QK_mask_tensor_list, VO_mask_tensor_list, keep_shape)

        self.keep_shape = keep_shape
        if save_model is True:
            return self.save_model()

    def iterative_pruning(self, dataloader, adaptor, batch_postprocessor=None, keep_shape=False, save_model=True, rewrite_cache=False) -> Optional[str]:

        target_QK_head_size = self.finegrained_pruning_config.target_QK_head_size
        target_VO_head_size = self.finegrained_pruning_config.target_VO_head_size

        n_iters = self.finegrained_pruning_config.n_iters
        multiple_of = self.finegrained_pruning_config.multiple_of

        QK_importance_fn = os.path.join(self.output_dir, f'QK_importance.pt')
        VO_importance_fn = os.path.join(self.output_dir,f'VO_importance.pt')

        if os.path.exists(QK_importance_fn) and os.path.exists(VO_importance_fn) and rewrite_cache is False:
            logger.info(f"Loading pre-cached QK importance score {QK_importance_fn}")
            QK_importance_list = torch.load(QK_importance_fn)
            logger.info(f"Loading pre-cached VO importance score {VO_importance_fn}")
            VO_importance_list = torch.load(VO_importance_fn)
        else:
            logger.info("Calculating QK importance and VO importance")
            if self.finegrained_pruning_config.use_logits:
                QK_importance_list, VO_importance_list = self.get_importance_score_with_logits(dataloader, adaptor, batch_postprocessor)
            else:
                QK_importance_list, VO_importance_list = self.get_importance_score(dataloader, adaptor, batch_postprocessor)
            QK_importance_list = [importance.cpu() for importance in QK_importance_list] # (num_layers) *(all_head_size,)
            VO_importance_list = [importance.cpu() for importance in VO_importance_list] # (num_layers) *(all_head_size,)
            # Save importance score
            logger.info("Save...")
            torch.save(QK_importance_list, QK_importance_fn)
            torch.save(VO_importance_list, VO_importance_fn)

        head_size = int(self.base_model.config.hidden_size / self.base_model.config.num_attention_heads)

        n_layers_QK_head_size = head_size #len(QK_importance_list) * head_size  #sum(n_layers *(head_size,)
        n_layers_VO_head_size = head_size #len(VO_importance_list) * head_size  #sum(n_layers *(head_size,)
        target_n_layers_QK_head_size = target_QK_head_size #len(QK_importance_list) * target_QK_head_size
        target_n_layers_VO_head_size = target_VO_head_size #len(VO_importance_list) * target_VO_head_size

        num_of_QK_per_iter = (n_layers_QK_head_size - target_n_layers_QK_head_size) // n_iters
        num_of_VO_per_iter = (n_layers_VO_head_size - target_n_layers_VO_head_size) // n_iters
        num_of_QK_res = (n_layers_QK_head_size - target_n_layers_QK_head_size) % n_iters
        num_of_VO_res = (n_layers_VO_head_size - target_n_layers_VO_head_size) % n_iters

        print (n_layers_QK_head_size,target_n_layers_QK_head_size,num_of_QK_per_iter,num_of_QK_res, head_size) #debug 

        dQK_size = n_layers_QK_head_size
        dVO_size = n_layers_VO_head_size

        for i in range(n_iters):
            logger.info(f'Number of pruning iterations: {i+1}/{n_iters}')
            if i > 0:
                logger.info("Calculating QK importance and VO importance")
                if self.finegrained_pruning_config.use_logits:
                    QK_importance_list, VO_importance_list = self.get_importance_score_with_logits(dataloader, adaptor, batch_postprocessor)
                else:
                    QK_importance_list, VO_importance_list = self.get_importance_score(dataloader, adaptor, batch_postprocessor)
                QK_importance_list = [importance.cpu() for importance in QK_importance_list] # (num_layers) *(all_head_size,)
                VO_importance_list = [importance.cpu() for importance in VO_importance_list] # (num_layers) *(all_head_size,)

                assert torch.all(QK_importance_list[5]==QK_importance_list[5]*self.QK_mask_list[5])
                assert torch.all(VO_importance_list[5]==VO_importance_list[5]*self.VO_mask_list[5])
                #head_importance *= self.head_mask
                #ffn_importance *= self.ffn_mask

            dQK_size -= num_of_QK_per_iter + 1 if i < num_of_QK_res else num_of_QK_per_iter
            dVO_size -= num_of_VO_per_iter + 1 if i < num_of_VO_res else num_of_VO_per_iter

            group_size = head_size

            print(dQK_size, dVO_size, n_layers_QK_head_size, n_layers_VO_head_size,[q.size() for q in QK_importance_list])

            self.QK_mask_list = generate_mask_v2(QK_importance_list, dQK_size, group_size)
            self.VO_mask_list = generate_mask_v2(VO_importance_list, dVO_size, group_size)

            logger.info(f"New QK size:{[m.sum().item() for m in self.QK_mask_list]}")
            logger.info(f"New VO size:{[m.sum().item() for m in self.VO_mask_list]}")

            if i==n_iters-1:
                self.prune_with_masks(keep_shape=keep_shape, save_model=False)
            else:
                self.prune_with_masks(keep_shape=True, save_model=False)

        #clear cache
        self.soft_labels = []
        self.shoule_cache_logits = True

        logger.info("QK and VO masks have been generated, can be accessed via self.QK_mask and self.VO_mask")
        if save_model is True:
            return self.save_model()


    def save_masks(self,name='mask.pt') -> str:
        save_dir = os.path.join(self.general_config.output_dir,f'QKVO_masks')
        os.makedirs(save_dir, exist_ok=True)
        torch.save((self.QK_mask_list,self.VO_mask_list),os.path.join(save_dir,f'{name}'))
        # save config
        logger.info(f"Masks have been saved to {save_dir}")

        return save_dir


    def save_model(self, dir_name=None) -> str:
        if self.keep_shape is False:
            QK_size = self.finegrained_pruning_config.target_QK_head_size
            VO_size = self.finegrained_pruning_config.target_VO_head_size
        else:
            QK_size = int(self.base_model.config.hidden_size / self.base_model.config.num_attention_heads)
            VO_size = int(self.base_model.config.hidden_size / self.base_model.config.num_attention_heads)

        if dir_name is None:
            save_dir = os.path.join(self.general_config.output_dir,f'pruned_QK{QK_size}VO{VO_size}')
        else:
            save_dir = os.path.join(self.general_config.output_dir,dir_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(),os.path.join(save_dir,'pytorch_model.bin'))
        # save config
        self.base_model.config.save_pretrained(save_dir)
        logger.info(f"Model and configuration have been saved to {save_dir}")

        return save_dir


    def reorder_attention_weights(self, QK_mask_list = None, VO_mask_list = None, keep_shape = False):
        # QK_mask : (n_layers, head_size)
        # VO_mask : (n_layers, head_size)
        assert QK_mask_list is not None or VO_mask_list is not None
        num_attention_heads = self.base_model.config.num_attention_heads
        # head_size = int(self.base_model.config.hidden_size / )

        if QK_mask_list is not None:
            n_layers = len(QK_mask_list)
            att_queries = self.model_structure.get_att_query(self.base_model, ignore_model_prefix=True)
            att_keys = self.model_structure.get_att_key(self.base_model, ignore_model_prefix=True)
            for layer_num in range(n_layers):
                
                query_weight = att_queries[layer_num].weight
                query_bias = att_queries[layer_num].bias
                key_weight = att_keys[layer_num].weight
                key_bias = att_keys[layer_num].bias

                hidden_size = self.base_model.config.hidden_size
                orig_head_size = self.base_model.config.hidden_size / self.base_model.config.num_attention_heads
                num_heads = query_weight.size(0) / orig_head_size


                query_weight, query_bias = rearange_weights(query_weight, query_bias,QK_mask_list[layer_num],1,keep_shape)
                att_queries[layer_num].weight = torch.nn.Parameter(query_weight.contiguous())
                att_queries[layer_num].bias = torch.nn.Parameter(query_bias.contiguous())

                key_weight, key_bias = rearange_weights(key_weight, key_bias,QK_mask_list[layer_num],1,keep_shape)
                att_keys[layer_num].weight = torch.nn.Parameter(key_weight.contiguous())
                att_keys[layer_num].bias = torch.nn.Parameter(key_bias.contiguous())

        if VO_mask_list is not None:
            n_layers = len(VO_mask_list)
            att_values = self.model_structure.get_att_value(self.base_model, ignore_model_prefix=True)
            att_outputs = self.model_structure.get_att_output(self.base_model, ignore_model_prefix=True)
            for layer_num in range(n_layers):
                value_weight = att_values[layer_num].weight
                value_bias = att_values[layer_num].bias
                output_weight = att_outputs[layer_num].weight

                hidden_size = self.base_model.config.hidden_size
                orig_head_size = self.base_model.config.hidden_size / self.base_model.config.num_attention_heads
                num_heads = value_weight.size(0) / orig_head_size


                value_weight, value_bias = rearange_weights(value_weight, value_bias,VO_mask_list[layer_num],1,keep_shape)
                att_values[layer_num].weight = torch.nn.Parameter(value_weight.contiguous())
                att_values[layer_num].bias = torch.nn.Parameter(value_bias.contiguous())

                output_weight = output_weight.transpose(0,1)
                output_weight, _ = rearange_weights(output_weight, None, VO_mask_list[layer_num],1,keep_shape)
                output_weight = output_weight.transpose(0,1)
                att_outputs[layer_num].weight = torch.nn.Parameter(output_weight.contiguous())


    def get_importance_score(self, dataloader,
                                adaptor=None, batch_postprocessor=None) -> torch.Tensor :
        model = self.model

        n_layers = self.model_structure.get_num_layers(self.base_model, ignore_model_prefix=True)
        n_heads = self.base_model.config.num_attention_heads
        intermediate_size = self.base_model.config.intermediate_size

        device = self.general_config.device

        logger.info("***** Running Forward and Backward to calcuate importance score*****")
        logger.info(" Length of dataloader = %d", len(dataloader))
        model.eval()

        head_importance = torch.zeros(n_layers, n_heads).to(device)

        #get ffn weights and bias
        att_Q_weights = []
        att_Q_bias = []
        att_K_weights = []
        att_K_bias = []
        att_O_weights = []

        att_queries = self.model_structure.get_att_query(self.base_model, ignore_model_prefix=True)
        att_keys = self.model_structure.get_att_key(self.base_model, ignore_model_prefix=True)
        att_outputs = self.model_structure.get_att_output(self.base_model, ignore_model_prefix=True)
        for layer_num in range(n_layers):
                att_Q_weights.append(att_queries[layer_num].weight) #.detach().to(device)
                att_Q_bias.append(att_queries[layer_num].bias)
                att_K_weights.append(att_keys[layer_num].weight) #.detach().to(device)
                att_K_bias.append(att_keys[layer_num].bias)
                att_O_weights.append(att_outputs[layer_num].weight)

        QK_importance_list = [torch.zeros(att_Q_weights[i].size(0)).to(device) for i in range(n_layers)]
        VO_importance_list = [torch.zeros(att_O_weights[i].size(1)).to(device) for i in range(n_layers)]

        num_examples = 0.0

        for batch in tqdm(dataloader, desc="Calculating IS with loss"):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)
            batch = move_to_device(batch, device)
            if isinstance(batch,abc.Mapping):
                outputs = model(**batch)
                batch_num_examples = len(list(batch.values())[0])
            else:
                outputs = model(*batch)
                batch_num_examples = len(batch[0])
            loss = infer_loss(outputs, adaptor)
            loss.backward()

            for layer_num in range(n_layers):
                O_weight = att_O_weights[layer_num]
                Q_weight = att_Q_weights[layer_num]
                K_weight = att_K_weights[layer_num]
                Q_bias = att_Q_bias[layer_num]
                K_bias = att_K_bias[layer_num]

                QK_importance_list[layer_num] += ((Q_weight.grad * Q_weight).sum(dim=1) + Q_bias.grad * Q_bias
                                                 +(K_weight.grad * K_weight).sum(dim=1) +K_bias.grad * K_bias).abs().detach()
                VO_importance_list[layer_num] += (O_weight.grad * O_weight).sum(dim=0).abs().detach()

            model.zero_grad()
            num_examples += batch_num_examples

        if self.shoule_cache_logits is True:
            self.shoule_cache_logits = False

        QK_importance_list = [importance/num_examples for importance in QK_importance_list]
        VO_importance_list = [importance/num_examples for importance in VO_importance_list]


        return QK_importance_list, VO_importance_list


    def get_importance_score_with_logits(self, dataloader,
                                adaptor=None, batch_postprocessor=None) -> torch.Tensor :
        model = self.model

        n_layers = self.model_structure.get_num_layers(self.base_model, ignore_model_prefix=True)
        n_heads = self.base_model.config.num_attention_heads
        intermediate_size = self.base_model.config.intermediate_size

        device = self.general_config.device

        logger.info("***** Running Forward and Backward to calcuate importance score*****")
        logger.info(" Length of dataloader = %d", len(dataloader))
        model.eval()
        self.model_rep.eval()

        #get ffn weights and bias
        att_Q_weights = []
        att_Q_bias = []
        att_K_weights = []
        att_K_bias = []
        att_O_weights = []

        att_queries = self.model_structure.get_att_query(self.base_model, ignore_model_prefix=True)
        att_keys = self.model_structure.get_att_key(self.base_model, ignore_model_prefix=True)
        att_outputs = self.model_structure.get_att_output(self.base_model, ignore_model_prefix=True)
        for layer_num in range(n_layers):
                att_Q_weights.append(att_queries[layer_num].weight) #.detach().to(device)
                att_Q_bias.append(att_queries[layer_num].bias)
                att_K_weights.append(att_keys[layer_num].weight) #.detach().to(device)
                att_K_bias.append(att_keys[layer_num].bias)
                att_O_weights.append(att_outputs[layer_num].weight)

        QK_importance_list = [torch.zeros(att_Q_weights[i].size(0)).to(device) for i in range(n_layers)]
        VO_importance_list = [torch.zeros(att_O_weights[i].size(1)).to(device) for i in range(n_layers)]

        num_examples = 0.0


        for idx,batch in enumerate(tqdm(dataloader, desc="Calculating IS with logits")):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)
            batch = move_to_device(batch, device)
            if isinstance(batch,abc.Mapping):
                outputs = model(**batch)
                batch_num_examples = len(list(batch.values())[0])
            else:
                outputs = model(*batch)
                batch_num_examples = len(batch[0])

            with torch.no_grad():
                outputs_rep = self.model_rep(**batch) if isinstance(batch,abc.Mapping) else self.model_rep(*batch)
            logits_rep = infer_logits(outputs_rep, adaptor)

            logits = infer_logits(outputs, adaptor)
            #if self.shoule_cache_logits is True: # cache soft labels if the cache is empty
            #    p = softmax(logits, dim=-1).detach()
            #    self.soft_labels.append(p)

            if isinstance(logits,(list,tuple)):
                entropy = 0
                for logits_p, logits_q in zip(logits_rep, logits):
                    current_p = softmax(logits_p, dim=-1).detach()
                    current_q = logits_q
                    entropy += -(log_softmax(current_q,dim=-1) * current_p).sum(dim=-1).mean()
            else:
                current_p = softmax(logits_rep, dim=-1).detach() #p = softmax(logits, dim=-1).detach() #self.soft_labels[idx]
                #current_p = self.soft_labels[idx]
                current_q = logits
                entropy = - (log_softmax(current_q,dim=-1) * current_p).sum(dim=-1).mean()
            entropy.backward()


            for layer_num in range(n_layers):
                O_weight = att_O_weights[layer_num]
                Q_weight = att_Q_weights[layer_num]
                K_weight = att_K_weights[layer_num]
                Q_bias = att_Q_bias[layer_num]
                K_bias = att_K_bias[layer_num]

                QK_importance_list[layer_num] += ((Q_weight.grad * Q_weight).sum(dim=1) + Q_bias.grad * Q_bias
                                                 +(K_weight.grad * K_weight).sum(dim=1) +K_bias.grad * K_bias).abs().detach()
                VO_importance_list[layer_num] += (O_weight.grad * O_weight).sum(dim=0).abs().detach()

            model.zero_grad()
            num_examples += batch_num_examples

        if self.shoule_cache_logits is True:
            self.shoule_cache_logits = False

        QK_importance_list = [importance/num_examples for importance in QK_importance_list]
        VO_importance_list = [importance/num_examples for importance in VO_importance_list]

        return QK_importance_list, VO_importance_list


def rearange_weights(weight, bias, mask, head_size, keep_shape = False):
    num_heads = mask.size(0)
    mask_dim3 = mask.view(num_heads,1,1).to(torch.bool) # 12,1,1 ?
    weight_dim3 = weight.view(num_heads,head_size,weight.size(1)) # 12,64,768
    if keep_shape is False:
        selected_weight = weight_dim3.masked_select(mask_dim3)
        new_num_heads = int(mask.sum().item())
    else:
        selected_weight = torch.mul(weight_dim3, mask_dim3)
        new_num_heads = num_heads

    ##reshape back
    selected_weight = selected_weight.view(new_num_heads*head_size, weight.size(1)).contiguous()

    selected_bias = None
    if bias is not None:
        mask_dim2 = mask.view(num_heads,1).to(torch.bool) # 12,1 ?
        bias_dim2 = bias.view(num_heads,head_size) #12,64
        if keep_shape == False:
            selected_bias = bias_dim2.masked_select(mask_dim2)
        else:
            selected_bias = torch.mul(bias_dim2, mask_dim2)
        selected_bias = selected_bias.view(new_num_heads*head_size).contiguous()

    return selected_weight, selected_bias


def generate_mask_v3(importance_list : List[torch.Tensor], target_head_size: int, group_size : int) -> List[torch.Tensor]:
    mask_list = []
    num_layers = len(importance_list)
    num_selected = 0


    while num_selected < target_head_size * num_layers:
        # step 1: select num_head dims from each layer
        for i in range(num_layers):
            importance = importance_list[i] #todo
            num_groups = int(importance.size(0) / group_size)
            grouped_importance = importance.view(num_groups, group_size)

            
            importance_order = torch.argsort(grouped_importance, dim=-1)

            #TODO

            mask = torch.ones_like(grouped_importance)
            for gi in range(num_groups):
                mask[gi][importance_order[gi][:-target_head_size]] = 0
            mask_list.append(mask.reshape(-1))

    return mask_list


def generate_mask_v2(importance_list : List[torch.Tensor], target_head_size: int, group_size : int) -> List[torch.Tensor]:
    mask_list = []
    num_layers = len(importance_list)
    for i in range(num_layers):
        importance = importance_list[i]
        num_groups = int(importance.size(0) / group_size)
        grouped_importance = importance.view(num_groups, group_size)
        importance_order = torch.argsort(grouped_importance, dim=-1)
        mask = torch.ones_like(grouped_importance)
        for gi in range(num_groups):
            mask[gi][importance_order[gi][:-target_head_size]] = 0
        mask_list.append(mask.reshape(-1))

    return mask_list

    if multiple_of == 1:
        importance_flat = importance.reshape(-1)
        importance_order = torch.argsort(importance_flat)   # ascending
        mask_flat = torch.ones_like(importance_flat)
        for pos in importance_order[:-total_target_size]:
            mask_flat[pos] = 0
        mask = mask_flat.reshape(importance.shape)
    else:
        num_layers = importance.size(0)
        num_groups = importance.size(1) // multiple_of
        importance_order_2d = torch.argsort(importance,dim=-1)
        importance_3d = torch.zeros(num_layers, num_groups, multiple_of).to(importance)
        for i, layer_order in enumerate(importance_order_2d):
            layer_sorted_by_importance = importance[i][layer_order].view(-1,multiple_of) # (num_head // multiple_of, multiple_of)
            importance_3d[i] = layer_sorted_by_importance
        importance_2d_order_2d = importance_order_2d.view(num_layers * num_groups, multiple_of)

        importance_3d_s_flat = importance_3d.sum(-1).view(-1) # num_layers * num_groups
        importance_3d_s_flat_order_flat = torch.argsort(importance_3d_s_flat)   # ascending

        total_group_target_size = total_target_size // multiple_of
        mask = torch.ones_like(importance)

        for pos in importance_3d_s_flat_order_flat[:-total_group_target_size]:
            x = int(pos) // num_groups
            mask[x,importance_2d_order_2d[pos]] = 0

    # check for disconnected graph
    mask_sum = mask.sum(-1)
    for i in range(len(mask_sum)):
        if mask_sum[i]==0:
            print("Warning")
            most_imp = torch.argmax(importance[i])
            mask[i][most_imp] = 1
    return mask
