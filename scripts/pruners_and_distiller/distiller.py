import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import os
from accelerate import Accelerator
logger = logging.getLogger(__name__)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

from .pruners import ISPruner, FineISPruner
from .utils import TrainingConfig, DistillationConfig, PruningConfig, DistillationContext, kd_ce_loss, hid_mse_loss
from .utils import schedule_threshold, select_logits_with_mask

def initializer_builder(std):
    _std = std
    def init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=_std)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    return init_weights
def linear_projection(dim_in, dim_out,init='gaussian'):
    model = torch.nn.Linear(in_features=dim_in, out_features=dim_out, bias=True)
    if init=='gaussian':
        initializer = initializer_builder(0.02)
        model.apply(initializer)
    elif init=='identity':
        torch.nn.init.zeros_(model.bias)
        torch.nn.init.eye_(model.weight)
    else:
        raise NotImplementedError
    return model


class PruneDistiller(DistillationContext):
    def __init__(self, train_config, distill_config: DistillationConfig, prune_config : PruningConfig, model_T, model_S, adaptor):
        super(PruneDistiller, self).__init__()
        self.t_config = train_config
        self.d_config : DistillationConfig = distill_config 
        self.p_config : PruningConfig = prune_config
        self.model_T = model_T
        self.model_S = model_S
        self.model = self.model_S
        self.adaptor = adaptor

        self.print_freq = 20
        self.tb_writer = None
        self.accelerator = None

        if self.p_config.pruner_type=='ISPruner':
            self.pruner = ISPruner(self.model)
        elif self.p_config.pruner_type=='FineISPruner':
            self.pruner = FineISPruner(self.model)
        else:
            raise ValueError

        self.projs = dict()
        if self.d_config.matching_layers is not None:
            for layer_s,_ in self.d_config.matching_layers: #range(0,13,2):
                self.projs[layer_s]=linear_projection(768,768,'identity')

        self.global_status = dict()
        self.metrics = {}

    def train(self, dataloader, optimizer, lr_scheduler, num_epochs, num_steps = None, max_grad_norm = None, callback=None, batch_postprocessor=None):

        mixed_precision = 'fp16' if self.t_config.fp16 is True else 'no'
        self.accelerator = Accelerator(mixed_precision=mixed_precision)

        if self.accelerator.is_main_process:
            self.tb_writer = SummaryWriter(log_dir = self.t_config.log_dir)
        self.device = self.accelerator.device
        for proj in (self.projs.values()):
            optimizer.add_param_group({**{'params':proj.parameters()},})

        self.model, self.model_T, optimizer, dataloader, lr_scheduler, *projs = self.accelerator.prepare(
            self.model, self.model_T, optimizer, dataloader, lr_scheduler, *list(self.projs.values()))
        for idx,proj in zip(self.projs.keys(),projs):
            self.projs[idx] = proj
        self.model_S = self.model

        self.pruner.initialize(self.model)

        if num_epochs is not None:
            num_epochs = int(num_epochs)
            self.train_epochs(dataloader, optimizer, lr_scheduler, num_epochs, max_grad_norm, callback, batch_postprocessor)


    def train_epochs(self, dataloader, optimizer, lr_scheduler, num_epochs, max_grad_norm = None, callback=None, batch_postprocessor=None):

        train_steps_per_epoch = len(dataloader) // self.t_config.gradient_accumulation_steps
        print_every = train_steps_per_epoch // self.print_freq
        if print_every == 0:
            print_every = train_steps_per_epoch
        checkpoints = [int(train_steps_per_epoch*ci/self.t_config.ckpt_frequency) for ci in range(self.t_config.ckpt_frequency)]

        total_global_steps = train_steps_per_epoch * num_epochs
        logger.info(f"Training steps per epoch: {train_steps_per_epoch}")
        logger.info(f"Training total global steps: {total_global_steps}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        global_step = 0

        scalar_total_loss = 0
        tqdm_disable = None if self.accelerator.is_main_process else True

        # only works with gradient_accumulation_steps==1 
        assert self.t_config.gradient_accumulation_steps == 1

        for current_epoch in tqdm(range(num_epochs),disable=tqdm_disable):

            logger.info(f"Epoch {current_epoch+1}")
            logger.info(f"Length of current epoch in forward batch: {len(dataloader)}")

            for forward_step, batch in tqdm(enumerate(dataloader),disable=tqdm_disable):

                #init
                optimizer.zero_grad()
                # forward and get loss
                batch = batch_postprocessor(batch) if batch_postprocessor is not None else batch
                ce_loss, matching_loss = self.train_on_batch(batch)
                if matching_loss is not None:
                    sum_loss = ce_loss+matching_loss
                else:
                    sum_loss = ce_loss
                scalar_total_loss += sum_loss.cpu().item()

                global_step += 1

                # backward
                if matching_loss is not None and (self.p_config.dbw is True):
                    self.accelerator.backward(ce_loss,retain_graph=True)
                    _,prune_ended = self.maybe_should_prune(global_step, total_global_steps,do_prune=False, do_gather=True,do_update=False, gamma=1)
                    self.accelerator.backward(matching_loss,retain_graph=False)
                    _,prune_ended = self.maybe_should_prune(global_step, total_global_steps,do_prune=True, do_gather=False, do_update=True, gamma=self.p_config.IS_gamma)
                else:
                    self.accelerator.backward(sum_loss)
                    _,prune_ended = self.maybe_should_prune(global_step, total_global_steps)
                #Tensorboard logging
                if self.accelerator.is_main_process:
                    self.tb_writer.add_scalar('scalar/total_loss', scalar_total_loss, global_step)
                # gradient clipping
                if max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                #optimizer step
                optimizer.step()
                lr_scheduler.step()

                if (global_step) % print_every == 0:
                    logger.info(f"Global step: {global_step}, epoch forward_step:{forward_step+1}")

                if (global_step % train_steps_per_epoch in checkpoints) \
                        and ((current_epoch+1) % self.t_config.ckpt_epoch_frequency==0 or current_epoch+1==num_epochs):

                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process and prune_ended:
                        logger.info(f"Saving at global step {global_step}, epoch forward_step {forward_step+1} epoch {current_epoch+1}")
                        coreModel = self.accelerator.unwrap_model(self.model)
                        state_dict = coreModel.state_dict()
                        self.accelerator.save(state_dict, os.path.join(self.t_config.output_dir,f"gs{global_step}.pt"))
                    self.accelerator.wait_for_everyone()
                    if callback is not None:
                        logger.info("Running callback function...")
                        res = callback(model=self.model, step=global_step)
                        self.metrics[global_step] = res
                        self.model.train()

            logger.info(f"Epoch {current_epoch+1} finished")

    def train_on_batch(self, batch) -> torch.Tensor:
        #batch = move_to_device(batch, self.t_config.device)
        if isinstance(batch,(list,tuple)):
            results_S = self.model(*batch)
            with torch.no_grad():
                results_T = self.model_T(*batch)
        else:
            results_S = self.model(**batch)
            with torch.no_grad():
                results_T = self.model_T(**batch)
        results_S = post_adaptor(self.adaptor(batch,results_S))
        results_T = post_adaptor(self.adaptor(batch,results_T))

        ce_loss, matching_loss = self.compute_loss(results_S,results_T)

        return ce_loss, matching_loss #, losses_dict


    def compute_loss(self, results_S, results_T):
        total_loss = 0
        matching_loss = None
        losses_dict = dict()
        logits_list_T = results_T['logits']  # list of tensor
        logits_list_S = results_S['logits']  # list of tensor

        if 'logits_mask' in results_S:
            masks_list_S = results_S['logits_mask']
            logits_list_S = select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
        if 'logits_mask' in results_T:
            masks_list_T = results_T['logits_mask']
            logits_list_T = select_logits_with_mask(logits_list_T,masks_list_T)  #(mask_sum, num_of_class)

        total_kd_loss = 0
        for l_T,l_S in zip(logits_list_T,logits_list_S):
            temperature = self.d_config.temperature
            total_kd_loss += kd_ce_loss(l_S, l_T, temperature)
        total_loss += total_kd_loss * self.d_config.kd_loss_weight
        losses_dict['unweighted_kd_loss'] = total_kd_loss

        if 'losses' in results_S:
            total_hl_loss = 0
            total_hl_loss = sum(loss.mean() for loss in results_S['losses'])  # in case of multi-GPU
            total_loss += total_hl_loss * self.d_config.hard_label_weight
            losses_dict['unweighted_hard_label_loss'] = total_hl_loss
        if 'hidden' in results_T and 'hidden' in results_S and (self.d_config.matching_layers is not None):
            matching_loss = 0

            loss_weight_pairs = []
            for layer_s, layer_t in self.d_config.matching_layers:
                inter_S = self.projs[layer_s](results_S['hidden'][layer_s])
                inter_T = results_T['hidden'][layer_t]
                inputs_mask_S = results_S.get('inputs_mask',None)
                loss_weight = 1

                match_loss = hid_mse_loss(inter_S, inter_T, mask=inputs_mask_S)
                loss_weight_pairs.append((match_loss,loss_weight))

            weights_sum = sum(w for _,w in loss_weight_pairs[1:])
            num_matchings = len(loss_weight_pairs) - 1 #excluding embeddings
            rescaled_weights = [w/weights_sum for _,w in loss_weight_pairs[1:]] #embddings + trm
            rescaled_weights_sum = sum(rescaled_weights)
            normalized_weights = [1] + [w/rescaled_weights_sum * num_matchings for w in rescaled_weights]

            self.global_status['normalized_weights'] = normalized_weights

            assert len(normalized_weights)==len(loss_weight_pairs)
            matching_loss += sum(p[0]*w for p,w in zip(loss_weight_pairs,normalized_weights))
        return total_loss, matching_loss 


    def maybe_should_prune(self,global_step, total_global_steps, do_gather=True, do_update=True, do_prune=True, gamma:float=1):
        pruning_frequency = self.p_config.pruning_frequency
        start_pruning_steps = int(self.p_config.start_pruning_at * total_global_steps)
        end_pruning_steps = int(self.p_config.end_pruning_at * total_global_steps)
        if do_gather is True:
            self.pruner.gather_IS(self.model, gamma,self.p_config.score_type, global_step>=start_pruning_steps)
        if do_update is True:
            self.pruner.update_IS(beta = self.p_config.IS_beta, alpha=self.p_config.IS_alpha, 
                                                                alpha_head=self.p_config.IS_alpha_head, 
                                                                alpha_ffn=self.p_config.IS_alpha_ffn,
                                                                alpha_mha=self.p_config.IS_alpha_mha)
        if do_prune is True:
            if (global_step % pruning_frequency == 0) and \
                global_step >= start_pruning_steps and global_step < end_pruning_steps:
                if self.p_config.pruner_type=='ISPruner' or self.p_config.pruner_type=='FineISPruner':
                    self.pruner.do_prune(self.model, global_step, total_global_steps, self.p_config)
                else:
                    raise NotImplementedError

        # logging
        if (global_step % 100 == 0):
            ffn_density = self.pruner.ffn_mask.sum()/self.pruner.ffn_mask.numel()
            if self.p_config.pruner_type=='ISPruner':
                head_density = self.pruner.head_mask.sum()/self.pruner.head_mask.numel()
                print(f"group density FFN/MHA {ffn_density:.4f} {head_density:.4f}")
                print(f"weighted density FFN/MHA {ffn_density * self.pruner.total_ffn_ratio + head_density * self.pruner.total_head_ratio:.4f}")
            elif self.p_config.pruner_type=='FineISPruner':
                qk_density = self.pruner.qk_mask.sum()/self.pruner.qk_mask.numel()
                vo_density = self.pruner.vo_mask.sum()/self.pruner.vo_mask.numel()
                print(f"Num zeros {self.pruner.pre_num_zeros / self.pruner.total_num}")
                print(f"group density FFN/QK/VO {ffn_density:.4f} {qk_density:.4f} {vo_density:.4f}")
                print(f"weighted density FFN/QK/VO {ffn_density * self.pruner.total_ffn_ratio + qk_density * self.pruner.total_qk_ratio + vo_density * self.pruner.total_vo_ratio:.4f}")

        return global_step >= start_pruning_steps, global_step >= end_pruning_steps

def post_adaptor(dict_object):
    if 'logits' in dict_object:
        logits = dict_object['logits']
        if not isinstance(logits,(list,tuple)):
            dict_object['logits'] = [ logits ]
    if 'logits_mask' in dict_object:
        logits_mask = dict_object['logits_mask']
        if not isinstance(logits_mask,(list,tuple)):
            dict_object['logits_mask'] = [ logits_mask ]
    if 'losses' in dict_object:
        losses = dict_object['losses']
        if not isinstance(losses,(list,tuple)):
            dict_object['losses'] = [ losses ]
    if 'labels' in dict_object:
        labels = dict_object['labels']
        if not isinstance(labels,(list,tuple)):
            dict_object['labels'] = [ labels ]
    return dict_object


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
