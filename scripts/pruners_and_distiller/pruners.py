import torch
from torch.nn.utils import prune as torch_prune
from .utils import schedule_threshold
from math import ceil

class ISPruner:
    def __init__(self,model):

        self.n_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.ffn_size = model.config.intermediate_size
        self.hidden_size = model.config.hidden_size
        self.total_head_size = model.config.hidden_size
        self.head_size = self.total_head_size / self.num_heads

        self.initialized = False
        self.ffn_order = None
        self.head_order = None
        self.ffn_mask = None
        self.head_mask = None
        self.ffn_importances = None
        self.head_importances = None

        self.has_started = False

    def initialize(self,model):
        if self.initialized is False:

            self.ffn_mask = torch.ones(self.n_layers, self.ffn_size).to(model.device)
            self.head_mask = torch.ones(self.n_layers, self.num_heads).to(model.device)
            self.ffn_importances = torch.zeros(self.n_layers, self.ffn_size).to(model.device)
            self.head_importances = torch.zeros(self.n_layers, self.num_heads).to(model.device)
            
            self.ffn_importances_updates = torch.zeros_like(self.ffn_importances)
            self.head_importances_updates = torch.zeros_like(self.head_importances)
            
            
            self.ffn_unit_size = self.hidden_size*2 + 1
            self.head_unit_size = self.hidden_size * self.head_size * 4 +  self.head_size * 3
            total_ffn_size = self.ffn_unit_size * self.ffn_importances.nelement()
            total_head_size = self.head_unit_size * self.head_importances.nelement()
            param_sizes = [self.ffn_unit_size] * self.ffn_importances.nelement() + [self.head_unit_size] * self.head_importances.nelement()
            total_params = sum(param_sizes)
            self.total_ffn_ratio = total_ffn_size / total_params
            self.total_head_ratio = total_head_size / total_params

            self.normed_param_sizes = torch.tensor([unit_size/total_params for unit_size in param_sizes])

            self.initialized = True

            self.pre_ffn_density = self.ffn_mask.nelement()
            self.pre_head_density = self.head_mask.nelement()
            self.pre_num_zeros = 0


    def do_prune(self,model,global_step, total_global_steps, p_config):
        
        pre_ffn_mask = self.ffn_mask.clone()
        pre_head_mask = self.head_mask.clone()

        if p_config.is_two_ratios is False:
            weights_ratio = schedule_threshold(global_step, total_global_steps, p_config)
            self.generate_mask(weights_ratio=weights_ratio,
                            is_global=p_config.is_global,
                            is_reweight=p_config.is_reweight)
        else:
            weights_ratio_FFN = schedule_threshold(global_step, total_global_steps, p_config, p_config.FFN_weights_ratio)
            weights_ratio_MHA = schedule_threshold(global_step, total_global_steps, p_config, p_config.MHA_weights_ratio)
            self.generate_mask_with_two_ratios(weights_ratio_FFN, weights_ratio_MHA)

        assert torch.all(((pre_ffn_mask==0)&(self.ffn_mask==0)) == (pre_ffn_mask==0))
        assert torch.all(((pre_head_mask==0)&(self.head_mask==0)) == (pre_head_mask==0))

        self.prune_model(model=model)


    def update_IS(self, beta = 0.99, alpha: float = 0.0001, alpha_head = None, alpha_ffn = None, alpha_mha = None):

        self.ffn_importances = self.ffn_importances * beta + self.ffn_importances_updates * (1 - beta)
        self.head_importances = self.head_importances * beta + self.head_importances_updates * (1 - beta)


        self.ffn_importances_updates.zero_()
        self.head_importances_updates.zero_()

        #debug
        assert torch.all(~torch.isnan(self.ffn_importances)),f"{torch.isnan(self.ffn_importances).sum()}"
        assert torch.all(~torch.isnan(self.head_importances)),f"{torch.isnan(self.head_importances).sum()}"


    def gather_IS(self, model, gamma : float = 1, score_type='grad', has_started = None):
        if has_started is not None:
            self.has_started = has_started
        #debug
        for idx,layer in reversed(list(enumerate(model.base_model.encoder.layer))):
            output = layer.attention.output.dense
            ffn2 = layer.output.dense
            if hasattr(ffn2,'weight_orig'):
                ffn2_weight = ffn2.weight_orig
                output_weight = output.weight_orig
            else:
                ffn2_weight = ffn2.weight
                output_weight = output.weight

            if not torch.all(~torch.isnan(ffn2_weight.grad)):
                print (f"Skip current batch. {idx} FFN has NaN {torch.isnan(ffn2_weight.grad).sum()}")
                return
            if not torch.all(~torch.isnan(output_weight.grad)):
                print (f"Skip current batch. {idx} Output has NaN {torch.isnan(output_weight.grad).sum()}")
                return


        for idx,layer in enumerate(model.base_model.encoder.layer):
            ffn2 = layer.output.dense
            output = layer.attention.output.dense

            if hasattr(ffn2,'weight_orig') and score_type=='grad':
                ffn2_weight = ffn2.weight_orig
                output_weight = output.weight_orig
            else:
                ffn2_weight = ffn2.weight
                output_weight = output.weight

            if score_type=='grad':
                self.ffn_importances_updates[idx] += gamma *  ((ffn2_weight.grad * ffn2_weight.data).sum(dim=0)).abs().detach()
                self.head_importances_updates[idx] += gamma * (output_weight.grad * output_weight.data).view(output_weight.size(0), self.num_heads, -1).sum(dim=(0,2)).abs().detach()
            elif score_type=='magnitude-sumabs':
                self.ffn_importances[idx] = ffn2_weight.data.sum(dim=0).abs()
                self.head_importances[idx] = output_weight.data.view(output_weight.size(0), self.num_heads, -1).sum(dim=(0,2)).abs()
            elif score_type=='magnitude-Linf':
                self.ffn_importances[idx] = ffn2_weight.data.abs().amax(dim=0)
                self.head_importances[idx] = output_weight.data.view(output_weight.size(0), self.num_heads, -1).abs().amax(dim=(0,2))
            elif score_type=='magnitude-L1':
                self.ffn_importances[idx] = ffn2_weight.data.abs().sum(dim=0)
                self.head_importances[idx] = output_weight.data.view(output_weight.size(0), self.num_heads, -1).abs().sum(dim=(0,2))
            elif score_type=='random':
                self.ffn_importances[idx] = (self.ffn_mask[idx]) * (torch.rand_like(self.ffn_importances[idx]) + 1)
                self.head_importances[idx] = (self.head_mask[idx]) * (torch.rand_like(self.head_importances[idx]) + 1)
            else:
                raise ValueError

    def generate_mask_with_two_ratios(self, weights_ratio_FFN, weights_ratio_MHA):

        ffn_density = ceil(self.ffn_mask.nelement() * weights_ratio_FFN) #warning: int -> ceil
        head_density = ceil(self.head_mask.nelement() * weights_ratio_MHA)
    
        if ffn_density < self.pre_ffn_density or head_density < self.pre_head_density:
            self.pre_ffn_density = ffn_density
            self.pre_head_density = head_density

            ffn_order = torch.argsort(self.ffn_importances.view(-1))
            head_order = torch.argsort(self.head_importances.view(-1))

            self.ffn_mask.fill_(1).view(-1)[ffn_order[:-ffn_density]] = 0
            self.head_mask.fill_(1).view(-1)[head_order[:-head_density]] = 0

        self.ffn_importances *= self.ffn_mask
        self.head_importances *= self.head_mask

    def generate_mask(self, weights_ratio, is_global=False, is_reweight=1):
        if is_global is True:
            all_importances = torch.cat([self.ffn_importances.view(-1) * is_reweight, self.head_importances.view(-1)])
            all_order = torch.argsort(all_importances)

            sorted_normed_param_sizes = self.normed_param_sizes[all_order]
            cumsum_sorted_normed_param_sizes = torch.cumsum(sorted_normed_param_sizes,dim=0)
            num_zeros = torch.searchsorted(cumsum_sorted_normed_param_sizes, 1 - weights_ratio).item()
            if num_zeros > self.pre_num_zeros:
                self.pre_num_zeros = num_zeros
                all_topk = all_order[:num_zeros]
                ffn_indices = all_topk[all_topk<self.ffn_importances.nelement()]
                head_indices = all_topk[all_topk>=self.ffn_importances.nelement()] - self.ffn_importances.nelement()

                self.ffn_mask.fill_(1).view(-1)[ffn_indices] = 0
                self.head_mask.fill_(1).view(-1)[head_indices] = 0
        else:
            ffn_density = int(self.ffn_mask.nelement() * weights_ratio)
            head_density = int(self.head_mask.nelement() * weights_ratio)
            if ffn_density < self.pre_ffn_density or head_density < self.pre_head_density:
                self.pre_ffn_density = ffn_density
                self.pre_head_density = head_density

                ffn_order = torch.argsort(self.ffn_importances.view(-1))
                head_order = torch.argsort(self.head_importances.view(-1))

                self.ffn_mask.fill_(1).view(-1)[ffn_order[:-ffn_density]] = 0
                self.head_mask.fill_(1).view(-1)[head_order[:-head_density]] = 0

        self.ffn_importances *= self.ffn_mask
        self.head_importances *= self.head_mask


    def prune_model(self, model):

        for idx,layer in enumerate(model.base_model.encoder.layer):
            ffn1 = layer.intermediate.dense
            ffn2 = layer.output.dense
            hidden_size = ffn1.weight.size(1)

            ffn_mask = self.ffn_mask[idx] #(3072,)
            w2_mask = ffn_mask.unsqueeze(0).expand(ffn2.weight.size(0),-1)

            if hasattr(ffn2,'weight_mask'):
                ffn2.weight_mask = w2_mask
            else:
                torch_prune.custom_from_mask(ffn2,'weight',w2_mask)

            output = layer.attention.output.dense

            head_mask = self.head_mask[idx]
            o_mask = head_mask.view(-1,1,1).expand(-1,hidden_size//head_mask.size(0),hidden_size).reshape(-1,hidden_size).transpose(0,1).contiguous()

            if hasattr(output,'weight_mask'):
                output.weight_mask = o_mask
            else:
                torch_prune.custom_from_mask(output,'weight',o_mask)


class FinePruner:
    initialized = False
    ffn_order = None
    qk_order = None
    vo_order = None
    ffn_mask = None
    qk_mask = None
    vo_mask = None
    @classmethod
    def do_prune(cls,model,weights_ratio):
        if cls.initialized is False:
            cls.n_layers = model.config.num_hidden_layers
            cls.ffn_size = model.config.intermediate_size
            cls.num_heads = model.config.num_attention_heads
            cls.hidden_size = model.config.hidden_size
            cls.total_head_size = model.config.hidden_size

            cls.ffn_order = torch.randperm(cls.n_layers*cls.ffn_size).to(model.device)
            cls.qk_order = torch.randperm(cls.n_layers*cls.total_head_size).to(model.device)
            cls.vo_order = torch.randperm(cls.n_layers*cls.total_head_size).to(model.device)
            cls.ffn_mask = torch.ones(cls.n_layers, cls.ffn_size).to(model.device)
            cls.qk_mask = torch.ones(cls.n_layers, cls.total_head_size).to(model.device)
            cls.vo_mask = torch.ones(cls.n_layers, cls.total_head_size).to(model.device)

            cls.initialized = True

        cls.generate_mask(weights_ratio=weights_ratio)
        print(f"do pruning wtih weights_ratio {weights_ratio}")
        print(f"sparsity FFN/QK/VO {1-cls.ffn_mask.sum()/cls.ffn_mask.nelement():.4f} {1-cls.qk_mask.sum()/cls.qk_mask.nelement():.4f} {1-cls.vo_mask.sum()/cls.vo_mask.nelement():.4f}")
        cls.prune_model(model=model)

    @classmethod
    def generate_mask(cls, weights_ratio):          
        #skip generating masks
        #let's test random masks
        ffn_mask_zeros_sum = int(cls.ffn_mask.nelement() * (1-weights_ratio))
        qk_mask_zeros_sum = int(cls.qk_mask.nelement() * (1-weights_ratio))
        vo_mask_zeros_sum = int(cls.vo_mask.nelement() * (1-weights_ratio))
        cls.ffn_mask.view(-1)[cls.ffn_order[:ffn_mask_zeros_sum]] = 0
        cls.qk_mask.view(-1)[cls.qk_order[:qk_mask_zeros_sum]] = 0
        cls.vo_mask.view(-1)[cls.vo_order[:vo_mask_zeros_sum]] = 0

    @classmethod
    def prune_model(cls, model):

        for idx,layer in enumerate(model.base_model.encoder.layer):
            ffn1 = layer.intermediate.dense
            ffn2 = layer.output.dense
            hidden_size = ffn1.weight.size(1)

            ffn_mask = cls.ffn_mask[idx] #(3072,)
            w2_mask = ffn_mask.unsqueeze(0).expand(ffn2.weight.size(0),-1)

            if hasattr(ffn2,'weight_mask'):
                ffn2.weight_mask = w2_mask
            else:
                torch_prune.custom_from_mask(ffn2,'weight',w2_mask)
            #w1_mask = ffn_mask.unsqueeze(1).expand(-1,ffn1.weight.size(1))
            #torch_prune.custom_from_mask(ffn1,'weight',w1_mask)
            #torch_prune.custom_from_mask(ffn1,'bias',ffn_mask)


            #-------attention part-----------#

            query = layer.attention.self.query
            key = layer.attention.self.key
            value = layer.attention.self.value
            output = layer.attention.output.dense

            qk_mask = cls.qk_mask[idx]
            vo_mask = cls.vo_mask[idx]
            qk_weight_mask = cls.qk_mask[idx].unsqueeze(1).expand(-1,query.weight.size(1))
            vo_weight_mask = cls.vo_mask[idx].unsqueeze(1).expand(-1,value.weight.size(1))
            #o_mask = head_mask.unsqueeze(0).expand(output.weight.size(0),-1)

            if hasattr(output,'weight_mask'):
                key.weight_mask = qk_weight_mask
                query.weight_mask = qk_weight_mask
                value.weight_mask = vo_weight_mask
                output.weight_mask = vo_weight_mask.transpose(0,1)
                key.bias_mask = qk_mask
                query.bias_mask = qk_mask
                value.bias_mask = vo_mask
            else:
                torch_prune.custom_from_mask(query,'weight',qk_weight_mask)
                torch_prune.custom_from_mask(key,'weight',qk_weight_mask)
                torch_prune.custom_from_mask(value,'weight',vo_weight_mask)
                torch_prune.custom_from_mask(output,'weight',vo_weight_mask.transpose(0,1))
                torch_prune.custom_from_mask(query,'bias',qk_mask)
                torch_prune.custom_from_mask(key,'bias',qk_mask)
                torch_prune.custom_from_mask(value,'bias',vo_mask)

            assert hasattr(layer.attention.self,'dynamic_attention_head_size')
            layer.attention.self.dynamic_attention_head_size = qk_mask.view(cls.num_heads,-1).sum(-1)

            if idx==6:
                #print(w2_mask.sum()/w2_mask.nelement(),o_mask.sum()/o_mask.nelement())
                #print(f"Layer {idx} ffn1-weight sparsity: {(ffn1.weight==0).sum()/ffn1.weight.nelement()}")
                print(f"Layer {idx} effective head size (new):", layer.attention.self.dynamic_attention_head_size)
                print(f"Layer {idx} ffn2-weight sparsity: {(ffn2.weight==0).sum()/ffn2.weight.nelement()}")
                print(f"Layer {idx} query-weight sparsity: {(query.weight==0).sum()/query.weight.nelement()}")
                print(f"Layer {idx} output-weight sparsity: {(output.weight==0).sum()/output.weight.nelement()}")


class FineISPruner:
    def __init__(self,model):

        self.n_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.ffn_size = model.config.intermediate_size
        self.hidden_size = model.config.hidden_size
        self.total_head_size = model.config.hidden_size
        self.head_size = self.total_head_size / self.num_heads

        self.initialized = False
        self.ffn_order = self.ffn_mask = self.ffn_importances = None
        self.qk_order = self.qk_mask = self.qk_importances = None
        self.vo_order = self.vo_mask = self.vo_importances = None

        self.has_started = False

    def initialize(self,model):
        if self.initialized is False:

            self.ffn_mask = torch.ones(self.n_layers, self.ffn_size).to(model.device)
            self.qk_mask = torch.ones(self.n_layers, self.total_head_size).to(model.device)
            self.vo_mask = torch.ones(self.n_layers, self.total_head_size).to(model.device)
            self.ffn_importances = torch.zeros(self.n_layers,self.ffn_size).to(model.device)
            self.qk_importances = torch.zeros(self.n_layers,self.total_head_size).to(model.device)
            self.vo_importances = torch.zeros(self.n_layers,self.total_head_size).to(model.device)

            #store IS updates
            self.ffn_importances_updates = torch.zeros_like(self.ffn_importances)
            self.qk_importances_updates = torch.zeros_like(self.qk_importances)
            self.vo_importances_updates = torch.zeros_like(self.vo_importances)

            self.ffn_unit_size = self.hidden_size*2 + 1
            self.qk_unit_size = self.hidden_size*2 + 2
            self.vo_unit_size = self.hidden_size*2 + 1
            self.total_ffn_size = self.ffn_unit_size * self.ffn_importances.numel()
            self.total_qk_size = self.qk_unit_size * self.qk_importances.numel()
            self.total_vo_size = self.vo_unit_size * self.vo_importances.numel()
            self.ffn_idx_start = 0
            self.qk_idx_start = self.ffn_idx_start + self.ffn_importances.numel()
            self.vo_idx_start = self.qk_idx_start + self.qk_importances.numel()
            self.head_idx_start = self.vo_idx_start + self.vo_importances.numel()


            param_sizes = [self.ffn_unit_size] * self.ffn_importances.numel() \
                        + [self.qk_unit_size]  * self.qk_importances.numel() \
                        + [self.vo_unit_size]  * self.vo_importances.numel()
            total_params = sum(param_sizes)
            self.total_ffn_ratio = self.total_ffn_size / total_params
            self.total_qk_ratio = self.total_qk_size / total_params
            self.total_vo_ratio = self.total_vo_size / total_params

            self.normed_param_sizes = torch.tensor([unit_size/total_params for unit_size in param_sizes])

            self.initialized = True

            self.pre_ffn_density = self.ffn_mask.numel()
            self.pre_qk_density = self.qk_mask.numel()
            self.pre_vo_density = self.vo_mask.numel()
            self.total_num = self.ffn_mask.numel() + self.qk_mask.numel() + self.vo_mask.numel()
            self.pre_num_zeros = 0


    def do_prune(self, model, global_step, total_global_steps, p_config):
        pre_ffn_mask = self.ffn_mask.clone()
        pre_qk_mask = self.qk_mask.clone()
        pre_vo_mask = self.vo_mask.clone()

        if p_config.is_two_ratios is False:
            weights_ratio = schedule_threshold(global_step, total_global_steps, p_config)
            self.generate_mask(weights_ratio=weights_ratio,
                            is_global=p_config.is_global,
                            is_reweight=p_config.is_reweight)
        else:
            raise NotImplementedError

        assert torch.all(((pre_ffn_mask==0)&(self.ffn_mask==0)) == (pre_ffn_mask==0))
        assert torch.all(((pre_qk_mask==0)&(self.qk_mask==0)) == (pre_qk_mask==0)), (pre_qk_mask.sum(),self.qk_mask.sum())
        assert torch.all(((pre_vo_mask==0)&(self.vo_mask==0)) == (pre_vo_mask==0)), (torch.where(pre_vo_mask==0),torch.where(self.vo_mask==0))
        self.prune_model(model=model, dynamic_head_size=p_config.dynamic_head_size)

    def update_IS(self, beta = 0.99, alpha: float = 0.0001, alpha_head = None, alpha_ffn = None, alpha_mha = None):

        alpha_head = alpha if alpha_head is None else alpha_head
        alpha_ffn = alpha if alpha_ffn is None else alpha_ffn
        alpha_mha = alpha if alpha_mha is None else alpha_mha


        vo_importances_updates_3d = self.vo_importances_updates.view(self.vo_mask.size(0),self.num_heads,-1)
        head_scale = torch.tanh(self.vo_mask.view(self.vo_mask.size(0),self.num_heads,-1).mean(-1,keepdim=True)/alpha_head)
        self.vo_importances_updates = (head_scale * vo_importances_updates_3d).view(self.vo_mask.size(0),-1)

        mha_scale = torch.tanh(self.vo_mask.mean(-1,keepdim=True)/alpha_mha)
        self.vo_importances_updates = (mha_scale * self.vo_importances_updates)


        ffn_scale = torch.tanh(self.ffn_mask.mean(-1,keepdim=True)/alpha_ffn)
        self.ffn_importances_updates = (ffn_scale * self.ffn_importances_updates)


        self.ffn_importances = self.ffn_importances * beta + self.ffn_importances_updates * (1 - beta)
        self.qk_importances = self.qk_importances * beta + self.qk_importances_updates * (1 - beta)
        self.vo_importances = self.vo_importances * beta + self.vo_importances_updates * (1 - beta)

        self.ffn_importances_updates.zero_()
        self.qk_importances_updates.zero_()
        self.vo_importances_updates.zero_()

        assert torch.all(~torch.isnan(self.ffn_importances)),f"{torch.isnan(self.ffn_importances).sum()}"
        assert torch.all(~torch.isnan(self.qk_importances)),f"{torch.isnan(self.qk_importances).sum()}"
        assert torch.all(~torch.isnan(self.vo_importances)),f"{torch.isnan(self.vo_importances).sum()}"


    def gather_IS(self, model, gamma : float = 1, score_type='grad', has_started = None):
        if has_started is not None:
            self.has_started = has_started
        for idx,layer in enumerate(model.base_model.encoder.layer):
            #ffn1 = layer.intermediate.dense
            ffn2 = layer.output.dense
            query = layer.attention.self.query
            output = layer.attention.output.dense
            if hasattr(ffn2,'weight_orig'): # and score_type=='grad':
                ffn2_weight = ffn2.weight_orig
                output_weight = output.weight_orig
                query_weight = query.weight_orig
                query_bias = query.bias_orig
            else:
                ffn2_weight = ffn2.weight
                output_weight = output.weight
                query_weight = query.weight
                query_bias = query.bias
            if torch.any(torch.isnan(ffn2_weight.grad)) or \
                torch.any(torch.isnan(output_weight.grad)) or \
                torch.any(torch.isnan(query_weight.grad)):
                print (f"Skip current batch. {idx} has NaN")
                return
        for idx,layer in enumerate(model.base_model.encoder.layer):
            ffn2 = layer.output.dense
            query = layer.attention.self.query
            output = layer.attention.output.dense
            if hasattr(ffn2,'weight_orig'): # and score_type=='grad':
                ffn2_weight = ffn2.weight_orig
                output_weight = output.weight_orig
                query_weight = query.weight_orig
                query_bias = query.bias_orig
            else:
                ffn2_weight = ffn2.weight
                output_weight = output.weight
                query_weight = query.weight
                query_bias = query.bias
            if score_type == 'grad':
                ffn2_grad_data = ffn2_weight.grad * ffn2_weight.data
                query_grad_data = query_weight.grad * query_weight.data
                output_grad_data = output_weight.grad * output_weight.data

                self.ffn_importances_updates[idx] += gamma * ((ffn2_grad_data).sum(dim=0)).abs().detach()
                self.qk_importances_updates[idx] += gamma * ((query_grad_data).sum(dim=1)+query_bias.grad * query_bias.data).abs().detach()
                self.vo_importances_updates[idx] += gamma * ((output_grad_data).sum(dim=0)).abs().detach()


            elif score_type=='random':
                self.ffn_importances_updates[idx] += (self.ffn_mask[idx]) * (torch.rand_like(self.ffn_importances[idx]))
                self.qk_importances_updates[idx] += (self.qk_mask[idx]) * (torch.rand_like(self.qk_importances[idx]))
                self.vo_importances_updates[idx] += (self.vo_mask[idx]) * (torch.rand_like(self.vo_importances[idx]))
            else:
                raise ValueError

    def generate_mask(self, weights_ratio, is_global=False, is_reweight=False):
        if is_global is True:
            all_importances = torch.cat([self.ffn_importances.view(-1), 
                                         self.qk_importances.view(-1),
                                         self.vo_importances.view(-1)])
            all_order = torch.argsort(all_importances)
            sorted_normed_param_sizes = self.normed_param_sizes[all_order]
            cumsum_sorted_normed_param_sizes = torch.cumsum(sorted_normed_param_sizes,dim=0)
            num_zeros = torch.searchsorted(cumsum_sorted_normed_param_sizes, 1 - weights_ratio).item()
            if num_zeros > self.pre_num_zeros:

                all_topk = all_order[:num_zeros]
                ffn_indices = all_topk[all_topk<self.qk_idx_start]
                qk_indices = all_topk[torch.logical_and(all_topk >= self.qk_idx_start, all_topk < self.vo_idx_start)] - self.qk_idx_start
                vo_indices = all_topk[all_topk>=self.vo_idx_start] - self.vo_idx_start

                self.ffn_mask.fill_(1).view(-1)[ffn_indices] = 0
                self.qk_mask.fill_(1).view(-1)[qk_indices] = 0
                self.vo_mask.fill_(1).view(-1)[vo_indices] = 0

                vo_mask_3d = self.vo_mask.view(self.vo_mask.size(0),self.num_heads,-1)
                qk_mask_3d = self.qk_mask.view(self.qk_mask.size(0),self.num_heads,-1)
                self.qk_mask = (torch.any(vo_mask_3d==1,dim=-1,keepdim=True) * qk_mask_3d).view(self.qk_mask.size(0),self.qk_mask.size(1))

                actual_num_zeros = (self.ffn_mask==0).sum() + (self.qk_mask==0).sum() + (self.vo_mask==0).sum()
                assert num_zeros<=actual_num_zeros
                self.pre_num_zeros = actual_num_zeros

        else:
            ffn_density = int(self.ffn_mask.numel() * weights_ratio)
            qk_density = int(self.qk_mask.numel() * weights_ratio)
            vo_density = int(self.vo_mask.numel() * weights_ratio)
            if ffn_density < self.pre_ffn_density or qk_density < self.pre_qk_density or vo_density < self.pre_vo_density:
                self.pre_ffn_density = ffn_density
                self.pre_qk_density = qk_density
                self.pre_vo_density = vo_density

                ffn_order = torch.argsort(self.ffn_importances.view(-1))
                qk_order = torch.argsort(self.qk_importances.view(-1))
                vo_order = torch.argsort(self.vo_importances.view(-1))

                self.ffn_mask.fill_(1).view(-1)[ffn_order[:-ffn_density]] = 0
                self.qk_mask.fill_(1).view(-1)[qk_order[:-qk_density]] = 0
                self.vo_mask.fill_(1).view(-1)[vo_order[:-vo_density]] = 0

        self.ffn_importances = torch.where(self.ffn_mask==1,self.ffn_importances, torch.tensor(-float('inf'),device=self.ffn_mask.device))
        self.qk_importances = torch.where(self.qk_mask==1,self.qk_importances, torch.tensor(-float('inf'),device=self.ffn_mask.device))
        self.vo_importances = torch.where(self.vo_mask==1,self.vo_importances, torch.tensor(-float('inf'),device=self.ffn_mask.device))

    def prune_model(self, model, dynamic_head_size = False):

        for idx,layer in enumerate(model.base_model.encoder.layer):
            ffn1 = layer.intermediate.dense
            ffn2 = layer.output.dense
            hidden_size = ffn1.weight.size(1)

            ffn_mask = self.ffn_mask[idx] #(3072,)
            w2_mask = ffn_mask.unsqueeze(0).expand(ffn2.weight.size(0),-1)

            if hasattr(ffn2,'weight_mask'):
                ffn2.weight_mask = w2_mask
            else:
                torch_prune.custom_from_mask(ffn2,'weight',w2_mask)

            #-------attention part----------#
            query = layer.attention.self.query
            key = layer.attention.self.key
            value = layer.attention.self.value
            output = layer.attention.output.dense

            qk_mask = self.qk_mask[idx]
            vo_mask = self.vo_mask[idx]
            qk_weight_mask = qk_mask.unsqueeze(1).expand(-1,query.weight.size(1))
            vo_weight_mask = vo_mask.unsqueeze(1).expand(-1,value.weight.size(1))

            if hasattr(output,'weight_mask'):
                key.weight_mask = qk_weight_mask
                query.weight_mask = qk_weight_mask
                value.weight_mask = vo_weight_mask
                output.weight_mask = vo_weight_mask.transpose(0,1)
                key.bias_mask = qk_mask
                query.bias_mask = qk_mask
                value.bias_mask = vo_mask
            else:
                torch_prune.custom_from_mask(query,'weight',qk_weight_mask)
                torch_prune.custom_from_mask(key,'weight',qk_weight_mask)
                torch_prune.custom_from_mask(value,'weight',vo_weight_mask)
                torch_prune.custom_from_mask(output,'weight',vo_weight_mask.transpose(0,1))
                torch_prune.custom_from_mask(query,'bias',qk_mask)
                torch_prune.custom_from_mask(key,'bias',qk_mask)
                torch_prune.custom_from_mask(value,'bias',vo_mask)

            if dynamic_head_size is True:
                assert hasattr(layer.attention.self,'dynamic_attention_head_size')
                head_size = qk_mask.view(self.num_heads,-1).sum(-1)
                dynamic_attention_head_size = torch.max(torch.ones_like(head_size),head_size)
                layer.attention.self.dynamic_attention_head_size = dynamic_attention_head_size