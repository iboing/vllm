#from transformers import LlamaForCausalLM
#from .llama import LlamaForCausalLM
from .llama_hf import LlamaForCausalLM
from .configuration_oursvd_llama import CovSVDLlamaConfig
import torch.nn as nn
import torch.nn.functional as F
import torch

##
from typing import Any, Dict, Iterable, List, Optional, Tuple
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

from vllm.config import LoRAConfig, CacheConfig
from vllm.model_executor.layers.linear import LinearMethodBase, MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.sampling_metadata import SamplingMetadata




class CovSVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.weight_residual = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_residual.requires_grad = False
        
    def forward(self, input):
        y = self.BLinear(input)
        y = self.ALinear(y) + F.linear(input, self.weight_residual)
        return y


class CovSVDLlamaForCausalLM(LlamaForCausalLM):
    config_class = CovSVDLlamaConfig
    def __init__(
        self, 
	config:CovSVDLlamaConfig,
    #linear_method: Optional[LinearMethodBase] = None,
    cache_config: Optional[CacheConfig] = None,
	quant_config: Optional[QuantizationConfig]=None,
	lora_config: Optional[LoRAConfig]=None,
    ):
        super().__init__(config)
        #self.truncation_ranks=config.truncation_ranks
        self.lora_r = config.lora_r

        '''
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.unpadded_vocab_size = config.vocab_size
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        
        '''

        full_name_dict = {module: name for name, module in self.named_modules()}
        linear_info = {}
        modules = [self]
        while len(modules) > 0:
            submodule = modules.pop()
            for name, raw_linear in submodule.named_children():
                #if isinstance(raw_linear, nn.Linear):
                if "proj" in name:                    
                    full_name = full_name_dict[raw_linear]
                    #print(full_name)
                    #linear_info[raw_linear] = {
                    #    "father": submodule,
                    #    "name": name,
                    #    "full_name": full_name,
                    #}
                    linear_info[full_name] = {
                        "father": submodule,
                        "name": name,
                        "module": raw_linear,
                    }
                else:
                    modules.append(raw_linear)
        
        del full_name_dict
        #print(linear_info)
        keys = list(linear_info.keys())
        #for name,module in self.named_modules():
        for name in keys:
            #print("name:",name)
            info = linear_info[name]
            module = info["module"]
            #print("module:",module)
            if "lm_head" not in name and isinstance(module, nn.Linear):
                #print("XXX --- init CovSVD model processed:", name)
                #info=linear_info[module]
                new_layer=CovSVDLinear(module.in_features, module.out_features, self.lora_r, bias=module.bias is not None)
                delattr(info["father"], info["name"])
                setattr(info["father"], info["name"], new_layer)
                del linear_info[name], module, info
                torch.cuda.empty_cache()

        print(self.model)

        
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        #print("pretrained CovSVD model param names", params_dict.keys())
        print("loading parameters ...")
        for name, loaded_weight in weights:
            #print(name)
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            '''
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]      # KeyError: 'model.layers.22.mlp.down_proj.ALinear.weight'
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            '''
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            del loaded_weight
            torch.cuda.empty_cache()
        print("load params finished")
        del weights
