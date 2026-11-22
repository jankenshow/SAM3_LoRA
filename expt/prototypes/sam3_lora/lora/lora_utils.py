"""Compatibility exports for archived prototypes; new code uses adapters/."""
from sam3_lora.adapters import *
from sam3_lora.adapters.checkpoint import adapter_state_dict as get_lora_state_dict, load_adapter_state as load_lora_state_dict

def inject_lora_into_model(model, config, verbose=True):
    return apply_lora_to_model(model, config)

def print_trainable_parameters(model):
    print(count_parameters(model))

def merge_lora_weights(model):
    merge_lora_linears(model)
    restore_multihead_attention(model)
    return model
