import torch
import json

# From Croco v2 repository
def interpolate_pos_embed(model, new_img_size):
    print(model, vars(model))
    keys = ['enc_pos_embed']+(['dec_pos_embed'] if hasattr(model,'dec_blocks') else [])
    for k in keys:
        pos_embed_checkpoint = getattr(model, k)
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens = 0 # no cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = (new_img_size[0]//model.patch_embed.patch_size[0],new_img_size[1]//model.patch_embed.patch_size[1])
        if orig_size != new_size[0] or orig_size != new_size[1]:
            print("Position interpolate %s from %dx%d to %dx%d" % (k, orig_size, orig_size, new_size[0], new_size[1]))
            extra_tokens = pos_embed_checkpoint[:num_extra_tokens,:]
            pos_tokens = pos_embed_checkpoint[num_extra_tokens:,:]
            pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            #checkpoint_model[k] = new_pos_embed.squeeze(0)
            setattr(model, k, new_pos_embed.squeeze(0))

def get_parameter_groups(model, weight_decay, layer_decay=1.0, skip_list=(), no_lr_scale_list=[]):
    parameter_group_names = {}
    parameter_group_vars = {}
    enc_depth, dec_depth = None, None
    # prepare layer decay values 
    assert layer_decay==1.0 or 0.<layer_decay<1.
    if layer_decay<1.:
        enc_depth = model.enc_depth
        dec_depth = model.dec_depth if hasattr(model, 'dec_blocks') else 0
        num_layers = enc_depth+dec_depth
        layer_decay_values = list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
        
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        # Assign weight decay values
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        # Assign layer ID for LR scaling
        if layer_decay<1.:
            skip_scale = False
            layer_id = _get_num_layer_for_vit(name, enc_depth, dec_depth)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f'{group_name}_no_lr_scale'
        else:
            layer_id = 0
            skip_scale = True

        if group_name not in parameter_group_names:
            if not skip_scale:
                scale = layer_decay_values[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

