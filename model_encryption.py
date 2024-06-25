import torch


def weight_extracting(embedding_module):
    ce_weight = None
    pos_weight = None

    with torch.no_grad():
        for name, param in embedding_module.named_parameters():
            # print(f"Name: {name}, Shape: {param.shape}")

            if name == "patch_embeddings.projection.weight":
                ce_weight = param.clone()

            if name == "position_embeddings":
                pos_weight = param.clone()

    return ce_weight, pos_weight


def weight_reloading(embedding_module, new_ce_weight, new_pos_weight):
    with torch.no_grad():
        for name, param in embedding_module.named_parameters():
            if name == "patch_embeddings.projection.weight":
                param.copy_(new_ce_weight)

            if name == "position_embeddings":
                param.copy_(new_pos_weight)

    return embedding_module


def cube_embedding_shuffling(ce_weight, shuffling_order):
    print("Shuffling weight of Patch embedding...")
    original_shape = ce_weight.shape
    shuffled_weight = torch.flatten(ce_weight, start_dim=1)
    shuffled_weight = shuffled_weight[..., shuffling_order]
    shuffled_weight = shuffled_weight.reshape(original_shape)

    return shuffled_weight


def pos_embedding_shuffling(pos_weight, shuffling_order):
    print("Shuffling weight of Position embedding...")
    # specify the elements except a from cls token
    shuffling_scope = pos_weight[:, 1:, :]
    pos_weight[:, 1:, :] = shuffling_scope[:, shuffling_order, :]
    return pos_weight