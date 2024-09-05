import torch

def mask_top_rate(data, kept_rate):
    """
    Given a batch of samples(in a tensor form) and a rate of kept values,
    return a mask tensor which has the same shape as the input tensor. And
    the positions where the input tensor's value is within its topk range
    determined by the kept rate are set to be 1, others are set to be 0.

    Args:
        data (Tensor): The input samples
        kept_rate (float): The kept rate. Of each sample in the batch, how
            much ratio of the values from top are kept.

    Returns:
        Tensor: The mask tensor in the same shape as the input tensor.
    """
    data_flattened = data.view(data.size(0), -1)
    values_kept, _ = data_flattened.topk(int(data_flattened.size(1)*kept_rate), dim=1)
    values_min, _ = torch.min(values_kept, dim=-1)
    values_min = values_min.unsqueeze(-1).repeat(1, data_flattened.size(-1))
    mask = torch.greater(data_flattened, values_min).float().view(data.size())
    return mask