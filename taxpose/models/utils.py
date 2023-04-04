import torch_geometric.data as tgd


def norm_scale_batch(batch):
    def _norm_scale_data(data):
        shift = data.pos.mean(dim=0)
        data.pos = data.pos - shift
        scale = data.pos.abs().max()
        data.pos = data.pos / scale

        return data

    return tgd.Batch.from_data_list(
        [_norm_scale_data(data) for data in batch.to_data_list()]
    )
