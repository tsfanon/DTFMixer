import random
import numpy as np
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_PEMS, Dataset_M4
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PEMS': Dataset_PEMS
}


def _build_loader(dataset, args, shuffle_flag, drop_last, collate_fn=None):
    pin_memory = bool(getattr(args, 'use_gpu', False))
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'shuffle': shuffle_flag,
        'num_workers': args.num_workers,
        'drop_last': drop_last,
        'pin_memory': pin_memory,
    }
    if collate_fn is not None:
        loader_kwargs['collate_fn'] = collate_fn

    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = getattr(args, 'prefetch_factor', 2)

        base_seed = getattr(args, 'seed', 0)

        def _worker_init_fn(worker_id):
            worker_seed = base_seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        loader_kwargs['worker_init_fn'] = _worker_init_fn

    return DataLoader(**loader_kwargs)


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    default_drop_last = True if flag == 'train' else False
    drop_last = getattr(args, 'drop_last', None)
    if drop_last is None:
        drop_last = default_drop_last
    freq = args.freq

    if args.task_name not in {'long_term_forecast', 'short_term_forecast'}:
        raise ValueError("Unsupported task_name. Please use: long_term_forecast or short_term_forecast.")

    if args.data == 'm4':
        drop_last = False
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    print(flag, len(data_set))
    data_loader = _build_loader(data_set, args, shuffle_flag, drop_last)
    return data_set, data_loader
