from mindspore.dataset import GeneratorDataset
from data_provider.data_loader_ms import Dataset_ETT_hour
from data_provider.data_loader import Dataset_Pred

data_dict = {
    'ETTh1': Dataset_ETT_hour,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # fix bug
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    dataset = GeneratorDataset(data_set.generator(), column_names=['seq_x', 'seq_y', 'seq_x_mark', 'seq_y_mark'])
    dataset = dataset.batch(batch_size, drop_remainder=drop_last)
    return data_set, dataset
