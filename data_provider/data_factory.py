import mindspore.dataset as ds
import numpy as np
from data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Solar

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'Solar': Dataset_Solar
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
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

    # 创建数据集实例
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

    # 使用 GeneratorDataset 将数据集转换为 MindSpore 可用的数据集
    # 需要确保 __getitem__ 返回的是 NumPy 格式的数据
    generator_ds = ds.GeneratorDataset(
        source=data_set,
        column_names=["data", "label"],  # 这里的 column_names 需要与 Dataset 的输出一致
        shuffle=shuffle_flag
    )

    # 设置 batch_size 和 drop_last 参数
    generator_ds = generator_ds.batch(batch_size=batch_size, drop_remainder=drop_last)

    # 返回数据集和加载器（在 MindSpore 中没有明确的 DataLoader，对应的是 Dataset 本身）
    return data_set, generator_ds

