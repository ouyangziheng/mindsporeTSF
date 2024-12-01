import mindspore
from mindspore import nn
from mindspore import Tensor

class Model(nn.Cell):
    def __init__(self, configs):
        super(Model, self).__init__()

        # 获取参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # 1D卷积层
        self.conv1d = nn.Conv1d(
            in_channels=1,              # 输入通道数
            out_channels=1,             # 输出通道数
            kernel_size=1 + 2 * (self.period_len // 2),  # 卷积核的大小
            stride=1,                   # 步长
            pad_mode="pad",             # 使用 "pad" 模式进行手动填充
            padding=self.period_len // 2,  # 填充大小                 # 禁用偏置项
        )

        # 根据模型类型选择对应的层
        if self.model_type == 'linear':
            self.linear = nn.Dense(self.seg_num_x, self.seg_num_y, has_bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.SequentialCell(
                nn.Dense(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Dense(self.d_model, self.seg_num_y)
            )

    def construct(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = mindspore.ops.ReduceMean(keep_dims=True)(x, 1)  # 使用 MindSpore 的均值计算
        x = (x - seq_mean).transpose(0, 2, 1)  # b,c,s -> b,s,c

        # 1D卷积聚合
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # 下采样操作: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).transpose(0, 2, 1)  # b,c,s -> bc,w,n

        # 稀疏预测
        if self.model_type == 'linear':
            y = self.linear(x)  # bc,w,m
        elif self.model_type == 'mlp':
            y = self.mlp(x)

        # 上采样: bc,w,m -> bc,m,w -> b,c,s
        y = y.transpose(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute 和去归一化
        y = y.transpose(0, 2, 1) + seq_mean

        return y
