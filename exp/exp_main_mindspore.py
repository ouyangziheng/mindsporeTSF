'''
main file written in mindspore
'''
import os
import time
import numpy as np
import warnings
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn, ops
from mindspore import context
from mindspore.train import Model
from mindspore.train.callback import EarlyStopping
from mindspore import Tensor
from utils.tools import adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from data_provider.data_factory_ms import data_provider
from models import SparseTSF_mindspore

warnings.filterwarnings('ignore')
class OneCycleLR:
    def __init__(self, optimizer, steps_per_epoch, pct_start, epochs, max_lr, min_lr=0):
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.epochs = epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = steps_per_epoch * epochs
        self.steps_up = int(self.total_steps * pct_start)  
        self.steps_down = self.total_steps - self.steps_up  
        self.base_lr = min_lr
    
    def get_lr(self, current_step):
        if current_step < self.steps_up:
            
            return self.base_lr + (self.max_lr - self.base_lr) * (current_step / self.steps_up)
        else:
            
            return self.max_lr - (self.max_lr - self.min_lr) * ((current_step - self.steps_up) / self.steps_down)
    
    def step(self, current_step):
        new_lr = self.get_lr(current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
class TrainOneStep(nn.Cell):
    def __init__(self,model,optimizer,criterion,args):
        super(TrainOneStep,self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.grad_fn = ops.GradOperation(get_by_list=True)
        self.weights = optimizer.parameters
        self.args = args
    def construct(self, batch_x,batch_y,batch_x_mark,batch_y_mark,dec_inp):
        def forward_fn(batch_x,batch_y,batch_x_mark,batch_y_mark,dec_inp):
            outputs = self.model(batch_x)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
            loss = self.criterion(outputs, batch_y)
            return loss
        loss = forward_fn(batch_x,batch_y,batch_x_mark,batch_y_mark,dec_inp)
        grads = self.grad_fn(forward_fn,self.weights)(batch_x,batch_y,batch_x_mark,batch_y_mark,dec_inp) 
        self.optimizer(grads)       
        return loss
    
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()  
        self.model = self._build_model() 
        # self.model.to(self.device) 

    def _build_model(self):
        raise NotImplementedError 
        return None

    def _acquire_device(self):
        """根据 args 配置选择设备"""
        if self.args.use_gpu:
            # 设置 MindSpore 使用的设备
            device = "GPU" if not self.args.use_multi_gpu else self.args.devices
            context.set_context(device_target=device)  # 配置 MindSpore 的设备目标
            print(f"Use GPU: {device}")
        else:
            context.set_context(device_target="CPU")  # 如果不使用 GPU，使用 CPU
            device = "CPU"
            print("Use CPU")
        return device

    def _get_data(self):
        """获取数据的占位方法，子类可以根据需求实现"""
        pass

    def vali(self):
        """验证模型的占位方法，子类可以根据需求实现"""
        pass

    def train(self):
        """训练模型的占位方法，子类可以根据需求实现"""
        pass

    def test(self):
        """测试模型的占位方法，子类可以根据需求实现"""
        pass

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.device = "GPU" if args.use_gpu else "CPU"
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU" if args.use_gpu else "CPU")

    def _build_model(self):
        model_dict = {
            'SparseTSF': SparseTSF_mindspore
        }
        model = model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = nn.Adam(self.model.trainable_params(), learning_rate=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.set_train(False)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = Tensor(batch_x, ms.float32)
            batch_y = Tensor(batch_y, ms.float32)
            batch_x_mark = Tensor(batch_x_mark, ms.float32)
            batch_y_mark = Tensor(batch_y_mark, ms.float32)
            dec_inp = ops.zeros_like(batch_y[:,-self.args.pred_len:,:])
            dec_inp = ops.Concat(1)([batch_y[:, :self.args.label_len, :], dec_inp])

            if self.args.use_amp:
                with ms.amp.auto_mixed_precision():
                    outputs = self.model(batch_x)
            else:
                outputs = self.model(batch_x)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            pred = outputs
            true = batch_y

            loss = criterion(pred, true)
            total_loss.append(loss.item())

        total_loss = np.mean(total_loss)
        self.model.set_train(True)
        return total_loss

    def forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark, dec_inp):
        pred_len = self.args.pred_len
        criterion = self._select_criterion()

        if self.args.use_amp:
            with ms.amp.auto_mixed_precision():
                outputs = self.model(batch_x)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            loss = criterion(outputs, batch_y)
        else:
            outputs = self.model(batch_x)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            loss = criterion(outputs, batch_y)

        return loss, outputs

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        model_optim = self._select_optimizer()
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        criterion = self._select_criterion()
        train_onestep = TrainOneStep(self.model, model_optim, criterion, self.args)
        scheduler = OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        best_vali_loss = float('inf')  

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.set_train(True)
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                batch_x = Tensor(batch_x, ms.float32)
                batch_y = Tensor(batch_y, ms.float32)
                batch_x_mark = Tensor(batch_x_mark, ms.float32)
                batch_y_mark = Tensor(batch_y_mark, ms.float32)
                batch_y = batch_y.asnumpy().astype(np.float32)

                dec_inp = Tensor(np.zeros((batch_y.shape[0], self.args.pred_len, batch_y.shape[2]), dtype=np.float32), ms.float32)
                batch_y = Tensor(batch_y, ms.float32)
                dec_inp = ops.Concat(1)([batch_y[:, :self.args.label_len, :], dec_inp])

                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                outputs = Tensor(outputs, ms.float32)
                loss = train_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    avg_loss = np.mean(train_loss)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * len(train_loader) - i)
                    print(f"Iteration {i + 1}, Epoch {epoch + 1} | Loss: {loss:.7f}")
                    print(f"Speed: {speed:.4f}s/iter; Estimated time left: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            avg_loss = np.mean(train_loss)
            epoch_duration = time.time() - epoch_time
            print(f"Epoch [{epoch+1}/{self.args.train_epochs}], Loss: {avg_loss:.4f}, Time: {epoch_duration:.2f}s")

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.7f}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")

            if vali_loss < best_vali_loss:  
                best_vali_loss = vali_loss
                best_model_path = os.path.join(path, 'best_model.ckpt')
                ms.save_checkpoint(self.model, best_model_path)  # 保存模型

            # 在此处可以使用 early_stopping 进行早期停止
            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

        # 加载最佳模型（即验证集损失最低的模型）
        self.model.set_train(False)
        # self.model.load_state_dict(ms.load_checkpoint(best_model_path))
         
        # param_dict = ms.load_checkpoint(path+'best_model.ckpt')
        # ms.load_param_into_net(self.model, param_dict)
        
        model_path = path+'/'+'checkpoint.ckpt'
        checkpoint_dir = os.path.dirname(model_path)
        if not os.path.exists(checkpoint_dir):
            print(f"路径 {checkpoint_dir} 不存在，正在创建...")
            os.makedirs(checkpoint_dir)

        # 加载模型参数
        if os.path.exists(model_path):
            param_dict = ms.load_checkpoint(model_path)  # 加载检查点
            ms.load_param_into_net(self.model, param_dict)  # 将参数加载到模型中
            print(f"成功加载模型：{model_path}")
        else:
            print(f"错误：找不到检查点文件：{model_path}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(ms.load_checkpoint(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.set_train(False)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            cast = ops.Cast()
            batch_x = cast(batch_x,ms.float32)
            batch_y = cast(batch_y,ms.float32)
            batch_x_mark = cast(batch_x_mark,ms.float32)
            batch_y_mark = cast(batch_y_mark,ms.float32)

            # dec_inp = Tensor(np.zeros_like(batch_y[:, -self.args.pred_len:, :]), ms.float32)
            # dec_inp = ops.Concat(1)([batch_y[:, :self.args.label_len, :], dec_inp])

            outputs = self.model(batch_x)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            pred = outputs.asnumpy()
            true = batch_y.asnumpy()

            preds.append(pred)
            trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe, *extra = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        
        return preds, trues
        

    def predict(self, setting, load=False):
        
        pred_data, pred_loader = self._get_data(flag='pred')


        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(ms.load_checkpoint(best_model_path))

        preds = []

        self.model.set_train(False)  
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = Tensor(batch_x, ms.float32).to(self.device)
            batch_y = Tensor(batch_y, ms.float32)
            batch_x_mark = Tensor(batch_x_mark, ms.float32).to(self.device)
            batch_y_mark = Tensor(batch_y_mark, ms.float32).to(self.device)

            dec_inp = Tensor(np.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]), ms.float32)
            dec_inp = ops.Concat(1)([batch_y[:, :self.args.label_len, :], dec_inp])  # 拼接输入

            with ms.no_grad():
                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            pred = outputs.asnumpy()
            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])  


        folder_path = os.path.join('./results', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)

        return preds
