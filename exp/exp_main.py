from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, PatchTST, SparseTSF
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.train.callback import LearningRateScheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SparseTSF': SparseTSF
        }
        model = model_dict[self.args.model].Model(self.args).to_float(mindspore.float32)
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
            batch_x = Tensor(batch_x, mindspore.float32)
            batch_y = Tensor(batch_y, mindspore.float32)

            batch_x_mark = Tensor(batch_x_mark, mindspore.float32)
            batch_y_mark = Tensor(batch_y_mark, mindspore.float32)

            # decoder input
            dec_inp = mindspore.ops.ZerosLike()(batch_y[:, -self.args.pred_len:, :])
            dec_inp = mindspore.ops.Concat(1)([batch_y[:, :self.args.label_len, :], dec_inp])
            
            if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                outputs = self.model(batch_x)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            loss = criterion(outputs, batch_y)
            total_loss.append(loss.asnumpy())
        
        total_loss = np.average(total_loss)
        self.model.set_train(True)
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = LearningRateScheduler(learning_rate=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.set_train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.clear_grad()
                batch_x = Tensor(batch_x, mindspore.float32)
                batch_y = Tensor(batch_y, mindspore.float32)
                batch_x_mark = Tensor(batch_x_mark, mindspore.float32)
                batch_y_mark = Tensor(batch_y_mark, mindspore.float32)

                # decoder input
                dec_inp = mindspore.ops.ZerosLike()(batch_y[:, -self.args.pred_len:, :])
                dec_inp = mindspore.ops.Concat(1)([batch_y[:, :self.args.label_len, :], dec_inp])

                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.asnumpy())

                loss.backward()
                model_optim.step()
                scheduler(model_optim)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.asnumpy()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.ckpt'
        mindspore.save_checkpoint(self.model, best_model_path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            mindspore.load_param_into_net(self.model, mindspore.load_checkpoint(os.path.join('./checkpoints/' + setting, 'checkpoint.ckpt')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.set_train(False)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = Tensor(batch_x, mindspore.float32)
            batch_y = Tensor(batch_y, mindspore.float32)

            batch_x_mark = Tensor(batch_x_mark, mindspore.float32)
            batch_y_mark = Tensor(batch_y_mark, mindspore.float32)

            # decoder input
            dec_inp = mindspore.ops.ZerosLike()(batch_y[:, -self.args.pred_len:, :])
            dec_inp = mindspore.ops.Concat(1)([batch_y[:, :self.args.label_len, :], dec_inp])

            if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                outputs = self.model(batch_x)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.asnumpy()
            batch_y = batch_y.asnumpy()

            preds.append(outputs)
            trues.append(batch_y)

            if i % 20 == 0:
                input = batch_x.asnumpy()
                gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # fix bug
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.ckpt'
            mindspore.load_param_into_net(self.model, mindspore.load_checkpoint(best_model_path))

        preds = []

        self.model.set_train(False)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = Tensor(batch_x, mindspore.float32)
            batch_y = Tensor(batch_y, mindspore.float32)
            batch_x_mark = Tensor(batch_x_mark, mindspore.float32)
            batch_y_mark = Tensor(batch_y_mark, mindspore.float32)

            # decoder input
            dec_inp = mindspore.ops.Zeros()([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]], mindspore.float32)
            dec_inp = mindspore.ops.Concat(1)([batch_y[:, :self.args.label_len, :], dec_inp])

            if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                outputs = self.model(batch_x)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            preds.append(outputs.asnumpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
