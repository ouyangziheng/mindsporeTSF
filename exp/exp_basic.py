import os
import mindspore
import mindspore.numpy as ms_np
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        # Placeholder for building model - replace with actual model instantiation
        print("Building the model...")
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            device_id = int(self.args.gpu) if not self.args.use_multi_gpu else [int(d) for d in self.args.devices.split(',')]
            context = mindspore.context
            context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=device_id)
            print(f'Use GPU: {device_id}')
        else:
            mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_target="CPU")
            print('Use CPU')
        return "device set"

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

# Example usage of Exp_Basic if needed
if __name__ == "__main__":
    class Args:
        use_gpu = True
        gpu = '0'
        use_multi_gpu = False
        devices = '0,1'
    
    args = Args()
    experiment = Exp_Basic(args)
    print(experiment.device)
