import os
import time
import torch
import random
import numpy as np
from alisuretool.Tools import Tools


class Parameters(object):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id,
                 epochs=1000, seed=41, init_lr=0.001, lr_reduce_factor=0.5, lr_schedule_patience=5,
                 min_lr=1e-5, weight_decay=0.0, print_epoch_interval=5, max_time=48):
        self.dataset = dataset
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        ##########################################################################
        self.epochs = epochs
        self.seed = seed
        self.init_lr = init_lr
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_schedule_patience = lr_schedule_patience
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.print_epoch_interval = print_epoch_interval
        self.max_time = max_time
        ##########################################################################
        self.device = self._gpu_setup(use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        self.root_log_dir, self.root_ckpt_dir = self._set_dir(self.model_name, self.dataset_name,
                                                              self.gpu_id, self.out_dir)
        self._set_seed(self.seed, device=self.device)
        ##########################################################################
        self.in_dim = self.dataset.train[0][0].ndata['feat'][0].size(0)
        self.in_dim_edge = self.dataset.train[0][0].edata['feat'][0].size(0)
        self.n_classes = len(np.unique(np.array(self.dataset.train[:][1])))
        ##########################################################################
        self.drop_last = True if self.model_name == 'DiffPool' else False
        ##########################################################################
        # self.L = 4
        # self.hidden_dim = 146
        # self.out_dim = 146
        # self.residual = True
        # self.readout = "mean"
        # self.in_feat_dropout = 0.0
        # self.dropout = 0.0
        # self.graph_norm = True
        # self.batch_norm = True
        # self.self_loop = False
        ##########################################################################
        pass

    def print_info(self):
        Tools.print()
        Tools.print("Dataset: {}, Model: {}".format(self.dataset_name, self.model_name))
        Tools.print("Training Graphs: {}".format(len(self.dataset.train)))
        Tools.print("Validation Graphs: {}".format(len(self.dataset.val)))
        Tools.print("Test Graphs: {}".format(len(self.dataset.test)))
        Tools.print("Number of Classes: {}".format(self.n_classes))

        Tools.print()
        Tools.print("Params: \n{}".format(str(self)))
        Tools.print()
        pass

    @staticmethod
    def _set_dir(model_name, dataset_name, gpu_id, out_dir):
        _file_name = "{}_{}_GPU{}_{}".format(model_name, dataset_name,
                                             gpu_id, time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y'))
        root_log_dir = Tools.new_dir("{}/logs/{}".format(out_dir, _file_name))
        root_ckpt_dir = Tools.new_dir("{}/checkpoints/{}".format(out_dir, _file_name))
        return root_log_dir, root_ckpt_dir

    @staticmethod
    def _gpu_setup(use_gpu, gpu_id):
        if torch.cuda.is_available() and use_gpu:
            Tools.print()
            Tools.print('Cuda available with GPU: {}'.format(torch.cuda.get_device_name(0)))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = torch.device("cuda")
        else:
            Tools.print()
            Tools.print('Cuda not available')
            device = torch.device("cpu")
        return device

    @staticmethod
    def _set_seed(seed, device):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            pass
        pass

    def __str__(self):
        n_str = ""
        p_n = 1
        for index, attr in enumerate(dir(self)):
            if attr[0] != "_":
                if attr == "dataset" or attr == "print_info":
                    continue
                n_str += attr + ":" + str(getattr(self, attr)) + " "
                if index % p_n == p_n - 1:
                    n_str += "\n"
                    pass
            pass
        return n_str

    pass


class ParametersGCN(Parameters):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id):
        super().__init__(dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id)

        ##########################################################################
        self.L = 4
        self.hidden_dim = 146
        self.out_dim = 146
        self.residual = True
        self.readout = "mean"
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.graph_norm = True
        self.batch_norm = True
        self.self_loop = False
        ##########################################################################
        if self.self_loop:
            Tools.print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            self.dataset.add_self_loops()
            pass
        ##########################################################################
        pass

    pass


class ParametersGAT(Parameters):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id):
        super().__init__(dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id)

        ##########################################################################
        self.L = 4
        self.hidden_dim = 19
        self.out_dim = 152
        self.residual = True
        self.readout = "mean"
        self.n_heads = 8
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.graph_norm = True
        self.batch_norm = True
        self.self_loop = False
        ##########################################################################
        if self.self_loop:
            Tools.print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            self.dataset.add_self_loops()
            pass
        ##########################################################################

        pass

    pass


class ParametersDiffPool(Parameters):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id):
        super().__init__(dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id)

        ##########################################################################
        self.L = 3
        self.hidden_dim = 32
        self.embedding_dim = 32 if self.dataset_name == "MNIST" else 16
        self.num_pool = 1
        self.residual = True
        self.readout = "mean"
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.graph_norm = True
        self.batch_norm = True
        self.sage_aggregator = "meanpool"
        self.data_mode = "default"
        self.linkpred = True
        self.cat = False
        ##########################################################################
        self.pool_ratio = 0.15
        _max_train = max([self.dataset.train[i][0].number_of_nodes() for i in range(len(self.dataset.train))])
        _max_test = max([self.dataset.test[i][0].number_of_nodes() for i in range(len(self.dataset.test))])
        self.assign_dim = int(max(_max_train, _max_test) * self.pool_ratio) * self.batch_size
        ##########################################################################
        pass

    pass


class ParametersGatedGCN(Parameters):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id):
        super().__init__(dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id)

        ##########################################################################
        self.L = 4
        self.hidden_dim = 70
        self.out_dim = 70
        self.residual = True
        self.readout = "mean"
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.graph_norm = True
        self.batch_norm = True
        ##########################################################################
        self.edge_feat = True
        ##########################################################################

        pass

    pass


class ParametersGIN(Parameters):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id):
        super().__init__(dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id)

        ##########################################################################
        self.L = 4
        self.hidden_dim = 110
        self.out_dim = 110
        self.residual = True
        self.readout = "sum"
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.graph_norm = True
        self.batch_norm = True
        ##########################################################################
        self.n_mlp_GIN = 2
        self.learn_eps_GIN = True
        self.neighbor_aggr_GIN = "sum"
        ##########################################################################
        pass

    pass


class ParametersGraphSage(Parameters):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id):
        super().__init__(dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id)

        ##########################################################################
        self.L = 4
        self.hidden_dim = 108
        self.out_dim = 108
        self.residual = True
        self.readout = "mean"
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.graph_norm = True
        self.batch_norm = True
        ##########################################################################
        self.sage_aggregator = "meanpool"
        ##########################################################################
        pass

    pass


class ParametersMoNet(Parameters):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id):
        super().__init__(dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id)

        ##########################################################################
        self.L = 4
        self.hidden_dim = 90
        self.out_dim = 90
        self.residual = True
        self.readout = "mean"
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        self.graph_norm = True
        self.batch_norm = True
        ##########################################################################
        self.kernel = 3
        self.pseudo_dim_MoNet = 2
        ##########################################################################
        pass

    pass


class ParametersMLP(Parameters):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id):
        super().__init__(dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id)

        ##########################################################################
        self.L = 4
        self.hidden_dim = 168
        self.out_dim = 168
        self.readout = "mean"
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        ##########################################################################
        self.gated = False
        ##########################################################################
        pass

    pass


class ParametersMLPGated(Parameters):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id):
        super().__init__(dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id)

        ##########################################################################
        self.L = 4
        self.hidden_dim = 150
        self.out_dim = 150
        self.readout = "mean"
        self.in_feat_dropout = 0.0
        self.dropout = 0.0
        ##########################################################################
        self.gated = True
        ##########################################################################
        pass

    pass
