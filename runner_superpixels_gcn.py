import os
import time
import glob
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from nets.load_net import gnn_model
from alisuretool.Tools import Tools
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from data.superpixels import SuperPixDataset


class Parameters(object):

    def __init__(self, dataset, model_name, dataset_name, out_dir, batch_size, use_gpu, gpu_id):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.dataset = dataset
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        self.seed = 41
        self.epochs = 1000
        self.batch_size = batch_size
        self.init_lr = 0.001
        self.lr_reduce_factor = 0.5
        self.lr_schedule_patience = 5
        self.min_lr = 1e-5
        self.weight_decay = 0.0
        self.print_epoch_interval = 5
        self.max_time = 48

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

        self.in_dim = self.dataset.train[0][0].ndata['feat'][0].size(0)
        self.in_dim_edge = self.dataset.train[0][0].edata['feat'][0].size(0)
        self.n_classes = len(np.unique(np.array(self.dataset.train[:][1])))

        ##########################################################################

        self.device = self._gpu_setup(use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        self.root_log_dir, self.root_ckpt_dir = self._set_dir()
        self._set_seed(self.seed, device=self.device)

        ##########################################################################

        self._set_self_loop()
        self.drop_last, self.assign_dim = self._set_diff_pool()

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

    def _set_self_loop(self):
        if self.model_name in ['GCN', 'GAT'] and hasattr(self, "self_loop") and self.self_loop:
            Tools.print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            self.dataset.add_self_loops()
            pass
        pass

    def _set_diff_pool(self):
        drop_last = True if self.model_name == 'DiffPool' else False
        assign_dim = None
        if self.model_name == 'DiffPool':
            if hasattr(self, 'pool_ratio'):
                max_train = max([self.dataset.train[i][0].number_of_nodes() for i in range(len(self.dataset.train))])
                max_test = max([self.dataset.test[i][0].number_of_nodes() for i in range(len(self.dataset.test))])
                assign_dim = int(max(max_train, max_test) * self.pool_ratio) * self.batch_size
            pass
        return drop_last, assign_dim

    def _set_dir(self):
        _file_name = "{}_{}_GPU{}_{}".format(self.model_name, self.dataset_name,
                                             self.gpu_id, time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y'))
        root_log_dir = Tools.new_dir("{}/logs/{}".format(self.out_dir, _file_name))
        root_ckpt_dir = Tools.new_dir("{}/checkpoints/{}".format(self.out_dir, _file_name))
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


class Runner(object):

    def __init__(self, params):
        self.params = params
        self.dataset = self.params.dataset

        self.writer = SummaryWriter(log_dir=os.path.join(self.params.root_log_dir, "RUN_" + str(0)))

        self.model = gnn_model(self.params).to(self.params.device)
        Tools.print("Total param: {}".format(self.view_model_param(self.model)))

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.params.init_lr,  weight_decay=self.params.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=self.params.lr_reduce_factor,
            patience=self.params.lr_schedule_patience, verbose=True)

        self.loss_ce = nn.CrossEntropyLoss()

        self.train_loader = DataLoader(self.dataset.train, batch_size=self.params.batch_size, shuffle=True,
                                       drop_last=self.params.drop_last, collate_fn=self.dataset.collate)
        self.val_loader = DataLoader(self.dataset.val, batch_size=self.params.batch_size, shuffle=False,
                                     drop_last=self.params.drop_last, collate_fn=self.dataset.collate)
        self.test_loader = DataLoader(self.dataset.test, batch_size=self.params.batch_size, shuffle=False,
                                      drop_last=self.params.drop_last, collate_fn=self.dataset.collate)
        pass

    def train_val_pipeline(self):
        t0, per_epoch_time = time.time(), []
        epoch_train_losses, epoch_val_losses, epoch_train_accs, epoch_val_accs = [], [], [], []
        for epoch in range(self.params.epochs):
            start = time.time()
            Tools.print()
            Tools.print("Start Epoch {}".format(epoch))

            epoch_train_loss, epoch_train_acc = self.train_epoch(self.params.device, self.train_loader)
            epoch_val_loss, epoch_val_acc = self.evaluate_network(self.params.device, self.val_loader)
            epoch_test_loss, epoch_test_acc = self.evaluate_network(self.params.device, self.test_loader)

            self.scheduler.step(epoch_val_loss)
            self.save_checkpoint(self.model, self.params.root_ckpt_dir, epoch)

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            epoch_train_accs.append(epoch_train_acc)
            epoch_val_accs.append(epoch_val_acc)

            self.writer.add_scalar('train/_loss', epoch_train_loss, epoch)
            self.writer.add_scalar('val/_loss', epoch_val_loss, epoch)
            self.writer.add_scalar('train/_acc', epoch_train_acc, epoch)
            self.writer.add_scalar('val/_acc', epoch_val_acc, epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            per_epoch_time.append(time.time() - start)
            Tools.print("time={:.4f}, lr={:.4f}, loss={:.4f}/{:.4f}/{:.4f}, acc={:.4f}/{:.4f}/{:.4f}".format(
                time.time() - start, self.optimizer.param_groups[0]['lr'], epoch_train_loss,
                epoch_val_loss, epoch_test_loss, epoch_train_acc, epoch_val_acc, epoch_test_acc))

            # Stop training
            if self.optimizer.param_groups[0]['lr'] < self.params.min_lr:
                Tools.print()
                Tools.print("\n!! LR EQUAL TO MIN LR SET.")
                break
            if time.time() - t0 > self.params.max_time * 3600:
                Tools.print()
                Tools.print("Max_time for training elapsed {:.2f} hours, so stopping".format(self.params.max_time))
                break

            pass

        _, val_acc = self.evaluate_network(self.params.device, self.val_loader)
        _, test_acc = self.evaluate_network(self.params.device, self.test_loader)
        _, train_acc = self.evaluate_network(self.params.device, self.train_loader)

        Tools.print()
        Tools.print("Val Accuracy: {:.4f}".format(val_acc))
        Tools.print("Test Accuracy: {:.4f}".format(test_acc))
        Tools.print("Train Accuracy: {:.4f}".format(train_acc))
        Tools.print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
        Tools.print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

        self.writer.close()
        pass

    def train_epoch(self, device, data_loader):
        self.model.train()

        epoch_loss, epoch_train_acc, nb_data = 0, 0, 0
        for _, (batch_graphs, batch_labels,
                batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt) in enumerate(data_loader):
            batch_nodes_feat = batch_graphs.ndata['feat'].to(device)  # num x feat
            batch_edges_feat = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            batch_nodes_num_norm_sqrt = batch_nodes_num_norm_sqrt.to(device)  # num x 1
            batch_edges_num_norm_sqrt = batch_edges_num_norm_sqrt.to(device)

            self.optimizer.zero_grad()

            batch_scores = self.model.forward(batch_graphs, batch_nodes_feat, batch_edges_feat,
                                              batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt)
            loss = self.loss_ce(batch_scores, batch_labels)

            loss.backward()
            self.optimizer.step()

            nb_data += batch_labels.size(0)
            epoch_loss += loss.detach().item()
            epoch_train_acc += self.accuracy(batch_scores, batch_labels)
            pass

        epoch_train_acc /= nb_data
        epoch_loss /= (len(data_loader) + 1)

        return epoch_loss, epoch_train_acc

    def evaluate_network(self, device, data_loader):
        self.model.eval()

        epoch_test_loss, epoch_test_acc, nb_data = 0, 0, 0
        with torch.no_grad():
            for _, (batch_graphs, batch_labels,
                    batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt) in enumerate(data_loader):
                batch_nodes_feat = batch_graphs.ndata['feat'].to(device)
                batch_edges_feat = batch_graphs.edata['feat'].to(device)
                batch_labels = batch_labels.to(device)
                batch_nodes_num_norm_sqrt = batch_nodes_num_norm_sqrt.to(device)
                batch_edges_num_norm_sqrt = batch_edges_num_norm_sqrt.to(device)

                batch_scores = self.model.forward(batch_graphs, batch_nodes_feat, batch_edges_feat,
                                                  batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt)
                loss = self.loss_ce(batch_scores, batch_labels)

                nb_data += batch_labels.size(0)
                epoch_test_loss += loss.detach().item()
                epoch_test_acc += self.accuracy(batch_scores, batch_labels)
                pass

            epoch_test_loss /= (len(data_loader) + 1)
            epoch_test_acc /= nb_data
            pass

        return epoch_test_loss, epoch_test_acc

    @staticmethod
    def save_checkpoint(model, root_ckpt_dir, epoch):
        torch.save(model.state_dict(), os.path.join(root_ckpt_dir, 'epoch_{}.pkl'.format(epoch)))
        for file in glob.glob(root_ckpt_dir + '/*.pkl'):
            if int(file.split('_')[-1].split('.')[0]) < epoch - 1:
                os.remove(file)
                pass
            pass
        pass

    @staticmethod
    def accuracy(scores, targets):
        return (scores.detach().argmax(dim=1) == targets).float().sum().item()

    @staticmethod
    def view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


if __name__ == '__main__':
    now_gpu_id = "0"
    now_batch_size = 128
    now_model_name = "GCN"  # GCN or MLP
    # now_model_name = "MLP"  # GCN or MLP
    now_dataset_name = "MNIST"  # MNIST or CIFAR10
    # now_dataset_name = "CIFAR10"  # MNIST or CIFAR10
    now_run_name = "{}_{}_demo".format(now_dataset_name, now_model_name)

    now_dataset = SuperPixDataset(now_dataset_name, data_file="/mnt/4T/ALISURE/GCN/{}.pkl".format(now_dataset_name))
    now_params = Parameters(dataset=now_dataset, dataset_name=now_dataset_name,
                            out_dir=Tools.new_dir("result/{}".format(now_run_name)),
                            model_name=now_model_name, batch_size=now_batch_size, use_gpu=True, gpu_id=now_gpu_id)
    now_params.print_info()

    runner = Runner(params=now_params)
    runner.train_val_pipeline()
