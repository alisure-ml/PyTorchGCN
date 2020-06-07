import os
import time
import glob
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from nets.load_net import gnn_model
from alisuretool.Tools import Tools
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from data.superpixels import SuperPixDataset
from parameters.load_parameter_abl import GNNParameter


class Runner(object):

    def __init__(self, params):
        self.params = params
        self.dataset = self.params.dataset

        self.writer = SummaryWriter(log_dir=os.path.join(self.params.root_log_dir, "RUN_" + str(0)))

        self.model = gnn_model(self.params).to(self.params.device)
        Tools.print("Total param: {}".format(self.view_model_param(self.model)))

        if self.params.is_sgd:
            self.lr_s = [[0, 0.1], [80, 0.01], [140, 0.001], [180, 0.0001]]
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.lr_s[0][0], momentum=0.9, weight_decay=5e-4)
        else:
            self.lr_s = [[0, 0.001], [25, 0.001], [50, 0.0003], [75, 0.0001]]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_s[0][0], weight_decay=0.0)

        self.loss_ce = nn.CrossEntropyLoss()

        self.train_loader = DataLoader(self.dataset.train, batch_size=self.params.batch_size, shuffle=True,
                                       drop_last=self.params.drop_last, collate_fn=self.dataset.collate)
        self.val_loader = DataLoader(self.dataset.val, batch_size=self.params.batch_size, shuffle=False,
                                     drop_last=self.params.drop_last, collate_fn=self.dataset.collate)
        self.test_loader = DataLoader(self.dataset.test, batch_size=self.params.batch_size, shuffle=False,
                                      drop_last=self.params.drop_last, collate_fn=self.dataset.collate)
        pass

    def _lr(self, epoch):
        # [[0, 0.001], [25, 0.001], [50, 0.0002], [75, 0.00004]]
        for lr in self.lr_s:
            if lr[0] == epoch:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr[1]
                pass
            pass
        pass

    def train_val_pipeline(self):
        t0, per_epoch_time = time.time(), []
        epoch_train_losses, epoch_val_losses, epoch_train_accs, epoch_val_accs = [], [], [], []
        for epoch in range(self.params.epochs):
            self._lr(epoch)

            start = time.time()
            Tools.print()
            Tools.print("Start Epoch {}".format(epoch))
            Tools.print('Epoch:{:02d},lr={:.4f}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            epoch_train_loss, epoch_train_acc = self.train_epoch(self.train_loader)
            epoch_val_loss, epoch_val_acc = self.evaluate_network(self.val_loader)
            epoch_test_loss, epoch_test_acc = self.evaluate_network(self.test_loader)

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
            pass

        _, val_acc = self.evaluate_network(self.val_loader)
        _, test_acc = self.evaluate_network(self.test_loader)
        _, train_acc = self.evaluate_network(self.train_loader)

        Tools.print()
        Tools.print("Val Accuracy: {:.4f}".format(val_acc))
        Tools.print("Test Accuracy: {:.4f}".format(test_acc))
        Tools.print("Train Accuracy: {:.4f}".format(train_acc))
        Tools.print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
        Tools.print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

        self.writer.close()
        pass

    def train_epoch(self, data_loader):
        self.model.train()

        epoch_loss, epoch_train_acc, nb_data = 0, 0, 0
        for _, (batch_graphs, batch_labels, batch_nodes_num_norm_sqrt,
                batch_edges_num_norm_sqrt) in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch_nodes_feat = batch_graphs.ndata['feat'].to(self.params.device)  # num x feat
            batch_edges_feat = batch_graphs.edata['feat'].to(self.params.device)
            batch_labels = batch_labels.to(self.params.device)
            batch_nodes_num_norm_sqrt = batch_nodes_num_norm_sqrt.to(self.params.device)  # num x 1
            batch_edges_num_norm_sqrt = batch_edges_num_norm_sqrt.to(self.params.device)

            self.optimizer.zero_grad()

            batch_scores = self.model.forward(batch_graphs, batch_nodes_feat, batch_edges_feat,
                                              batch_nodes_num_norm_sqrt, batch_edges_num_norm_sqrt)
            loss = self.loss_ce(batch_scores, batch_labels)

            loss.backward()
            self.optimizer.step()

            nb_data += batch_labels.size(0)
            epoch_loss += loss.detach().item()
            epoch_train_acc += self.accuracy(batch_scores, batch_labels)

            # Tools.print("{} {}/{}".format(_, epoch_loss / (_ + 1), loss.detach().item()))
            pass

        epoch_train_acc /= nb_data
        epoch_loss /= (len(data_loader) + 1)

        return epoch_loss, epoch_train_acc

    def evaluate_network(self, data_loader):
        self.model.eval()

        epoch_test_loss, epoch_test_acc, nb_data = 0, 0, 0
        with torch.no_grad():
            for _, (batch_graphs, batch_labels, batch_nodes_num_norm_sqrt,
                    batch_edges_num_norm_sqrt) in tqdm(enumerate(data_loader), total=len(data_loader)):
                batch_nodes_feat = batch_graphs.ndata['feat'].to(self.params.device)
                batch_edges_feat = batch_graphs.edata['feat'].to(self.params.device)
                batch_labels = batch_labels.to(self.params.device)
                batch_nodes_num_norm_sqrt = batch_nodes_num_norm_sqrt.to(self.params.device)
                batch_edges_num_norm_sqrt = batch_edges_num_norm_sqrt.to(self.params.device)

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
    """
    MNIST GCN 0.9321 0.9074 0.9028
    """

    """
    CIFAR OLD       GCN 0.6328/0.5714/0.5418
    CIFAR New       GCN 0.6556/0.5596/0.5538
    CIFAR New GraphSage 0.8324/0.6664/0.6608
    CIFAR New  GatedGCN 0.7421/0.6938/0.6869
    
    CIFAR  78506       GCN 0.6307/0.5658/0.5571
    CIFAR 343978  GatedGCN 0.9959/0.7146/0.7097
    CIFAR  79018       GAT 0.7997/0.6666/0.6636
    CIFAR 209650     MoNet
    CIFAR 598538  DiffPool 0.7626/0.6536/0.6550
    CIFAR 144042 GraphSage 0.8631/0.6760/0.6685
    CIFAR 138786       GIN 0.6794/0.5992/0.5870
    
    MNIST 15898        GCN 0.8838/0.8914/0.8826
    MNIST 66330   GatedGCN 0.9999/0.9748/0.9737
    MNIST 16090        GAT 0.9683/0.9576/0.9526
    MNIST 139146  DiffPool 0.9985/0.9730/0.9685
    MNIST 28186  GraphSage 0.9929/0.9668/0.9675
    MNIST 40528      MoNet
    MNIST 27937       GIN  0.9701/0.9546/0.9546
    
    MNIST 20186        GCN 0.9135/0.9068/0.8980
    MNIST 87386   GatedGCN 0.9995/0.9766/0.9772
    MNIST 20442        GAT
    MNIST 151690  DiffPool 0.9996/0.9744/0.9731
    MNIST 36570  GraphSage 0.9967/0.9708/0.9696
    MNIST 53026     MoNet
    MNIST 36514        GIN 0.9690/0.9526/0.9517
    """

    now_use_gpu = True
    # now_gpu_id = "0"
    now_gpu_id = "1"
    now_model_name = "MoNet"  # GCN GatedGCN GAT GraphSage GIN MoNet DiffPool MLP MLPGated
    now_dataset_name = "MNIST"  # MNIST CIFAR10
    now_run_name = "{}_{}_128".format(now_dataset_name, now_model_name)
    # now_data_file = "D:\data\GCN\{}.pkl".format(now_dataset_name)
    now_data_file = "./data/{}.pkl".format(now_dataset_name)

    now_batch_size = 64
    now_dataset = SuperPixDataset(now_dataset_name, data_file=now_data_file)
    now_params = GNNParameter(now_model_name, dataset=now_dataset, dataset_name=now_dataset_name,
                              out_dir=Tools.new_dir("result/{}".format(now_run_name)), model_name=now_model_name,
                              batch_size=now_batch_size, use_gpu=now_use_gpu, gpu_id=now_gpu_id)
    now_params.print_info()

    runner = Runner(params=now_params)
    runner.train_val_pipeline()
