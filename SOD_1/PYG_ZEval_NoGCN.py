import os
import torch
import numpy as np
from PIL import Image
from SODData import SODData
from alisuretool.Tools import Tools
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


def gpu_setup(use_gpu, gpu_id):
    if torch.cuda.is_available() and use_gpu:
        Tools.print()
        Tools.print('Cuda available with GPU: {}'.format(torch.cuda.get_device_name(0)))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        Tools.print()
        Tools.print('Cuda not available')
        device = torch.device("cpu")
    return device


class MyEvalDataset(Dataset):

    def __init__(self, image_name_list, label_name_list=None, min_size=256):
        super().__init__()
        self.min_size = min_size
        self.image_name_list = image_name_list
        self.label_name_list = label_name_list
        pass

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        # 读数据
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image_name = self.image_name_list[idx]

        # 限制最小大小
        if image.size[0] < self.min_size or image.size[1] < self.min_size:
            if image.size[0] < image.size[1]:
                image = image.resize((self.min_size, int(self.min_size / image.size[0] * image.size[1])))
            else:
                image = image.resize((int(self.min_size / image.size[1] * image.size[0]), self.min_size))
            pass

        if self.label_name_list is not None:
            label = Image.open(self.label_name_list[idx]).convert("L").resize(image.size)
        else:
            label = Image.new("L", size=image.size)

        label = np.asarray(label) / 255
        # 归一化
        _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_data = transforms.Compose([transforms.ToTensor(), _normalize])(image).unsqueeze(dim=0)

        # 返回
        return img_data, label, image_name

    @staticmethod
    def collate_fn(samples):
        images, labels, image_name = map(list, zip(*samples))
        images = torch.cat(images)
        return images, labels, image_name

    pass


class RunnerSPE(object):

    def __init__(self, min_size=256, use_gpu=True, gpu_id="1"):
        self.min_size = min_size

        self.device = gpu_setup(use_gpu=use_gpu, gpu_id=gpu_id)
        self.model = MyGCNNet().to(self.device)

        Tools.print("Total param: {}".format(self._view_model_param(self.model)))
        self._print_network(self.model)
        pass

    def load_model(self, model_file_name):
        ckpt = torch.load(model_file_name, map_location=self.device)

        self.model.load_state_dict(ckpt, strict=False)
        Tools.print('Load Model: {}'.format(model_file_name))
        pass

    def save_result(self, model_file=None, image_name_list=None, label_name_list=None, save_path=None):
        assert image_name_list is not None
        if model_file is not None:
            self.load_model(model_file_name=model_file)
        self.model.train()

        # 统计
        Tools.print()
        th_num = 25
        nb_data = 0
        epoch_test_mae, epoch_test_mae2 = 0.0, 0.0
        epoch_test_prec, epoch_test_recall = np.zeros(shape=(th_num,)) + 1e-6, np.zeros(shape=(th_num,)) + 1e-6

        dataset = MyEvalDataset(image_name_list, label_name_list=label_name_list, min_size=self.min_size)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)
        tr_num = len(loader)
        with torch.no_grad():
            for i, (images, labels, image_names) in enumerate(loader):
                # Data
                images = images.float().to(self.device)
                labels_sod = torch.unsqueeze(torch.Tensor(labels), dim=1).to(self.device)

                _, sod_logits_sigmoid = self.model.forward(images)

                labels_sod_val = labels_sod.cpu().detach().numpy()
                sod_logits_sigmoid_val = sod_logits_sigmoid.cpu().detach().numpy()

                # Stat
                nb_data += images.size(0)

                # cal 1
                mae = self._eval_mae(sod_logits_sigmoid_val, labels_sod_val)
                prec, recall = self._eval_pr(sod_logits_sigmoid_val, labels_sod_val, th_num)
                epoch_test_mae += mae
                epoch_test_prec += prec
                epoch_test_recall += recall

                if save_path is not None:
                    im_size = Image.open(image_names[0]).size
                    image_name = os.path.splitext(os.path.basename(image_names[0]))[0]

                    save_file_name = Tools.new_dir(os.path.join(save_path, "SOD", "{}.png".format(image_name)))
                    sod_result = torch.squeeze(sod_logits_sigmoid).detach().cpu().numpy()
                    Image.fromarray(np.asarray(sod_result * 255, dtype=np.uint8)).resize(im_size).save(save_file_name)
                    pass

                # Print
                if i % 500 == 0:
                    Tools.print("{:4d}-{:4d} sod-mse={:.4f}({:.4f})".format(i, len(loader), mae, epoch_test_mae/(i+1)))
                    pass
                pass
            pass

        avg_mae, avg_prec, avg_recall = epoch_test_mae/tr_num, epoch_test_prec/tr_num, epoch_test_recall/tr_num
        score = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)

        Tools.print('{} sod-mae-score={:.4f}-{:.4f}'.format(save_path, avg_mae, score.max()))
        pass

    @staticmethod
    def _print_network(model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        Tools.print(model)
        Tools.print("The number of parameters: {}".format(num_params))
        pass

    @staticmethod
    def _eval_mae(y_pred, y):
        return np.abs(y_pred - y).mean()

    @staticmethod
    def _eval_pr(y_pred, y, th_num=100):
        prec, recall = np.zeros(shape=(th_num,)), np.zeros(shape=(th_num,))
        th_list = np.linspace(0, 1 - 1e-10, th_num)
        for i in range(th_num):
            y_temp = y_pred >= th_list[i]
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
            pass
        return prec, recall

    @staticmethod
    def _view_model_param(model):
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        return total_param

    pass


"""
DUTS-TR sod-mae-score=0.0083-0.9872 gcn-mae-score=0.0378-0.9249
DUTS-TE sod-mae-score=0.0376-0.8817 gcn-mae-score=0.0729-0.7472
model_name = "PYG_GCNAtt_NoAddGCN_NoAttRes"
from PYG_GCNAtt_NoAddGCN_NoAttRes import MyGCNNet, MyDataset
model_file = "/media/ubuntu/data1/ALISURE/PyTorchGCN/SOD_1/ckpt/PYG_GCNAtt_NoAddGCN_NoAttRes/0/epoch_29.pkl"

result_path = "/media/ubuntu/data1/ALISURE/PyTorchGCN_Result"
_data_root_path = "/media/ubuntu/data1/ALISURE"


model_name = "PYG_ChangeGCN_GCNAtt_NoAddGCN_NoAttRes"
from PYG_ChangeGCN_GCNAtt_NoAddGCN_NoAttRes import MyGCNNet, MyDataset
model_file = "/mnt/4T/ALISURE/GCN/PyTorchGCN/SOD_1/ckpt/PYG_ChangeGCN_GCNAtt_NoAddGCN_NoAttRes/0/epoch_24.pkl"

result_path = "/mnt/4T/ALISURE/GCN/PyTorchGCN_Result"
_data_root_path = "/mnt/4T/Data/SOD"

DUTS-TE sod-mae-score=0.0367-0.8845 gcn-mae-score=0.0745-0.7489
DUTS-TR sod-mae-score=0.0088-0.9868 gcn-mae-score=0.0440-0.9183
model_name = "PYG_GCNAtt_NoAddGCN_NoAttRes_NewPool"
from PYG_GCNAtt_NoAddGCN_NoAttRes_NewPool import MyGCNNet, MyDataset
model_file = "/media/ubuntu/data1/ALISURE/PyTorchGCN/SOD_1/ckpt/PYG_GCNAtt_NoAddGCN_NoAttRes_NewPool/2/epoch_28.pkl"


DUTS-TE sod-mae-score=0.0369-0.8863 gcn-mae-score=0.0764-0.7460
model_name = "PYG_GCNAtt_NoAddGCN_NoAttRes_Sigmoid"
from PYG_GCNAtt_NoAddGCN_NoAttRes_Sigmoid import MyGCNNet, MyDataset
model_file = "/media/ubuntu/data1/ALISURE/PyTorchGCN/SOD_1/ckpt/PYG_GCNAtt_NoAddGCN_NoAttRes_Sigmoid/32/epoch_29.pkl"

"""


if __name__ == '__main__':
    model_name = "FPN_Baseline2"
    from FPN_Baseline import MyGCNNet, MyDataset
    model_file = "/media/ubuntu/data1/ALISURE/PyTorchGCN/SOD_1/ckpt/FPN_Baseline/21/epoch_27.pkl"
    # model_name = "PYG_NoGCN_NewPool_Sigmoid"
    # from PYG_NoGCN_NewPool_Sigmoid import MyGCNNet, MyDataset
    # model_file = "/media/ubuntu/data1/ALISURE/PyTorchGCN/SOD_1/ckpt/PYG_NoGCN_NewPool_Sigmoid/10/epoch_29.pkl"

    result_path = "/media/ubuntu/data1/ALISURE/PyTorchGCN_Result"
    # _data_root_path = "/mnt/4T/Data/SOD"
    # _data_root_path = "/media/ubuntu/data1/ALISURE"
    _data_root_path = "/media/ubuntu/ALISURE/data/SOD"

    _gpu_id = "2"

    _use_gpu = True
    _improved = True
    _has_bn = True
    _has_residual = True
    _is_normalize = True
    _concat = True

    runner = RunnerSPE(use_gpu=_use_gpu, gpu_id=_gpu_id)

    sod_data = SODData(data_root_path=_data_root_path)
    # for data_set in [sod_data.cssd, sod_data.ecssd, sod_data.msra_1000_asd, sod_data.msra10k,
    #                  sod_data.msra_b, sod_data.sed2, sod_data.dut_dmron_5166, sod_data.hku_is,
    #                  sod_data.sod, sod_data.thur15000, sod_data.pascal1500, sod_data.pascal_s,
    #                  sod_data.judd, sod_data.duts_te, sod_data.duts_tr, sod_data.cub_200_2011]:
    # for data_set in [sod_data.duts_te, sod_data.duts_tr]:
    for data_set in [sod_data.sed1, sod_data.dut_dmron_5168]:
        img_name_list, lbl_name_list, dataset_name_list = data_set()
        Tools.print("Begin eval {} {}".format(dataset_name_list[0], len(img_name_list)))

        runner.save_result(model_file=model_file, image_name_list=img_name_list, label_name_list=lbl_name_list,
                           save_path="{}/{}/{}".format(result_path, model_name, dataset_name_list[0]))
        pass
    pass
