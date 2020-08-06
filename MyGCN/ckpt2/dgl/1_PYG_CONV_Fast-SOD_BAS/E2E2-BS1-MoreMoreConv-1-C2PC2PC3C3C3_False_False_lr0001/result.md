/home/ubuntu/anaconda3/envs/alisure36torch/bin/python /mnt/4T/ALISURE/GCN/PyTorchGCN/MyGCN/SPERunner_1_PYG_CONV_Fast_SOD_SGPU_E2E_BS1_MoreConv.py
2020-08-04 10:55:17 name:E2E2-BS1-MoreMoreConv-1-C2PC2PC3C3C3_False_False_lr0001 epochs:50 ckpt:./ckpt2/dgl/1_PYG_CONV_Fast-SOD_BAS/E2E2-BS1-MoreMoreConv-1-C2PC2PC3C3C3_False_False_lr0001 sp size:4 down_ratio:4 workers:16 gpu:0 has_mask:False has_residual:True is_normalize:True has_bn:True improved:True concat:True is_sgd:False weight_decay:0.0

2020-08-04 10:55:17 Cuda available with GPU: GeForce GTX 1080
2020-08-04 10:55:23 Total param: 55297472 lr_s=[[0, 0.0001], [20, 1e-05], [35, 1e-06]] Optimizer=Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.0001
    weight_decay: 0.0
)
2020-08-04 10:55:23 MyGCNNet(
  (model_conv): CONVNet(
    (features): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (9): ReLU(inplace=True)
      (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (12): ReLU(inplace=True)
      (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (16): ReLU(inplace=True)
      (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (19): ReLU(inplace=True)
      (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (22): ReLU(inplace=True)
      (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (26): ReLU(inplace=True)
      (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (29): ReLU(inplace=True)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (32): ReLU(inplace=True)
      (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (36): ReLU(inplace=True)
      (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (39): ReLU(inplace=True)
      (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (42): ReLU(inplace=True)
    )
    (max_pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (model_gnn1): SAGENet1(
    (relu): ReLU()
    (gcn_list): ModuleList(
      (0): SAGEConv(512, 512)
      (1): SAGEConv(512, 512)
    )
    (bn_list): ModuleList(
      (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (model_gnn2): SAGENet2(
    (embedding_h): Linear(in_features=512, out_features=512, bias=True)
    (relu): ReLU()
    (gcn_list): ModuleList(
      (0): SAGEConv(512, 512)
      (1): SAGEConv(512, 512)
      (2): SAGEConv(512, 1024)
      (3): SAGEConv(1024, 1024)
    )
    (bn_list): ModuleList(
      (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (skip_connect_list): ModuleList(
      (0): Linear(in_features=512, out_features=256, bias=False)
      (1): Linear(in_features=1024, out_features=256, bias=False)
    )
    (skip_connect_bn_list): ModuleList(
      (0): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (readout_mlp): Linear(in_features=512, out_features=1, bias=False)
  )
  (model_sod): SODNet(
    (conv_sod_gcn2_1): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_sod_gcn2_2): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_sod_gcn1_1): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_sod_gcn1_2): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv4_1): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv4_2): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv3_1): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv3_2): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv2_1): ConvBlock(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv2_2): ConvBlock(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv1_1): ConvBlock(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv1_2): ConvBlock(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_sod_gcn_1): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_sod_gcn_2): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv4_1): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv4_2): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv3_1): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv3_2): ConvBlock(
      (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv2_1): ConvBlock(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv2_2): ConvBlock(
      (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv1_1): ConvBlock(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv1_2): ConvBlock(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_final_1): ConvBlock(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_final_2): ConvBlock(
      (conv): Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    )
  )
)
2020-08-04 10:55:23 The number of parameters: 55297472

2020-08-04 10:55:23 Start Epoch 0
2020-08-04 10:55:23 Epoch:00,lr=0.0001
2020-08-04 10:55:24    0-10553 loss=1.4065(0.7126+0.6939)-1.4065(0.7126+0.6939) sod-mse=0.4973(0.4973) gcn-mse=0.4806(0.4806) gcn-final-mse=0.4865(0.5018)
2020-08-04 11:01:05 1000-10553 loss=0.2951(0.1609+0.1342)-0.6762(0.3189+0.3573) sod-mse=0.1016(0.2193) gcn-mse=0.1050(0.1859) gcn-final-mse=0.1863(0.2001)
2020-08-04 11:06:43 2000-10553 loss=0.8665(0.4302+0.4363)-0.6038(0.2935+0.3102) sod-mse=0.2563(0.1888) gcn-mse=0.2315(0.1666) gcn-final-mse=0.1668(0.1807)
2020-08-04 11:12:17 3000-10553 loss=0.1197(0.0724+0.0473)-0.5558(0.2747+0.2811) sod-mse=0.0299(0.1696) gcn-mse=0.0313(0.1543) gcn-final-mse=0.1544(0.1684)
2020-08-04 11:17:47 4000-10553 loss=0.3615(0.1591+0.2025)-0.5324(0.2648+0.2676) sod-mse=0.1572(0.1599) gcn-mse=0.0964(0.1473) gcn-final-mse=0.1475(0.1614)
2020-08-04 11:23:17 5000-10553 loss=0.2370(0.1407+0.0963)-0.5123(0.2570+0.2553) sod-mse=0.0787(0.1520) gcn-mse=0.0884(0.1419) gcn-final-mse=0.1420(0.1560)
2020-08-04 11:28:47 6000-10553 loss=0.2898(0.1438+0.1460)-0.4968(0.2508+0.2460) sod-mse=0.0944(0.1458) gcn-mse=0.0728(0.1374) gcn-final-mse=0.1374(0.1514)
2020-08-04 11:33:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 11:34:17 7000-10553 loss=0.9334(0.4813+0.4521)-0.4845(0.2456+0.2389) sod-mse=0.2661(0.1410) gcn-mse=0.2558(0.1338) gcn-final-mse=0.1339(0.1478)
2020-08-04 11:36:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 11:39:48 8000-10553 loss=0.1677(0.0925+0.0753)-0.4754(0.2419+0.2335) sod-mse=0.0546(0.1375) gcn-mse=0.0423(0.1313) gcn-final-mse=0.1313(0.1452)
2020-08-04 11:45:20 9000-10553 loss=1.2780(0.5567+0.7214)-0.4644(0.2373+0.2271) sod-mse=0.2341(0.1335) gcn-mse=0.2205(0.1283) gcn-final-mse=0.1283(0.1423)
2020-08-04 11:47:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 11:50:50 10000-10553 loss=0.3025(0.1586+0.1439)-0.4549(0.2332+0.2217) sod-mse=0.1028(0.1301) gcn-mse=0.0972(0.1254) gcn-final-mse=0.1254(0.1394)

2020-08-04 11:53:54    0-5019 loss=1.2242(0.5438+0.6804)-1.2242(0.5438+0.6804) sod-mse=0.2390(0.2390) gcn-mse=0.2188(0.2188) gcn-final-mse=0.2103(0.2209)
2020-08-04 11:56:21 1000-5019 loss=0.1466(0.0970+0.0496)-0.4602(0.2449+0.2152) sod-mse=0.0455(0.1141) gcn-mse=0.0725(0.1302) gcn-final-mse=0.1304(0.1445)
2020-08-04 11:58:44 2000-5019 loss=0.5578(0.2614+0.2964)-0.4666(0.2476+0.2190) sod-mse=0.1222(0.1154) gcn-mse=0.1338(0.1312) gcn-final-mse=0.1314(0.1454)
2020-08-04 12:01:09 3000-5019 loss=0.0836(0.0553+0.0282)-0.4714(0.2502+0.2212) sod-mse=0.0189(0.1168) gcn-mse=0.0273(0.1327) gcn-final-mse=0.1330(0.1470)
2020-08-04 12:02:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 12:03:32 4000-5019 loss=0.2933(0.1624+0.1309)-0.4707(0.2500+0.2207) sod-mse=0.0754(0.1169) gcn-mse=0.0865(0.1328) gcn-final-mse=0.1331(0.1471)
2020-08-04 12:04:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 12:05:55 5000-5019 loss=0.6744(0.3282+0.3462)-0.4710(0.2503+0.2207) sod-mse=0.1358(0.1169) gcn-mse=0.1492(0.1329) gcn-final-mse=0.1332(0.1471)
2020-08-04 12:05:57 E: 0, Train sod-mae-score=0.1290-0.8480 gcn-mae-score=0.1245-0.8384 gcn-final-mse-score=0.1245-0.8420(0.1385/0.8420) loss=0.4522(0.2320+0.2201)
2020-08-04 12:05:57 E: 0, Test  sod-mae-score=0.1169-0.7260 gcn-mae-score=0.1329-0.6801 gcn-final-mse-score=0.1331-0.6869(0.1471/0.6869) loss=0.4707(0.2502+0.2205)

2020-08-04 12:05:57 Start Epoch 1
2020-08-04 12:05:57 Epoch:01,lr=0.0001
2020-08-04 12:05:58    0-10553 loss=0.3113(0.1509+0.1604)-0.3113(0.1509+0.1604) sod-mse=0.0687(0.0687) gcn-mse=0.0724(0.0724) gcn-final-mse=0.0687(0.0849)
2020-08-04 12:11:29 1000-10553 loss=0.7377(0.3470+0.3907)-0.3526(0.1896+0.1629) sod-mse=0.1774(0.0920) gcn-mse=0.1741(0.0970) gcn-final-mse=0.0968(0.1114)
2020-08-04 12:17:00 2000-10553 loss=0.6085(0.2883+0.3202)-0.3563(0.1918+0.1645) sod-mse=0.1965(0.0930) gcn-mse=0.1746(0.0979) gcn-final-mse=0.0978(0.1122)
2020-08-04 12:22:28 3000-10553 loss=0.1414(0.0898+0.0516)-0.3648(0.1947+0.1701) sod-mse=0.0409(0.0963) gcn-mse=0.0404(0.0995) gcn-final-mse=0.0994(0.1138)
2020-08-04 12:24:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 12:28:00 4000-10553 loss=0.2265(0.0978+0.1288)-0.3609(0.1928+0.1681) sod-mse=0.0986(0.0953) gcn-mse=0.0746(0.0984) gcn-final-mse=0.0983(0.1126)
2020-08-04 12:33:33 5000-10553 loss=0.2413(0.1311+0.1102)-0.3618(0.1932+0.1686) sod-mse=0.0644(0.0958) gcn-mse=0.0577(0.0986) gcn-final-mse=0.0985(0.1128)
2020-08-04 12:36:43 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 12:39:02 6000-10553 loss=0.2942(0.1605+0.1336)-0.3577(0.1911+0.1666) sod-mse=0.0914(0.0945) gcn-mse=0.0931(0.0972) gcn-final-mse=0.0971(0.1113)
2020-08-04 12:44:32 7000-10553 loss=0.1199(0.0711+0.0488)-0.3556(0.1903+0.1653) sod-mse=0.0291(0.0939) gcn-mse=0.0336(0.0969) gcn-final-mse=0.0967(0.1109)
2020-08-04 12:48:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 12:50:01 8000-10553 loss=0.1714(0.1123+0.0591)-0.3533(0.1892+0.1641) sod-mse=0.0430(0.0930) gcn-mse=0.0626(0.0961) gcn-final-mse=0.0959(0.1102)
2020-08-04 12:55:31 9000-10553 loss=0.0355(0.0292+0.0062)-0.3499(0.1879+0.1619) sod-mse=0.0040(0.0918) gcn-mse=0.0213(0.0953) gcn-final-mse=0.0952(0.1094)
2020-08-04 13:01:01 10000-10553 loss=0.0404(0.0305+0.0099)-0.3470(0.1867+0.1603) sod-mse=0.0072(0.0908) gcn-mse=0.0173(0.0945) gcn-final-mse=0.0943(0.1086)

2020-08-04 13:04:04    0-5019 loss=1.1255(0.5300+0.5954)-1.1255(0.5300+0.5954) sod-mse=0.1968(0.1968) gcn-mse=0.1891(0.1891) gcn-final-mse=0.1816(0.1912)
2020-08-04 13:06:28 1000-5019 loss=0.0721(0.0490+0.0231)-0.4247(0.2178+0.2070) sod-mse=0.0211(0.0957) gcn-mse=0.0295(0.1063) gcn-final-mse=0.1069(0.1194)
2020-08-04 13:08:51 2000-5019 loss=0.9445(0.3871+0.5574)-0.4313(0.2208+0.2105) sod-mse=0.1418(0.0969) gcn-mse=0.1362(0.1080) gcn-final-mse=0.1086(0.1211)
2020-08-04 13:11:14 3000-5019 loss=0.0753(0.0504+0.0249)-0.4396(0.2249+0.2147) sod-mse=0.0151(0.0985) gcn-mse=0.0237(0.1097) gcn-final-mse=0.1104(0.1229)
2020-08-04 13:12:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 13:13:37 4000-5019 loss=0.2807(0.1566+0.1241)-0.4398(0.2252+0.2146) sod-mse=0.0623(0.0987) gcn-mse=0.0714(0.1100) gcn-final-mse=0.1106(0.1231)
2020-08-04 13:14:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 13:15:59 5000-5019 loss=1.0273(0.4223+0.6050)-0.4400(0.2256+0.2144) sod-mse=0.1222(0.0988) gcn-mse=0.1346(0.1101) gcn-final-mse=0.1107(0.1232)
2020-08-04 13:16:02 E: 1, Train sod-mae-score=0.0902-0.8964 gcn-mae-score=0.0940-0.8676 gcn-final-mse-score=0.0939-0.8708(0.1081/0.8708) loss=0.3450(0.1859+0.1591)
2020-08-04 13:16:02 E: 1, Test  sod-mae-score=0.0987-0.7479 gcn-mae-score=0.1101-0.6934 gcn-final-mse-score=0.1107-0.6993(0.1231/0.6993) loss=0.4397(0.2255+0.2142)

2020-08-04 13:16:02 Start Epoch 2
2020-08-04 13:16:02 Epoch:02,lr=0.0001
2020-08-04 13:16:03    0-10553 loss=0.0611(0.0436+0.0175)-0.0611(0.0436+0.0175) sod-mse=0.0129(0.0129) gcn-mse=0.0305(0.0305) gcn-final-mse=0.0297(0.0345)
2020-08-04 13:21:35 1000-10553 loss=0.1275(0.0586+0.0689)-0.3168(0.1736+0.1433) sod-mse=0.0614(0.0793) gcn-mse=0.0417(0.0854) gcn-final-mse=0.0852(0.0995)
2020-08-04 13:27:07 2000-10553 loss=0.1332(0.0965+0.0367)-0.3100(0.1701+0.1399) sod-mse=0.0277(0.0778) gcn-mse=0.0396(0.0843) gcn-final-mse=0.0840(0.0983)
2020-08-04 13:32:38 3000-10553 loss=0.6912(0.3595+0.3317)-0.3113(0.1708+0.1405) sod-mse=0.1850(0.0783) gcn-mse=0.2013(0.0844) gcn-final-mse=0.0842(0.0986)
2020-08-04 13:38:07 4000-10553 loss=0.1201(0.0706+0.0495)-0.3100(0.1701+0.1398) sod-mse=0.0335(0.0781) gcn-mse=0.0291(0.0843) gcn-final-mse=0.0841(0.0985)
2020-08-04 13:38:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 13:43:36 5000-10553 loss=0.3303(0.1828+0.1475)-0.3065(0.1687+0.1377) sod-mse=0.1038(0.0768) gcn-mse=0.0950(0.0832) gcn-final-mse=0.0830(0.0975)
2020-08-04 13:44:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 13:49:07 6000-10553 loss=0.1158(0.0689+0.0468)-0.3027(0.1669+0.1358) sod-mse=0.0397(0.0757) gcn-mse=0.0528(0.0821) gcn-final-mse=0.0819(0.0964)
2020-08-04 13:54:38 7000-10553 loss=0.1100(0.0678+0.0422)-0.3033(0.1672+0.1361) sod-mse=0.0312(0.0758) gcn-mse=0.0381(0.0823) gcn-final-mse=0.0820(0.0964)
2020-08-04 13:54:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 14:00:10 8000-10553 loss=0.1218(0.0703+0.0515)-0.3042(0.1675+0.1366) sod-mse=0.0287(0.0763) gcn-mse=0.0295(0.0825) gcn-final-mse=0.0822(0.0966)
2020-08-04 14:05:42 9000-10553 loss=0.2546(0.1542+0.1003)-0.3015(0.1663+0.1353) sod-mse=0.0718(0.0755) gcn-mse=0.0744(0.0818) gcn-final-mse=0.0815(0.0960)
2020-08-04 14:11:11 10000-10553 loss=0.0772(0.0497+0.0275)-0.2986(0.1650+0.1336) sod-mse=0.0186(0.0746) gcn-mse=0.0240(0.0811) gcn-final-mse=0.0808(0.0952)

2020-08-04 14:14:15    0-5019 loss=1.0447(0.5508+0.4939)-1.0447(0.5508+0.4939) sod-mse=0.2207(0.2207) gcn-mse=0.2001(0.2001) gcn-final-mse=0.1892(0.1977)
2020-08-04 14:16:40 1000-5019 loss=0.0821(0.0446+0.0375)-0.4033(0.2132+0.1901) sod-mse=0.0336(0.1093) gcn-mse=0.0261(0.1042) gcn-final-mse=0.1042(0.1178)
2020-08-04 14:19:03 2000-5019 loss=0.6222(0.3125+0.3097)-0.4079(0.2159+0.1921) sod-mse=0.1439(0.1108) gcn-mse=0.1299(0.1060) gcn-final-mse=0.1061(0.1197)
2020-08-04 14:21:28 3000-5019 loss=0.0705(0.0431+0.0274)-0.4108(0.2179+0.1930) sod-mse=0.0193(0.1118) gcn-mse=0.0159(0.1073) gcn-final-mse=0.1075(0.1210)
2020-08-04 14:22:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 14:23:52 4000-5019 loss=0.2567(0.1346+0.1220)-0.4111(0.2181+0.1930) sod-mse=0.0737(0.1118) gcn-mse=0.0652(0.1074) gcn-final-mse=0.1076(0.1211)
2020-08-04 14:24:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 14:26:16 5000-5019 loss=0.7912(0.3515+0.4397)-0.4125(0.2189+0.1935) sod-mse=0.1257(0.1120) gcn-mse=0.1276(0.1078) gcn-final-mse=0.1078(0.1214)
2020-08-04 14:26:18 E: 2, Train sod-mae-score=0.0749-0.9122 gcn-mae-score=0.0814-0.8810 gcn-final-mse-score=0.0811-0.8840(0.0955/0.8840) loss=0.2996(0.1654+0.1342)
2020-08-04 14:26:18 E: 2, Test  sod-mae-score=0.1120-0.7636 gcn-mae-score=0.1077-0.7038 gcn-final-mse-score=0.1078-0.7110(0.1213/0.7110) loss=0.4121(0.2188+0.1934)

2020-08-04 14:26:18 Start Epoch 3
2020-08-04 14:26:18 Epoch:03,lr=0.0001
2020-08-04 14:26:20    0-10553 loss=0.4988(0.2283+0.2706)-0.4988(0.2283+0.2706) sod-mse=0.1222(0.1222) gcn-mse=0.1116(0.1116) gcn-final-mse=0.1152(0.1230)
2020-08-04 14:27:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 14:31:51 1000-10553 loss=0.3026(0.1839+0.1187)-0.2661(0.1498+0.1162) sod-mse=0.0755(0.0649) gcn-mse=0.0967(0.0718) gcn-final-mse=0.0716(0.0862)
2020-08-04 14:37:22 2000-10553 loss=0.1859(0.1093+0.0766)-0.2793(0.1551+0.1242) sod-mse=0.0460(0.0689) gcn-mse=0.0430(0.0749) gcn-final-mse=0.0746(0.0891)
2020-08-04 14:42:53 3000-10553 loss=0.1786(0.1203+0.0583)-0.2739(0.1528+0.1211) sod-mse=0.0483(0.0672) gcn-mse=0.0818(0.0737) gcn-final-mse=0.0734(0.0880)
2020-08-04 14:48:24 4000-10553 loss=0.5842(0.2788+0.3054)-0.2773(0.1542+0.1231) sod-mse=0.1384(0.0683) gcn-mse=0.1274(0.0743) gcn-final-mse=0.0740(0.0886)
2020-08-04 14:53:53 5000-10553 loss=0.0879(0.0633+0.0246)-0.2767(0.1539+0.1227) sod-mse=0.0150(0.0681) gcn-mse=0.0212(0.0741) gcn-final-mse=0.0739(0.0884)
2020-08-04 14:59:22 6000-10553 loss=0.2543(0.1360+0.1182)-0.2755(0.1533+0.1222) sod-mse=0.0589(0.0678) gcn-mse=0.0541(0.0739) gcn-final-mse=0.0736(0.0882)
2020-08-04 15:04:53 7000-10553 loss=0.0785(0.0563+0.0222)-0.2748(0.1531+0.1217) sod-mse=0.0144(0.0675) gcn-mse=0.0217(0.0738) gcn-final-mse=0.0736(0.0882)
2020-08-04 15:06:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 15:07:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 15:10:25 8000-10553 loss=0.0530(0.0393+0.0138)-0.2730(0.1523+0.1207) sod-mse=0.0104(0.0669) gcn-mse=0.0128(0.0732) gcn-final-mse=0.0730(0.0876)
2020-08-04 15:15:56 9000-10553 loss=0.1252(0.0899+0.0353)-0.2740(0.1528+0.1212) sod-mse=0.0282(0.0672) gcn-mse=0.0589(0.0735) gcn-final-mse=0.0733(0.0878)
2020-08-04 15:21:28 10000-10553 loss=0.0893(0.0621+0.0272)-0.2754(0.1535+0.1220) sod-mse=0.0226(0.0676) gcn-mse=0.0385(0.0738) gcn-final-mse=0.0736(0.0881)

2020-08-04 15:24:31    0-5019 loss=0.9398(0.5436+0.3962)-0.9398(0.5436+0.3962) sod-mse=0.1487(0.1487) gcn-mse=0.1775(0.1775) gcn-final-mse=0.1721(0.1844)
2020-08-04 15:26:55 1000-5019 loss=0.0899(0.0521+0.0377)-0.3950(0.2008+0.1941) sod-mse=0.0326(0.1014) gcn-mse=0.0322(0.0950) gcn-final-mse=0.0952(0.1091)
2020-08-04 15:29:18 2000-5019 loss=0.4734(0.2640+0.2094)-0.4105(0.2076+0.2029) sod-mse=0.1071(0.1044) gcn-mse=0.1049(0.0976) gcn-final-mse=0.0979(0.1118)
2020-08-04 15:31:42 3000-5019 loss=0.0581(0.0395+0.0186)-0.4209(0.2119+0.2090) sod-mse=0.0096(0.1063) gcn-mse=0.0122(0.0994) gcn-final-mse=0.0996(0.1135)
2020-08-04 15:33:01 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 15:34:05 4000-5019 loss=0.1449(0.0958+0.0491)-0.4183(0.2107+0.2076) sod-mse=0.0294(0.1062) gcn-mse=0.0364(0.0992) gcn-final-mse=0.0994(0.1133)
2020-08-04 15:34:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 15:36:27 5000-5019 loss=0.6852(0.3226+0.3626)-0.4180(0.2109+0.2071) sod-mse=0.1685(0.1061) gcn-mse=0.1414(0.0993) gcn-final-mse=0.0994(0.1133)
2020-08-04 15:36:29 E: 3, Train sod-mae-score=0.0673-0.9207 gcn-mae-score=0.0735-0.8889 gcn-final-mse-score=0.0733-0.8919(0.0879/0.8919) loss=0.2743(0.1529+0.1214)
2020-08-04 15:36:29 E: 3, Test  sod-mae-score=0.1061-0.7922 gcn-mae-score=0.0993-0.7372 gcn-final-mse-score=0.0994-0.7441(0.1133/0.7441) loss=0.4178(0.2108+0.2070)

2020-08-04 15:36:29 Start Epoch 4
2020-08-04 15:36:29 Epoch:04,lr=0.0001
2020-08-04 15:36:31    0-10553 loss=0.0686(0.0484+0.0202)-0.0686(0.0484+0.0202) sod-mse=0.0170(0.0170) gcn-mse=0.0279(0.0279) gcn-final-mse=0.0294(0.0383)
2020-08-04 15:42:02 1000-10553 loss=0.1340(0.0903+0.0437)-0.2423(0.1384+0.1038) sod-mse=0.0314(0.0577) gcn-mse=0.0458(0.0654) gcn-final-mse=0.0652(0.0798)
2020-08-04 15:46:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 15:47:35 2000-10553 loss=0.1214(0.0786+0.0428)-0.2471(0.1404+0.1067) sod-mse=0.0257(0.0592) gcn-mse=0.0304(0.0663) gcn-final-mse=0.0660(0.0807)
2020-08-04 15:53:06 3000-10553 loss=0.0676(0.0431+0.0245)-0.2518(0.1424+0.1094) sod-mse=0.0149(0.0600) gcn-mse=0.0220(0.0670) gcn-final-mse=0.0667(0.0815)
2020-08-04 15:58:37 4000-10553 loss=0.1527(0.1021+0.0506)-0.2503(0.1417+0.1085) sod-mse=0.0383(0.0596) gcn-mse=0.0520(0.0667) gcn-final-mse=0.0663(0.0811)
2020-08-04 16:04:07 5000-10553 loss=0.0269(0.0189+0.0080)-0.2505(0.1419+0.1086) sod-mse=0.0062(0.0597) gcn-mse=0.0139(0.0667) gcn-final-mse=0.0664(0.0811)
2020-08-04 16:09:37 6000-10553 loss=0.0711(0.0495+0.0217)-0.2520(0.1426+0.1093) sod-mse=0.0152(0.0604) gcn-mse=0.0235(0.0674) gcn-final-mse=0.0670(0.0817)
2020-08-04 16:15:08 7000-10553 loss=0.5954(0.3168+0.2786)-0.2514(0.1424+0.1090) sod-mse=0.1767(0.0602) gcn-mse=0.1487(0.0672) gcn-final-mse=0.0668(0.0815)
2020-08-04 16:20:37 8000-10553 loss=0.4728(0.2347+0.2381)-0.2528(0.1431+0.1097) sod-mse=0.1179(0.0607) gcn-mse=0.1058(0.0677) gcn-final-mse=0.0673(0.0820)
2020-08-04 16:21:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 16:24:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 16:26:06 9000-10553 loss=0.0713(0.0477+0.0236)-0.2536(0.1434+0.1101) sod-mse=0.0191(0.0609) gcn-mse=0.0259(0.0678) gcn-final-mse=0.0675(0.0822)
2020-08-04 16:31:35 10000-10553 loss=3.4543(1.5305+1.9238)-0.2543(0.1437+0.1105) sod-mse=0.3984(0.0610) gcn-mse=0.3777(0.0679) gcn-final-mse=0.0676(0.0824)

2020-08-04 16:34:38    0-5019 loss=0.7566(0.4197+0.3370)-0.7566(0.4197+0.3370) sod-mse=0.1074(0.1074) gcn-mse=0.1374(0.1374) gcn-final-mse=0.1294(0.1414)
2020-08-04 16:37:02 1000-5019 loss=0.0461(0.0347+0.0115)-0.3698(0.1948+0.1750) sod-mse=0.0101(0.0839) gcn-mse=0.0163(0.0901) gcn-final-mse=0.0902(0.1031)
2020-08-04 16:39:26 2000-5019 loss=0.6555(0.3032+0.3524)-0.3750(0.1974+0.1776) sod-mse=0.1002(0.0856) gcn-mse=0.0971(0.0917) gcn-final-mse=0.0918(0.1047)
2020-08-04 16:41:49 3000-5019 loss=0.0600(0.0387+0.0213)-0.3821(0.2005+0.1816) sod-mse=0.0095(0.0871) gcn-mse=0.0120(0.0931) gcn-final-mse=0.0932(0.1060)
2020-08-04 16:43:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 16:44:12 4000-5019 loss=0.2053(0.1266+0.0786)-0.3811(0.2001+0.1810) sod-mse=0.0426(0.0869) gcn-mse=0.0489(0.0930) gcn-final-mse=0.0931(0.1060)
2020-08-04 16:44:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 16:46:36 5000-5019 loss=0.5188(0.2476+0.2711)-0.3794(0.1997+0.1797) sod-mse=0.1072(0.0867) gcn-mse=0.1122(0.0930) gcn-final-mse=0.0930(0.1059)
2020-08-04 16:46:38 E: 4, Train sod-mae-score=0.0609-0.9267 gcn-mae-score=0.0679-0.8948 gcn-final-mse-score=0.0676-0.8977(0.0823/0.8977) loss=0.2538(0.1436+0.1102)
2020-08-04 16:46:38 E: 4, Test  sod-mae-score=0.0867-0.7880 gcn-mae-score=0.0930-0.7315 gcn-final-mse-score=0.0930-0.7373(0.1059/0.7373) loss=0.3793(0.1997+0.1796)

2020-08-04 16:46:38 Start Epoch 5
2020-08-04 16:46:38 Epoch:05,lr=0.0001
2020-08-04 16:46:40    0-10553 loss=0.1740(0.0962+0.0778)-0.1740(0.0962+0.0778) sod-mse=0.0289(0.0289) gcn-mse=0.0336(0.0336) gcn-final-mse=0.0323(0.0469)
2020-08-04 16:52:13 1000-10553 loss=0.2754(0.1587+0.1167)-0.2327(0.1339+0.0988) sod-mse=0.0874(0.0547) gcn-mse=0.0917(0.0625) gcn-final-mse=0.0622(0.0768)
2020-08-04 16:57:43 2000-10553 loss=0.0930(0.0724+0.0206)-0.2282(0.1315+0.0967) sod-mse=0.0133(0.0534) gcn-mse=0.0294(0.0612) gcn-final-mse=0.0609(0.0757)
2020-08-04 16:58:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 17:03:15 3000-10553 loss=0.4543(0.2332+0.2211)-0.2321(0.1333+0.0988) sod-mse=0.1316(0.0545) gcn-mse=0.1279(0.0619) gcn-final-mse=0.0616(0.0764)
2020-08-04 17:04:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 17:08:46 4000-10553 loss=0.1112(0.0669+0.0443)-0.2308(0.1327+0.0981) sod-mse=0.0210(0.0542) gcn-mse=0.0234(0.0617) gcn-final-mse=0.0614(0.0762)
2020-08-04 17:14:16 5000-10553 loss=0.1051(0.0594+0.0457)-0.2299(0.1322+0.0977) sod-mse=0.0336(0.0539) gcn-mse=0.0248(0.0614) gcn-final-mse=0.0611(0.0759)
2020-08-04 17:19:46 6000-10553 loss=0.2317(0.1384+0.0933)-0.2329(0.1335+0.0994) sod-mse=0.0744(0.0547) gcn-mse=0.0767(0.0619) gcn-final-mse=0.0615(0.0763)
2020-08-04 17:25:16 7000-10553 loss=0.2077(0.1261+0.0816)-0.2341(0.1342+0.0999) sod-mse=0.0644(0.0551) gcn-mse=0.0677(0.0623) gcn-final-mse=0.0619(0.0767)
2020-08-04 17:30:46 8000-10553 loss=0.0914(0.0606+0.0308)-0.2334(0.1337+0.0997) sod-mse=0.0161(0.0549) gcn-mse=0.0201(0.0621) gcn-final-mse=0.0618(0.0765)
2020-08-04 17:36:15 9000-10553 loss=0.1296(0.0827+0.0468)-0.2347(0.1342+0.1005) sod-mse=0.0390(0.0554) gcn-mse=0.0554(0.0624) gcn-final-mse=0.0621(0.0769)
2020-08-04 17:41:45 10000-10553 loss=0.1324(0.0789+0.0534)-0.2352(0.1343+0.1008) sod-mse=0.0307(0.0555) gcn-mse=0.0306(0.0624) gcn-final-mse=0.0621(0.0769)
2020-08-04 17:43:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg

2020-08-04 17:44:49    0-5019 loss=0.7147(0.4513+0.2634)-0.7147(0.4513+0.2634) sod-mse=0.1085(0.1085) gcn-mse=0.1368(0.1368) gcn-final-mse=0.1295(0.1406)
2020-08-04 17:47:14 1000-5019 loss=0.0647(0.0404+0.0243)-0.3608(0.1883+0.1724) sod-mse=0.0218(0.0850) gcn-mse=0.0201(0.0838) gcn-final-mse=0.0840(0.0973)
2020-08-04 17:49:38 2000-5019 loss=0.5264(0.2812+0.2453)-0.3696(0.1911+0.1785) sod-mse=0.1015(0.0866) gcn-mse=0.0970(0.0854) gcn-final-mse=0.0855(0.0988)
2020-08-04 17:52:02 3000-5019 loss=0.0517(0.0356+0.0161)-0.3746(0.1935+0.1811) sod-mse=0.0079(0.0879) gcn-mse=0.0086(0.0865) gcn-final-mse=0.0867(0.1000)
2020-08-04 17:53:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 17:54:26 4000-5019 loss=0.1422(0.0875+0.0547)-0.3740(0.1932+0.1808) sod-mse=0.0314(0.0880) gcn-mse=0.0326(0.0867) gcn-final-mse=0.0868(0.1001)
2020-08-04 17:55:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 17:56:49 5000-5019 loss=0.8002(0.3779+0.4223)-0.3721(0.1923+0.1797) sod-mse=0.1453(0.0878) gcn-mse=0.1326(0.0866) gcn-final-mse=0.0867(0.0999)
2020-08-04 17:56:52 E: 5, Train sod-mae-score=0.0554-0.9324 gcn-mae-score=0.0623-0.9002 gcn-final-mse-score=0.0620-0.9032(0.0768/0.9032) loss=0.2347(0.1341+0.1006)
2020-08-04 17:56:52 E: 5, Test  sod-mae-score=0.0878-0.8044 gcn-mae-score=0.0866-0.7450 gcn-final-mse-score=0.0867-0.7513(0.0999/0.7513) loss=0.3719(0.1923+0.1796)

2020-08-04 17:56:52 Start Epoch 6
2020-08-04 17:56:52 Epoch:06,lr=0.0001
2020-08-04 17:56:53    0-10553 loss=0.3657(0.1839+0.1818)-0.3657(0.1839+0.1818) sod-mse=0.0776(0.0776) gcn-mse=0.0758(0.0758) gcn-final-mse=0.0802(0.1042)
2020-08-04 18:02:25 1000-10553 loss=0.3049(0.1736+0.1314)-0.2177(0.1250+0.0926) sod-mse=0.0774(0.0506) gcn-mse=0.0884(0.0567) gcn-final-mse=0.0564(0.0714)
2020-08-04 18:07:55 2000-10553 loss=0.2350(0.1331+0.1020)-0.2185(0.1260+0.0925) sod-mse=0.0373(0.0506) gcn-mse=0.0475(0.0575) gcn-final-mse=0.0572(0.0722)
2020-08-04 18:09:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 18:13:25 3000-10553 loss=0.3151(0.1652+0.1499)-0.2206(0.1270+0.0936) sod-mse=0.0942(0.0517) gcn-mse=0.0914(0.0583) gcn-final-mse=0.0580(0.0730)
2020-08-04 18:18:56 4000-10553 loss=0.0442(0.0285+0.0158)-0.2206(0.1271+0.0935) sod-mse=0.0113(0.0515) gcn-mse=0.0136(0.0584) gcn-final-mse=0.0581(0.0730)
2020-08-04 18:24:28 5000-10553 loss=0.0555(0.0373+0.0181)-0.2182(0.1261+0.0921) sod-mse=0.0099(0.0508) gcn-mse=0.0182(0.0578) gcn-final-mse=0.0575(0.0724)
2020-08-04 18:29:58 6000-10553 loss=0.0835(0.0569+0.0266)-0.2180(0.1261+0.0920) sod-mse=0.0129(0.0507) gcn-mse=0.0161(0.0578) gcn-final-mse=0.0575(0.0723)
2020-08-04 18:31:59 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 18:35:29 7000-10553 loss=0.0415(0.0262+0.0154)-0.2189(0.1265+0.0924) sod-mse=0.0095(0.0508) gcn-mse=0.0126(0.0580) gcn-final-mse=0.0577(0.0725)
2020-08-04 18:40:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 18:41:01 8000-10553 loss=0.2013(0.1263+0.0751)-0.2202(0.1271+0.0930) sod-mse=0.0444(0.0511) gcn-mse=0.0577(0.0583) gcn-final-mse=0.0580(0.0729)
2020-08-04 18:46:30 9000-10553 loss=0.1454(0.0836+0.0618)-0.2200(0.1270+0.0930) sod-mse=0.0431(0.0510) gcn-mse=0.0342(0.0581) gcn-final-mse=0.0578(0.0727)
2020-08-04 18:51:59 10000-10553 loss=0.0489(0.0400+0.0089)-0.2204(0.1272+0.0932) sod-mse=0.0053(0.0511) gcn-mse=0.0096(0.0582) gcn-final-mse=0.0579(0.0728)

2020-08-04 18:55:02    0-5019 loss=0.3026(0.1805+0.1221)-0.3026(0.1805+0.1221) sod-mse=0.0734(0.0734) gcn-mse=0.0896(0.0896) gcn-final-mse=0.0850(0.0983)
2020-08-04 18:57:27 1000-5019 loss=0.0617(0.0407+0.0210)-0.3669(0.1921+0.1747) sod-mse=0.0192(0.0959) gcn-mse=0.0222(0.0923) gcn-final-mse=0.0923(0.1063)
2020-08-04 18:59:52 2000-5019 loss=0.5269(0.2968+0.2301)-0.3777(0.1966+0.1811) sod-mse=0.1044(0.0979) gcn-mse=0.1049(0.0942) gcn-final-mse=0.0941(0.1081)
2020-08-04 19:02:16 3000-5019 loss=0.0507(0.0364+0.0143)-0.3839(0.1992+0.1847) sod-mse=0.0079(0.0996) gcn-mse=0.0098(0.0956) gcn-final-mse=0.0954(0.1095)
2020-08-04 19:03:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 19:04:40 4000-5019 loss=0.1451(0.0979+0.0473)-0.3831(0.1990+0.1841) sod-mse=0.0305(0.0996) gcn-mse=0.0407(0.0958) gcn-final-mse=0.0956(0.1097)
2020-08-04 19:05:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 19:07:03 5000-5019 loss=0.4563(0.2225+0.2338)-0.3817(0.1984+0.1833) sod-mse=0.1297(0.0995) gcn-mse=0.1074(0.0958) gcn-final-mse=0.0956(0.1096)
2020-08-04 19:07:06 E: 6, Train sod-mae-score=0.0512-0.9366 gcn-mae-score=0.0582-0.9044 gcn-final-mse-score=0.0579-0.9073(0.0728/0.9073) loss=0.2205(0.1272+0.0933)
2020-08-04 19:07:06 E: 6, Test  sod-mae-score=0.0995-0.8032 gcn-mae-score=0.0958-0.7368 gcn-final-mse-score=0.0956-0.7424(0.1096/0.7424) loss=0.3815(0.1983+0.1832)

2020-08-04 19:07:06 Start Epoch 7
2020-08-04 19:07:06 Epoch:07,lr=0.0001
2020-08-04 19:07:07    0-10553 loss=0.1270(0.0835+0.0435)-0.1270(0.0835+0.0435) sod-mse=0.0365(0.0365) gcn-mse=0.0438(0.0438) gcn-final-mse=0.0405(0.0552)
2020-08-04 19:12:38 1000-10553 loss=0.0387(0.0226+0.0160)-0.1965(0.1145+0.0820) sod-mse=0.0135(0.0442) gcn-mse=0.0043(0.0510) gcn-final-mse=0.0509(0.0655)
2020-08-04 19:16:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 19:18:09 2000-10553 loss=0.0420(0.0298+0.0122)-0.1916(0.1127+0.0789) sod-mse=0.0075(0.0427) gcn-mse=0.0081(0.0500) gcn-final-mse=0.0498(0.0646)
2020-08-04 19:23:40 3000-10553 loss=0.1382(0.0933+0.0449)-0.2001(0.1170+0.0831) sod-mse=0.0306(0.0452) gcn-mse=0.0342(0.0525) gcn-final-mse=0.0522(0.0669)
2020-08-04 19:29:10 4000-10553 loss=0.0496(0.0360+0.0136)-0.2053(0.1196+0.0857) sod-mse=0.0076(0.0466) gcn-mse=0.0101(0.0537) gcn-final-mse=0.0534(0.0682)
2020-08-04 19:34:40 5000-10553 loss=0.1471(0.1030+0.0440)-0.2046(0.1194+0.0852) sod-mse=0.0295(0.0465) gcn-mse=0.0490(0.0537) gcn-final-mse=0.0534(0.0683)
2020-08-04 19:40:11 6000-10553 loss=0.0703(0.0527+0.0176)-0.2042(0.1193+0.0850) sod-mse=0.0081(0.0464) gcn-mse=0.0153(0.0536) gcn-final-mse=0.0533(0.0681)
2020-08-04 19:45:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 19:45:40 7000-10553 loss=0.0335(0.0278+0.0057)-0.2032(0.1188+0.0844) sod-mse=0.0046(0.0461) gcn-mse=0.0106(0.0534) gcn-final-mse=0.0530(0.0679)
2020-08-04 19:51:10 8000-10553 loss=0.2760(0.1861+0.0899)-0.2046(0.1194+0.0852) sod-mse=0.0579(0.0465) gcn-mse=0.0719(0.0536) gcn-final-mse=0.0532(0.0681)
2020-08-04 19:56:39 9000-10553 loss=0.1159(0.0761+0.0398)-0.2065(0.1203+0.0862) sod-mse=0.0242(0.0471) gcn-mse=0.0281(0.0540) gcn-final-mse=0.0537(0.0686)
2020-08-04 19:59:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 20:02:09 10000-10553 loss=0.0577(0.0446+0.0132)-0.2078(0.1209+0.0870) sod-mse=0.0077(0.0475) gcn-mse=0.0101(0.0543) gcn-final-mse=0.0540(0.0689)

2020-08-04 20:05:15    0-5019 loss=0.7364(0.4425+0.2938)-0.7364(0.4425+0.2938) sod-mse=0.0996(0.0996) gcn-mse=0.1126(0.1126) gcn-final-mse=0.1045(0.1156)
2020-08-04 20:07:43 1000-5019 loss=0.0630(0.0443+0.0188)-0.3182(0.1718+0.1464) sod-mse=0.0168(0.0798) gcn-mse=0.0254(0.0815) gcn-final-mse=0.0816(0.0953)
2020-08-04 20:10:07 2000-5019 loss=0.4214(0.2247+0.1967)-0.3220(0.1734+0.1485) sod-mse=0.1001(0.0804) gcn-mse=0.0914(0.0822) gcn-final-mse=0.0823(0.0960)
2020-08-04 20:12:32 3000-5019 loss=0.0530(0.0376+0.0155)-0.3302(0.1771+0.1530) sod-mse=0.0085(0.0817) gcn-mse=0.0110(0.0834) gcn-final-mse=0.0834(0.0971)
2020-08-04 20:13:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 20:14:56 4000-5019 loss=0.1783(0.1175+0.0608)-0.3287(0.1766+0.1522) sod-mse=0.0345(0.0816) gcn-mse=0.0473(0.0833) gcn-final-mse=0.0834(0.0971)
2020-08-04 20:15:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 20:17:19 5000-5019 loss=0.6310(0.2851+0.3459)-0.3288(0.1767+0.1521) sod-mse=0.1669(0.0817) gcn-mse=0.1416(0.0834) gcn-final-mse=0.0835(0.0971)
2020-08-04 20:17:21 E: 7, Train sod-mae-score=0.0477-0.9398 gcn-mae-score=0.0545-0.9080 gcn-final-mse-score=0.0542-0.9108(0.0691/0.9108) loss=0.2087(0.1213+0.0873)
2020-08-04 20:17:21 E: 7, Test  sod-mae-score=0.0816-0.8179 gcn-mae-score=0.0834-0.7561 gcn-final-mse-score=0.0834-0.7618(0.0971/0.7618) loss=0.3285(0.1766+0.1520)

2020-08-04 20:17:21 Start Epoch 8
2020-08-04 20:17:21 Epoch:08,lr=0.0001
2020-08-04 20:17:23    0-10553 loss=0.0692(0.0494+0.0198)-0.0692(0.0494+0.0198) sod-mse=0.0149(0.0149) gcn-mse=0.0179(0.0179) gcn-final-mse=0.0166(0.0307)
2020-08-04 20:22:55 1000-10553 loss=0.0469(0.0391+0.0078)-0.1854(0.1106+0.0748) sod-mse=0.0060(0.0404) gcn-mse=0.0189(0.0486) gcn-final-mse=0.0482(0.0631)
2020-08-04 20:28:27 2000-10553 loss=0.1723(0.0933+0.0790)-0.1871(0.1110+0.0761) sod-mse=0.0502(0.0409) gcn-mse=0.0472(0.0484) gcn-final-mse=0.0480(0.0631)
2020-08-04 20:33:58 3000-10553 loss=0.4633(0.2470+0.2163)-0.1866(0.1105+0.0761) sod-mse=0.1226(0.0411) gcn-mse=0.1327(0.0482) gcn-final-mse=0.0478(0.0628)
2020-08-04 20:39:29 4000-10553 loss=0.1108(0.0647+0.0461)-0.1852(0.1099+0.0753) sod-mse=0.0344(0.0407) gcn-mse=0.0287(0.0480) gcn-final-mse=0.0476(0.0627)
2020-08-04 20:44:57 5000-10553 loss=0.1580(0.0997+0.0583)-0.1880(0.1114+0.0767) sod-mse=0.0243(0.0415) gcn-mse=0.0374(0.0487) gcn-final-mse=0.0483(0.0633)
2020-08-04 20:50:30 6000-10553 loss=0.1328(0.0702+0.0626)-0.1900(0.1123+0.0777) sod-mse=0.0448(0.0420) gcn-mse=0.0260(0.0491) gcn-final-mse=0.0488(0.0637)
2020-08-04 20:56:00 7000-10553 loss=0.2868(0.1693+0.1175)-0.1915(0.1131+0.0784) sod-mse=0.0806(0.0424) gcn-mse=0.0930(0.0496) gcn-final-mse=0.0492(0.0642)
2020-08-04 21:01:29 8000-10553 loss=0.0915(0.0636+0.0279)-0.1930(0.1139+0.0790) sod-mse=0.0169(0.0429) gcn-mse=0.0250(0.0501) gcn-final-mse=0.0498(0.0647)
2020-08-04 21:03:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 21:03:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 21:06:57 9000-10553 loss=0.1851(0.1285+0.0566)-0.1944(0.1145+0.0799) sod-mse=0.0384(0.0432) gcn-mse=0.0626(0.0504) gcn-final-mse=0.0500(0.0650)
2020-08-04 21:07:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 21:12:29 10000-10553 loss=0.0534(0.0381+0.0152)-0.1958(0.1152+0.0806) sod-mse=0.0123(0.0436) gcn-mse=0.0133(0.0507) gcn-final-mse=0.0503(0.0654)

2020-08-04 21:15:33    0-5019 loss=0.7639(0.4138+0.3502)-0.7639(0.4138+0.3502) sod-mse=0.1091(0.1091) gcn-mse=0.1126(0.1126) gcn-final-mse=0.1065(0.1182)
2020-08-04 21:17:58 1000-5019 loss=0.0358(0.0292+0.0066)-0.3407(0.1809+0.1597) sod-mse=0.0056(0.0692) gcn-mse=0.0109(0.0767) gcn-final-mse=0.0766(0.0895)
2020-08-04 21:20:20 2000-5019 loss=0.6050(0.2858+0.3192)-0.3491(0.1843+0.1648) sod-mse=0.0932(0.0711) gcn-mse=0.0896(0.0784) gcn-final-mse=0.0783(0.0912)
2020-08-04 21:22:43 3000-5019 loss=0.0487(0.0356+0.0131)-0.3555(0.1868+0.1687) sod-mse=0.0067(0.0719) gcn-mse=0.0089(0.0792) gcn-final-mse=0.0791(0.0920)
2020-08-04 21:24:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 21:25:06 4000-5019 loss=0.1344(0.0854+0.0490)-0.3547(0.1869+0.1679) sod-mse=0.0269(0.0717) gcn-mse=0.0269(0.0792) gcn-final-mse=0.0792(0.0920)
2020-08-04 21:25:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 21:27:28 5000-5019 loss=0.8333(0.3152+0.5181)-0.3542(0.1870+0.1672) sod-mse=0.1160(0.0718) gcn-mse=0.1133(0.0795) gcn-final-mse=0.0794(0.0923)
2020-08-04 21:27:31 E: 8, Train sod-mae-score=0.0439-0.9444 gcn-mae-score=0.0509-0.9123 gcn-final-mse-score=0.0505-0.9151(0.0656/0.9151) loss=0.1967(0.1156+0.0811)
2020-08-04 21:27:31 E: 8, Test  sod-mae-score=0.0718-0.8174 gcn-mae-score=0.0796-0.7566 gcn-final-mse-score=0.0794-0.7630(0.0923/0.7630) loss=0.3539(0.1869+0.1670)

2020-08-04 21:27:31 Start Epoch 9
2020-08-04 21:27:31 Epoch:09,lr=0.0001
2020-08-04 21:27:32    0-10553 loss=0.0694(0.0501+0.0194)-0.0694(0.0501+0.0194) sod-mse=0.0131(0.0131) gcn-mse=0.0158(0.0158) gcn-final-mse=0.0165(0.0335)
2020-08-04 21:33:02 1000-10553 loss=0.0445(0.0191+0.0254)-0.1748(0.1049+0.0699) sod-mse=0.0088(0.0368) gcn-mse=0.0076(0.0447) gcn-final-mse=0.0442(0.0593)
2020-08-04 21:38:33 2000-10553 loss=0.0444(0.0308+0.0136)-0.1791(0.1070+0.0721) sod-mse=0.0069(0.0385) gcn-mse=0.0108(0.0460) gcn-final-mse=0.0455(0.0607)
2020-08-04 21:44:03 3000-10553 loss=0.0893(0.0550+0.0343)-0.1836(0.1088+0.0748) sod-mse=0.0282(0.0401) gcn-mse=0.0310(0.0469) gcn-final-mse=0.0464(0.0614)
2020-08-04 21:49:34 4000-10553 loss=0.0815(0.0527+0.0288)-0.1828(0.1082+0.0746) sod-mse=0.0219(0.0400) gcn-mse=0.0235(0.0467) gcn-final-mse=0.0463(0.0613)
2020-08-04 21:55:07 5000-10553 loss=0.1818(0.1091+0.0727)-0.1846(0.1092+0.0754) sod-mse=0.0380(0.0405) gcn-mse=0.0498(0.0473) gcn-final-mse=0.0469(0.0620)
2020-08-04 21:55:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 21:56:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 22:00:36 6000-10553 loss=0.5123(0.2646+0.2477)-0.1872(0.1107+0.0765) sod-mse=0.1267(0.0412) gcn-mse=0.1206(0.0481) gcn-final-mse=0.0477(0.0629)
2020-08-04 22:06:07 7000-10553 loss=0.2145(0.0894+0.1250)-0.1868(0.1106+0.0762) sod-mse=0.0433(0.0411) gcn-mse=0.0358(0.0480) gcn-final-mse=0.0476(0.0628)
2020-08-04 22:11:36 8000-10553 loss=0.0933(0.0586+0.0347)-0.1854(0.1098+0.0755) sod-mse=0.0195(0.0407) gcn-mse=0.0194(0.0475) gcn-final-mse=0.0471(0.0623)
2020-08-04 22:17:07 9000-10553 loss=0.2480(0.1309+0.1171)-0.1865(0.1104+0.0761) sod-mse=0.0478(0.0411) gcn-mse=0.0448(0.0478) gcn-final-mse=0.0474(0.0626)
2020-08-04 22:22:38 10000-10553 loss=0.0380(0.0306+0.0075)-0.1858(0.1101+0.0758) sod-mse=0.0046(0.0409) gcn-mse=0.0136(0.0477) gcn-final-mse=0.0473(0.0624)
2020-08-04 22:23:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg

2020-08-04 22:25:40    0-5019 loss=0.8167(0.4248+0.3919)-0.8167(0.4248+0.3919) sod-mse=0.1022(0.1022) gcn-mse=0.1147(0.1147) gcn-final-mse=0.1078(0.1175)
2020-08-04 22:28:06 1000-5019 loss=0.0823(0.0598+0.0224)-0.3450(0.1816+0.1634) sod-mse=0.0201(0.0700) gcn-mse=0.0385(0.0766) gcn-final-mse=0.0769(0.0899)
2020-08-04 22:30:29 2000-5019 loss=0.7280(0.3578+0.3702)-0.3468(0.1827+0.1641) sod-mse=0.1024(0.0708) gcn-mse=0.1008(0.0774) gcn-final-mse=0.0777(0.0907)
2020-08-04 22:32:54 3000-5019 loss=0.0473(0.0349+0.0123)-0.3513(0.1846+0.1667) sod-mse=0.0061(0.0714) gcn-mse=0.0079(0.0781) gcn-final-mse=0.0784(0.0914)
2020-08-04 22:34:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 22:35:18 4000-5019 loss=0.1460(0.1018+0.0442)-0.3469(0.1830+0.1640) sod-mse=0.0278(0.0711) gcn-mse=0.0435(0.0779) gcn-final-mse=0.0782(0.0912)
2020-08-04 22:35:59 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 22:37:41 5000-5019 loss=0.5629(0.2087+0.3541)-0.3466(0.1830+0.1637) sod-mse=0.1059(0.0712) gcn-mse=0.1018(0.0780) gcn-final-mse=0.0782(0.0912)
2020-08-04 22:37:43 E: 9, Train sod-mae-score=0.0409-0.9476 gcn-mae-score=0.0477-0.9157 gcn-final-mse-score=0.0473-0.9186(0.0624/0.9186) loss=0.1858(0.1101+0.0758)
2020-08-04 22:37:43 E: 9, Test  sod-mae-score=0.0712-0.8082 gcn-mae-score=0.0781-0.7443 gcn-final-mse-score=0.0782-0.7498(0.0912/0.7498) loss=0.3464(0.1829+0.1635)

2020-08-04 22:37:43 Start Epoch 10
2020-08-04 22:37:43 Epoch:10,lr=0.0001
2020-08-04 22:37:45    0-10553 loss=0.2895(0.1508+0.1387)-0.2895(0.1508+0.1387) sod-mse=0.0623(0.0623) gcn-mse=0.0638(0.0638) gcn-final-mse=0.0678(0.0812)
2020-08-04 22:43:14 1000-10553 loss=0.2584(0.1446+0.1139)-0.1695(0.1024+0.0672) sod-mse=0.0709(0.0366) gcn-mse=0.0750(0.0442) gcn-final-mse=0.0439(0.0590)
2020-08-04 22:48:43 2000-10553 loss=0.0776(0.0509+0.0267)-0.1743(0.1046+0.0697) sod-mse=0.0188(0.0381) gcn-mse=0.0212(0.0453) gcn-final-mse=0.0449(0.0600)
2020-08-04 22:53:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 22:54:13 3000-10553 loss=0.1019(0.0688+0.0330)-0.1774(0.1063+0.0711) sod-mse=0.0172(0.0389) gcn-mse=0.0192(0.0460) gcn-final-mse=0.0456(0.0607)
2020-08-04 22:59:43 4000-10553 loss=0.1095(0.0785+0.0310)-0.1752(0.1052+0.0700) sod-mse=0.0166(0.0380) gcn-mse=0.0271(0.0452) gcn-final-mse=0.0448(0.0600)
2020-08-04 23:05:16 5000-10553 loss=0.0882(0.0582+0.0300)-0.1723(0.1037+0.0686) sod-mse=0.0183(0.0371) gcn-mse=0.0221(0.0443) gcn-final-mse=0.0439(0.0590)
2020-08-04 23:10:48 6000-10553 loss=0.0722(0.0452+0.0270)-0.1729(0.1039+0.0690) sod-mse=0.0157(0.0373) gcn-mse=0.0203(0.0444) gcn-final-mse=0.0440(0.0591)
2020-08-04 23:12:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 23:16:17 7000-10553 loss=0.1281(0.0940+0.0341)-0.1750(0.1050+0.0700) sod-mse=0.0239(0.0378) gcn-mse=0.0455(0.0449) gcn-final-mse=0.0444(0.0596)
2020-08-04 23:20:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 23:21:48 8000-10553 loss=0.2630(0.1739+0.0891)-0.1750(0.1051+0.0699) sod-mse=0.0551(0.0377) gcn-mse=0.0676(0.0449) gcn-final-mse=0.0445(0.0597)
2020-08-04 23:27:18 9000-10553 loss=0.0912(0.0566+0.0345)-0.1760(0.1054+0.0705) sod-mse=0.0275(0.0381) gcn-mse=0.0340(0.0451) gcn-final-mse=0.0447(0.0599)
2020-08-04 23:32:47 10000-10553 loss=0.0858(0.0671+0.0187)-0.1771(0.1060+0.0711) sod-mse=0.0159(0.0384) gcn-mse=0.0298(0.0454) gcn-final-mse=0.0450(0.0601)

2020-08-04 23:35:50    0-5019 loss=0.5322(0.2755+0.2567)-0.5322(0.2755+0.2567) sod-mse=0.0966(0.0966) gcn-mse=0.0849(0.0849) gcn-final-mse=0.0793(0.0925)
2020-08-04 23:38:14 1000-5019 loss=0.0537(0.0392+0.0145)-0.3328(0.1789+0.1539) sod-mse=0.0129(0.0756) gcn-mse=0.0202(0.0812) gcn-final-mse=0.0815(0.0948)
2020-08-04 23:40:36 2000-5019 loss=1.0014(0.4564+0.5450)-0.3366(0.1814+0.1552) sod-mse=0.1195(0.0769) gcn-mse=0.1212(0.0828) gcn-final-mse=0.0831(0.0964)
2020-08-04 23:43:00 3000-5019 loss=0.0556(0.0384+0.0172)-0.3396(0.1829+0.1567) sod-mse=0.0108(0.0774) gcn-mse=0.0118(0.0835) gcn-final-mse=0.0838(0.0971)
2020-08-04 23:44:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 23:45:23 4000-5019 loss=0.1896(0.1165+0.0731)-0.3363(0.1815+0.1548) sod-mse=0.0435(0.0769) gcn-mse=0.0494(0.0831) gcn-final-mse=0.0834(0.0967)
2020-08-04 23:46:05 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 23:47:46 5000-5019 loss=0.5697(0.2285+0.3412)-0.3363(0.1816+0.1547) sod-mse=0.1285(0.0768) gcn-mse=0.1174(0.0832) gcn-final-mse=0.0835(0.0967)
2020-08-04 23:47:48 E:10, Train sod-mae-score=0.0385-0.9500 gcn-mae-score=0.0455-0.9179 gcn-final-mse-score=0.0450-0.9206(0.0602/0.9206) loss=0.1775(0.1062+0.0713)
2020-08-04 23:47:48 E:10, Test  sod-mae-score=0.0768-0.8064 gcn-mae-score=0.0832-0.7304 gcn-final-mse-score=0.0835-0.7354(0.0967/0.7354) loss=0.3361(0.1815+0.1546)

2020-08-04 23:47:48 Start Epoch 11
2020-08-04 23:47:48 Epoch:11,lr=0.0001
2020-08-04 23:47:50    0-10553 loss=0.1207(0.0753+0.0455)-0.1207(0.0753+0.0455) sod-mse=0.0340(0.0340) gcn-mse=0.0350(0.0350) gcn-final-mse=0.0309(0.0412)
2020-08-04 23:53:21 1000-10553 loss=0.1193(0.0748+0.0445)-0.1585(0.0969+0.0616) sod-mse=0.0299(0.0330) gcn-mse=0.0339(0.0407) gcn-final-mse=0.0402(0.0554)
2020-08-04 23:54:58 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 23:58:51 2000-10553 loss=0.0837(0.0593+0.0244)-0.1589(0.0969+0.0620) sod-mse=0.0174(0.0332) gcn-mse=0.0228(0.0404) gcn-final-mse=0.0399(0.0553)
2020-08-05 00:04:21 3000-10553 loss=0.5239(0.2970+0.2269)-0.1628(0.0988+0.0640) sod-mse=0.1033(0.0343) gcn-mse=0.1149(0.0413) gcn-final-mse=0.0408(0.0561)
2020-08-05 00:09:50 4000-10553 loss=0.7590(0.3760+0.3830)-0.1617(0.0981+0.0636) sod-mse=0.1371(0.0339) gcn-mse=0.1356(0.0407) gcn-final-mse=0.0403(0.0555)
2020-08-05 00:15:19 5000-10553 loss=0.1790(0.1061+0.0730)-0.1647(0.0997+0.0650) sod-mse=0.0439(0.0348) gcn-mse=0.0452(0.0417) gcn-final-mse=0.0413(0.0565)
2020-08-05 00:20:50 6000-10553 loss=0.1693(0.1219+0.0475)-0.1640(0.0993+0.0647) sod-mse=0.0337(0.0348) gcn-mse=0.0605(0.0417) gcn-final-mse=0.0413(0.0565)
2020-08-05 00:26:20 7000-10553 loss=0.0652(0.0478+0.0174)-0.1644(0.0996+0.0648) sod-mse=0.0106(0.0348) gcn-mse=0.0152(0.0418) gcn-final-mse=0.0414(0.0566)
2020-08-05 00:31:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 00:31:49 8000-10553 loss=0.1017(0.0757+0.0260)-0.1657(0.1003+0.0654) sod-mse=0.0164(0.0351) gcn-mse=0.0211(0.0421) gcn-final-mse=0.0416(0.0569)
2020-08-05 00:37:19 9000-10553 loss=0.1100(0.0723+0.0377)-0.1667(0.1008+0.0659) sod-mse=0.0228(0.0355) gcn-mse=0.0336(0.0424) gcn-final-mse=0.0420(0.0573)
2020-08-05 00:42:49 10000-10553 loss=0.0734(0.0473+0.0261)-0.1667(0.1009+0.0659) sod-mse=0.0203(0.0354) gcn-mse=0.0262(0.0424) gcn-final-mse=0.0420(0.0573)
2020-08-05 00:45:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg

2020-08-05 00:45:55    0-5019 loss=0.5437(0.3560+0.1878)-0.5437(0.3560+0.1878) sod-mse=0.0901(0.0901) gcn-mse=0.0924(0.0924) gcn-final-mse=0.0857(0.0965)
2020-08-05 00:48:21 1000-5019 loss=0.0712(0.0412+0.0300)-0.3207(0.1786+0.1420) sod-mse=0.0274(0.0833) gcn-mse=0.0219(0.0744) gcn-final-mse=0.0748(0.0882)
2020-08-05 00:50:45 2000-5019 loss=0.6343(0.3704+0.2639)-0.3311(0.1834+0.1478) sod-mse=0.1182(0.0859) gcn-mse=0.1106(0.0768) gcn-final-mse=0.0771(0.0905)
2020-08-05 00:53:09 3000-5019 loss=0.0522(0.0370+0.0152)-0.3340(0.1850+0.1490) sod-mse=0.0092(0.0867) gcn-mse=0.0104(0.0776) gcn-final-mse=0.0780(0.0913)
2020-08-05 00:54:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 00:55:33 4000-5019 loss=0.1274(0.0844+0.0430)-0.3331(0.1848+0.1483) sod-mse=0.0291(0.0866) gcn-mse=0.0306(0.0777) gcn-final-mse=0.0780(0.0913)
2020-08-05 00:56:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 00:57:56 5000-5019 loss=0.4529(0.2285+0.2244)-0.3318(0.1843+0.1476) sod-mse=0.1300(0.0865) gcn-mse=0.1158(0.0777) gcn-final-mse=0.0780(0.0912)
2020-08-05 00:57:58 E:11, Train sod-mae-score=0.0354-0.9529 gcn-mae-score=0.0423-0.9210 gcn-final-mse-score=0.0419-0.9238(0.0572/0.9238) loss=0.1665(0.1007+0.0658)
2020-08-05 00:57:58 E:11, Test  sod-mae-score=0.0865-0.8256 gcn-mae-score=0.0777-0.7301 gcn-final-mse-score=0.0779-0.7354(0.0912/0.7354) loss=0.3317(0.1842+0.1475)

2020-08-05 00:57:58 Start Epoch 12
2020-08-05 00:57:58 Epoch:12,lr=0.0001
2020-08-05 00:57:59    0-10553 loss=0.1123(0.0676+0.0447)-0.1123(0.0676+0.0447) sod-mse=0.0351(0.0351) gcn-mse=0.0329(0.0329) gcn-final-mse=0.0327(0.0451)
2020-08-05 01:03:29 1000-10553 loss=0.2220(0.1311+0.0909)-0.1545(0.0950+0.0595) sod-mse=0.0593(0.0320) gcn-mse=0.0591(0.0391) gcn-final-mse=0.0386(0.0539)
2020-08-05 01:08:58 2000-10553 loss=0.0191(0.0142+0.0049)-0.1594(0.0972+0.0622) sod-mse=0.0030(0.0335) gcn-mse=0.0036(0.0404) gcn-final-mse=0.0400(0.0551)
2020-08-05 01:14:30 3000-10553 loss=0.1254(0.0745+0.0509)-0.1592(0.0970+0.0622) sod-mse=0.0203(0.0334) gcn-mse=0.0264(0.0402) gcn-final-mse=0.0397(0.0550)
2020-08-05 01:16:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 01:20:01 4000-10553 loss=0.0534(0.0442+0.0091)-0.1587(0.0968+0.0620) sod-mse=0.0047(0.0331) gcn-mse=0.0079(0.0401) gcn-final-mse=0.0396(0.0549)
2020-08-05 01:25:31 5000-10553 loss=0.0644(0.0458+0.0186)-0.1590(0.0967+0.0623) sod-mse=0.0123(0.0331) gcn-mse=0.0257(0.0398) gcn-final-mse=0.0394(0.0547)
2020-08-05 01:31:02 6000-10553 loss=0.1606(0.1090+0.0516)-0.1578(0.0962+0.0615) sod-mse=0.0236(0.0328) gcn-mse=0.0372(0.0396) gcn-final-mse=0.0391(0.0545)
2020-08-05 01:34:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 01:36:31 7000-10553 loss=0.1182(0.0849+0.0333)-0.1579(0.0963+0.0616) sod-mse=0.0177(0.0328) gcn-mse=0.0288(0.0396) gcn-final-mse=0.0391(0.0545)
2020-08-05 01:37:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 01:42:00 8000-10553 loss=0.1004(0.0721+0.0282)-0.1590(0.0969+0.0622) sod-mse=0.0156(0.0331) gcn-mse=0.0217(0.0399) gcn-final-mse=0.0394(0.0547)
2020-08-05 01:47:29 9000-10553 loss=0.1557(0.0913+0.0644)-0.1596(0.0972+0.0624) sod-mse=0.0298(0.0332) gcn-mse=0.0331(0.0401) gcn-final-mse=0.0396(0.0549)
2020-08-05 01:52:58 10000-10553 loss=0.0629(0.0476+0.0153)-0.1597(0.0973+0.0624) sod-mse=0.0101(0.0333) gcn-mse=0.0150(0.0401) gcn-final-mse=0.0397(0.0550)

2020-08-05 01:56:02    0-5019 loss=1.0844(0.5751+0.5093)-1.0844(0.5751+0.5093) sod-mse=0.1290(0.1290) gcn-mse=0.1460(0.1460) gcn-final-mse=0.1387(0.1505)
2020-08-05 01:58:29 1000-5019 loss=0.0393(0.0326+0.0067)-0.3778(0.1932+0.1846) sod-mse=0.0057(0.0691) gcn-mse=0.0139(0.0754) gcn-final-mse=0.0754(0.0891)
2020-08-05 02:00:54 2000-5019 loss=0.7169(0.3349+0.3820)-0.3845(0.1960+0.1885) sod-mse=0.1131(0.0706) gcn-mse=0.1088(0.0768) gcn-final-mse=0.0769(0.0905)
2020-08-05 02:03:18 3000-5019 loss=0.0508(0.0367+0.0142)-0.3893(0.1976+0.1918) sod-mse=0.0080(0.0714) gcn-mse=0.0102(0.0778) gcn-final-mse=0.0779(0.0915)
2020-08-05 02:04:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 02:05:43 4000-5019 loss=0.1498(0.0966+0.0532)-0.3865(0.1967+0.1899) sod-mse=0.0269(0.0710) gcn-mse=0.0341(0.0777) gcn-final-mse=0.0778(0.0914)
2020-08-05 02:06:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 02:08:07 5000-5019 loss=1.0122(0.4116+0.6006)-0.3860(0.1965+0.1894) sod-mse=0.1129(0.0712) gcn-mse=0.1023(0.0779) gcn-final-mse=0.0779(0.0915)
2020-08-05 02:08:09 E:12, Train sod-mae-score=0.0332-0.9555 gcn-mae-score=0.0401-0.9238 gcn-final-mse-score=0.0396-0.9266(0.0550/0.9266) loss=0.1594(0.0971+0.0623)
2020-08-05 02:08:09 E:12, Test  sod-mae-score=0.0712-0.8092 gcn-mae-score=0.0779-0.7456 gcn-final-mse-score=0.0779-0.7513(0.0915/0.7513) loss=0.3859(0.1965+0.1894)

2020-08-05 02:08:09 Start Epoch 13
2020-08-05 02:08:09 Epoch:13,lr=0.0001
2020-08-05 02:08:11    0-10553 loss=0.1295(0.0925+0.0369)-0.1295(0.0925+0.0369) sod-mse=0.0243(0.0243) gcn-mse=0.0408(0.0408) gcn-final-mse=0.0383(0.0555)
2020-08-05 02:13:41 1000-10553 loss=0.1185(0.0896+0.0289)-0.1458(0.0910+0.0547) sod-mse=0.0151(0.0288) gcn-mse=0.0215(0.0360) gcn-final-mse=0.0354(0.0512)
2020-08-05 02:19:11 2000-10553 loss=0.0626(0.0465+0.0160)-0.1507(0.0934+0.0573) sod-mse=0.0096(0.0305) gcn-mse=0.0118(0.0375) gcn-final-mse=0.0369(0.0526)
2020-08-05 02:24:42 3000-10553 loss=0.2100(0.1301+0.0798)-0.1460(0.0910+0.0551) sod-mse=0.0637(0.0292) gcn-mse=0.0885(0.0363) gcn-final-mse=0.0357(0.0512)
2020-08-05 02:25:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 02:30:15 4000-10553 loss=0.1345(0.0903+0.0442)-0.1462(0.0910+0.0552) sod-mse=0.0275(0.0293) gcn-mse=0.0295(0.0363) gcn-final-mse=0.0357(0.0513)
2020-08-05 02:35:45 5000-10553 loss=0.1772(0.1130+0.0642)-0.1479(0.0918+0.0561) sod-mse=0.0423(0.0296) gcn-mse=0.0605(0.0367) gcn-final-mse=0.0361(0.0516)
2020-08-05 02:41:16 6000-10553 loss=0.1769(0.1056+0.0714)-0.1515(0.0936+0.0579) sod-mse=0.0547(0.0307) gcn-mse=0.0677(0.0378) gcn-final-mse=0.0372(0.0527)
2020-08-05 02:46:50 7000-10553 loss=0.1017(0.0655+0.0362)-0.1508(0.0933+0.0575) sod-mse=0.0196(0.0306) gcn-mse=0.0233(0.0376) gcn-final-mse=0.0371(0.0526)
2020-08-05 02:48:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 02:52:22 8000-10553 loss=0.0611(0.0421+0.0189)-0.1505(0.0930+0.0575) sod-mse=0.0161(0.0305) gcn-mse=0.0173(0.0375) gcn-final-mse=0.0370(0.0525)
2020-08-05 02:57:54 9000-10553 loss=0.1446(0.1054+0.0392)-0.1512(0.0932+0.0580) sod-mse=0.0169(0.0308) gcn-mse=0.0266(0.0376) gcn-final-mse=0.0371(0.0526)
2020-08-05 03:03:25 10000-10553 loss=0.0792(0.0543+0.0249)-0.1511(0.0931+0.0580) sod-mse=0.0166(0.0308) gcn-mse=0.0163(0.0376) gcn-final-mse=0.0371(0.0526)
2020-08-05 03:04:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg

2020-08-05 03:06:29    0-5019 loss=0.7222(0.3610+0.3612)-0.7222(0.3610+0.3612) sod-mse=0.1091(0.1091) gcn-mse=0.1146(0.1146) gcn-final-mse=0.1065(0.1175)
2020-08-05 03:08:54 1000-5019 loss=0.0377(0.0307+0.0070)-0.3318(0.1726+0.1593) sod-mse=0.0059(0.0644) gcn-mse=0.0115(0.0669) gcn-final-mse=0.0669(0.0801)
2020-08-05 03:11:18 2000-5019 loss=0.5217(0.2709+0.2508)-0.3290(0.1716+0.1574) sod-mse=0.0930(0.0649) gcn-mse=0.0931(0.0677) gcn-final-mse=0.0677(0.0809)
2020-08-05 03:13:41 3000-5019 loss=0.0481(0.0347+0.0135)-0.3318(0.1730+0.1588) sod-mse=0.0073(0.0653) gcn-mse=0.0082(0.0681) gcn-final-mse=0.0682(0.0813)
2020-08-05 03:15:01 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 03:16:05 4000-5019 loss=0.1497(0.0969+0.0528)-0.3299(0.1723+0.1575) sod-mse=0.0304(0.0653) gcn-mse=0.0359(0.0682) gcn-final-mse=0.0682(0.0813)
2020-08-05 03:16:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 03:18:29 5000-5019 loss=0.9103(0.3793+0.5310)-0.3307(0.1727+0.1580) sod-mse=0.1164(0.0655) gcn-mse=0.1221(0.0684) gcn-final-mse=0.0684(0.0815)
2020-08-05 03:18:32 E:13, Train sod-mae-score=0.0311-0.9577 gcn-mae-score=0.0378-0.9260 gcn-final-mse-score=0.0373-0.9288(0.0527/0.9288) loss=0.1518(0.0934+0.0584)
2020-08-05 03:18:32 E:13, Test  sod-mae-score=0.0654-0.8279 gcn-mae-score=0.0684-0.7660 gcn-final-mse-score=0.0684-0.7721(0.0815/0.7721) loss=0.3305(0.1727+0.1579)

2020-08-05 03:18:32 Start Epoch 14
2020-08-05 03:18:32 Epoch:14,lr=0.0001
2020-08-05 03:18:33    0-10553 loss=0.2590(0.1671+0.0919)-0.2590(0.1671+0.0919) sod-mse=0.0672(0.0672) gcn-mse=0.0965(0.0965) gcn-final-mse=0.0919(0.1033)
2020-08-05 03:23:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 03:24:07 1000-10553 loss=0.0945(0.0640+0.0305)-0.1433(0.0889+0.0544) sod-mse=0.0199(0.0287) gcn-mse=0.0235(0.0351) gcn-final-mse=0.0345(0.0502)
2020-08-05 03:29:39 2000-10553 loss=0.0925(0.0590+0.0334)-0.1478(0.0907+0.0571) sod-mse=0.0192(0.0301) gcn-mse=0.0209(0.0362) gcn-final-mse=0.0356(0.0510)
2020-08-05 03:35:12 3000-10553 loss=0.0980(0.0628+0.0352)-0.1425(0.0882+0.0543) sod-mse=0.0234(0.0286) gcn-mse=0.0208(0.0348) gcn-final-mse=0.0342(0.0497)
2020-08-05 03:39:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 03:40:42 4000-10553 loss=0.1540(0.0938+0.0602)-0.1436(0.0889+0.0547) sod-mse=0.0378(0.0290) gcn-mse=0.0489(0.0354) gcn-final-mse=0.0348(0.0502)
2020-08-05 03:45:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 03:46:13 5000-10553 loss=0.3793(0.2008+0.1786)-0.1451(0.0897+0.0554) sod-mse=0.1004(0.0295) gcn-mse=0.0965(0.0358) gcn-final-mse=0.0353(0.0507)
2020-08-05 03:51:44 6000-10553 loss=0.1010(0.0642+0.0367)-0.1441(0.0893+0.0548) sod-mse=0.0267(0.0291) gcn-mse=0.0246(0.0355) gcn-final-mse=0.0349(0.0504)
2020-08-05 03:57:14 7000-10553 loss=0.0443(0.0368+0.0075)-0.1440(0.0893+0.0547) sod-mse=0.0047(0.0291) gcn-mse=0.0126(0.0356) gcn-final-mse=0.0350(0.0505)
2020-08-05 04:02:46 8000-10553 loss=0.1556(0.0969+0.0588)-0.1437(0.0891+0.0546) sod-mse=0.0308(0.0290) gcn-mse=0.0417(0.0354) gcn-final-mse=0.0349(0.0503)
2020-08-05 04:08:17 9000-10553 loss=0.0891(0.0623+0.0268)-0.1454(0.0901+0.0553) sod-mse=0.0161(0.0294) gcn-mse=0.0241(0.0359) gcn-final-mse=0.0354(0.0509)
2020-08-05 04:13:50 10000-10553 loss=0.2226(0.1266+0.0960)-0.1456(0.0902+0.0554) sod-mse=0.0431(0.0295) gcn-mse=0.0408(0.0359) gcn-final-mse=0.0354(0.0509)

2020-08-05 04:16:54    0-5019 loss=0.9017(0.5559+0.3458)-0.9017(0.5559+0.3458) sod-mse=0.1050(0.1050) gcn-mse=0.1232(0.1232) gcn-final-mse=0.1172(0.1308)
2020-08-05 04:19:20 1000-5019 loss=0.0404(0.0314+0.0090)-0.3689(0.1987+0.1702) sod-mse=0.0079(0.0706) gcn-mse=0.0132(0.0746) gcn-final-mse=0.0744(0.0876)
2020-08-05 04:21:45 2000-5019 loss=0.4939(0.2535+0.2404)-0.3801(0.2043+0.1757) sod-mse=0.0918(0.0732) gcn-mse=0.0894(0.0772) gcn-final-mse=0.0770(0.0901)
2020-08-05 04:24:09 3000-5019 loss=0.0483(0.0357+0.0127)-0.3877(0.2075+0.1802) sod-mse=0.0066(0.0744) gcn-mse=0.0084(0.0783) gcn-final-mse=0.0782(0.0912)
2020-08-05 04:25:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 04:26:33 4000-5019 loss=0.1507(0.1089+0.0417)-0.3848(0.2061+0.1786) sod-mse=0.0262(0.0743) gcn-mse=0.0433(0.0782) gcn-final-mse=0.0781(0.0911)
2020-08-05 04:27:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 04:28:57 5000-5019 loss=0.9936(0.4436+0.5500)-0.3833(0.2055+0.1778) sod-mse=0.1154(0.0745) gcn-mse=0.1199(0.0785) gcn-final-mse=0.0783(0.0913)
2020-08-05 04:28:59 E:14, Train sod-mae-score=0.0297-0.9598 gcn-mae-score=0.0361-0.9277 gcn-final-mse-score=0.0356-0.9304(0.0512/0.9304) loss=0.1465(0.0907+0.0558)
2020-08-05 04:28:59 E:14, Test  sod-mae-score=0.0745-0.8121 gcn-mae-score=0.0785-0.7513 gcn-final-mse-score=0.0783-0.7576(0.0913/0.7576) loss=0.3829(0.2053+0.1776)

2020-08-05 04:28:59 Start Epoch 15
2020-08-05 04:28:59 Epoch:15,lr=0.0001
2020-08-05 04:29:01    0-10553 loss=0.0644(0.0473+0.0171)-0.0644(0.0473+0.0171) sod-mse=0.0099(0.0099) gcn-mse=0.0162(0.0162) gcn-final-mse=0.0144(0.0257)
2020-08-05 04:34:32 1000-10553 loss=0.1119(0.0808+0.0311)-0.1341(0.0847+0.0494) sod-mse=0.0211(0.0263) gcn-mse=0.0276(0.0326) gcn-final-mse=0.0319(0.0478)
2020-08-05 04:40:03 2000-10553 loss=0.0619(0.0448+0.0171)-0.1296(0.0826+0.0469) sod-mse=0.0081(0.0250) gcn-mse=0.0141(0.0318) gcn-final-mse=0.0311(0.0469)
2020-08-05 04:45:33 3000-10553 loss=0.0414(0.0330+0.0084)-0.1315(0.0835+0.0480) sod-mse=0.0042(0.0254) gcn-mse=0.0069(0.0322) gcn-final-mse=0.0316(0.0473)
2020-08-05 04:51:05 4000-10553 loss=0.1021(0.0649+0.0372)-0.1335(0.0843+0.0492) sod-mse=0.0251(0.0259) gcn-mse=0.0279(0.0325) gcn-final-mse=0.0319(0.0476)
2020-08-05 04:52:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 04:56:35 5000-10553 loss=0.0835(0.0646+0.0189)-0.1347(0.0849+0.0498) sod-mse=0.0110(0.0263) gcn-mse=0.0189(0.0327) gcn-final-mse=0.0321(0.0477)
2020-08-05 05:02:08 6000-10553 loss=0.0700(0.0503+0.0197)-0.1366(0.0858+0.0507) sod-mse=0.0129(0.0268) gcn-mse=0.0141(0.0333) gcn-final-mse=0.0327(0.0483)
2020-08-05 05:07:41 7000-10553 loss=0.0524(0.0403+0.0122)-0.1386(0.0868+0.0518) sod-mse=0.0066(0.0274) gcn-mse=0.0133(0.0338) gcn-final-mse=0.0333(0.0489)
2020-08-05 05:12:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 05:13:13 8000-10553 loss=0.3270(0.1495+0.1774)-0.1399(0.0875+0.0524) sod-mse=0.1045(0.0278) gcn-mse=0.0799(0.0342) gcn-final-mse=0.0337(0.0493)
2020-08-05 05:17:58 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 05:18:45 9000-10553 loss=0.0953(0.0500+0.0453)-0.1397(0.0874+0.0523) sod-mse=0.0300(0.0277) gcn-mse=0.0216(0.0342) gcn-final-mse=0.0336(0.0492)
2020-08-05 05:24:17 10000-10553 loss=0.1794(0.1000+0.0794)-0.1399(0.0875+0.0524) sod-mse=0.0496(0.0278) gcn-mse=0.0562(0.0343) gcn-final-mse=0.0337(0.0493)

2020-08-05 05:27:20    0-5019 loss=0.9077(0.5474+0.3603)-0.9077(0.5474+0.3603) sod-mse=0.1046(0.1046) gcn-mse=0.1180(0.1180) gcn-final-mse=0.1101(0.1216)
2020-08-05 05:29:45 1000-5019 loss=0.0353(0.0277+0.0076)-0.3750(0.1966+0.1784) sod-mse=0.0061(0.0713) gcn-mse=0.0097(0.0758) gcn-final-mse=0.0758(0.0889)
2020-08-05 05:32:08 2000-5019 loss=0.7183(0.3879+0.3303)-0.3813(0.1994+0.1820) sod-mse=0.1022(0.0729) gcn-mse=0.1055(0.0773) gcn-final-mse=0.0774(0.0904)
2020-08-05 05:34:31 3000-5019 loss=0.0506(0.0359+0.0147)-0.3845(0.2007+0.1838) sod-mse=0.0079(0.0740) gcn-mse=0.0095(0.0784) gcn-final-mse=0.0785(0.0915)
2020-08-05 05:35:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 05:36:55 4000-5019 loss=0.1693(0.1057+0.0635)-0.3820(0.1997+0.1823) sod-mse=0.0383(0.0738) gcn-mse=0.0439(0.0783) gcn-final-mse=0.0783(0.0913)
2020-08-05 05:37:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 05:39:18 5000-5019 loss=0.6529(0.2805+0.3724)-0.3818(0.1998+0.1820) sod-mse=0.1024(0.0739) gcn-mse=0.1007(0.0785) gcn-final-mse=0.0785(0.0915)
2020-08-05 05:39:20 E:15, Train sod-mae-score=0.0282-0.9617 gcn-mae-score=0.0346-0.9305 gcn-final-mse-score=0.0341-0.9332(0.0496/0.9332) loss=0.1415(0.0882+0.0533)
2020-08-05 05:39:20 E:15, Test  sod-mae-score=0.0739-0.8003 gcn-mae-score=0.0785-0.7439 gcn-final-mse-score=0.0785-0.7497(0.0915/0.7497) loss=0.3814(0.1997+0.1818)

2020-08-05 05:39:20 Start Epoch 16
2020-08-05 05:39:20 Epoch:16,lr=0.0001
2020-08-05 05:39:22    0-10553 loss=0.0512(0.0352+0.0160)-0.0512(0.0352+0.0160) sod-mse=0.0102(0.0102) gcn-mse=0.0097(0.0097) gcn-final-mse=0.0099(0.0254)
2020-08-05 05:44:53 1000-10553 loss=0.0674(0.0464+0.0210)-0.1271(0.0808+0.0462) sod-mse=0.0130(0.0243) gcn-mse=0.0111(0.0309) gcn-final-mse=0.0305(0.0461)
2020-08-05 05:45:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 05:50:24 2000-10553 loss=0.0734(0.0546+0.0187)-0.1267(0.0807+0.0460) sod-mse=0.0122(0.0243) gcn-mse=0.0234(0.0309) gcn-final-mse=0.0303(0.0459)
2020-08-05 05:54:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 05:55:58 3000-10553 loss=0.2666(0.1510+0.1157)-0.1300(0.0825+0.0475) sod-mse=0.0812(0.0252) gcn-mse=0.0885(0.0319) gcn-final-mse=0.0313(0.0469)
2020-08-05 06:01:30 4000-10553 loss=0.1727(0.0794+0.0933)-0.1315(0.0833+0.0483) sod-mse=0.0534(0.0255) gcn-mse=0.0355(0.0321) gcn-final-mse=0.0316(0.0472)
2020-08-05 06:07:03 5000-10553 loss=0.0502(0.0340+0.0162)-0.1310(0.0830+0.0480) sod-mse=0.0095(0.0255) gcn-mse=0.0112(0.0321) gcn-final-mse=0.0315(0.0471)
2020-08-05 06:12:34 6000-10553 loss=0.2300(0.1451+0.0850)-0.1335(0.0842+0.0493) sod-mse=0.0586(0.0262) gcn-mse=0.0681(0.0326) gcn-final-mse=0.0321(0.0477)
2020-08-05 06:18:05 7000-10553 loss=0.2045(0.1277+0.0768)-0.1337(0.0844+0.0493) sod-mse=0.0368(0.0263) gcn-mse=0.0472(0.0327) gcn-final-mse=0.0322(0.0478)
2020-08-05 06:23:35 8000-10553 loss=0.0641(0.0501+0.0140)-0.1339(0.0845+0.0495) sod-mse=0.0082(0.0263) gcn-mse=0.0107(0.0327) gcn-final-mse=0.0322(0.0478)
2020-08-05 06:29:07 9000-10553 loss=0.1356(0.0921+0.0435)-0.1350(0.0850+0.0499) sod-mse=0.0280(0.0265) gcn-mse=0.0296(0.0330) gcn-final-mse=0.0325(0.0481)
2020-08-05 06:30:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 06:34:38 10000-10553 loss=0.0402(0.0301+0.0101)-0.1358(0.0854+0.0505) sod-mse=0.0061(0.0268) gcn-mse=0.0079(0.0332) gcn-final-mse=0.0326(0.0482)

2020-08-05 06:37:43    0-5019 loss=0.7790(0.4283+0.3507)-0.7790(0.4283+0.3507) sod-mse=0.0815(0.0815) gcn-mse=0.0871(0.0871) gcn-final-mse=0.0811(0.0922)
2020-08-05 06:40:08 1000-5019 loss=0.0475(0.0397+0.0079)-0.3716(0.1863+0.1853) sod-mse=0.0066(0.0676) gcn-mse=0.0190(0.0720) gcn-final-mse=0.0717(0.0850)
2020-08-05 06:42:32 2000-5019 loss=0.6319(0.2989+0.3331)-0.3827(0.1902+0.1925) sod-mse=0.0833(0.0690) gcn-mse=0.0873(0.0732) gcn-final-mse=0.0729(0.0862)
2020-08-05 06:44:57 3000-5019 loss=0.0497(0.0361+0.0136)-0.3888(0.1923+0.1965) sod-mse=0.0063(0.0705) gcn-mse=0.0082(0.0743) gcn-final-mse=0.0742(0.0874)
2020-08-05 06:46:18 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 06:47:22 4000-5019 loss=0.1101(0.0766+0.0334)-0.3871(0.1916+0.1955) sod-mse=0.0152(0.0701) gcn-mse=0.0206(0.0741) gcn-final-mse=0.0739(0.0871)
2020-08-05 06:48:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 06:49:46 5000-5019 loss=1.3298(0.5515+0.7783)-0.3849(0.1910+0.1939) sod-mse=0.1230(0.0700) gcn-mse=0.1174(0.0742) gcn-final-mse=0.0739(0.0871)
2020-08-05 06:49:48 E:16, Train sod-mae-score=0.0271-0.9624 gcn-mae-score=0.0334-0.9305 gcn-final-mse-score=0.0328-0.9332(0.0484/0.9332) loss=0.1369(0.0859+0.0510)
2020-08-05 06:49:48 E:16, Test  sod-mae-score=0.0699-0.8221 gcn-mae-score=0.0742-0.7647 gcn-final-mse-score=0.0739-0.7706(0.0871/0.7706) loss=0.3847(0.1909+0.1937)

2020-08-05 06:49:48 Start Epoch 17
2020-08-05 06:49:48 Epoch:17,lr=0.0001
2020-08-05 06:49:49    0-10553 loss=0.1574(0.1170+0.0403)-0.1574(0.1170+0.0403) sod-mse=0.0224(0.0224) gcn-mse=0.0330(0.0330) gcn-final-mse=0.0286(0.0699)
2020-08-05 06:50:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 06:55:21 1000-10553 loss=0.1117(0.0777+0.0340)-0.1268(0.0818+0.0451) sod-mse=0.0192(0.0241) gcn-mse=0.0215(0.0313) gcn-final-mse=0.0307(0.0465)
2020-08-05 06:56:50 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 07:00:51 2000-10553 loss=0.0878(0.0622+0.0256)-0.1243(0.0804+0.0439) sod-mse=0.0169(0.0232) gcn-mse=0.0199(0.0300) gcn-final-mse=0.0294(0.0453)
2020-08-05 07:06:23 3000-10553 loss=0.3315(0.1846+0.1470)-0.1263(0.0812+0.0451) sod-mse=0.0545(0.0237) gcn-mse=0.0676(0.0305) gcn-final-mse=0.0299(0.0457)
2020-08-05 07:11:53 4000-10553 loss=0.0601(0.0397+0.0204)-0.1296(0.0826+0.0471) sod-mse=0.0140(0.0248) gcn-mse=0.0154(0.0313) gcn-final-mse=0.0307(0.0464)
2020-08-05 07:12:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 07:17:25 5000-10553 loss=0.0968(0.0567+0.0401)-0.1282(0.0819+0.0463) sod-mse=0.0260(0.0244) gcn-mse=0.0306(0.0311) gcn-final-mse=0.0305(0.0462)
2020-08-05 07:22:58 6000-10553 loss=0.0770(0.0594+0.0176)-0.1301(0.0828+0.0473) sod-mse=0.0141(0.0250) gcn-mse=0.0257(0.0315) gcn-final-mse=0.0310(0.0467)
2020-08-05 07:28:29 7000-10553 loss=0.0624(0.0461+0.0163)-0.1300(0.0827+0.0473) sod-mse=0.0084(0.0250) gcn-mse=0.0140(0.0315) gcn-final-mse=0.0309(0.0466)
2020-08-05 07:34:00 8000-10553 loss=0.1208(0.0928+0.0280)-0.1321(0.0837+0.0484) sod-mse=0.0189(0.0256) gcn-mse=0.0297(0.0321) gcn-final-mse=0.0315(0.0472)
2020-08-05 07:39:33 9000-10553 loss=0.0561(0.0410+0.0151)-0.1315(0.0834+0.0481) sod-mse=0.0115(0.0254) gcn-mse=0.0182(0.0319) gcn-final-mse=0.0313(0.0470)
2020-08-05 07:45:03 10000-10553 loss=0.1334(0.0795+0.0539)-0.1323(0.0837+0.0486) sod-mse=0.0226(0.0256) gcn-mse=0.0288(0.0320) gcn-final-mse=0.0315(0.0472)

2020-08-05 07:48:08    0-5019 loss=0.9216(0.5436+0.3781)-0.9216(0.5436+0.3781) sod-mse=0.0937(0.0937) gcn-mse=0.1081(0.1081) gcn-final-mse=0.1005(0.1127)
2020-08-05 07:50:33 1000-5019 loss=0.0415(0.0340+0.0074)-0.3394(0.1797+0.1597) sod-mse=0.0064(0.0578) gcn-mse=0.0154(0.0631) gcn-final-mse=0.0632(0.0771)
2020-08-05 07:52:57 2000-5019 loss=0.8266(0.4395+0.3871)-0.3491(0.1827+0.1664) sod-mse=0.0949(0.0594) gcn-mse=0.0986(0.0645) gcn-final-mse=0.0646(0.0783)
2020-08-05 07:55:21 3000-5019 loss=0.0470(0.0349+0.0120)-0.3556(0.1855+0.1701) sod-mse=0.0063(0.0607) gcn-mse=0.0085(0.0658) gcn-final-mse=0.0659(0.0796)
2020-08-05 07:56:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 07:57:45 4000-5019 loss=0.1049(0.0747+0.0302)-0.3507(0.1835+0.1672) sod-mse=0.0154(0.0601) gcn-mse=0.0218(0.0654) gcn-final-mse=0.0655(0.0792)
2020-08-05 07:58:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 08:00:07 5000-5019 loss=1.1771(0.5127+0.6644)-0.3531(0.1847+0.1685) sod-mse=0.1431(0.0605) gcn-mse=0.1380(0.0658) gcn-final-mse=0.0658(0.0794)
2020-08-05 08:00:10 E:17, Train sod-mae-score=0.0256-0.9639 gcn-mae-score=0.0320-0.9323 gcn-final-mse-score=0.0315-0.9350(0.0472/0.9350) loss=0.1322(0.0836+0.0485)
2020-08-05 08:00:10 E:17, Test  sod-mae-score=0.0605-0.8386 gcn-mae-score=0.0658-0.7769 gcn-final-mse-score=0.0658-0.7830(0.0794/0.7830) loss=0.3529(0.1846+0.1683)

2020-08-05 08:00:10 Start Epoch 18
2020-08-05 08:00:10 Epoch:18,lr=0.0001
2020-08-05 08:00:11    0-10553 loss=0.0511(0.0378+0.0133)-0.0511(0.0378+0.0133) sod-mse=0.0067(0.0067) gcn-mse=0.0128(0.0128) gcn-final-mse=0.0127(0.0226)
2020-08-05 08:05:45 1000-10553 loss=0.0377(0.0295+0.0082)-0.1282(0.0822+0.0461) sod-mse=0.0041(0.0243) gcn-mse=0.0046(0.0311) gcn-final-mse=0.0305(0.0463)
2020-08-05 08:11:17 2000-10553 loss=0.0946(0.0668+0.0279)-0.1250(0.0801+0.0449) sod-mse=0.0152(0.0236) gcn-mse=0.0207(0.0300) gcn-final-mse=0.0294(0.0449)
2020-08-05 08:16:48 3000-10553 loss=0.0971(0.0685+0.0286)-0.1237(0.0794+0.0443) sod-mse=0.0200(0.0232) gcn-mse=0.0265(0.0295) gcn-final-mse=0.0289(0.0446)
2020-08-05 08:18:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 08:22:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 08:22:19 4000-10553 loss=0.1187(0.0920+0.0267)-0.1239(0.0794+0.0445) sod-mse=0.0176(0.0233) gcn-mse=0.0318(0.0294) gcn-final-mse=0.0288(0.0445)
2020-08-05 08:27:52 5000-10553 loss=0.0712(0.0541+0.0171)-0.1264(0.0805+0.0458) sod-mse=0.0087(0.0240) gcn-mse=0.0205(0.0301) gcn-final-mse=0.0295(0.0452)
2020-08-05 08:32:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 08:33:24 6000-10553 loss=0.1095(0.0758+0.0337)-0.1257(0.0803+0.0453) sod-mse=0.0200(0.0238) gcn-mse=0.0193(0.0300) gcn-final-mse=0.0294(0.0452)
2020-08-05 08:38:55 7000-10553 loss=0.1510(0.0968+0.0542)-0.1261(0.0806+0.0455) sod-mse=0.0354(0.0239) gcn-mse=0.0369(0.0302) gcn-final-mse=0.0296(0.0454)
2020-08-05 08:44:26 8000-10553 loss=0.1242(0.0603+0.0639)-0.1263(0.0805+0.0458) sod-mse=0.0484(0.0240) gcn-mse=0.0221(0.0302) gcn-final-mse=0.0296(0.0454)
2020-08-05 08:50:03 9000-10553 loss=0.1242(0.0959+0.0283)-0.1270(0.0808+0.0462) sod-mse=0.0136(0.0242) gcn-mse=0.0227(0.0303) gcn-final-mse=0.0297(0.0455)
2020-08-05 08:55:37 10000-10553 loss=0.1835(0.1159+0.0676)-0.1275(0.0811+0.0464) sod-mse=0.0435(0.0243) gcn-mse=0.0405(0.0305) gcn-final-mse=0.0299(0.0457)

2020-08-05 08:58:40    0-5019 loss=0.7155(0.3866+0.3289)-0.7155(0.3866+0.3289) sod-mse=0.1005(0.1005) gcn-mse=0.1006(0.1006) gcn-final-mse=0.0937(0.1073)
2020-08-05 09:01:06 1000-5019 loss=0.0359(0.0291+0.0067)-0.3463(0.1784+0.1678) sod-mse=0.0057(0.0627) gcn-mse=0.0107(0.0664) gcn-final-mse=0.0663(0.0797)
2020-08-05 09:03:30 2000-5019 loss=0.6687(0.3280+0.3407)-0.3577(0.1845+0.1732) sod-mse=0.0946(0.0644) gcn-mse=0.0867(0.0684) gcn-final-mse=0.0683(0.0817)
2020-08-05 09:05:54 3000-5019 loss=0.0457(0.0347+0.0111)-0.3593(0.1854+0.1738) sod-mse=0.0056(0.0647) gcn-mse=0.0079(0.0691) gcn-final-mse=0.0691(0.0823)
2020-08-05 09:07:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 09:08:19 4000-5019 loss=0.1157(0.0790+0.0367)-0.3583(0.1851+0.1732) sod-mse=0.0193(0.0647) gcn-mse=0.0231(0.0692) gcn-final-mse=0.0692(0.0824)
2020-08-05 09:09:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 09:10:42 5000-5019 loss=1.3167(0.5788+0.7379)-0.3557(0.1842+0.1716) sod-mse=0.0958(0.0647) gcn-mse=0.1015(0.0692) gcn-final-mse=0.0691(0.0823)
2020-08-05 09:10:45 E:18, Train sod-mae-score=0.0245-0.9658 gcn-mae-score=0.0306-0.9344 gcn-final-mse-score=0.0300-0.9370(0.0458/0.9370) loss=0.1279(0.0814+0.0465)
2020-08-05 09:10:45 E:18, Test  sod-mae-score=0.0647-0.8232 gcn-mae-score=0.0692-0.7661 gcn-final-mse-score=0.0691-0.7719(0.0823/0.7719) loss=0.3554(0.1840+0.1714)

2020-08-05 09:10:45 Start Epoch 19
2020-08-05 09:10:45 Epoch:19,lr=0.0001
2020-08-05 09:10:46    0-10553 loss=0.1330(0.0830+0.0500)-0.1330(0.0830+0.0500) sod-mse=0.0244(0.0244) gcn-mse=0.0234(0.0234) gcn-final-mse=0.0259(0.0462)
2020-08-05 09:12:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 09:16:17 1000-10553 loss=0.0691(0.0471+0.0220)-0.1186(0.0772+0.0414) sod-mse=0.0131(0.0217) gcn-mse=0.0142(0.0284) gcn-final-mse=0.0277(0.0435)
2020-08-05 09:21:50 2000-10553 loss=0.1078(0.0702+0.0376)-0.1171(0.0761+0.0410) sod-mse=0.0202(0.0212) gcn-mse=0.0300(0.0276) gcn-final-mse=0.0269(0.0427)
2020-08-05 09:27:23 3000-10553 loss=0.0824(0.0606+0.0218)-0.1208(0.0777+0.0431) sod-mse=0.0154(0.0225) gcn-mse=0.0305(0.0287) gcn-final-mse=0.0281(0.0437)
2020-08-05 09:32:54 4000-10553 loss=0.0784(0.0637+0.0147)-0.1202(0.0775+0.0427) sod-mse=0.0089(0.0223) gcn-mse=0.0296(0.0286) gcn-final-mse=0.0280(0.0436)
2020-08-05 09:36:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 09:38:26 5000-10553 loss=0.0387(0.0284+0.0103)-0.1236(0.0792+0.0444) sod-mse=0.0059(0.0231) gcn-mse=0.0081(0.0292) gcn-final-mse=0.0286(0.0444)
2020-08-05 09:43:55 6000-10553 loss=0.2124(0.1190+0.0934)-0.1249(0.0799+0.0450) sod-mse=0.0499(0.0235) gcn-mse=0.0476(0.0297) gcn-final-mse=0.0291(0.0448)
2020-08-05 09:49:26 7000-10553 loss=0.0882(0.0585+0.0298)-0.1238(0.0794+0.0445) sod-mse=0.0156(0.0232) gcn-mse=0.0310(0.0294) gcn-final-mse=0.0288(0.0446)
2020-08-05 09:54:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 09:54:57 8000-10553 loss=0.0629(0.0433+0.0196)-0.1239(0.0794+0.0445) sod-mse=0.0116(0.0233) gcn-mse=0.0139(0.0294) gcn-final-mse=0.0288(0.0446)
2020-08-05 10:00:28 9000-10553 loss=0.1581(0.1145+0.0436)-0.1242(0.0795+0.0447) sod-mse=0.0324(0.0234) gcn-mse=0.0660(0.0294) gcn-final-mse=0.0288(0.0447)
2020-08-05 10:06:00 10000-10553 loss=0.1062(0.0732+0.0330)-0.1248(0.0798+0.0450) sod-mse=0.0187(0.0235) gcn-mse=0.0218(0.0296) gcn-final-mse=0.0290(0.0448)

2020-08-05 10:09:06    0-5019 loss=0.9498(0.5472+0.4026)-0.9498(0.5472+0.4026) sod-mse=0.1088(0.1088) gcn-mse=0.1225(0.1225) gcn-final-mse=0.1148(0.1284)
2020-08-05 10:11:31 1000-5019 loss=0.0377(0.0309+0.0068)-0.3672(0.1886+0.1785) sod-mse=0.0057(0.0642) gcn-mse=0.0125(0.0695) gcn-final-mse=0.0693(0.0832)
2020-08-05 10:13:55 2000-5019 loss=0.8047(0.3922+0.4124)-0.3814(0.1943+0.1871) sod-mse=0.1030(0.0670) gcn-mse=0.1001(0.0721) gcn-final-mse=0.0719(0.0857)
2020-08-05 10:16:21 3000-5019 loss=0.0469(0.0348+0.0121)-0.3816(0.1945+0.1872) sod-mse=0.0060(0.0672) gcn-mse=0.0075(0.0724) gcn-final-mse=0.0723(0.0861)
2020-08-05 10:17:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 10:18:45 4000-5019 loss=0.1337(0.0864+0.0474)-0.3773(0.1925+0.1847) sod-mse=0.0296(0.0666) gcn-mse=0.0310(0.0720) gcn-final-mse=0.0719(0.0856)
2020-08-05 10:19:27 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 10:21:08 5000-5019 loss=1.0783(0.4397+0.6386)-0.3781(0.1929+0.1851) sod-mse=0.0884(0.0668) gcn-mse=0.0943(0.0723) gcn-final-mse=0.0720(0.0858)
2020-08-05 10:21:11 E:19, Train sod-mae-score=0.0237-0.9667 gcn-mae-score=0.0298-0.9353 gcn-final-mse-score=0.0292-0.9380(0.0450/0.9380) loss=0.1253(0.0801+0.0453)
2020-08-05 10:21:11 E:19, Test  sod-mae-score=0.0668-0.8145 gcn-mae-score=0.0722-0.7587 gcn-final-mse-score=0.0720-0.7649(0.0858/0.7649) loss=0.3776(0.1928+0.1849)

2020-08-05 10:21:11 Start Epoch 20
2020-08-05 10:21:11 Epoch:20,lr=0.0000
2020-08-05 10:21:12    0-10553 loss=0.1059(0.0797+0.0263)-0.1059(0.0797+0.0263) sod-mse=0.0164(0.0164) gcn-mse=0.0337(0.0337) gcn-final-mse=0.0343(0.0467)
2020-08-05 10:26:43 1000-10553 loss=0.0613(0.0473+0.0140)-0.1125(0.0737+0.0388) sod-mse=0.0060(0.0204) gcn-mse=0.0120(0.0270) gcn-final-mse=0.0264(0.0425)
2020-08-05 10:32:12 2000-10553 loss=0.1010(0.0740+0.0270)-0.1095(0.0722+0.0373) sod-mse=0.0152(0.0195) gcn-mse=0.0209(0.0259) gcn-final-mse=0.0253(0.0414)
2020-08-05 10:34:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 10:37:43 3000-10553 loss=0.0913(0.0674+0.0239)-0.1076(0.0712+0.0364) sod-mse=0.0146(0.0190) gcn-mse=0.0150(0.0252) gcn-final-mse=0.0246(0.0406)
2020-08-05 10:43:16 4000-10553 loss=0.0656(0.0422+0.0234)-0.1062(0.0705+0.0358) sod-mse=0.0099(0.0186) gcn-mse=0.0098(0.0247) gcn-final-mse=0.0241(0.0401)
2020-08-05 10:48:47 5000-10553 loss=0.0634(0.0459+0.0175)-0.1057(0.0702+0.0355) sod-mse=0.0092(0.0185) gcn-mse=0.0121(0.0244) gcn-final-mse=0.0237(0.0398)
2020-08-05 10:54:18 6000-10553 loss=0.0809(0.0556+0.0253)-0.1051(0.0698+0.0352) sod-mse=0.0152(0.0183) gcn-mse=0.0198(0.0241) gcn-final-mse=0.0235(0.0395)
2020-08-05 10:58:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 10:59:49 7000-10553 loss=0.0764(0.0588+0.0176)-0.1041(0.0694+0.0347) sod-mse=0.0088(0.0181) gcn-mse=0.0153(0.0238) gcn-final-mse=0.0232(0.0393)
2020-08-05 11:05:20 8000-10553 loss=0.3191(0.1588+0.1604)-0.1034(0.0689+0.0344) sod-mse=0.0540(0.0178) gcn-mse=0.0542(0.0235) gcn-final-mse=0.0228(0.0390)
2020-08-05 11:10:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 11:10:52 9000-10553 loss=0.0376(0.0236+0.0140)-0.1029(0.0687+0.0342) sod-mse=0.0089(0.0177) gcn-mse=0.0072(0.0233) gcn-final-mse=0.0226(0.0388)
2020-08-05 11:16:23 10000-10553 loss=0.0514(0.0387+0.0127)-0.1027(0.0686+0.0342) sod-mse=0.0075(0.0177) gcn-mse=0.0101(0.0232) gcn-final-mse=0.0225(0.0387)

2020-08-05 11:19:28    0-5019 loss=0.9073(0.4885+0.4187)-0.9073(0.4885+0.4187) sod-mse=0.0991(0.0991) gcn-mse=0.1064(0.1064) gcn-final-mse=0.0985(0.1110)
2020-08-05 11:21:53 1000-5019 loss=0.0310(0.0256+0.0054)-0.3290(0.1683+0.1606) sod-mse=0.0044(0.0540) gcn-mse=0.0075(0.0579) gcn-final-mse=0.0577(0.0714)
2020-08-05 11:24:17 2000-5019 loss=0.7830(0.3895+0.3935)-0.3378(0.1726+0.1652) sod-mse=0.0976(0.0562) gcn-mse=0.0935(0.0603) gcn-final-mse=0.0602(0.0738)
2020-08-05 11:26:40 3000-5019 loss=0.0450(0.0338+0.0112)-0.3371(0.1721+0.1650) sod-mse=0.0055(0.0561) gcn-mse=0.0073(0.0603) gcn-final-mse=0.0602(0.0738)
2020-08-05 11:28:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 11:29:04 4000-5019 loss=0.1097(0.0751+0.0346)-0.3335(0.1705+0.1630) sod-mse=0.0182(0.0557) gcn-mse=0.0208(0.0600) gcn-final-mse=0.0599(0.0735)
2020-08-05 11:29:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 11:31:27 5000-5019 loss=1.2818(0.5361+0.7457)-0.3330(0.1705+0.1625) sod-mse=0.0978(0.0558) gcn-mse=0.1096(0.0602) gcn-final-mse=0.0600(0.0736)
2020-08-05 11:31:29 E:20, Train sod-mae-score=0.0176-0.9741 gcn-mae-score=0.0231-0.9439 gcn-final-mse-score=0.0224-0.9465(0.0386/0.9465) loss=0.1024(0.0684+0.0340)
2020-08-05 11:31:29 E:20, Test  sod-mae-score=0.0558-0.8430 gcn-mae-score=0.0602-0.7838 gcn-final-mse-score=0.0600-0.7895(0.0735/0.7895) loss=0.3328(0.1704+0.1624)

2020-08-05 11:31:29 Start Epoch 21
2020-08-05 11:31:29 Epoch:21,lr=0.0000
2020-08-05 11:31:31    0-10553 loss=0.0449(0.0353+0.0096)-0.0449(0.0353+0.0096) sod-mse=0.0049(0.0049) gcn-mse=0.0072(0.0072) gcn-final-mse=0.0061(0.0195)
2020-08-05 11:32:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 11:37:00 1000-10553 loss=0.0807(0.0546+0.0261)-0.0911(0.0626+0.0286) sod-mse=0.0142(0.0146) gcn-mse=0.0175(0.0195) gcn-final-mse=0.0189(0.0352)
2020-08-05 11:42:33 2000-10553 loss=0.0215(0.0160+0.0054)-0.0924(0.0632+0.0292) sod-mse=0.0037(0.0149) gcn-mse=0.0060(0.0197) gcn-final-mse=0.0191(0.0355)
2020-08-05 11:48:03 3000-10553 loss=0.0724(0.0510+0.0214)-0.0918(0.0627+0.0291) sod-mse=0.0114(0.0148) gcn-mse=0.0122(0.0195) gcn-final-mse=0.0188(0.0353)
2020-08-05 11:53:35 4000-10553 loss=0.1456(0.1065+0.0390)-0.0919(0.0627+0.0291) sod-mse=0.0200(0.0148) gcn-mse=0.0387(0.0196) gcn-final-mse=0.0189(0.0353)
2020-08-05 11:59:08 5000-10553 loss=0.1025(0.0634+0.0391)-0.0921(0.0629+0.0292) sod-mse=0.0193(0.0149) gcn-mse=0.0231(0.0196) gcn-final-mse=0.0189(0.0353)
2020-08-05 12:00:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 12:04:41 6000-10553 loss=0.0310(0.0221+0.0089)-0.0926(0.0631+0.0294) sod-mse=0.0061(0.0149) gcn-mse=0.0054(0.0197) gcn-final-mse=0.0190(0.0354)
2020-08-05 12:10:12 7000-10553 loss=0.0613(0.0431+0.0182)-0.0928(0.0632+0.0296) sod-mse=0.0082(0.0150) gcn-mse=0.0112(0.0197) gcn-final-mse=0.0190(0.0355)
2020-08-05 12:15:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 12:15:42 8000-10553 loss=0.0388(0.0322+0.0066)-0.0928(0.0632+0.0296) sod-mse=0.0025(0.0151) gcn-mse=0.0061(0.0197) gcn-final-mse=0.0190(0.0354)
2020-08-05 12:21:13 9000-10553 loss=0.1094(0.0769+0.0325)-0.0935(0.0635+0.0300) sod-mse=0.0185(0.0152) gcn-mse=0.0236(0.0198) gcn-final-mse=0.0191(0.0355)
2020-08-05 12:26:44 10000-10553 loss=0.0616(0.0395+0.0220)-0.0938(0.0636+0.0301) sod-mse=0.0119(0.0153) gcn-mse=0.0096(0.0198) gcn-final-mse=0.0191(0.0355)

2020-08-05 12:29:47    0-5019 loss=0.9473(0.5401+0.4071)-0.9473(0.5401+0.4071) sod-mse=0.0969(0.0969) gcn-mse=0.1066(0.1066) gcn-final-mse=0.0987(0.1110)
2020-08-05 12:32:12 1000-5019 loss=0.0304(0.0253+0.0052)-0.3458(0.1781+0.1678) sod-mse=0.0043(0.0555) gcn-mse=0.0072(0.0588) gcn-final-mse=0.0587(0.0723)
2020-08-05 12:34:36 2000-5019 loss=0.7397(0.3815+0.3582)-0.3557(0.1826+0.1731) sod-mse=0.0917(0.0575) gcn-mse=0.0906(0.0610) gcn-final-mse=0.0609(0.0744)
2020-08-05 12:37:00 3000-5019 loss=0.0446(0.0336+0.0110)-0.3550(0.1821+0.1729) sod-mse=0.0053(0.0576) gcn-mse=0.0067(0.0612) gcn-final-mse=0.0610(0.0746)
2020-08-05 12:38:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 12:39:24 4000-5019 loss=0.1029(0.0716+0.0313)-0.3507(0.1802+0.1705) sod-mse=0.0161(0.0571) gcn-mse=0.0181(0.0609) gcn-final-mse=0.0607(0.0742)
2020-08-05 12:40:06 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 12:41:47 5000-5019 loss=1.3633(0.5800+0.7833)-0.3489(0.1796+0.1693) sod-mse=0.1073(0.0571) gcn-mse=0.1136(0.0609) gcn-final-mse=0.0606(0.0741)
2020-08-05 12:41:49 E:21, Train sod-mae-score=0.0153-0.9772 gcn-mae-score=0.0199-0.9475 gcn-final-mse-score=0.0192-0.9502(0.0356/0.9502) loss=0.0939(0.0637+0.0302)
2020-08-05 12:41:49 E:21, Test  sod-mae-score=0.0571-0.8413 gcn-mae-score=0.0609-0.7819 gcn-final-mse-score=0.0606-0.7876(0.0741/0.7876) loss=0.3486(0.1795+0.1692)

2020-08-05 12:41:49 Start Epoch 22
2020-08-05 12:41:49 Epoch:22,lr=0.0000
2020-08-05 12:41:51    0-10553 loss=0.1763(0.1102+0.0661)-0.1763(0.1102+0.0661) sod-mse=0.0316(0.0316) gcn-mse=0.0530(0.0530) gcn-final-mse=0.0488(0.0545)
2020-08-05 12:43:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 12:47:22 1000-10553 loss=0.0838(0.0651+0.0187)-0.0883(0.0607+0.0276) sod-mse=0.0136(0.0144) gcn-mse=0.0176(0.0188) gcn-final-mse=0.0181(0.0346)
2020-08-05 12:52:55 2000-10553 loss=0.1563(0.1089+0.0474)-0.0900(0.0617+0.0283) sod-mse=0.0217(0.0145) gcn-mse=0.0232(0.0189) gcn-final-mse=0.0182(0.0347)
2020-08-05 12:53:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 12:58:31 3000-10553 loss=0.2466(0.1617+0.0849)-0.0897(0.0616+0.0281) sod-mse=0.0557(0.0144) gcn-mse=0.0646(0.0188) gcn-final-mse=0.0181(0.0347)
2020-08-05 13:04:04 4000-10553 loss=0.0901(0.0634+0.0267)-0.0894(0.0614+0.0279) sod-mse=0.0104(0.0143) gcn-mse=0.0127(0.0187) gcn-final-mse=0.0180(0.0345)
2020-08-05 13:09:35 5000-10553 loss=0.0624(0.0458+0.0166)-0.0895(0.0615+0.0281) sod-mse=0.0091(0.0144) gcn-mse=0.0097(0.0187) gcn-final-mse=0.0180(0.0345)
2020-08-05 13:15:07 6000-10553 loss=0.0525(0.0392+0.0133)-0.0893(0.0614+0.0279) sod-mse=0.0074(0.0143) gcn-mse=0.0124(0.0187) gcn-final-mse=0.0179(0.0344)
2020-08-05 13:20:38 7000-10553 loss=0.0597(0.0430+0.0167)-0.0898(0.0616+0.0282) sod-mse=0.0051(0.0144) gcn-mse=0.0055(0.0187) gcn-final-mse=0.0180(0.0345)
2020-08-05 13:21:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 13:26:11 8000-10553 loss=0.0651(0.0431+0.0220)-0.0897(0.0615+0.0282) sod-mse=0.0108(0.0144) gcn-mse=0.0123(0.0186) gcn-final-mse=0.0179(0.0345)
2020-08-05 13:31:41 9000-10553 loss=0.0863(0.0608+0.0255)-0.0899(0.0616+0.0283) sod-mse=0.0158(0.0144) gcn-mse=0.0228(0.0187) gcn-final-mse=0.0179(0.0345)
2020-08-05 13:37:13 10000-10553 loss=0.0637(0.0467+0.0170)-0.0901(0.0617+0.0284) sod-mse=0.0098(0.0144) gcn-mse=0.0123(0.0187) gcn-final-mse=0.0180(0.0346)

2020-08-05 13:40:17    0-5019 loss=0.9870(0.5639+0.4231)-0.9870(0.5639+0.4231) sod-mse=0.0954(0.0954) gcn-mse=0.1047(0.1047) gcn-final-mse=0.0965(0.1075)
2020-08-05 13:42:44 1000-5019 loss=0.0299(0.0249+0.0049)-0.3520(0.1768+0.1752) sod-mse=0.0040(0.0535) gcn-mse=0.0067(0.0574) gcn-final-mse=0.0572(0.0708)
2020-08-05 13:45:08 2000-5019 loss=0.6887(0.3580+0.3307)-0.3631(0.1812+0.1818) sod-mse=0.0862(0.0554) gcn-mse=0.0866(0.0595) gcn-final-mse=0.0593(0.0728)
2020-08-05 13:47:32 3000-5019 loss=0.0445(0.0335+0.0109)-0.3590(0.1795+0.1796) sod-mse=0.0052(0.0551) gcn-mse=0.0068(0.0593) gcn-final-mse=0.0591(0.0726)
2020-08-05 13:48:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 13:49:56 4000-5019 loss=0.1007(0.0704+0.0304)-0.3545(0.1776+0.1769) sod-mse=0.0150(0.0546) gcn-mse=0.0166(0.0589) gcn-final-mse=0.0588(0.0722)
2020-08-05 13:50:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 13:52:20 5000-5019 loss=1.3899(0.5749+0.8150)-0.3534(0.1771+0.1762) sod-mse=0.1142(0.0546) gcn-mse=0.1173(0.0590) gcn-final-mse=0.0588(0.0722)
2020-08-05 13:52:23 E:22, Train sod-mae-score=0.0144-0.9782 gcn-mae-score=0.0187-0.9486 gcn-final-mse-score=0.0180-0.9511(0.0345/0.9511) loss=0.0900(0.0617+0.0283)
2020-08-05 13:52:23 E:22, Test  sod-mae-score=0.0546-0.8447 gcn-mae-score=0.0590-0.7864 gcn-final-mse-score=0.0588-0.7920(0.0722/0.7920) loss=0.3532(0.1771+0.1761)

2020-08-05 13:52:23 Start Epoch 23
2020-08-05 13:52:23 Epoch:23,lr=0.0000
2020-08-05 13:52:24    0-10553 loss=0.1536(0.0972+0.0564)-0.1536(0.0972+0.0564) sod-mse=0.0303(0.0303) gcn-mse=0.0388(0.0388) gcn-final-mse=0.0350(0.0533)
2020-08-05 13:57:55 1000-10553 loss=0.0567(0.0390+0.0177)-0.0868(0.0602+0.0267) sod-mse=0.0065(0.0133) gcn-mse=0.0063(0.0177) gcn-final-mse=0.0170(0.0335)
2020-08-05 14:00:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 14:03:25 2000-10553 loss=0.0601(0.0447+0.0154)-0.0900(0.0619+0.0281) sod-mse=0.0083(0.0140) gcn-mse=0.0085(0.0183) gcn-final-mse=0.0176(0.0343)
2020-08-05 14:08:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 14:08:55 3000-10553 loss=0.0415(0.0333+0.0081)-0.0893(0.0614+0.0279) sod-mse=0.0037(0.0140) gcn-mse=0.0059(0.0182) gcn-final-mse=0.0175(0.0342)
2020-08-05 14:12:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 14:14:25 4000-10553 loss=0.1500(0.0914+0.0587)-0.0892(0.0615+0.0278) sod-mse=0.0366(0.0140) gcn-mse=0.0341(0.0183) gcn-final-mse=0.0175(0.0343)
2020-08-05 14:19:57 5000-10553 loss=0.0416(0.0302+0.0113)-0.0891(0.0613+0.0278) sod-mse=0.0049(0.0140) gcn-mse=0.0052(0.0182) gcn-final-mse=0.0174(0.0342)
2020-08-05 14:25:29 6000-10553 loss=0.0832(0.0525+0.0307)-0.0884(0.0609+0.0275) sod-mse=0.0132(0.0139) gcn-mse=0.0117(0.0181) gcn-final-mse=0.0173(0.0340)
2020-08-05 14:31:01 7000-10553 loss=0.0855(0.0551+0.0303)-0.0880(0.0607+0.0273) sod-mse=0.0086(0.0138) gcn-mse=0.0107(0.0180) gcn-final-mse=0.0172(0.0339)
2020-08-05 14:36:31 8000-10553 loss=0.0544(0.0417+0.0127)-0.0883(0.0608+0.0275) sod-mse=0.0088(0.0138) gcn-mse=0.0117(0.0180) gcn-final-mse=0.0173(0.0340)
2020-08-05 14:42:04 9000-10553 loss=0.0698(0.0496+0.0202)-0.0883(0.0608+0.0275) sod-mse=0.0117(0.0138) gcn-mse=0.0160(0.0180) gcn-final-mse=0.0172(0.0339)
2020-08-05 14:47:36 10000-10553 loss=0.0575(0.0443+0.0132)-0.0880(0.0606+0.0274) sod-mse=0.0076(0.0138) gcn-mse=0.0100(0.0179) gcn-final-mse=0.0172(0.0338)

2020-08-05 14:50:43    0-5019 loss=0.9914(0.5846+0.4068)-0.9914(0.5846+0.4068) sod-mse=0.0967(0.0967) gcn-mse=0.1050(0.1050) gcn-final-mse=0.0969(0.1093)
2020-08-05 14:53:09 1000-5019 loss=0.0293(0.0241+0.0051)-0.3457(0.1779+0.1678) sod-mse=0.0042(0.0526) gcn-mse=0.0063(0.0559) gcn-final-mse=0.0558(0.0692)
2020-08-05 14:55:33 2000-5019 loss=0.8637(0.4345+0.4292)-0.3553(0.1818+0.1735) sod-mse=0.0969(0.0544) gcn-mse=0.0927(0.0577) gcn-final-mse=0.0576(0.0710)
2020-08-05 14:57:57 3000-5019 loss=0.0440(0.0331+0.0109)-0.3525(0.1802+0.1724) sod-mse=0.0053(0.0541) gcn-mse=0.0065(0.0576) gcn-final-mse=0.0574(0.0708)
2020-08-05 14:59:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 15:00:22 4000-5019 loss=0.1007(0.0703+0.0304)-0.3490(0.1786+0.1704) sod-mse=0.0154(0.0538) gcn-mse=0.0163(0.0574) gcn-final-mse=0.0572(0.0706)
2020-08-05 15:01:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 15:02:45 5000-5019 loss=1.3840(0.6159+0.7681)-0.3472(0.1779+0.1693) sod-mse=0.0971(0.0538) gcn-mse=0.1061(0.0574) gcn-final-mse=0.0571(0.0705)
2020-08-05 15:02:48 E:23, Train sod-mae-score=0.0137-0.9794 gcn-mae-score=0.0179-0.9497 gcn-final-mse-score=0.0171-0.9522(0.0337/0.9522) loss=0.0877(0.0604+0.0273)
2020-08-05 15:02:48 E:23, Test  sod-mae-score=0.0537-0.8493 gcn-mae-score=0.0574-0.7893 gcn-final-mse-score=0.0571-0.7951(0.0705/0.7951) loss=0.3470(0.1778+0.1691)

2020-08-05 15:02:48 Start Epoch 24
2020-08-05 15:02:48 Epoch:24,lr=0.0000
2020-08-05 15:02:49    0-10553 loss=0.0964(0.0724+0.0239)-0.0964(0.0724+0.0239) sod-mse=0.0136(0.0136) gcn-mse=0.0232(0.0232) gcn-final-mse=0.0232(0.0438)
2020-08-05 15:08:22 1000-10553 loss=0.0907(0.0605+0.0302)-0.0886(0.0613+0.0272) sod-mse=0.0164(0.0139) gcn-mse=0.0169(0.0181) gcn-final-mse=0.0174(0.0343)
2020-08-05 15:13:53 2000-10553 loss=0.0493(0.0363+0.0130)-0.0859(0.0599+0.0260) sod-mse=0.0076(0.0132) gcn-mse=0.0088(0.0176) gcn-final-mse=0.0168(0.0335)
2020-08-05 15:19:26 3000-10553 loss=0.0690(0.0554+0.0136)-0.0860(0.0598+0.0262) sod-mse=0.0091(0.0133) gcn-mse=0.0223(0.0176) gcn-final-mse=0.0168(0.0335)
2020-08-05 15:24:56 4000-10553 loss=0.0593(0.0399+0.0194)-0.0854(0.0594+0.0260) sod-mse=0.0097(0.0132) gcn-mse=0.0111(0.0174) gcn-final-mse=0.0166(0.0332)
2020-08-05 15:30:27 5000-10553 loss=0.1238(0.0807+0.0431)-0.0857(0.0595+0.0262) sod-mse=0.0204(0.0132) gcn-mse=0.0253(0.0174) gcn-final-mse=0.0166(0.0332)
2020-08-05 15:32:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 15:35:58 6000-10553 loss=0.0342(0.0291+0.0051)-0.0853(0.0593+0.0261) sod-mse=0.0029(0.0132) gcn-mse=0.0123(0.0173) gcn-final-mse=0.0165(0.0331)
2020-08-05 15:41:28 7000-10553 loss=0.0211(0.0156+0.0055)-0.0855(0.0593+0.0261) sod-mse=0.0031(0.0132) gcn-mse=0.0026(0.0173) gcn-final-mse=0.0165(0.0332)
2020-08-05 15:44:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 15:47:01 8000-10553 loss=0.0667(0.0442+0.0224)-0.0852(0.0592+0.0260) sod-mse=0.0107(0.0132) gcn-mse=0.0135(0.0172) gcn-final-mse=0.0164(0.0331)
2020-08-05 15:52:32 9000-10553 loss=0.1372(0.0948+0.0424)-0.0853(0.0592+0.0261) sod-mse=0.0215(0.0132) gcn-mse=0.0264(0.0172) gcn-final-mse=0.0164(0.0331)
2020-08-05 15:55:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 15:58:02 10000-10553 loss=0.2064(0.1326+0.0738)-0.0859(0.0595+0.0264) sod-mse=0.0339(0.0133) gcn-mse=0.0417(0.0173) gcn-final-mse=0.0166(0.0332)

2020-08-05 16:01:08    0-5019 loss=1.0204(0.6033+0.4171)-1.0204(0.6033+0.4171) sod-mse=0.0973(0.0973) gcn-mse=0.1044(0.1044) gcn-final-mse=0.0961(0.1083)
2020-08-05 16:03:34 1000-5019 loss=0.0290(0.0241+0.0049)-0.3529(0.1804+0.1725) sod-mse=0.0041(0.0528) gcn-mse=0.0062(0.0563) gcn-final-mse=0.0561(0.0697)
2020-08-05 16:05:58 2000-5019 loss=0.7692(0.4007+0.3685)-0.3636(0.1847+0.1789) sod-mse=0.0848(0.0544) gcn-mse=0.0881(0.0581) gcn-final-mse=0.0578(0.0714)
2020-08-05 16:08:22 3000-5019 loss=0.0439(0.0331+0.0107)-0.3602(0.1829+0.1773) sod-mse=0.0050(0.0541) gcn-mse=0.0067(0.0578) gcn-final-mse=0.0577(0.0712)
2020-08-05 16:09:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 16:10:47 4000-5019 loss=0.0993(0.0699+0.0295)-0.3560(0.1811+0.1749) sod-mse=0.0147(0.0538) gcn-mse=0.0162(0.0576) gcn-final-mse=0.0574(0.0710)
2020-08-05 16:11:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 16:13:11 5000-5019 loss=1.3675(0.5877+0.7797)-0.3540(0.1803+0.1738) sod-mse=0.1012(0.0537) gcn-mse=0.1087(0.0576) gcn-final-mse=0.0574(0.0709)
2020-08-05 16:13:13 E:24, Train sod-mae-score=0.0133-0.9799 gcn-mae-score=0.0173-0.9507 gcn-final-mse-score=0.0165-0.9532(0.0332/0.9532) loss=0.0859(0.0595+0.0264)
2020-08-05 16:13:13 E:24, Test  sod-mae-score=0.0537-0.8478 gcn-mae-score=0.0576-0.7866 gcn-final-mse-score=0.0573-0.7922(0.0709/0.7922) loss=0.3538(0.1802+0.1736)

2020-08-05 16:13:13 Start Epoch 25
2020-08-05 16:13:13 Epoch:25,lr=0.0000
2020-08-05 16:13:14    0-10553 loss=0.0540(0.0415+0.0125)-0.0540(0.0415+0.0125) sod-mse=0.0064(0.0064) gcn-mse=0.0125(0.0125) gcn-final-mse=0.0105(0.0204)
2020-08-05 16:18:46 1000-10553 loss=0.1250(0.0732+0.0518)-0.0827(0.0576+0.0251) sod-mse=0.0193(0.0128) gcn-mse=0.0194(0.0166) gcn-final-mse=0.0159(0.0326)
2020-08-05 16:24:19 2000-10553 loss=0.0437(0.0313+0.0124)-0.0831(0.0581+0.0250) sod-mse=0.0070(0.0127) gcn-mse=0.0108(0.0166) gcn-final-mse=0.0159(0.0327)
2020-08-05 16:26:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 16:29:50 3000-10553 loss=0.0288(0.0238+0.0050)-0.0826(0.0577+0.0248) sod-mse=0.0036(0.0126) gcn-mse=0.0033(0.0165) gcn-final-mse=0.0158(0.0325)
2020-08-05 16:35:19 4000-10553 loss=0.0998(0.0762+0.0236)-0.0834(0.0582+0.0252) sod-mse=0.0114(0.0127) gcn-mse=0.0207(0.0165) gcn-final-mse=0.0158(0.0326)
2020-08-05 16:39:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 16:40:52 5000-10553 loss=0.0161(0.0118+0.0043)-0.0835(0.0582+0.0253) sod-mse=0.0026(0.0127) gcn-mse=0.0018(0.0166) gcn-final-mse=0.0158(0.0325)
2020-08-05 16:46:23 6000-10553 loss=0.0591(0.0481+0.0110)-0.0837(0.0583+0.0254) sod-mse=0.0052(0.0128) gcn-mse=0.0169(0.0166) gcn-final-mse=0.0159(0.0326)
2020-08-05 16:49:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 16:51:55 7000-10553 loss=0.0509(0.0371+0.0138)-0.0838(0.0584+0.0254) sod-mse=0.0059(0.0128) gcn-mse=0.0063(0.0166) gcn-final-mse=0.0159(0.0327)
2020-08-05 16:57:26 8000-10553 loss=0.0357(0.0286+0.0070)-0.0837(0.0583+0.0254) sod-mse=0.0030(0.0128) gcn-mse=0.0046(0.0166) gcn-final-mse=0.0159(0.0326)
2020-08-05 17:02:58 9000-10553 loss=0.0498(0.0353+0.0145)-0.0837(0.0583+0.0254) sod-mse=0.0087(0.0128) gcn-mse=0.0114(0.0166) gcn-final-mse=0.0158(0.0326)
2020-08-05 17:08:27 10000-10553 loss=0.1580(0.1062+0.0518)-0.0838(0.0584+0.0254) sod-mse=0.0294(0.0128) gcn-mse=0.0444(0.0166) gcn-final-mse=0.0159(0.0326)

2020-08-05 17:11:31    0-5019 loss=0.9862(0.5691+0.4170)-0.9862(0.5691+0.4170) sod-mse=0.0961(0.0961) gcn-mse=0.1037(0.1037) gcn-final-mse=0.0955(0.1074)
2020-08-05 17:13:57 1000-5019 loss=0.0286(0.0237+0.0049)-0.3600(0.1811+0.1789) sod-mse=0.0040(0.0523) gcn-mse=0.0058(0.0553) gcn-final-mse=0.0551(0.0686)
2020-08-05 17:16:22 2000-5019 loss=0.7889(0.4067+0.3821)-0.3695(0.1848+0.1847) sod-mse=0.0882(0.0536) gcn-mse=0.0871(0.0568) gcn-final-mse=0.0566(0.0700)
2020-08-05 17:18:47 3000-5019 loss=0.0440(0.0333+0.0107)-0.3655(0.1828+0.1827) sod-mse=0.0050(0.0534) gcn-mse=0.0068(0.0566) gcn-final-mse=0.0565(0.0698)
2020-08-05 17:20:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 17:21:11 4000-5019 loss=0.1000(0.0696+0.0304)-0.3607(0.1809+0.1798) sod-mse=0.0151(0.0531) gcn-mse=0.0155(0.0564) gcn-final-mse=0.0562(0.0696)
2020-08-05 17:21:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 17:23:34 5000-5019 loss=1.4208(0.5856+0.8352)-0.3591(0.1803+0.1788) sod-mse=0.0963(0.0531) gcn-mse=0.1063(0.0564) gcn-final-mse=0.0562(0.0695)
2020-08-05 17:23:37 E:25, Train sod-mae-score=0.0129-0.9805 gcn-mae-score=0.0167-0.9509 gcn-final-mse-score=0.0160-0.9534(0.0327/0.9534) loss=0.0842(0.0586+0.0257)
2020-08-05 17:23:37 E:25, Test  sod-mae-score=0.0531-0.8488 gcn-mae-score=0.0564-0.7897 gcn-final-mse-score=0.0562-0.7953(0.0695/0.7953) loss=0.3588(0.1802+0.1787)

2020-08-05 17:23:37 Start Epoch 26
2020-08-05 17:23:37 Epoch:26,lr=0.0000
2020-08-05 17:23:38    0-10553 loss=0.0811(0.0621+0.0190)-0.0811(0.0621+0.0190) sod-mse=0.0106(0.0106) gcn-mse=0.0142(0.0142) gcn-final-mse=0.0134(0.0355)
2020-08-05 17:27:18 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 17:29:11 1000-10553 loss=0.0680(0.0517+0.0163)-0.0807(0.0567+0.0240) sod-mse=0.0090(0.0121) gcn-mse=0.0131(0.0159) gcn-final-mse=0.0151(0.0318)
2020-08-05 17:34:43 2000-10553 loss=0.0376(0.0308+0.0068)-0.0812(0.0570+0.0241) sod-mse=0.0035(0.0123) gcn-mse=0.0038(0.0161) gcn-final-mse=0.0153(0.0321)
2020-08-05 17:40:14 3000-10553 loss=0.0614(0.0456+0.0158)-0.0819(0.0575+0.0244) sod-mse=0.0068(0.0123) gcn-mse=0.0091(0.0162) gcn-final-mse=0.0155(0.0323)
2020-08-05 17:45:46 4000-10553 loss=0.0536(0.0387+0.0150)-0.0818(0.0574+0.0244) sod-mse=0.0068(0.0123) gcn-mse=0.0117(0.0162) gcn-final-mse=0.0154(0.0322)
2020-08-05 17:51:16 5000-10553 loss=0.0437(0.0334+0.0103)-0.0820(0.0575+0.0245) sod-mse=0.0072(0.0123) gcn-mse=0.0140(0.0162) gcn-final-mse=0.0155(0.0322)
2020-08-05 17:56:48 6000-10553 loss=0.1256(0.0839+0.0417)-0.0821(0.0576+0.0245) sod-mse=0.0197(0.0123) gcn-mse=0.0183(0.0162) gcn-final-mse=0.0154(0.0322)
2020-08-05 18:02:19 7000-10553 loss=0.0570(0.0427+0.0143)-0.0821(0.0576+0.0245) sod-mse=0.0077(0.0123) gcn-mse=0.0148(0.0161) gcn-final-mse=0.0154(0.0322)
2020-08-05 18:07:50 8000-10553 loss=0.0701(0.0548+0.0154)-0.0820(0.0575+0.0245) sod-mse=0.0107(0.0123) gcn-mse=0.0172(0.0161) gcn-final-mse=0.0154(0.0322)
2020-08-05 18:08:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 18:13:22 9000-10553 loss=0.0882(0.0601+0.0281)-0.0821(0.0576+0.0245) sod-mse=0.0177(0.0123) gcn-mse=0.0120(0.0161) gcn-final-mse=0.0154(0.0322)
2020-08-05 18:13:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 18:18:52 10000-10553 loss=0.0909(0.0613+0.0296)-0.0823(0.0576+0.0247) sod-mse=0.0130(0.0124) gcn-mse=0.0189(0.0162) gcn-final-mse=0.0154(0.0322)

2020-08-05 18:21:56    0-5019 loss=1.0205(0.6325+0.3880)-1.0205(0.6325+0.3880) sod-mse=0.0954(0.0954) gcn-mse=0.1020(0.1020) gcn-final-mse=0.0937(0.1064)
2020-08-05 18:24:21 1000-5019 loss=0.0280(0.0232+0.0048)-0.3646(0.1864+0.1782) sod-mse=0.0039(0.0531) gcn-mse=0.0054(0.0565) gcn-final-mse=0.0563(0.0699)
2020-08-05 18:26:44 2000-5019 loss=0.7499(0.4077+0.3422)-0.3750(0.1901+0.1849) sod-mse=0.0870(0.0546) gcn-mse=0.0881(0.0580) gcn-final-mse=0.0577(0.0712)
2020-08-05 18:29:08 3000-5019 loss=0.0438(0.0330+0.0107)-0.3737(0.1892+0.1845) sod-mse=0.0049(0.0545) gcn-mse=0.0064(0.0580) gcn-final-mse=0.0579(0.0713)
2020-08-05 18:30:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 18:31:32 4000-5019 loss=0.0989(0.0682+0.0306)-0.3696(0.1874+0.1822) sod-mse=0.0151(0.0542) gcn-mse=0.0141(0.0578) gcn-final-mse=0.0576(0.0710)
2020-08-05 18:32:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 18:33:55 5000-5019 loss=1.3190(0.5594+0.7596)-0.3680(0.1866+0.1814) sod-mse=0.1115(0.0542) gcn-mse=0.1146(0.0578) gcn-final-mse=0.0575(0.0710)
2020-08-05 18:33:57 E:26, Train sod-mae-score=0.0124-0.9814 gcn-mae-score=0.0162-0.9517 gcn-final-mse-score=0.0154-0.9542(0.0322/0.9542) loss=0.0824(0.0577+0.0247)
2020-08-05 18:33:57 E:26, Test  sod-mae-score=0.0542-0.8457 gcn-mae-score=0.0578-0.7863 gcn-final-mse-score=0.0575-0.7920(0.0710/0.7920) loss=0.3678(0.1865+0.1813)

2020-08-05 18:33:57 Start Epoch 27
2020-08-05 18:33:57 Epoch:27,lr=0.0000
2020-08-05 18:33:59    0-10553 loss=0.1034(0.0704+0.0329)-0.1034(0.0704+0.0329) sod-mse=0.0154(0.0154) gcn-mse=0.0259(0.0259) gcn-final-mse=0.0226(0.0356)
2020-08-05 18:39:31 1000-10553 loss=0.0326(0.0244+0.0083)-0.0799(0.0564+0.0235) sod-mse=0.0045(0.0117) gcn-mse=0.0059(0.0154) gcn-final-mse=0.0146(0.0316)
2020-08-05 18:41:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 18:45:02 2000-10553 loss=0.0358(0.0276+0.0082)-0.0810(0.0570+0.0241) sod-mse=0.0037(0.0120) gcn-mse=0.0055(0.0156) gcn-final-mse=0.0149(0.0320)
2020-08-05 18:50:35 3000-10553 loss=0.1375(0.0893+0.0482)-0.0810(0.0569+0.0241) sod-mse=0.0209(0.0120) gcn-mse=0.0223(0.0156) gcn-final-mse=0.0149(0.0319)
2020-08-05 18:52:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 18:56:05 4000-10553 loss=0.1090(0.0781+0.0309)-0.0807(0.0568+0.0239) sod-mse=0.0163(0.0120) gcn-mse=0.0233(0.0156) gcn-final-mse=0.0149(0.0318)
2020-08-05 19:01:37 5000-10553 loss=0.0937(0.0624+0.0313)-0.0808(0.0567+0.0241) sod-mse=0.0168(0.0120) gcn-mse=0.0184(0.0157) gcn-final-mse=0.0150(0.0318)
2020-08-05 19:07:12 6000-10553 loss=0.0850(0.0538+0.0312)-0.0807(0.0567+0.0239) sod-mse=0.0152(0.0120) gcn-mse=0.0180(0.0157) gcn-final-mse=0.0149(0.0317)
2020-08-05 19:10:05 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 19:12:43 7000-10553 loss=0.0424(0.0344+0.0079)-0.0810(0.0570+0.0240) sod-mse=0.0034(0.0120) gcn-mse=0.0048(0.0157) gcn-final-mse=0.0150(0.0318)
2020-08-05 19:18:16 8000-10553 loss=0.1810(0.1079+0.0731)-0.0807(0.0568+0.0239) sod-mse=0.0261(0.0120) gcn-mse=0.0313(0.0157) gcn-final-mse=0.0149(0.0317)
2020-08-05 19:23:46 9000-10553 loss=0.6971(0.3485+0.3486)-0.0806(0.0567+0.0238) sod-mse=0.0833(0.0120) gcn-mse=0.0845(0.0156) gcn-final-mse=0.0149(0.0317)
2020-08-05 19:29:17 10000-10553 loss=0.0752(0.0593+0.0159)-0.0808(0.0568+0.0240) sod-mse=0.0122(0.0120) gcn-mse=0.0146(0.0157) gcn-final-mse=0.0149(0.0318)

2020-08-05 19:32:21    0-5019 loss=1.0211(0.6248+0.3963)-1.0211(0.6248+0.3963) sod-mse=0.0970(0.0970) gcn-mse=0.1052(0.1052) gcn-final-mse=0.0968(0.1097)
2020-08-05 19:34:49 1000-5019 loss=0.0280(0.0231+0.0050)-0.3708(0.1870+0.1838) sod-mse=0.0040(0.0530) gcn-mse=0.0052(0.0558) gcn-final-mse=0.0556(0.0692)
2020-08-05 19:37:14 2000-5019 loss=0.8555(0.4415+0.4141)-0.3804(0.1905+0.1899) sod-mse=0.0921(0.0547) gcn-mse=0.0908(0.0576) gcn-final-mse=0.0573(0.0709)
2020-08-05 19:39:40 3000-5019 loss=0.0440(0.0330+0.0110)-0.3773(0.1888+0.1885) sod-mse=0.0052(0.0546) gcn-mse=0.0061(0.0575) gcn-final-mse=0.0573(0.0708)
2020-08-05 19:41:01 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 19:42:05 4000-5019 loss=0.0973(0.0682+0.0291)-0.3731(0.1870+0.1861) sod-mse=0.0144(0.0542) gcn-mse=0.0144(0.0572) gcn-final-mse=0.0570(0.0705)
2020-08-05 19:42:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 19:44:30 5000-5019 loss=1.3875(0.6017+0.7858)-0.3717(0.1863+0.1854) sod-mse=0.1108(0.0542) gcn-mse=0.1141(0.0572) gcn-final-mse=0.0569(0.0704)
2020-08-05 19:44:33 E:27, Train sod-mae-score=0.0120-0.9817 gcn-mae-score=0.0157-0.9525 gcn-final-mse-score=0.0149-0.9549(0.0318/0.9549) loss=0.0808(0.0569+0.0240)
2020-08-05 19:44:33 E:27, Test  sod-mae-score=0.0542-0.8460 gcn-mae-score=0.0572-0.7876 gcn-final-mse-score=0.0569-0.7935(0.0704/0.7935) loss=0.3715(0.1862+0.1853)

2020-08-05 19:44:33 Start Epoch 28
2020-08-05 19:44:33 Epoch:28,lr=0.0000
2020-08-05 19:44:34    0-10553 loss=0.0430(0.0316+0.0114)-0.0430(0.0316+0.0114) sod-mse=0.0067(0.0067) gcn-mse=0.0084(0.0084) gcn-final-mse=0.0073(0.0155)
2020-08-05 19:50:07 1000-10553 loss=0.1005(0.0647+0.0358)-0.0804(0.0570+0.0234) sod-mse=0.0184(0.0116) gcn-mse=0.0264(0.0154) gcn-final-mse=0.0146(0.0317)
2020-08-05 19:55:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 19:55:38 2000-10553 loss=0.0981(0.0683+0.0298)-0.0800(0.0566+0.0234) sod-mse=0.0168(0.0118) gcn-mse=0.0171(0.0154) gcn-final-mse=0.0146(0.0317)
2020-08-05 20:00:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 20:01:10 3000-10553 loss=0.0850(0.0637+0.0214)-0.0793(0.0562+0.0231) sod-mse=0.0123(0.0116) gcn-mse=0.0160(0.0152) gcn-final-mse=0.0144(0.0315)
2020-08-05 20:06:41 4000-10553 loss=0.0942(0.0645+0.0297)-0.0791(0.0560+0.0231) sod-mse=0.0142(0.0116) gcn-mse=0.0176(0.0151) gcn-final-mse=0.0144(0.0314)
2020-08-05 20:06:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 20:12:12 5000-10553 loss=0.1391(0.1080+0.0312)-0.0790(0.0559+0.0231) sod-mse=0.0148(0.0116) gcn-mse=0.0329(0.0151) gcn-final-mse=0.0144(0.0314)
2020-08-05 20:17:43 6000-10553 loss=0.0726(0.0583+0.0143)-0.0793(0.0561+0.0233) sod-mse=0.0061(0.0116) gcn-mse=0.0154(0.0152) gcn-final-mse=0.0144(0.0314)
2020-08-05 20:23:12 7000-10553 loss=0.0981(0.0658+0.0323)-0.0795(0.0561+0.0234) sod-mse=0.0166(0.0117) gcn-mse=0.0191(0.0152) gcn-final-mse=0.0144(0.0314)
2020-08-05 20:28:44 8000-10553 loss=0.0744(0.0597+0.0147)-0.0796(0.0562+0.0234) sod-mse=0.0076(0.0117) gcn-mse=0.0174(0.0153) gcn-final-mse=0.0145(0.0314)
2020-08-05 20:34:16 9000-10553 loss=0.0623(0.0466+0.0157)-0.0795(0.0561+0.0234) sod-mse=0.0088(0.0117) gcn-mse=0.0086(0.0152) gcn-final-mse=0.0145(0.0314)
2020-08-05 20:39:48 10000-10553 loss=0.0873(0.0680+0.0193)-0.0797(0.0562+0.0235) sod-mse=0.0116(0.0117) gcn-mse=0.0207(0.0153) gcn-final-mse=0.0145(0.0314)

2020-08-05 20:42:54    0-5019 loss=1.1697(0.6876+0.4821)-1.1697(0.6876+0.4821) sod-mse=0.1002(0.1002) gcn-mse=0.1061(0.1061) gcn-final-mse=0.0976(0.1097)
2020-08-05 20:45:21 1000-5019 loss=0.0275(0.0230+0.0044)-0.3961(0.1925+0.2036) sod-mse=0.0035(0.0516) gcn-mse=0.0052(0.0555) gcn-final-mse=0.0553(0.0686)
2020-08-05 20:47:45 2000-5019 loss=1.1001(0.5152+0.5849)-0.4026(0.1946+0.2080) sod-mse=0.0969(0.0529) gcn-mse=0.0950(0.0570) gcn-final-mse=0.0568(0.0700)
2020-08-05 20:50:10 3000-5019 loss=0.0438(0.0332+0.0106)-0.3988(0.1925+0.2064) sod-mse=0.0049(0.0528) gcn-mse=0.0062(0.0568) gcn-final-mse=0.0567(0.0699)
2020-08-05 20:51:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 20:52:34 4000-5019 loss=0.1003(0.0701+0.0302)-0.3936(0.1905+0.2031) sod-mse=0.0145(0.0524) gcn-mse=0.0154(0.0565) gcn-final-mse=0.0563(0.0695)
2020-08-05 20:53:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 20:54:58 5000-5019 loss=1.5561(0.6422+0.9138)-0.3924(0.1899+0.2025) sod-mse=0.1036(0.0524) gcn-mse=0.1108(0.0565) gcn-final-mse=0.0563(0.0695)
2020-08-05 20:55:01 E:28, Train sod-mae-score=0.0117-0.9822 gcn-mae-score=0.0153-0.9532 gcn-final-mse-score=0.0145-0.9557(0.0314/0.9557) loss=0.0796(0.0562+0.0234)
2020-08-05 20:55:01 E:28, Test  sod-mae-score=0.0523-0.8468 gcn-mae-score=0.0565-0.7876 gcn-final-mse-score=0.0563-0.7934(0.0695/0.7934) loss=0.3922(0.1899+0.2023)

2020-08-05 20:55:01 Start Epoch 29
2020-08-05 20:55:01 Epoch:29,lr=0.0000
2020-08-05 20:55:02    0-10553 loss=0.0510(0.0341+0.0170)-0.0510(0.0341+0.0170) sod-mse=0.0071(0.0071) gcn-mse=0.0081(0.0081) gcn-final-mse=0.0068(0.0158)
2020-08-05 21:00:32 1000-10553 loss=0.0531(0.0404+0.0127)-0.0823(0.0575+0.0248) sod-mse=0.0067(0.0121) gcn-mse=0.0087(0.0154) gcn-final-mse=0.0147(0.0319)
2020-08-05 21:06:02 2000-10553 loss=0.0989(0.0781+0.0208)-0.0809(0.0570+0.0239) sod-mse=0.0139(0.0118) gcn-mse=0.0196(0.0153) gcn-final-mse=0.0146(0.0318)
2020-08-05 21:11:33 3000-10553 loss=0.0576(0.0444+0.0132)-0.0795(0.0561+0.0234) sod-mse=0.0071(0.0115) gcn-mse=0.0115(0.0150) gcn-final-mse=0.0143(0.0313)
2020-08-05 21:14:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 21:17:05 4000-10553 loss=0.0916(0.0582+0.0334)-0.0808(0.0568+0.0240) sod-mse=0.0173(0.0118) gcn-mse=0.0155(0.0152) gcn-final-mse=0.0144(0.0315)
2020-08-05 21:22:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 21:22:38 5000-10553 loss=0.1112(0.0909+0.0202)-0.0804(0.0567+0.0238) sod-mse=0.0153(0.0117) gcn-mse=0.0251(0.0152) gcn-final-mse=0.0144(0.0315)
2020-08-05 21:28:11 6000-10553 loss=0.0308(0.0264+0.0043)-0.0799(0.0564+0.0235) sod-mse=0.0033(0.0116) gcn-mse=0.0060(0.0151) gcn-final-mse=0.0143(0.0314)
2020-08-05 21:30:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 21:33:43 7000-10553 loss=0.0729(0.0510+0.0218)-0.0796(0.0562+0.0234) sod-mse=0.0100(0.0116) gcn-mse=0.0124(0.0150) gcn-final-mse=0.0142(0.0313)
2020-08-05 21:39:15 8000-10553 loss=0.0835(0.0649+0.0186)-0.0792(0.0559+0.0232) sod-mse=0.0124(0.0115) gcn-mse=0.0136(0.0150) gcn-final-mse=0.0142(0.0312)
2020-08-05 21:44:46 9000-10553 loss=0.0803(0.0626+0.0177)-0.0793(0.0560+0.0233) sod-mse=0.0097(0.0116) gcn-mse=0.0151(0.0150) gcn-final-mse=0.0142(0.0312)
2020-08-05 21:50:17 10000-10553 loss=0.0656(0.0489+0.0167)-0.0791(0.0559+0.0233) sod-mse=0.0082(0.0115) gcn-mse=0.0114(0.0150) gcn-final-mse=0.0142(0.0312)

2020-08-05 21:53:22    0-5019 loss=1.1057(0.6772+0.4285)-1.1057(0.6772+0.4285) sod-mse=0.0997(0.0997) gcn-mse=0.1056(0.1056) gcn-final-mse=0.0973(0.1104)
2020-08-05 21:55:47 1000-5019 loss=0.0281(0.0231+0.0050)-0.3845(0.1927+0.1918) sod-mse=0.0040(0.0524) gcn-mse=0.0051(0.0559) gcn-final-mse=0.0557(0.0692)
2020-08-05 21:58:11 2000-5019 loss=0.8692(0.4399+0.4294)-0.3946(0.1964+0.1982) sod-mse=0.0916(0.0537) gcn-mse=0.0895(0.0574) gcn-final-mse=0.0572(0.0706)
2020-08-05 22:00:35 3000-5019 loss=0.0442(0.0331+0.0111)-0.3921(0.1947+0.1974) sod-mse=0.0052(0.0536) gcn-mse=0.0062(0.0574) gcn-final-mse=0.0572(0.0706)
2020-08-05 22:01:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 22:03:00 4000-5019 loss=0.0977(0.0683+0.0294)-0.3877(0.1929+0.1948) sod-mse=0.0143(0.0532) gcn-mse=0.0143(0.0570) gcn-final-mse=0.0568(0.0702)
2020-08-05 22:03:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 22:05:23 5000-5019 loss=1.3163(0.5042+0.8122)-0.3862(0.1922+0.1940) sod-mse=0.1052(0.0532) gcn-mse=0.1082(0.0571) gcn-final-mse=0.0568(0.0701)
2020-08-05 22:05:26 E:29, Train sod-mae-score=0.0115-0.9826 gcn-mae-score=0.0150-0.9536 gcn-final-mse-score=0.0142-0.9561(0.0312/0.9561) loss=0.0791(0.0559+0.0232)
2020-08-05 22:05:26 E:29, Test  sod-mae-score=0.0532-0.8449 gcn-mae-score=0.0571-0.7860 gcn-final-mse-score=0.0568-0.7918(0.0701/0.7918) loss=0.3861(0.1921+0.1939)

2020-08-05 22:05:26 Start Epoch 30
2020-08-05 22:05:26 Epoch:30,lr=0.0000
2020-08-05 22:05:27    0-10553 loss=0.0919(0.0625+0.0294)-0.0919(0.0625+0.0294) sod-mse=0.0118(0.0118) gcn-mse=0.0128(0.0128) gcn-final-mse=0.0126(0.0339)
2020-08-05 22:10:59 1000-10553 loss=0.0752(0.0505+0.0247)-0.0757(0.0542+0.0215) sod-mse=0.0133(0.0108) gcn-mse=0.0189(0.0141) gcn-final-mse=0.0134(0.0304)
2020-08-05 22:11:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 22:16:31 2000-10553 loss=0.0947(0.0674+0.0273)-0.0768(0.0547+0.0220) sod-mse=0.0148(0.0109) gcn-mse=0.0159(0.0142) gcn-final-mse=0.0135(0.0306)
2020-08-05 22:22:05 3000-10553 loss=0.0850(0.0570+0.0280)-0.0770(0.0548+0.0222) sod-mse=0.0169(0.0110) gcn-mse=0.0167(0.0144) gcn-final-mse=0.0136(0.0306)
2020-08-05 22:27:36 4000-10553 loss=0.0751(0.0529+0.0223)-0.0771(0.0549+0.0223) sod-mse=0.0107(0.0111) gcn-mse=0.0127(0.0144) gcn-final-mse=0.0137(0.0307)
2020-08-05 22:33:09 5000-10553 loss=0.0750(0.0506+0.0243)-0.0772(0.0549+0.0223) sod-mse=0.0140(0.0111) gcn-mse=0.0102(0.0144) gcn-final-mse=0.0137(0.0308)
2020-08-05 22:38:40 6000-10553 loss=0.0644(0.0420+0.0224)-0.0774(0.0551+0.0223) sod-mse=0.0074(0.0111) gcn-mse=0.0079(0.0145) gcn-final-mse=0.0137(0.0308)
2020-08-05 22:42:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 22:44:11 7000-10553 loss=0.1648(0.1215+0.0433)-0.0774(0.0551+0.0223) sod-mse=0.0197(0.0111) gcn-mse=0.0284(0.0145) gcn-final-mse=0.0137(0.0308)
2020-08-05 22:49:42 8000-10553 loss=0.0424(0.0346+0.0079)-0.0777(0.0552+0.0225) sod-mse=0.0034(0.0112) gcn-mse=0.0042(0.0145) gcn-final-mse=0.0138(0.0309)
2020-08-05 22:51:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 22:55:12 9000-10553 loss=0.0766(0.0530+0.0235)-0.0778(0.0553+0.0225) sod-mse=0.0138(0.0112) gcn-mse=0.0128(0.0146) gcn-final-mse=0.0138(0.0309)
2020-08-05 23:00:43 10000-10553 loss=0.0350(0.0258+0.0092)-0.0778(0.0553+0.0226) sod-mse=0.0043(0.0112) gcn-mse=0.0068(0.0146) gcn-final-mse=0.0138(0.0309)

2020-08-05 23:03:47    0-5019 loss=1.0616(0.6330+0.4286)-1.0616(0.6330+0.4286) sod-mse=0.0979(0.0979) gcn-mse=0.1026(0.1026) gcn-final-mse=0.0943(0.1062)
2020-08-05 23:06:14 1000-5019 loss=0.0278(0.0232+0.0047)-0.3760(0.1872+0.1889) sod-mse=0.0037(0.0510) gcn-mse=0.0053(0.0543) gcn-final-mse=0.0541(0.0676)
2020-08-05 23:08:38 2000-5019 loss=0.9007(0.4468+0.4539)-0.3849(0.1906+0.1942) sod-mse=0.0914(0.0526) gcn-mse=0.0904(0.0560) gcn-final-mse=0.0557(0.0692)
2020-08-05 23:11:02 3000-5019 loss=0.0436(0.0330+0.0106)-0.3819(0.1888+0.1931) sod-mse=0.0048(0.0524) gcn-mse=0.0062(0.0559) gcn-final-mse=0.0557(0.0692)
2020-08-05 23:12:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 23:13:26 4000-5019 loss=0.0978(0.0682+0.0296)-0.3771(0.1869+0.1902) sod-mse=0.0140(0.0521) gcn-mse=0.0142(0.0556) gcn-final-mse=0.0553(0.0688)
2020-08-05 23:14:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 23:15:49 5000-5019 loss=1.5134(0.6480+0.8654)-0.3751(0.1861+0.1890) sod-mse=0.0975(0.0520) gcn-mse=0.1063(0.0555) gcn-final-mse=0.0553(0.0687)
2020-08-05 23:15:52 E:30, Train sod-mae-score=0.0112-0.9830 gcn-mae-score=0.0146-0.9540 gcn-final-mse-score=0.0138-0.9564(0.0309/0.9564) loss=0.0777(0.0552+0.0225)
2020-08-05 23:15:52 E:30, Test  sod-mae-score=0.0520-0.8490 gcn-mae-score=0.0555-0.7891 gcn-final-mse-score=0.0553-0.7948(0.0687/0.7948) loss=0.3749(0.1860+0.1889)

2020-08-05 23:15:52 Start Epoch 31
2020-08-05 23:15:52 Epoch:31,lr=0.0000
2020-08-05 23:15:53    0-10553 loss=0.1144(0.0793+0.0351)-0.1144(0.0793+0.0351) sod-mse=0.0185(0.0185) gcn-mse=0.0232(0.0232) gcn-final-mse=0.0231(0.0494)
2020-08-05 23:21:24 1000-10553 loss=0.0688(0.0481+0.0207)-0.0746(0.0533+0.0212) sod-mse=0.0117(0.0104) gcn-mse=0.0188(0.0138) gcn-final-mse=0.0130(0.0298)
2020-08-05 23:26:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 23:26:56 2000-10553 loss=0.1494(0.0901+0.0592)-0.0773(0.0547+0.0226) sod-mse=0.0214(0.0111) gcn-mse=0.0280(0.0144) gcn-final-mse=0.0136(0.0305)
2020-08-05 23:32:28 3000-10553 loss=0.0223(0.0155+0.0068)-0.0762(0.0542+0.0220) sod-mse=0.0031(0.0109) gcn-mse=0.0023(0.0142) gcn-final-mse=0.0135(0.0303)
2020-08-05 23:32:58 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 23:37:59 4000-10553 loss=0.0591(0.0456+0.0135)-0.0762(0.0543+0.0220) sod-mse=0.0091(0.0109) gcn-mse=0.0115(0.0143) gcn-final-mse=0.0135(0.0304)
2020-08-05 23:43:28 5000-10553 loss=0.0351(0.0265+0.0086)-0.0762(0.0544+0.0219) sod-mse=0.0037(0.0109) gcn-mse=0.0079(0.0143) gcn-final-mse=0.0135(0.0305)
2020-08-05 23:48:59 6000-10553 loss=0.1021(0.0686+0.0335)-0.0763(0.0545+0.0218) sod-mse=0.0099(0.0109) gcn-mse=0.0097(0.0143) gcn-final-mse=0.0135(0.0305)
2020-08-05 23:54:31 7000-10553 loss=0.0260(0.0199+0.0060)-0.0765(0.0546+0.0220) sod-mse=0.0036(0.0109) gcn-mse=0.0040(0.0143) gcn-final-mse=0.0135(0.0305)
2020-08-06 00:00:02 8000-10553 loss=0.0224(0.0165+0.0059)-0.0767(0.0547+0.0220) sod-mse=0.0041(0.0110) gcn-mse=0.0050(0.0143) gcn-final-mse=0.0135(0.0306)
2020-08-06 00:02:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 00:05:34 9000-10553 loss=0.0723(0.0542+0.0181)-0.0768(0.0547+0.0221) sod-mse=0.0087(0.0110) gcn-mse=0.0170(0.0143) gcn-final-mse=0.0135(0.0306)
2020-08-06 00:11:06 10000-10553 loss=0.0389(0.0305+0.0084)-0.0767(0.0546+0.0221) sod-mse=0.0057(0.0110) gcn-mse=0.0089(0.0143) gcn-final-mse=0.0135(0.0306)

2020-08-06 00:14:11    0-5019 loss=0.9590(0.6076+0.3513)-0.9590(0.6076+0.3513) sod-mse=0.0969(0.0969) gcn-mse=0.1014(0.1014) gcn-final-mse=0.0930(0.1056)
2020-08-06 00:16:38 1000-5019 loss=0.0283(0.0229+0.0054)-0.3579(0.1881+0.1698) sod-mse=0.0045(0.0536) gcn-mse=0.0051(0.0556) gcn-final-mse=0.0554(0.0690)
2020-08-06 00:19:03 2000-5019 loss=0.8656(0.4687+0.3969)-0.3651(0.1910+0.1741) sod-mse=0.0937(0.0550) gcn-mse=0.0915(0.0574) gcn-final-mse=0.0572(0.0707)
2020-08-06 00:21:27 3000-5019 loss=0.0441(0.0332+0.0109)-0.3613(0.1888+0.1726) sod-mse=0.0053(0.0547) gcn-mse=0.0064(0.0572) gcn-final-mse=0.0571(0.0706)
2020-08-06 00:22:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 00:23:50 4000-5019 loss=0.0978(0.0685+0.0293)-0.3592(0.1879+0.1714) sod-mse=0.0146(0.0545) gcn-mse=0.0147(0.0570) gcn-final-mse=0.0568(0.0703)
2020-08-06 00:24:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 00:26:13 5000-5019 loss=1.2563(0.5838+0.6725)-0.3571(0.1869+0.1702) sod-mse=0.1026(0.0544) gcn-mse=0.1071(0.0570) gcn-final-mse=0.0567(0.0702)
2020-08-06 00:26:16 E:31, Train sod-mae-score=0.0111-0.9832 gcn-mae-score=0.0144-0.9536 gcn-final-mse-score=0.0136-0.9562(0.0307/0.9562) loss=0.0775(0.0550+0.0225)
2020-08-06 00:26:16 E:31, Test  sod-mae-score=0.0544-0.8433 gcn-mae-score=0.0570-0.7836 gcn-final-mse-score=0.0567-0.7893(0.0702/0.7893) loss=0.3571(0.1869+0.1702)

2020-08-06 00:26:16 Start Epoch 32
2020-08-06 00:26:16 Epoch:32,lr=0.0000
2020-08-06 00:26:17    0-10553 loss=0.0751(0.0554+0.0197)-0.0751(0.0554+0.0197) sod-mse=0.0135(0.0135) gcn-mse=0.0162(0.0162) gcn-final-mse=0.0142(0.0340)
2020-08-06 00:31:50 1000-10553 loss=0.0354(0.0281+0.0073)-0.0759(0.0544+0.0215) sod-mse=0.0038(0.0111) gcn-mse=0.0056(0.0143) gcn-final-mse=0.0135(0.0303)
2020-08-06 00:37:20 2000-10553 loss=0.0916(0.0661+0.0256)-0.0760(0.0544+0.0216) sod-mse=0.0130(0.0109) gcn-mse=0.0132(0.0142) gcn-final-mse=0.0134(0.0303)
2020-08-06 00:42:53 3000-10553 loss=0.0507(0.0361+0.0146)-0.0758(0.0543+0.0215) sod-mse=0.0095(0.0108) gcn-mse=0.0078(0.0141) gcn-final-mse=0.0133(0.0302)
2020-08-06 00:48:24 4000-10553 loss=0.0655(0.0497+0.0159)-0.0760(0.0543+0.0216) sod-mse=0.0081(0.0109) gcn-mse=0.0150(0.0141) gcn-final-mse=0.0132(0.0302)
2020-08-06 00:53:53 5000-10553 loss=0.0671(0.0487+0.0184)-0.0763(0.0545+0.0218) sod-mse=0.0092(0.0109) gcn-mse=0.0094(0.0141) gcn-final-mse=0.0133(0.0304)
2020-08-06 00:59:25 6000-10553 loss=0.1081(0.0748+0.0334)-0.0765(0.0546+0.0219) sod-mse=0.0171(0.0110) gcn-mse=0.0184(0.0142) gcn-final-mse=0.0134(0.0305)
2020-08-06 01:02:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 01:04:57 7000-10553 loss=0.0721(0.0575+0.0146)-0.0765(0.0546+0.0218) sod-mse=0.0107(0.0109) gcn-mse=0.0179(0.0142) gcn-final-mse=0.0134(0.0305)
2020-08-06 01:10:30 8000-10553 loss=0.0482(0.0364+0.0119)-0.0766(0.0546+0.0220) sod-mse=0.0081(0.0109) gcn-mse=0.0067(0.0141) gcn-final-mse=0.0134(0.0305)
2020-08-06 01:16:02 9000-10553 loss=0.0475(0.0320+0.0155)-0.0765(0.0545+0.0220) sod-mse=0.0110(0.0109) gcn-mse=0.0116(0.0141) gcn-final-mse=0.0133(0.0304)
2020-08-06 01:16:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 01:21:32 10000-10553 loss=0.0866(0.0697+0.0169)-0.0764(0.0544+0.0220) sod-mse=0.0087(0.0109) gcn-mse=0.0226(0.0141) gcn-final-mse=0.0133(0.0304)
2020-08-06 01:22:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg

2020-08-06 01:24:37    0-5019 loss=1.1286(0.6754+0.4532)-1.1286(0.6754+0.4532) sod-mse=0.1027(0.1027) gcn-mse=0.1088(0.1088) gcn-final-mse=0.1005(0.1128)
2020-08-06 01:27:02 1000-5019 loss=0.0271(0.0226+0.0045)-0.3897(0.1928+0.1969) sod-mse=0.0036(0.0523) gcn-mse=0.0048(0.0554) gcn-final-mse=0.0552(0.0687)
2020-08-06 01:29:26 2000-5019 loss=0.9031(0.4471+0.4560)-0.3998(0.1968+0.2030) sod-mse=0.0872(0.0535) gcn-mse=0.0889(0.0568) gcn-final-mse=0.0566(0.0700)
2020-08-06 01:31:49 3000-5019 loss=0.0434(0.0329+0.0105)-0.3969(0.1948+0.2021) sod-mse=0.0048(0.0533) gcn-mse=0.0062(0.0567) gcn-final-mse=0.0566(0.0700)
2020-08-06 01:33:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 01:34:14 4000-5019 loss=0.0982(0.0687+0.0295)-0.3930(0.1933+0.1997) sod-mse=0.0141(0.0531) gcn-mse=0.0143(0.0565) gcn-final-mse=0.0563(0.0697)
2020-08-06 01:34:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 01:36:37 5000-5019 loss=1.4995(0.6218+0.8777)-0.3906(0.1922+0.1983) sod-mse=0.1132(0.0530) gcn-mse=0.1174(0.0564) gcn-final-mse=0.0562(0.0696)
2020-08-06 01:36:39 E:32, Train sod-mae-score=0.0109-0.9835 gcn-mae-score=0.0141-0.9543 gcn-final-mse-score=0.0133-0.9568(0.0304/0.9568) loss=0.0763(0.0544+0.0219)
2020-08-06 01:36:39 E:32, Test  sod-mae-score=0.0530-0.8441 gcn-mae-score=0.0564-0.7839 gcn-final-mse-score=0.0562-0.7895(0.0696/0.7895) loss=0.3904(0.1922+0.1982)

2020-08-06 01:36:39 Start Epoch 33
2020-08-06 01:36:39 Epoch:33,lr=0.0000
2020-08-06 01:36:41    0-10553 loss=0.0682(0.0510+0.0172)-0.0682(0.0510+0.0172) sod-mse=0.0092(0.0092) gcn-mse=0.0139(0.0139) gcn-final-mse=0.0130(0.0306)
2020-08-06 01:42:15 1000-10553 loss=0.0646(0.0494+0.0152)-0.0741(0.0528+0.0213) sod-mse=0.0092(0.0106) gcn-mse=0.0171(0.0136) gcn-final-mse=0.0128(0.0294)
2020-08-06 01:47:45 2000-10553 loss=0.0655(0.0491+0.0164)-0.0743(0.0532+0.0211) sod-mse=0.0059(0.0105) gcn-mse=0.0126(0.0137) gcn-final-mse=0.0129(0.0297)
2020-08-06 01:48:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 01:53:14 3000-10553 loss=0.0338(0.0269+0.0069)-0.0748(0.0535+0.0213) sod-mse=0.0037(0.0106) gcn-mse=0.0046(0.0137) gcn-final-mse=0.0130(0.0299)
2020-08-06 01:56:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 01:58:48 4000-10553 loss=0.0488(0.0367+0.0120)-0.0750(0.0536+0.0213) sod-mse=0.0064(0.0106) gcn-mse=0.0051(0.0138) gcn-final-mse=0.0131(0.0301)
2020-08-06 02:04:18 5000-10553 loss=0.2040(0.1411+0.0629)-0.0751(0.0538+0.0213) sod-mse=0.0317(0.0106) gcn-mse=0.0379(0.0138) gcn-final-mse=0.0131(0.0301)
2020-08-06 02:09:51 6000-10553 loss=0.0558(0.0437+0.0120)-0.0756(0.0542+0.0214) sod-mse=0.0064(0.0107) gcn-mse=0.0079(0.0139) gcn-final-mse=0.0132(0.0303)
2020-08-06 02:15:23 7000-10553 loss=0.3057(0.1909+0.1148)-0.0756(0.0542+0.0214) sod-mse=0.0682(0.0106) gcn-mse=0.0802(0.0139) gcn-final-mse=0.0132(0.0304)
2020-08-06 02:20:55 8000-10553 loss=0.0297(0.0229+0.0068)-0.0756(0.0542+0.0215) sod-mse=0.0030(0.0106) gcn-mse=0.0029(0.0139) gcn-final-mse=0.0131(0.0303)
2020-08-06 02:26:27 9000-10553 loss=0.0707(0.0538+0.0169)-0.0753(0.0540+0.0213) sod-mse=0.0092(0.0106) gcn-mse=0.0112(0.0138) gcn-final-mse=0.0131(0.0302)
2020-08-06 02:31:58 10000-10553 loss=0.0686(0.0546+0.0140)-0.0756(0.0541+0.0215) sod-mse=0.0061(0.0107) gcn-mse=0.0102(0.0139) gcn-final-mse=0.0131(0.0302)
2020-08-06 02:34:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg

2020-08-06 02:35:03    0-5019 loss=1.2044(0.7341+0.4702)-1.2044(0.7341+0.4702) sod-mse=0.1029(0.1029) gcn-mse=0.1138(0.1138) gcn-final-mse=0.1057(0.1171)
2020-08-06 02:37:29 1000-5019 loss=0.0269(0.0224+0.0045)-0.4026(0.1984+0.2042) sod-mse=0.0036(0.0525) gcn-mse=0.0045(0.0555) gcn-final-mse=0.0553(0.0686)
2020-08-06 02:39:55 2000-5019 loss=0.9157(0.4537+0.4620)-0.4097(0.2014+0.2083) sod-mse=0.0872(0.0536) gcn-mse=0.0870(0.0569) gcn-final-mse=0.0567(0.0699)
2020-08-06 02:42:24 3000-5019 loss=0.0441(0.0333+0.0108)-0.4065(0.1992+0.2073) sod-mse=0.0049(0.0536) gcn-mse=0.0064(0.0569) gcn-final-mse=0.0567(0.0699)
2020-08-06 02:43:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 02:44:53 4000-5019 loss=0.0989(0.0689+0.0300)-0.4017(0.1973+0.2043) sod-mse=0.0145(0.0532) gcn-mse=0.0141(0.0565) gcn-final-mse=0.0563(0.0696)
2020-08-06 02:45:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 02:47:21 5000-5019 loss=1.5087(0.6123+0.8964)-0.3989(0.1962+0.2026) sod-mse=0.1072(0.0531) gcn-mse=0.1108(0.0564) gcn-final-mse=0.0562(0.0694)
2020-08-06 02:47:23 E:33, Train sod-mae-score=0.0107-0.9838 gcn-mae-score=0.0139-0.9547 gcn-final-mse-score=0.0131-0.9573(0.0303/0.9573) loss=0.0756(0.0541+0.0215)
2020-08-06 02:47:23 E:33, Test  sod-mae-score=0.0530-0.8437 gcn-mae-score=0.0564-0.7863 gcn-final-mse-score=0.0562-0.7920(0.0694/0.7920) loss=0.3987(0.1962+0.2025)

2020-08-06 02:47:23 Start Epoch 34
2020-08-06 02:47:23 Epoch:34,lr=0.0000
2020-08-06 02:47:24    0-10553 loss=0.0671(0.0502+0.0169)-0.0671(0.0502+0.0169) sod-mse=0.0074(0.0074) gcn-mse=0.0122(0.0122) gcn-final-mse=0.0122(0.0332)
2020-08-06 02:51:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 02:52:57 1000-10553 loss=0.0790(0.0563+0.0227)-0.0742(0.0535+0.0207) sod-mse=0.0119(0.0103) gcn-mse=0.0137(0.0137) gcn-final-mse=0.0129(0.0300)
2020-08-06 02:55:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 02:58:28 2000-10553 loss=0.1126(0.0743+0.0383)-0.0739(0.0531+0.0207) sod-mse=0.0214(0.0103) gcn-mse=0.0227(0.0135) gcn-final-mse=0.0127(0.0298)
2020-08-06 03:03:59 3000-10553 loss=0.0152(0.0131+0.0022)-0.0745(0.0534+0.0210) sod-mse=0.0014(0.0104) gcn-mse=0.0014(0.0135) gcn-final-mse=0.0127(0.0299)
2020-08-06 03:09:30 4000-10553 loss=0.0958(0.0619+0.0339)-0.0762(0.0543+0.0219) sod-mse=0.0180(0.0108) gcn-mse=0.0170(0.0138) gcn-final-mse=0.0130(0.0303)
2020-08-06 03:15:00 5000-10553 loss=0.0570(0.0394+0.0176)-0.0759(0.0542+0.0217) sod-mse=0.0078(0.0107) gcn-mse=0.0077(0.0138) gcn-final-mse=0.0130(0.0303)
2020-08-06 03:20:30 6000-10553 loss=0.0989(0.0789+0.0200)-0.0754(0.0539+0.0215) sod-mse=0.0090(0.0106) gcn-mse=0.0179(0.0137) gcn-final-mse=0.0129(0.0302)
2020-08-06 03:26:02 7000-10553 loss=0.0720(0.0483+0.0237)-0.0753(0.0539+0.0214) sod-mse=0.0127(0.0106) gcn-mse=0.0088(0.0137) gcn-final-mse=0.0129(0.0301)
2020-08-06 03:31:35 8000-10553 loss=0.1154(0.0841+0.0314)-0.0752(0.0538+0.0214) sod-mse=0.0190(0.0106) gcn-mse=0.0217(0.0136) gcn-final-mse=0.0129(0.0301)
2020-08-06 03:37:05 9000-10553 loss=0.0610(0.0442+0.0168)-0.0751(0.0538+0.0214) sod-mse=0.0101(0.0106) gcn-mse=0.0107(0.0136) gcn-final-mse=0.0128(0.0301)
2020-08-06 03:39:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 03:42:37 10000-10553 loss=0.0725(0.0491+0.0234)-0.0749(0.0536+0.0213) sod-mse=0.0116(0.0105) gcn-mse=0.0114(0.0136) gcn-final-mse=0.0128(0.0300)

2020-08-06 03:45:43    0-5019 loss=1.1019(0.6669+0.4350)-1.1019(0.6669+0.4350) sod-mse=0.0976(0.0976) gcn-mse=0.1059(0.1059) gcn-final-mse=0.0971(0.1082)
2020-08-06 03:48:08 1000-5019 loss=0.0269(0.0226+0.0044)-0.3912(0.1935+0.1977) sod-mse=0.0035(0.0524) gcn-mse=0.0046(0.0554) gcn-final-mse=0.0551(0.0686)
2020-08-06 03:50:32 2000-5019 loss=0.8699(0.4353+0.4346)-0.3985(0.1966+0.2018) sod-mse=0.0855(0.0538) gcn-mse=0.0873(0.0569) gcn-final-mse=0.0567(0.0700)
2020-08-06 03:52:55 3000-5019 loss=0.0432(0.0329+0.0104)-0.3955(0.1946+0.2009) sod-mse=0.0046(0.0535) gcn-mse=0.0060(0.0567) gcn-final-mse=0.0565(0.0699)
2020-08-06 03:54:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 03:55:18 4000-5019 loss=0.0993(0.0695+0.0298)-0.3915(0.1930+0.1985) sod-mse=0.0146(0.0532) gcn-mse=0.0146(0.0564) gcn-final-mse=0.0562(0.0695)
2020-08-06 03:56:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 03:57:42 5000-5019 loss=1.4861(0.5928+0.8933)-0.3894(0.1921+0.1973) sod-mse=0.0986(0.0530) gcn-mse=0.1046(0.0563) gcn-final-mse=0.0561(0.0694)
2020-08-06 03:57:44 E:34, Train sod-mae-score=0.0105-0.9838 gcn-mae-score=0.0136-0.9547 gcn-final-mse-score=0.0128-0.9572(0.0300/0.9572) loss=0.0749(0.0537+0.0212)
2020-08-06 03:57:44 E:34, Test  sod-mae-score=0.0530-0.8453 gcn-mae-score=0.0563-0.7866 gcn-final-mse-score=0.0561-0.7924(0.0694/0.7924) loss=0.3892(0.1920+0.1972)

2020-08-06 03:57:44 Start Epoch 35
2020-08-06 03:57:44 Epoch:35,lr=0.0000
2020-08-06 03:57:45    0-10553 loss=0.0781(0.0566+0.0215)-0.0781(0.0566+0.0215) sod-mse=0.0122(0.0122) gcn-mse=0.0168(0.0168) gcn-final-mse=0.0172(0.0395)
2020-08-06 04:03:17 1000-10553 loss=0.0550(0.0462+0.0088)-0.0743(0.0536+0.0207) sod-mse=0.0043(0.0102) gcn-mse=0.0114(0.0132) gcn-final-mse=0.0125(0.0300)
2020-08-06 04:08:47 2000-10553 loss=0.0762(0.0548+0.0214)-0.0740(0.0533+0.0207) sod-mse=0.0082(0.0102) gcn-mse=0.0150(0.0132) gcn-final-mse=0.0125(0.0298)
2020-08-06 04:13:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 04:14:18 3000-10553 loss=0.0300(0.0245+0.0056)-0.0734(0.0530+0.0204) sod-mse=0.0022(0.0101) gcn-mse=0.0038(0.0132) gcn-final-mse=0.0124(0.0297)
2020-08-06 04:19:50 4000-10553 loss=0.0397(0.0325+0.0072)-0.0728(0.0527+0.0201) sod-mse=0.0034(0.0100) gcn-mse=0.0069(0.0131) gcn-final-mse=0.0123(0.0296)
2020-08-06 04:25:23 5000-10553 loss=0.0749(0.0517+0.0232)-0.0730(0.0528+0.0203) sod-mse=0.0119(0.0101) gcn-mse=0.0115(0.0131) gcn-final-mse=0.0124(0.0296)
2020-08-06 04:26:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 04:30:54 6000-10553 loss=0.0599(0.0481+0.0118)-0.0730(0.0527+0.0203) sod-mse=0.0057(0.0101) gcn-mse=0.0101(0.0131) gcn-final-mse=0.0123(0.0296)
2020-08-06 04:36:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 04:36:26 7000-10553 loss=0.0919(0.0684+0.0235)-0.0729(0.0527+0.0203) sod-mse=0.0119(0.0101) gcn-mse=0.0125(0.0131) gcn-final-mse=0.0123(0.0296)
2020-08-06 04:42:00 8000-10553 loss=0.0399(0.0316+0.0083)-0.0730(0.0527+0.0203) sod-mse=0.0038(0.0101) gcn-mse=0.0050(0.0131) gcn-final-mse=0.0123(0.0296)
2020-08-06 04:47:32 9000-10553 loss=0.0156(0.0134+0.0022)-0.0734(0.0529+0.0205) sod-mse=0.0014(0.0101) gcn-mse=0.0021(0.0131) gcn-final-mse=0.0124(0.0296)
2020-08-06 04:53:05 10000-10553 loss=0.0875(0.0721+0.0154)-0.0734(0.0529+0.0205) sod-mse=0.0098(0.0101) gcn-mse=0.0205(0.0131) gcn-final-mse=0.0124(0.0296)

2020-08-06 04:56:10    0-5019 loss=1.1552(0.6980+0.4572)-1.1552(0.6980+0.4572) sod-mse=0.1012(0.1012) gcn-mse=0.1077(0.1077) gcn-final-mse=0.0991(0.1114)
2020-08-06 04:58:36 1000-5019 loss=0.0270(0.0226+0.0045)-0.3968(0.1951+0.2017) sod-mse=0.0036(0.0517) gcn-mse=0.0047(0.0548) gcn-final-mse=0.0546(0.0681)
2020-08-06 05:01:01 2000-5019 loss=0.9210(0.4529+0.4681)-0.4041(0.1981+0.2060) sod-mse=0.0891(0.0529) gcn-mse=0.0886(0.0562) gcn-final-mse=0.0560(0.0694)
2020-08-06 05:03:25 3000-5019 loss=0.0435(0.0329+0.0106)-0.4016(0.1962+0.2053) sod-mse=0.0048(0.0528) gcn-mse=0.0060(0.0561) gcn-final-mse=0.0559(0.0693)
2020-08-06 05:04:44 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 05:05:48 4000-5019 loss=0.0983(0.0687+0.0296)-0.3967(0.1943+0.2023) sod-mse=0.0143(0.0524) gcn-mse=0.0141(0.0558) gcn-final-mse=0.0555(0.0690)
2020-08-06 05:06:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 05:08:11 5000-5019 loss=1.5137(0.6151+0.8986)-0.3943(0.1933+0.2010) sod-mse=0.0980(0.0523) gcn-mse=0.1048(0.0557) gcn-final-mse=0.0554(0.0688)
2020-08-06 05:08:13 E:35, Train sod-mae-score=0.0101-0.9844 gcn-mae-score=0.0131-0.9554 gcn-final-mse-score=0.0124-0.9579(0.0297/0.9579) loss=0.0734(0.0529+0.0205)
2020-08-06 05:08:13 E:35, Test  sod-mae-score=0.0523-0.8475 gcn-mae-score=0.0557-0.7881 gcn-final-mse-score=0.0554-0.7938(0.0688/0.7938) loss=0.3941(0.1932+0.2009)

2020-08-06 05:08:13 Start Epoch 36
2020-08-06 05:08:13 Epoch:36,lr=0.0000
2020-08-06 05:08:14    0-10553 loss=0.1147(0.0916+0.0232)-0.1147(0.0916+0.0232) sod-mse=0.0111(0.0111) gcn-mse=0.0204(0.0204) gcn-final-mse=0.0176(0.0494)
2020-08-06 05:13:48 1000-10553 loss=0.1102(0.0911+0.0191)-0.0753(0.0537+0.0216) sod-mse=0.0099(0.0105) gcn-mse=0.0145(0.0134) gcn-final-mse=0.0127(0.0301)
2020-08-06 05:19:18 2000-10553 loss=0.0798(0.0613+0.0185)-0.0740(0.0532+0.0208) sod-mse=0.0104(0.0103) gcn-mse=0.0169(0.0132) gcn-final-mse=0.0125(0.0298)
2020-08-06 05:24:49 3000-10553 loss=0.0580(0.0428+0.0152)-0.0734(0.0529+0.0205) sod-mse=0.0105(0.0102) gcn-mse=0.0116(0.0131) gcn-final-mse=0.0123(0.0297)
2020-08-06 05:30:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 05:30:21 4000-10553 loss=0.0610(0.0477+0.0133)-0.0741(0.0534+0.0207) sod-mse=0.0055(0.0102) gcn-mse=0.0075(0.0132) gcn-final-mse=0.0125(0.0300)
2020-08-06 05:35:55 5000-10553 loss=0.1424(0.1040+0.0384)-0.0737(0.0532+0.0205) sod-mse=0.0185(0.0102) gcn-mse=0.0245(0.0132) gcn-final-mse=0.0124(0.0299)
2020-08-06 05:41:25 6000-10553 loss=0.0513(0.0443+0.0070)-0.0735(0.0530+0.0205) sod-mse=0.0053(0.0102) gcn-mse=0.0066(0.0131) gcn-final-mse=0.0124(0.0298)
2020-08-06 05:46:53 7000-10553 loss=0.0567(0.0427+0.0140)-0.0732(0.0528+0.0204) sod-mse=0.0076(0.0101) gcn-mse=0.0091(0.0130) gcn-final-mse=0.0123(0.0297)
2020-08-06 05:52:24 8000-10553 loss=0.0276(0.0224+0.0052)-0.0731(0.0528+0.0203) sod-mse=0.0026(0.0101) gcn-mse=0.0034(0.0130) gcn-final-mse=0.0122(0.0296)
2020-08-06 05:54:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 05:57:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 05:57:55 9000-10553 loss=0.0718(0.0525+0.0193)-0.0730(0.0527+0.0203) sod-mse=0.0110(0.0100) gcn-mse=0.0167(0.0130) gcn-final-mse=0.0122(0.0296)
2020-08-06 06:03:26 10000-10553 loss=0.1172(0.0868+0.0304)-0.0731(0.0528+0.0203) sod-mse=0.0180(0.0101) gcn-mse=0.0254(0.0130) gcn-final-mse=0.0122(0.0296)

2020-08-06 06:06:30    0-5019 loss=1.1552(0.7003+0.4549)-1.1552(0.7003+0.4549) sod-mse=0.1009(0.1009) gcn-mse=0.1076(0.1076) gcn-final-mse=0.0990(0.1116)
2020-08-06 06:08:57 1000-5019 loss=0.0271(0.0226+0.0045)-0.4020(0.1967+0.2054) sod-mse=0.0036(0.0515) gcn-mse=0.0047(0.0547) gcn-final-mse=0.0545(0.0680)
2020-08-06 06:11:24 2000-5019 loss=0.9336(0.4637+0.4700)-0.4098(0.1998+0.2100) sod-mse=0.0900(0.0527) gcn-mse=0.0890(0.0561) gcn-final-mse=0.0559(0.0692)
2020-08-06 06:13:50 3000-5019 loss=0.0436(0.0329+0.0107)-0.4070(0.1978+0.2092) sod-mse=0.0048(0.0526) gcn-mse=0.0061(0.0560) gcn-final-mse=0.0558(0.0692)
2020-08-06 06:15:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 06:16:17 4000-5019 loss=0.0978(0.0685+0.0293)-0.4019(0.1958+0.2061) sod-mse=0.0141(0.0522) gcn-mse=0.0139(0.0556) gcn-final-mse=0.0554(0.0688)
2020-08-06 06:17:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 06:18:44 5000-5019 loss=1.5227(0.6278+0.8948)-0.3994(0.1947+0.2047) sod-mse=0.0979(0.0521) gcn-mse=0.1050(0.0556) gcn-final-mse=0.0553(0.0686)
2020-08-06 06:18:47 E:36, Train sod-mae-score=0.0100-0.9847 gcn-mae-score=0.0130-0.9558 gcn-final-mse-score=0.0122-0.9582(0.0296/0.9582) loss=0.0730(0.0527+0.0203)
2020-08-06 06:18:47 E:36, Test  sod-mae-score=0.0521-0.8475 gcn-mae-score=0.0556-0.7875 gcn-final-mse-score=0.0553-0.7932(0.0686/0.7932) loss=0.3992(0.1946+0.2046)

2020-08-06 06:18:47 Start Epoch 37
2020-08-06 06:18:47 Epoch:37,lr=0.0000
2020-08-06 06:18:48    0-10553 loss=0.1054(0.0831+0.0222)-0.1054(0.0831+0.0222) sod-mse=0.0092(0.0092) gcn-mse=0.0140(0.0140) gcn-final-mse=0.0127(0.0393)
2020-08-06 06:24:23 1000-10553 loss=0.0462(0.0377+0.0085)-0.0724(0.0522+0.0202) sod-mse=0.0040(0.0099) gcn-mse=0.0067(0.0128) gcn-final-mse=0.0120(0.0292)
2020-08-06 06:29:59 2000-10553 loss=0.0873(0.0628+0.0245)-0.0723(0.0523+0.0200) sod-mse=0.0116(0.0099) gcn-mse=0.0148(0.0129) gcn-final-mse=0.0121(0.0294)
2020-08-06 06:35:34 3000-10553 loss=0.1039(0.0684+0.0355)-0.0725(0.0524+0.0202) sod-mse=0.0143(0.0099) gcn-mse=0.0150(0.0129) gcn-final-mse=0.0122(0.0293)
2020-08-06 06:39:01 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 06:41:08 4000-10553 loss=0.0636(0.0498+0.0138)-0.0726(0.0523+0.0202) sod-mse=0.0069(0.0099) gcn-mse=0.0137(0.0129) gcn-final-mse=0.0121(0.0293)
2020-08-06 06:46:43 5000-10553 loss=0.0829(0.0627+0.0202)-0.0730(0.0527+0.0203) sod-mse=0.0086(0.0100) gcn-mse=0.0176(0.0130) gcn-final-mse=0.0122(0.0295)
2020-08-06 06:52:18 6000-10553 loss=0.0795(0.0551+0.0244)-0.0729(0.0526+0.0202) sod-mse=0.0109(0.0100) gcn-mse=0.0172(0.0129) gcn-final-mse=0.0122(0.0295)
2020-08-06 06:57:54 7000-10553 loss=0.0814(0.0546+0.0267)-0.0730(0.0527+0.0203) sod-mse=0.0136(0.0100) gcn-mse=0.0157(0.0130) gcn-final-mse=0.0122(0.0295)
2020-08-06 06:57:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 07:03:30 8000-10553 loss=0.0448(0.0315+0.0133)-0.0730(0.0527+0.0203) sod-mse=0.0059(0.0100) gcn-mse=0.0059(0.0129) gcn-final-mse=0.0122(0.0295)
2020-08-06 07:09:04 9000-10553 loss=0.0364(0.0280+0.0084)-0.0728(0.0526+0.0202) sod-mse=0.0063(0.0100) gcn-mse=0.0129(0.0129) gcn-final-mse=0.0122(0.0295)
2020-08-06 07:14:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 07:14:39 10000-10553 loss=0.0544(0.0386+0.0158)-0.0728(0.0526+0.0202) sod-mse=0.0072(0.0100) gcn-mse=0.0087(0.0129) gcn-final-mse=0.0122(0.0295)

2020-08-06 07:17:46    0-5019 loss=1.1699(0.7056+0.4643)-1.1699(0.7056+0.4643) sod-mse=0.1012(0.1012) gcn-mse=0.1084(0.1084) gcn-final-mse=0.0996(0.1118)
2020-08-06 07:20:11 1000-5019 loss=0.0269(0.0225+0.0045)-0.4063(0.1971+0.2092) sod-mse=0.0035(0.0514) gcn-mse=0.0046(0.0546) gcn-final-mse=0.0544(0.0678)
2020-08-06 07:22:36 2000-5019 loss=0.9562(0.4717+0.4845)-0.4140(0.2002+0.2138) sod-mse=0.0906(0.0526) gcn-mse=0.0899(0.0560) gcn-final-mse=0.0558(0.0691)
2020-08-06 07:25:01 3000-5019 loss=0.0437(0.0331+0.0105)-0.4110(0.1981+0.2129) sod-mse=0.0047(0.0525) gcn-mse=0.0061(0.0559) gcn-final-mse=0.0557(0.0690)
2020-08-06 07:26:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 07:27:27 4000-5019 loss=0.0981(0.0687+0.0294)-0.4059(0.1961+0.2098) sod-mse=0.0141(0.0521) gcn-mse=0.0143(0.0555) gcn-final-mse=0.0553(0.0686)
2020-08-06 07:28:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 07:29:52 5000-5019 loss=1.5523(0.6356+0.9166)-0.4034(0.1950+0.2084) sod-mse=0.0981(0.0520) gcn-mse=0.1049(0.0555) gcn-final-mse=0.0552(0.0685)
2020-08-06 07:29:55 E:37, Train sod-mae-score=0.0100-0.9847 gcn-mae-score=0.0129-0.9557 gcn-final-mse-score=0.0122-0.9582(0.0295/0.9582) loss=0.0727(0.0526+0.0201)
2020-08-06 07:29:55 E:37, Test  sod-mae-score=0.0520-0.8470 gcn-mae-score=0.0555-0.7869 gcn-final-mse-score=0.0552-0.7927(0.0685/0.7927) loss=0.4032(0.1950+0.2083)

2020-08-06 07:29:55 Start Epoch 38
2020-08-06 07:29:55 Epoch:38,lr=0.0000
2020-08-06 07:29:56    0-10553 loss=0.0815(0.0613+0.0202)-0.0815(0.0613+0.0202) sod-mse=0.0107(0.0107) gcn-mse=0.0132(0.0132) gcn-final-mse=0.0127(0.0349)
2020-08-06 07:35:29 1000-10553 loss=0.0817(0.0628+0.0189)-0.0719(0.0522+0.0197) sod-mse=0.0095(0.0098) gcn-mse=0.0137(0.0127) gcn-final-mse=0.0119(0.0293)
2020-08-06 07:41:03 2000-10553 loss=0.0839(0.0624+0.0215)-0.0709(0.0516+0.0194) sod-mse=0.0103(0.0096) gcn-mse=0.0125(0.0125) gcn-final-mse=0.0118(0.0291)
2020-08-06 07:46:34 3000-10553 loss=0.0692(0.0556+0.0136)-0.0724(0.0525+0.0200) sod-mse=0.0061(0.0098) gcn-mse=0.0091(0.0129) gcn-final-mse=0.0121(0.0294)
2020-08-06 07:52:06 4000-10553 loss=0.0308(0.0240+0.0068)-0.0723(0.0524+0.0199) sod-mse=0.0031(0.0099) gcn-mse=0.0042(0.0129) gcn-final-mse=0.0121(0.0295)
2020-08-06 07:57:38 5000-10553 loss=0.0444(0.0323+0.0121)-0.0725(0.0525+0.0200) sod-mse=0.0071(0.0099) gcn-mse=0.0109(0.0129) gcn-final-mse=0.0122(0.0295)
2020-08-06 07:58:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 08:03:13 6000-10553 loss=0.0675(0.0531+0.0145)-0.0725(0.0525+0.0200) sod-mse=0.0082(0.0099) gcn-mse=0.0115(0.0129) gcn-final-mse=0.0122(0.0295)
2020-08-06 08:06:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 08:08:45 7000-10553 loss=0.0276(0.0227+0.0049)-0.0725(0.0525+0.0199) sod-mse=0.0036(0.0099) gcn-mse=0.0030(0.0129) gcn-final-mse=0.0121(0.0295)
2020-08-06 08:14:16 8000-10553 loss=0.0915(0.0641+0.0274)-0.0724(0.0524+0.0200) sod-mse=0.0154(0.0099) gcn-mse=0.0207(0.0128) gcn-final-mse=0.0121(0.0294)
2020-08-06 08:18:05 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 08:19:48 9000-10553 loss=0.0611(0.0462+0.0148)-0.0722(0.0523+0.0199) sod-mse=0.0072(0.0098) gcn-mse=0.0095(0.0128) gcn-final-mse=0.0120(0.0293)
2020-08-06 08:25:21 10000-10553 loss=0.0847(0.0654+0.0193)-0.0725(0.0525+0.0200) sod-mse=0.0090(0.0099) gcn-mse=0.0129(0.0128) gcn-final-mse=0.0121(0.0294)

2020-08-06 08:28:24    0-5019 loss=1.1668(0.7107+0.4561)-1.1668(0.7107+0.4561) sod-mse=0.1012(0.1012) gcn-mse=0.1086(0.1086) gcn-final-mse=0.0999(0.1123)
2020-08-06 08:30:52 1000-5019 loss=0.0270(0.0225+0.0045)-0.4046(0.1972+0.2074) sod-mse=0.0036(0.0516) gcn-mse=0.0046(0.0547) gcn-final-mse=0.0545(0.0679)
2020-08-06 08:33:20 2000-5019 loss=0.9490(0.4703+0.4786)-0.4123(0.2003+0.2120) sod-mse=0.0911(0.0527) gcn-mse=0.0900(0.0561) gcn-final-mse=0.0558(0.0692)
2020-08-06 08:35:47 3000-5019 loss=0.0437(0.0331+0.0106)-0.4094(0.1982+0.2112) sod-mse=0.0048(0.0526) gcn-mse=0.0062(0.0560) gcn-final-mse=0.0558(0.0691)
2020-08-06 08:37:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 08:38:15 4000-5019 loss=0.0979(0.0687+0.0292)-0.4042(0.1961+0.2081) sod-mse=0.0141(0.0522) gcn-mse=0.0143(0.0556) gcn-final-mse=0.0554(0.0688)
2020-08-06 08:38:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 08:40:41 5000-5019 loss=1.5317(0.6355+0.8962)-0.4018(0.1950+0.2067) sod-mse=0.0975(0.0521) gcn-mse=0.1049(0.0555) gcn-final-mse=0.0553(0.0686)
2020-08-06 08:40:44 E:38, Train sod-mae-score=0.0099-0.9848 gcn-mae-score=0.0129-0.9557 gcn-final-mse-score=0.0121-0.9582(0.0294/0.9582) loss=0.0725(0.0525+0.0200)
2020-08-06 08:40:44 E:38, Test  sod-mae-score=0.0521-0.8467 gcn-mae-score=0.0555-0.7868 gcn-final-mse-score=0.0553-0.7926(0.0686/0.7926) loss=0.4016(0.1950+0.2066)

2020-08-06 08:40:44 Start Epoch 39
2020-08-06 08:40:44 Epoch:39,lr=0.0000
2020-08-06 08:40:45    0-10553 loss=0.1108(0.0887+0.0221)-0.1108(0.0887+0.0221) sod-mse=0.0111(0.0111) gcn-mse=0.0167(0.0167) gcn-final-mse=0.0151(0.0529)
2020-08-06 08:43:01 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 08:46:21 1000-10553 loss=0.0756(0.0606+0.0150)-0.0720(0.0521+0.0199) sod-mse=0.0104(0.0100) gcn-mse=0.0146(0.0128) gcn-final-mse=0.0120(0.0294)
2020-08-06 08:51:57 2000-10553 loss=0.1012(0.0681+0.0331)-0.0731(0.0526+0.0206) sod-mse=0.0192(0.0100) gcn-mse=0.0196(0.0128) gcn-final-mse=0.0121(0.0293)
2020-08-06 08:52:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 08:57:30 3000-10553 loss=0.0640(0.0463+0.0177)-0.0731(0.0528+0.0203) sod-mse=0.0092(0.0100) gcn-mse=0.0147(0.0129) gcn-final-mse=0.0121(0.0295)
2020-08-06 09:00:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 09:03:04 4000-10553 loss=0.0391(0.0307+0.0085)-0.0729(0.0528+0.0202) sod-mse=0.0039(0.0099) gcn-mse=0.0047(0.0129) gcn-final-mse=0.0122(0.0296)
2020-08-06 09:08:38 5000-10553 loss=0.0958(0.0606+0.0352)-0.0728(0.0527+0.0201) sod-mse=0.0123(0.0099) gcn-mse=0.0159(0.0128) gcn-final-mse=0.0121(0.0295)
2020-08-06 09:14:13 6000-10553 loss=0.0628(0.0459+0.0169)-0.0726(0.0526+0.0200) sod-mse=0.0088(0.0099) gcn-mse=0.0130(0.0128) gcn-final-mse=0.0121(0.0295)
2020-08-06 09:19:49 7000-10553 loss=0.0893(0.0700+0.0193)-0.0726(0.0526+0.0200) sod-mse=0.0083(0.0099) gcn-mse=0.0145(0.0129) gcn-final-mse=0.0121(0.0295)
2020-08-06 09:25:21 8000-10553 loss=0.1198(0.0697+0.0501)-0.0724(0.0525+0.0199) sod-mse=0.0180(0.0099) gcn-mse=0.0182(0.0128) gcn-final-mse=0.0121(0.0294)
2020-08-06 09:30:58 9000-10553 loss=0.0744(0.0504+0.0241)-0.0724(0.0524+0.0200) sod-mse=0.0111(0.0099) gcn-mse=0.0095(0.0128) gcn-final-mse=0.0121(0.0294)
2020-08-06 09:36:33 10000-10553 loss=0.0424(0.0321+0.0103)-0.0723(0.0524+0.0199) sod-mse=0.0050(0.0099) gcn-mse=0.0053(0.0128) gcn-final-mse=0.0121(0.0294)

2020-08-06 09:39:39    0-5019 loss=1.1771(0.7107+0.4665)-1.1771(0.7107+0.4665) sod-mse=0.1018(0.1018) gcn-mse=0.1092(0.1092) gcn-final-mse=0.1006(0.1128)
2020-08-06 09:42:04 1000-5019 loss=0.0269(0.0224+0.0044)-0.4110(0.1985+0.2125) sod-mse=0.0035(0.0513) gcn-mse=0.0045(0.0544) gcn-final-mse=0.0542(0.0675)
2020-08-06 09:44:28 2000-5019 loss=0.9566(0.4743+0.4823)-0.4185(0.2016+0.2169) sod-mse=0.0904(0.0524) gcn-mse=0.0894(0.0558) gcn-final-mse=0.0556(0.0688)
2020-08-06 09:46:52 3000-5019 loss=0.0438(0.0331+0.0106)-0.4154(0.1993+0.2160) sod-mse=0.0047(0.0522) gcn-mse=0.0061(0.0557) gcn-final-mse=0.0555(0.0687)
2020-08-06 09:48:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 09:49:17 4000-5019 loss=0.0979(0.0686+0.0293)-0.4099(0.1972+0.2127) sod-mse=0.0140(0.0519) gcn-mse=0.0142(0.0553) gcn-final-mse=0.0551(0.0683)
2020-08-06 09:49:59 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 09:51:42 5000-5019 loss=1.5681(0.6444+0.9237)-0.4075(0.1961+0.2114) sod-mse=0.0963(0.0518) gcn-mse=0.1045(0.0552) gcn-final-mse=0.0550(0.0682)
2020-08-06 09:51:44 E:39, Train sod-mae-score=0.0098-0.9849 gcn-mae-score=0.0128-0.9558 gcn-final-mse-score=0.0120-0.9582(0.0294/0.9582) loss=0.0723(0.0524+0.0199)
2020-08-06 09:51:44 E:39, Test  sod-mae-score=0.0518-0.8473 gcn-mae-score=0.0552-0.7877 gcn-final-mse-score=0.0550-0.7935(0.0682/0.7935) loss=0.4073(0.1961+0.2112)

2020-08-06 09:51:44 Start Epoch 40
2020-08-06 09:51:44 Epoch:40,lr=0.0000
2020-08-06 09:51:46    0-10553 loss=0.0555(0.0401+0.0155)-0.0555(0.0401+0.0155) sod-mse=0.0071(0.0071) gcn-mse=0.0080(0.0080) gcn-final-mse=0.0074(0.0192)
2020-08-06 09:57:18 1000-10553 loss=0.0889(0.0611+0.0278)-0.0717(0.0520+0.0197) sod-mse=0.0114(0.0100) gcn-mse=0.0139(0.0129) gcn-final-mse=0.0121(0.0293)
2020-08-06 09:58:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 10:02:54 2000-10553 loss=0.0491(0.0364+0.0127)-0.0733(0.0529+0.0204) sod-mse=0.0067(0.0101) gcn-mse=0.0091(0.0130) gcn-final-mse=0.0123(0.0295)
2020-08-06 10:08:25 3000-10553 loss=0.0472(0.0368+0.0104)-0.0724(0.0525+0.0200) sod-mse=0.0046(0.0099) gcn-mse=0.0050(0.0129) gcn-final-mse=0.0121(0.0294)
2020-08-06 10:13:59 4000-10553 loss=0.0481(0.0372+0.0109)-0.0723(0.0524+0.0199) sod-mse=0.0049(0.0099) gcn-mse=0.0082(0.0129) gcn-final-mse=0.0121(0.0294)
2020-08-06 10:19:32 5000-10553 loss=0.0826(0.0610+0.0216)-0.0724(0.0525+0.0199) sod-mse=0.0115(0.0099) gcn-mse=0.0129(0.0128) gcn-final-mse=0.0121(0.0294)
2020-08-06 10:20:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 10:25:03 6000-10553 loss=0.0738(0.0478+0.0260)-0.0720(0.0522+0.0198) sod-mse=0.0091(0.0098) gcn-mse=0.0095(0.0127) gcn-final-mse=0.0120(0.0292)
2020-08-06 10:30:36 7000-10553 loss=0.0957(0.0677+0.0280)-0.0717(0.0521+0.0196) sod-mse=0.0146(0.0097) gcn-mse=0.0161(0.0127) gcn-final-mse=0.0119(0.0292)
2020-08-06 10:36:10 8000-10553 loss=0.1614(0.0991+0.0622)-0.0720(0.0522+0.0197) sod-mse=0.0299(0.0098) gcn-mse=0.0321(0.0127) gcn-final-mse=0.0119(0.0293)
2020-08-06 10:41:41 9000-10553 loss=0.0700(0.0489+0.0212)-0.0721(0.0523+0.0198) sod-mse=0.0114(0.0098) gcn-mse=0.0106(0.0127) gcn-final-mse=0.0120(0.0293)
2020-08-06 10:46:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 10:47:13 10000-10553 loss=0.1602(0.1137+0.0465)-0.0720(0.0523+0.0198) sod-mse=0.0272(0.0098) gcn-mse=0.0429(0.0127) gcn-final-mse=0.0120(0.0293)

2020-08-06 10:50:19    0-5019 loss=1.1944(0.7178+0.4766)-1.1944(0.7178+0.4766) sod-mse=0.1039(0.1039) gcn-mse=0.1108(0.1108) gcn-final-mse=0.1024(0.1143)
2020-08-06 10:52:47 1000-5019 loss=0.0269(0.0224+0.0045)-0.4113(0.1986+0.2127) sod-mse=0.0036(0.0512) gcn-mse=0.0045(0.0544) gcn-final-mse=0.0541(0.0676)
2020-08-06 10:55:14 2000-5019 loss=0.9739(0.4807+0.4932)-0.4177(0.2014+0.2163) sod-mse=0.0925(0.0523) gcn-mse=0.0903(0.0557) gcn-final-mse=0.0555(0.0688)
2020-08-06 10:57:41 3000-5019 loss=0.0436(0.0331+0.0106)-0.4145(0.1991+0.2154) sod-mse=0.0048(0.0521) gcn-mse=0.0061(0.0555) gcn-final-mse=0.0554(0.0687)
2020-08-06 10:59:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 11:00:09 4000-5019 loss=0.0976(0.0685+0.0291)-0.4088(0.1969+0.2120) sod-mse=0.0140(0.0517) gcn-mse=0.0142(0.0552) gcn-final-mse=0.0550(0.0683)
2020-08-06 11:00:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 11:02:35 5000-5019 loss=1.5769(0.6563+0.9205)-0.4066(0.1958+0.2108) sod-mse=0.0955(0.0517) gcn-mse=0.1044(0.0551) gcn-final-mse=0.0548(0.0682)
2020-08-06 11:02:37 E:40, Train sod-mae-score=0.0098-0.9849 gcn-mae-score=0.0127-0.9557 gcn-final-mse-score=0.0120-0.9582(0.0293/0.9582) loss=0.0722(0.0523+0.0199)
2020-08-06 11:02:37 E:40, Test  sod-mae-score=0.0516-0.8481 gcn-mae-score=0.0551-0.7881 gcn-final-mse-score=0.0548-0.7940(0.0682/0.7940) loss=0.4064(0.1957+0.2107)

2020-08-06 11:02:37 Start Epoch 41
2020-08-06 11:02:37 Epoch:41,lr=0.0000
2020-08-06 11:02:39    0-10553 loss=0.0539(0.0384+0.0156)-0.0539(0.0384+0.0156) sod-mse=0.0069(0.0069) gcn-mse=0.0075(0.0075) gcn-final-mse=0.0064(0.0193)
2020-08-06 11:08:15 1000-10553 loss=0.0685(0.0407+0.0278)-0.0740(0.0534+0.0206) sod-mse=0.0135(0.0100) gcn-mse=0.0113(0.0129) gcn-final-mse=0.0121(0.0294)
2020-08-06 11:10:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 11:13:52 2000-10553 loss=0.0396(0.0291+0.0105)-0.0723(0.0524+0.0199) sod-mse=0.0058(0.0098) gcn-mse=0.0090(0.0127) gcn-final-mse=0.0119(0.0292)
2020-08-06 11:19:26 3000-10553 loss=0.1069(0.0740+0.0328)-0.0711(0.0517+0.0194) sod-mse=0.0162(0.0096) gcn-mse=0.0175(0.0125) gcn-final-mse=0.0117(0.0290)
2020-08-06 11:22:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 11:25:00 4000-10553 loss=0.6045(0.3177+0.2868)-0.0712(0.0517+0.0194) sod-mse=0.0449(0.0096) gcn-mse=0.0382(0.0125) gcn-final-mse=0.0117(0.0289)
2020-08-06 11:30:40 5000-10553 loss=0.1043(0.0753+0.0290)-0.0712(0.0517+0.0194) sod-mse=0.0157(0.0096) gcn-mse=0.0194(0.0125) gcn-final-mse=0.0117(0.0291)
2020-08-06 11:36:15 6000-10553 loss=0.0557(0.0393+0.0164)-0.0716(0.0520+0.0196) sod-mse=0.0093(0.0097) gcn-mse=0.0108(0.0126) gcn-final-mse=0.0118(0.0291)
2020-08-06 11:41:50 7000-10553 loss=0.0611(0.0439+0.0172)-0.0718(0.0521+0.0197) sod-mse=0.0085(0.0097) gcn-mse=0.0119(0.0126) gcn-final-mse=0.0118(0.0292)
2020-08-06 11:47:25 8000-10553 loss=0.0411(0.0312+0.0099)-0.0718(0.0521+0.0197) sod-mse=0.0049(0.0097) gcn-mse=0.0085(0.0126) gcn-final-mse=0.0119(0.0292)
2020-08-06 11:53:00 9000-10553 loss=0.0555(0.0463+0.0092)-0.0718(0.0521+0.0197) sod-mse=0.0068(0.0098) gcn-mse=0.0126(0.0127) gcn-final-mse=0.0119(0.0293)
2020-08-06 11:55:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 11:58:35 10000-10553 loss=0.0686(0.0490+0.0196)-0.0720(0.0522+0.0198) sod-mse=0.0094(0.0098) gcn-mse=0.0098(0.0127) gcn-final-mse=0.0119(0.0293)

2020-08-06 12:01:42    0-5019 loss=1.1868(0.7144+0.4724)-1.1868(0.7144+0.4724) sod-mse=0.1025(0.1025) gcn-mse=0.1100(0.1100) gcn-final-mse=0.1015(0.1134)
2020-08-06 12:04:07 1000-5019 loss=0.0269(0.0224+0.0045)-0.4130(0.1992+0.2138) sod-mse=0.0036(0.0512) gcn-mse=0.0045(0.0545) gcn-final-mse=0.0543(0.0677)
2020-08-06 12:06:32 2000-5019 loss=0.9567(0.4753+0.4814)-0.4206(0.2023+0.2183) sod-mse=0.0908(0.0524) gcn-mse=0.0894(0.0559) gcn-final-mse=0.0556(0.0689)
2020-08-06 12:08:56 3000-5019 loss=0.0437(0.0332+0.0106)-0.4174(0.2000+0.2174) sod-mse=0.0047(0.0521) gcn-mse=0.0061(0.0557) gcn-final-mse=0.0555(0.0688)
2020-08-06 12:10:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 12:11:20 4000-5019 loss=0.0976(0.0685+0.0291)-0.4118(0.1978+0.2140) sod-mse=0.0140(0.0518) gcn-mse=0.0141(0.0553) gcn-final-mse=0.0551(0.0684)
2020-08-06 12:12:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 12:13:44 5000-5019 loss=1.5659(0.6513+0.9146)-0.4096(0.1968+0.2128) sod-mse=0.0969(0.0517) gcn-mse=0.1048(0.0553) gcn-final-mse=0.0550(0.0683)
2020-08-06 12:13:47 E:41, Train sod-mae-score=0.0098-0.9851 gcn-mae-score=0.0127-0.9562 gcn-final-mse-score=0.0119-0.9586(0.0292/0.9586) loss=0.0719(0.0522+0.0197)
2020-08-06 12:13:47 E:41, Test  sod-mae-score=0.0517-0.8470 gcn-mae-score=0.0553-0.7872 gcn-final-mse-score=0.0550-0.7930(0.0683/0.7930) loss=0.4094(0.1967+0.2127)

2020-08-06 12:13:47 Start Epoch 42
2020-08-06 12:13:47 Epoch:42,lr=0.0000
2020-08-06 12:13:48    0-10553 loss=0.0406(0.0308+0.0098)-0.0406(0.0308+0.0098) sod-mse=0.0039(0.0039) gcn-mse=0.0036(0.0036) gcn-final-mse=0.0036(0.0215)
2020-08-06 12:19:20 1000-10553 loss=0.0263(0.0188+0.0076)-0.0724(0.0523+0.0201) sod-mse=0.0047(0.0098) gcn-mse=0.0067(0.0126) gcn-final-mse=0.0119(0.0293)
2020-08-06 12:21:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 12:24:53 2000-10553 loss=0.0617(0.0466+0.0151)-0.0722(0.0522+0.0200) sod-mse=0.0062(0.0098) gcn-mse=0.0089(0.0126) gcn-final-mse=0.0119(0.0293)
2020-08-06 12:30:24 3000-10553 loss=0.0487(0.0380+0.0107)-0.0718(0.0520+0.0198) sod-mse=0.0067(0.0097) gcn-mse=0.0087(0.0125) gcn-final-mse=0.0118(0.0291)
2020-08-06 12:32:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 12:35:55 4000-10553 loss=0.1148(0.0821+0.0328)-0.0720(0.0521+0.0199) sod-mse=0.0188(0.0098) gcn-mse=0.0227(0.0126) gcn-final-mse=0.0119(0.0292)
2020-08-06 12:41:26 5000-10553 loss=0.0527(0.0420+0.0107)-0.0725(0.0524+0.0201) sod-mse=0.0050(0.0099) gcn-mse=0.0083(0.0127) gcn-final-mse=0.0119(0.0293)
2020-08-06 12:46:56 6000-10553 loss=0.0571(0.0455+0.0116)-0.0723(0.0523+0.0200) sod-mse=0.0087(0.0098) gcn-mse=0.0111(0.0127) gcn-final-mse=0.0119(0.0292)
2020-08-06 12:52:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 12:52:29 7000-10553 loss=0.0404(0.0343+0.0062)-0.0720(0.0522+0.0198) sod-mse=0.0022(0.0098) gcn-mse=0.0037(0.0127) gcn-final-mse=0.0119(0.0292)
2020-08-06 12:57:58 8000-10553 loss=0.1471(0.0962+0.0509)-0.0719(0.0521+0.0198) sod-mse=0.0199(0.0098) gcn-mse=0.0203(0.0126) gcn-final-mse=0.0119(0.0292)
2020-08-06 13:03:31 9000-10553 loss=0.0697(0.0530+0.0167)-0.0721(0.0522+0.0199) sod-mse=0.0086(0.0098) gcn-mse=0.0120(0.0127) gcn-final-mse=0.0119(0.0292)
2020-08-06 13:09:02 10000-10553 loss=0.0790(0.0609+0.0181)-0.0720(0.0522+0.0198) sod-mse=0.0089(0.0098) gcn-mse=0.0138(0.0127) gcn-final-mse=0.0119(0.0292)

2020-08-06 13:12:05    0-5019 loss=1.1813(0.7097+0.4715)-1.1813(0.7097+0.4715) sod-mse=0.1015(0.1015) gcn-mse=0.1089(0.1089) gcn-final-mse=0.1003(0.1122)
2020-08-06 13:14:33 1000-5019 loss=0.0267(0.0223+0.0044)-0.4185(0.2003+0.2181) sod-mse=0.0035(0.0514) gcn-mse=0.0044(0.0545) gcn-final-mse=0.0543(0.0676)
2020-08-06 13:17:00 2000-5019 loss=0.9631(0.4747+0.4884)-0.4261(0.2035+0.2226) sod-mse=0.0903(0.0525) gcn-mse=0.0890(0.0559) gcn-final-mse=0.0557(0.0689)
2020-08-06 13:19:26 3000-5019 loss=0.0437(0.0331+0.0106)-0.4228(0.2011+0.2216) sod-mse=0.0047(0.0523) gcn-mse=0.0061(0.0558) gcn-final-mse=0.0556(0.0688)
2020-08-06 13:20:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 13:21:53 4000-5019 loss=0.0976(0.0685+0.0291)-0.4171(0.1989+0.2181) sod-mse=0.0139(0.0519) gcn-mse=0.0141(0.0554) gcn-final-mse=0.0552(0.0684)
2020-08-06 13:22:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 13:24:19 5000-5019 loss=1.5796(0.6502+0.9294)-0.4146(0.1979+0.2168) sod-mse=0.0988(0.0518) gcn-mse=0.1054(0.0553) gcn-final-mse=0.0551(0.0682)
2020-08-06 13:24:21 E:42, Train sod-mae-score=0.0098-0.9850 gcn-mae-score=0.0127-0.9560 gcn-final-mse-score=0.0119-0.9585(0.0292/0.9585) loss=0.0719(0.0522+0.0198)
2020-08-06 13:24:21 E:42, Test  sod-mae-score=0.0518-0.8462 gcn-mae-score=0.0553-0.7873 gcn-final-mse-score=0.0551-0.7930(0.0682/0.7930) loss=0.4144(0.1978+0.2167)

2020-08-06 13:24:21 Start Epoch 43
2020-08-06 13:24:21 Epoch:43,lr=0.0000
2020-08-06 13:24:23    0-10553 loss=0.0774(0.0563+0.0211)-0.0774(0.0563+0.0211) sod-mse=0.0106(0.0106) gcn-mse=0.0131(0.0131) gcn-final-mse=0.0121(0.0355)
2020-08-06 13:26:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 13:29:59 1000-10553 loss=0.0622(0.0464+0.0158)-0.0719(0.0523+0.0197) sod-mse=0.0084(0.0097) gcn-mse=0.0142(0.0126) gcn-final-mse=0.0119(0.0294)
2020-08-06 13:35:32 2000-10553 loss=0.1375(0.1046+0.0329)-0.0712(0.0518+0.0193) sod-mse=0.0156(0.0095) gcn-mse=0.0195(0.0126) gcn-final-mse=0.0118(0.0291)
2020-08-06 13:41:06 3000-10553 loss=0.0835(0.0647+0.0188)-0.0710(0.0517+0.0193) sod-mse=0.0129(0.0095) gcn-mse=0.0153(0.0125) gcn-final-mse=0.0117(0.0290)
2020-08-06 13:43:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 13:46:40 4000-10553 loss=0.0689(0.0562+0.0128)-0.0716(0.0521+0.0195) sod-mse=0.0092(0.0096) gcn-mse=0.0136(0.0125) gcn-final-mse=0.0118(0.0291)
2020-08-06 13:52:15 5000-10553 loss=0.0657(0.0481+0.0177)-0.0714(0.0519+0.0195) sod-mse=0.0075(0.0096) gcn-mse=0.0077(0.0125) gcn-final-mse=0.0118(0.0290)
2020-08-06 13:57:51 6000-10553 loss=0.1193(0.0750+0.0443)-0.0713(0.0519+0.0194) sod-mse=0.0181(0.0096) gcn-mse=0.0176(0.0125) gcn-final-mse=0.0118(0.0291)
2020-08-06 14:03:25 7000-10553 loss=0.0456(0.0295+0.0161)-0.0714(0.0520+0.0195) sod-mse=0.0071(0.0096) gcn-mse=0.0067(0.0125) gcn-final-mse=0.0118(0.0291)
2020-08-06 14:08:59 8000-10553 loss=0.0697(0.0534+0.0163)-0.0716(0.0520+0.0196) sod-mse=0.0070(0.0096) gcn-mse=0.0090(0.0126) gcn-final-mse=0.0118(0.0292)
2020-08-06 14:14:34 9000-10553 loss=0.0915(0.0694+0.0221)-0.0717(0.0521+0.0196) sod-mse=0.0133(0.0097) gcn-mse=0.0216(0.0126) gcn-final-mse=0.0119(0.0292)
2020-08-06 14:14:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 14:20:09 10000-10553 loss=0.0554(0.0442+0.0113)-0.0717(0.0521+0.0196) sod-mse=0.0076(0.0097) gcn-mse=0.0086(0.0126) gcn-final-mse=0.0119(0.0292)

2020-08-06 14:23:16    0-5019 loss=1.1759(0.7075+0.4684)-1.1759(0.7075+0.4684) sod-mse=0.1018(0.1018) gcn-mse=0.1088(0.1088) gcn-final-mse=0.1002(0.1122)
2020-08-06 14:25:43 1000-5019 loss=0.0268(0.0224+0.0044)-0.4150(0.1994+0.2155) sod-mse=0.0035(0.0513) gcn-mse=0.0045(0.0544) gcn-final-mse=0.0541(0.0676)
2020-08-06 14:28:09 2000-5019 loss=0.9724(0.4799+0.4925)-0.4224(0.2026+0.2198) sod-mse=0.0915(0.0524) gcn-mse=0.0898(0.0558) gcn-final-mse=0.0555(0.0688)
2020-08-06 14:30:36 3000-5019 loss=0.0438(0.0332+0.0106)-0.4193(0.2004+0.2190) sod-mse=0.0048(0.0522) gcn-mse=0.0062(0.0556) gcn-final-mse=0.0554(0.0687)
2020-08-06 14:31:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 14:33:02 4000-5019 loss=0.0975(0.0684+0.0291)-0.4137(0.1982+0.2156) sod-mse=0.0140(0.0518) gcn-mse=0.0140(0.0553) gcn-final-mse=0.0551(0.0684)
2020-08-06 14:33:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 14:35:29 5000-5019 loss=1.5878(0.6662+0.9216)-0.4114(0.1971+0.2143) sod-mse=0.0981(0.0518) gcn-mse=0.1050(0.0552) gcn-final-mse=0.0549(0.0682)
2020-08-06 14:35:31 E:43, Train sod-mae-score=0.0097-0.9851 gcn-mae-score=0.0126-0.9562 gcn-final-mse-score=0.0119-0.9587(0.0292/0.9587) loss=0.0718(0.0521+0.0196)
2020-08-06 14:35:31 E:43, Test  sod-mae-score=0.0517-0.8465 gcn-mae-score=0.0552-0.7870 gcn-final-mse-score=0.0549-0.7928(0.0682/0.7928) loss=0.4112(0.1970+0.2142)

2020-08-06 14:35:31 Start Epoch 44
2020-08-06 14:35:31 Epoch:44,lr=0.0000
2020-08-06 14:35:33    0-10553 loss=0.0463(0.0375+0.0088)-0.0463(0.0375+0.0088) sod-mse=0.0066(0.0066) gcn-mse=0.0103(0.0103) gcn-final-mse=0.0114(0.0247)
2020-08-06 14:41:09 1000-10553 loss=0.1192(0.0840+0.0352)-0.0703(0.0512+0.0192) sod-mse=0.0193(0.0096) gcn-mse=0.0233(0.0123) gcn-final-mse=0.0116(0.0289)
2020-08-06 14:42:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 14:42:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 14:46:44 2000-10553 loss=0.0804(0.0595+0.0209)-0.0711(0.0518+0.0193) sod-mse=0.0105(0.0096) gcn-mse=0.0150(0.0124) gcn-final-mse=0.0117(0.0291)
2020-08-06 14:52:19 3000-10553 loss=0.0724(0.0545+0.0180)-0.0712(0.0518+0.0193) sod-mse=0.0072(0.0096) gcn-mse=0.0102(0.0124) gcn-final-mse=0.0117(0.0290)
2020-08-06 14:57:54 4000-10553 loss=0.0485(0.0352+0.0133)-0.0714(0.0519+0.0195) sod-mse=0.0073(0.0096) gcn-mse=0.0085(0.0125) gcn-final-mse=0.0117(0.0291)
2020-08-06 15:03:27 5000-10553 loss=0.0545(0.0437+0.0108)-0.0718(0.0522+0.0196) sod-mse=0.0047(0.0096) gcn-mse=0.0152(0.0125) gcn-final-mse=0.0118(0.0293)
2020-08-06 15:09:02 6000-10553 loss=0.1079(0.0741+0.0338)-0.0720(0.0523+0.0197) sod-mse=0.0165(0.0097) gcn-mse=0.0195(0.0126) gcn-final-mse=0.0118(0.0292)
2020-08-06 15:14:37 7000-10553 loss=0.0962(0.0751+0.0212)-0.0718(0.0522+0.0196) sod-mse=0.0101(0.0096) gcn-mse=0.0140(0.0126) gcn-final-mse=0.0118(0.0292)
2020-08-06 15:18:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 15:20:15 8000-10553 loss=0.1032(0.0794+0.0238)-0.0717(0.0521+0.0196) sod-mse=0.0149(0.0096) gcn-mse=0.0236(0.0126) gcn-final-mse=0.0118(0.0292)
2020-08-06 15:25:51 9000-10553 loss=0.0993(0.0707+0.0286)-0.0717(0.0521+0.0196) sod-mse=0.0116(0.0097) gcn-mse=0.0164(0.0126) gcn-final-mse=0.0118(0.0292)
2020-08-06 15:31:26 10000-10553 loss=0.0618(0.0500+0.0118)-0.0717(0.0521+0.0196) sod-mse=0.0055(0.0097) gcn-mse=0.0081(0.0126) gcn-final-mse=0.0118(0.0292)

2020-08-06 15:34:34    0-5019 loss=1.2031(0.7198+0.4833)-1.2031(0.7198+0.4833) sod-mse=0.1026(0.1026) gcn-mse=0.1100(0.1100) gcn-final-mse=0.1014(0.1134)
2020-08-06 15:36:59 1000-5019 loss=0.0269(0.0224+0.0045)-0.4189(0.1996+0.2192) sod-mse=0.0036(0.0513) gcn-mse=0.0045(0.0544) gcn-final-mse=0.0542(0.0676)
2020-08-06 15:39:23 2000-5019 loss=0.9787(0.4780+0.5007)-0.4259(0.2027+0.2232) sod-mse=0.0908(0.0524) gcn-mse=0.0898(0.0558) gcn-final-mse=0.0555(0.0689)
2020-08-06 15:41:48 3000-5019 loss=0.0437(0.0331+0.0106)-0.4228(0.2005+0.2223) sod-mse=0.0047(0.0521) gcn-mse=0.0061(0.0556) gcn-final-mse=0.0554(0.0688)
2020-08-06 15:43:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 15:44:13 4000-5019 loss=0.0972(0.0684+0.0288)-0.4170(0.1982+0.2187) sod-mse=0.0139(0.0518) gcn-mse=0.0140(0.0553) gcn-final-mse=0.0551(0.0684)
2020-08-06 15:44:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 15:46:37 5000-5019 loss=1.6228(0.6624+0.9604)-0.4146(0.1972+0.2174) sod-mse=0.0984(0.0517) gcn-mse=0.1056(0.0552) gcn-final-mse=0.0549(0.0683)
2020-08-06 15:46:40 E:44, Train sod-mae-score=0.0097-0.9851 gcn-mae-score=0.0126-0.9559 gcn-final-mse-score=0.0118-0.9584(0.0292/0.9584) loss=0.0717(0.0521+0.0196)
2020-08-06 15:46:40 E:44, Test  sod-mae-score=0.0517-0.8466 gcn-mae-score=0.0552-0.7871 gcn-final-mse-score=0.0549-0.7929(0.0683/0.7929) loss=0.4144(0.1971+0.2173)

2020-08-06 15:46:40 Start Epoch 45
2020-08-06 15:46:40 Epoch:45,lr=0.0000
2020-08-06 15:46:41    0-10553 loss=0.0589(0.0453+0.0136)-0.0589(0.0453+0.0136) sod-mse=0.0095(0.0095) gcn-mse=0.0117(0.0117) gcn-final-mse=0.0129(0.0356)
2020-08-06 15:52:14 1000-10553 loss=0.0464(0.0358+0.0106)-0.0701(0.0510+0.0192) sod-mse=0.0051(0.0094) gcn-mse=0.0100(0.0123) gcn-final-mse=0.0116(0.0288)
2020-08-06 15:54:27 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 15:57:44 2000-10553 loss=0.0909(0.0625+0.0284)-0.0711(0.0516+0.0195) sod-mse=0.0171(0.0096) gcn-mse=0.0161(0.0125) gcn-final-mse=0.0117(0.0290)
2020-08-06 16:03:16 3000-10553 loss=0.1070(0.0672+0.0397)-0.0711(0.0517+0.0194) sod-mse=0.0196(0.0096) gcn-mse=0.0193(0.0124) gcn-final-mse=0.0117(0.0291)
2020-08-06 16:08:46 4000-10553 loss=0.1024(0.0610+0.0414)-0.0716(0.0519+0.0196) sod-mse=0.0186(0.0096) gcn-mse=0.0201(0.0125) gcn-final-mse=0.0118(0.0290)
2020-08-06 16:14:17 5000-10553 loss=0.0756(0.0572+0.0184)-0.0715(0.0520+0.0195) sod-mse=0.0100(0.0096) gcn-mse=0.0133(0.0125) gcn-final-mse=0.0118(0.0292)
2020-08-06 16:19:49 6000-10553 loss=0.0504(0.0393+0.0111)-0.0714(0.0519+0.0195) sod-mse=0.0052(0.0096) gcn-mse=0.0126(0.0125) gcn-final-mse=0.0118(0.0291)
2020-08-06 16:21:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 16:25:21 7000-10553 loss=0.1104(0.0896+0.0208)-0.0713(0.0518+0.0195) sod-mse=0.0113(0.0096) gcn-mse=0.0205(0.0125) gcn-final-mse=0.0117(0.0291)
2020-08-06 16:30:52 8000-10553 loss=0.0387(0.0299+0.0088)-0.0714(0.0519+0.0195) sod-mse=0.0038(0.0096) gcn-mse=0.0044(0.0125) gcn-final-mse=0.0117(0.0291)
2020-08-06 16:32:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 16:36:21 9000-10553 loss=0.0401(0.0292+0.0109)-0.0714(0.0518+0.0195) sod-mse=0.0067(0.0096) gcn-mse=0.0086(0.0125) gcn-final-mse=0.0117(0.0291)
2020-08-06 16:41:56 10000-10553 loss=0.0510(0.0370+0.0139)-0.0715(0.0519+0.0195) sod-mse=0.0081(0.0096) gcn-mse=0.0067(0.0125) gcn-final-mse=0.0117(0.0291)
