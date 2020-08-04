/home/ubuntu/anaconda3/envs/alisure36torch/bin/python /mnt/4T/ALISURE/GCN/PyTorchGCN/MyGCN/SPERunner_1_PYG_CONV_Fast_SOD_SGPU_E2E_BS1_MoreConv.py
2020-08-03 12:20:21 name:E2E2-BS1-MoreConv-1-C2PC2PC3C3_False_False_lr0001 epochs:30 ckpt:./ckpt2/dgl/1_PYG_CONV_Fast-SOD_BAS/E2E2-BS1-MoreConv-1-C2PC2PC3C3_False_False_lr0001 sp size:4 down_ratio:4 workers:16 gpu:1 has_mask:False has_residual:True is_normalize:True has_bn:True improved:True concat:True is_sgd:False weight_decay:0.0

2020-08-03 12:20:21 Cuda available with GPU: GeForce GTX 1080
2020-08-03 12:20:27 Total param: 25501376 lr_s=[[0, 0.0001], [20, 1e-05]] Optimizer=Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.0001
    weight_decay: 0.0
)
2020-08-03 12:20:27 MyGCNNet(
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
    (conv_sod_gcn2): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_sod_gcn1): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv3): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv2): ConvBlock(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv1): ConvBlock(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_sod_gcn): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv3): ConvBlock(
      (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv2): ConvBlock(
      (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (cat_conv1): ConvBlock(
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
2020-08-03 12:20:27 The number of parameters: 25501376

2020-08-03 12:20:27 Start Epoch 0
2020-08-03 12:20:27 Epoch:00,lr=0.0001
2020-08-03 12:20:28    0-10553 loss=1.3924(0.6986+0.6939)-1.3924(0.6986+0.6939) sod-mse=0.4950(0.4950) gcn-mse=0.4787(0.4787) gcn-final-mse=0.4829(0.4909)
2020-08-03 12:23:38 1000-10553 loss=0.3286(0.1782+0.1504)-0.6896(0.3387+0.3509) sod-mse=0.0990(0.2174) gcn-mse=0.1044(0.1992) gcn-final-mse=0.1995(0.2130)
2020-08-03 12:26:51 2000-10553 loss=0.1113(0.0678+0.0435)-0.5994(0.2999+0.2995) sod-mse=0.0332(0.1815) gcn-mse=0.0505(0.1722) gcn-final-mse=0.1723(0.1855)
2020-08-03 12:28:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 12:30:03 3000-10553 loss=0.4826(0.2484+0.2341)-0.5540(0.2796+0.2744) sod-mse=0.1911(0.1648) gcn-mse=0.1944(0.1583) gcn-final-mse=0.1584(0.1719)
2020-08-03 12:31:58 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 12:33:14 4000-10553 loss=0.1216(0.0838+0.0378)-0.5268(0.2683+0.2585) sod-mse=0.0225(0.1543) gcn-mse=0.0426(0.1503) gcn-final-mse=0.1503(0.1638)
2020-08-03 12:36:25 5000-10553 loss=0.1299(0.0911+0.0388)-0.5119(0.2621+0.2498) sod-mse=0.0277(0.1484) gcn-mse=0.0395(0.1455) gcn-final-mse=0.1456(0.1593)
2020-08-03 12:39:38 6000-10553 loss=0.5032(0.2833+0.2198)-0.4921(0.2535+0.2386) sod-mse=0.1527(0.1410) gcn-mse=0.1560(0.1399) gcn-final-mse=0.1400(0.1537)
2020-08-03 12:42:51 7000-10553 loss=0.3711(0.1744+0.1967)-0.4762(0.2465+0.2297) sod-mse=0.0966(0.1352) gcn-mse=0.0879(0.1350) gcn-final-mse=0.1351(0.1489)
2020-08-03 12:46:04 8000-10553 loss=0.3059(0.1512+0.1547)-0.4642(0.2413+0.2229) sod-mse=0.1179(0.1309) gcn-mse=0.0821(0.1315) gcn-final-mse=0.1315(0.1454)
2020-08-03 12:49:17 9000-10553 loss=0.1529(0.0807+0.0722)-0.4551(0.2373+0.2179) sod-mse=0.0568(0.1276) gcn-mse=0.0481(0.1287) gcn-final-mse=0.1287(0.1427)
2020-08-03 12:52:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 12:52:30 10000-10553 loss=0.4431(0.2679+0.1752)-0.4465(0.2335+0.2130) sod-mse=0.1474(0.1246) gcn-mse=0.1762(0.1260) gcn-final-mse=0.1261(0.1401)

2020-08-03 12:54:16    0-5019 loss=0.6295(0.3741+0.2554)-0.6295(0.3741+0.2554) sod-mse=0.1560(0.1560) gcn-mse=0.1696(0.1696) gcn-final-mse=0.1626(0.1747)
2020-08-03 12:55:52 1000-5019 loss=0.2225(0.1096+0.1129)-0.5213(0.2702+0.2511) sod-mse=0.1007(0.1654) gcn-mse=0.0828(0.1500) gcn-final-mse=0.1498(0.1639)
2020-08-03 12:57:28 2000-5019 loss=0.5442(0.2856+0.2587)-0.5293(0.2737+0.2556) sod-mse=0.1619(0.1674) gcn-mse=0.1419(0.1518) gcn-final-mse=0.1515(0.1656)
2020-08-03 12:59:03 3000-5019 loss=0.0992(0.0629+0.0363)-0.5375(0.2776+0.2599) sod-mse=0.0295(0.1693) gcn-mse=0.0359(0.1536) gcn-final-mse=0.1533(0.1674)
2020-08-03 12:59:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 13:00:38 4000-5019 loss=0.2489(0.1307+0.1182)-0.5379(0.2776+0.2603) sod-mse=0.0922(0.1698) gcn-mse=0.0725(0.1540) gcn-final-mse=0.1537(0.1678)
2020-08-03 13:01:06 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 13:02:13 5000-5019 loss=0.8292(0.4212+0.4080)-0.5386(0.2781+0.2605) sod-mse=0.2188(0.1700) gcn-mse=0.1957(0.1542) gcn-final-mse=0.1539(0.1679)
2020-08-03 13:02:14 E: 0, Train sod-mae-score=0.1229-0.8573 gcn-mae-score=0.1245-0.8365 gcn-final-mse-score=0.1246-0.8400(0.1386/0.8400) loss=0.4418(0.2314+0.2104)
2020-08-03 13:02:14 E: 0, Test  sod-mae-score=0.1699-0.7124 gcn-mae-score=0.1541-0.6455 gcn-final-mse-score=0.1538-0.6511(0.1679/0.6511) loss=0.5382(0.2779+0.2603)

2020-08-03 13:02:14 Start Epoch 1
2020-08-03 13:02:14 Epoch:01,lr=0.0001
2020-08-03 13:02:16    0-10553 loss=0.2575(0.1551+0.1024)-0.2575(0.1551+0.1024) sod-mse=0.0751(0.0751) gcn-mse=0.0769(0.0769) gcn-final-mse=0.0749(0.0875)
2020-08-03 13:05:29 1000-10553 loss=0.1517(0.0831+0.0687)-0.3486(0.1902+0.1584) sod-mse=0.0576(0.0900) gcn-mse=0.0420(0.0967) gcn-final-mse=0.0965(0.1111)
2020-08-03 13:06:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 13:08:44 2000-10553 loss=0.1415(0.0964+0.0451)-0.3383(0.1846+0.1537) sod-mse=0.0334(0.0870) gcn-mse=0.0467(0.0940) gcn-final-mse=0.0938(0.1082)
2020-08-03 13:11:57 3000-10553 loss=0.1569(0.1027+0.0543)-0.3442(0.1868+0.1574) sod-mse=0.0367(0.0889) gcn-mse=0.0547(0.0946) gcn-final-mse=0.0944(0.1087)
2020-08-03 13:15:12 4000-10553 loss=0.5278(0.3085+0.2193)-0.3394(0.1846+0.1548) sod-mse=0.1456(0.0875) gcn-mse=0.1730(0.0933) gcn-final-mse=0.0931(0.1075)
2020-08-03 13:17:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 13:17:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 13:18:27 5000-10553 loss=0.1190(0.0788+0.0402)-0.3420(0.1857+0.1563) sod-mse=0.0310(0.0883) gcn-mse=0.0428(0.0938) gcn-final-mse=0.0936(0.1079)
2020-08-03 13:21:41 6000-10553 loss=0.1639(0.0989+0.0650)-0.3370(0.1833+0.1537) sod-mse=0.0421(0.0866) gcn-mse=0.0372(0.0924) gcn-final-mse=0.0922(0.1065)
2020-08-03 13:24:58 7000-10553 loss=0.1242(0.0731+0.0511)-0.3351(0.1825+0.1526) sod-mse=0.0360(0.0861) gcn-mse=0.0346(0.0919) gcn-final-mse=0.0917(0.1060)
2020-08-03 13:28:15 8000-10553 loss=0.1143(0.0699+0.0444)-0.3328(0.1816+0.1512) sod-mse=0.0196(0.0854) gcn-mse=0.0267(0.0914) gcn-final-mse=0.0912(0.1055)
2020-08-03 13:31:29 9000-10553 loss=0.2598(0.1621+0.0977)-0.3311(0.1808+0.1503) sod-mse=0.0730(0.0849) gcn-mse=0.0863(0.0908) gcn-final-mse=0.0906(0.1049)
2020-08-03 13:34:45 10000-10553 loss=0.2748(0.1530+0.1218)-0.3302(0.1803+0.1499) sod-mse=0.0935(0.0845) gcn-mse=0.0872(0.0904) gcn-final-mse=0.0902(0.1046)

2020-08-03 13:36:33    0-5019 loss=0.7710(0.4523+0.3187)-0.7710(0.4523+0.3187) sod-mse=0.1103(0.1103) gcn-mse=0.1277(0.1277) gcn-final-mse=0.1212(0.1322)
2020-08-03 13:38:08 1000-5019 loss=0.0730(0.0505+0.0225)-0.3931(0.2071+0.1860) sod-mse=0.0202(0.1071) gcn-mse=0.0314(0.1095) gcn-final-mse=0.1095(0.1240)
2020-08-03 13:39:42 2000-5019 loss=0.5034(0.2648+0.2386)-0.4038(0.2117+0.1921) sod-mse=0.1265(0.1094) gcn-mse=0.1289(0.1116) gcn-final-mse=0.1114(0.1258)
2020-08-03 13:41:17 3000-5019 loss=0.0615(0.0433+0.0182)-0.4118(0.2156+0.1962) sod-mse=0.0104(0.1111) gcn-mse=0.0173(0.1132) gcn-final-mse=0.1131(0.1274)
2020-08-03 13:42:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 13:42:51 4000-5019 loss=0.1634(0.1045+0.0590)-0.4096(0.2146+0.1951) sod-mse=0.0384(0.1108) gcn-mse=0.0490(0.1129) gcn-final-mse=0.1128(0.1272)
2020-08-03 13:43:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 13:44:26 5000-5019 loss=0.9573(0.4297+0.5276)-0.4095(0.2147+0.1948) sod-mse=0.1452(0.1110) gcn-mse=0.1454(0.1132) gcn-final-mse=0.1131(0.1274)
2020-08-03 13:44:27 E: 1, Train sod-mae-score=0.0844-0.9021 gcn-mae-score=0.0904-0.8707 gcn-final-mse-score=0.0902-0.8738(0.1046/0.8738) loss=0.3299(0.1802+0.1497)
2020-08-03 13:44:27 E: 1, Test  sod-mae-score=0.1109-0.7717 gcn-mae-score=0.1132-0.7198 gcn-final-mse-score=0.1130-0.7253(0.1274/0.7253) loss=0.4092(0.2146+0.1947)

2020-08-03 13:44:27 Start Epoch 2
2020-08-03 13:44:27 Epoch:02,lr=0.0001
2020-08-03 13:44:29    0-10553 loss=0.7545(0.3915+0.3630)-0.7545(0.3915+0.3630) sod-mse=0.2275(0.2275) gcn-mse=0.2287(0.2287) gcn-final-mse=0.2118(0.2270)
2020-08-03 13:47:41 1000-10553 loss=0.5915(0.3023+0.2892)-0.2921(0.1625+0.1297) sod-mse=0.1501(0.0731) gcn-mse=0.1432(0.0804) gcn-final-mse=0.0800(0.0943)
2020-08-03 13:49:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 13:50:54 2000-10553 loss=0.0522(0.0436+0.0086)-0.3012(0.1666+0.1346) sod-mse=0.0046(0.0758) gcn-mse=0.0145(0.0818) gcn-final-mse=0.0815(0.0960)
2020-08-03 13:54:06 3000-10553 loss=0.5133(0.2467+0.2665)-0.3007(0.1665+0.1342) sod-mse=0.1112(0.0755) gcn-mse=0.1246(0.0823) gcn-final-mse=0.0821(0.0965)
2020-08-03 13:57:17 4000-10553 loss=0.1167(0.0747+0.0420)-0.2934(0.1630+0.1304) sod-mse=0.0298(0.0731) gcn-mse=0.0315(0.0801) gcn-final-mse=0.0799(0.0943)
2020-08-03 14:00:30 5000-10553 loss=1.1465(0.5804+0.5661)-0.2903(0.1614+0.1290) sod-mse=0.2262(0.0724) gcn-mse=0.2066(0.0793) gcn-final-mse=0.0790(0.0934)
2020-08-03 14:03:41 6000-10553 loss=0.2800(0.1411+0.1389)-0.2903(0.1612+0.1291) sod-mse=0.0717(0.0725) gcn-mse=0.0876(0.0791) gcn-final-mse=0.0789(0.0932)
2020-08-03 14:06:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 14:06:54 7000-10553 loss=1.1800(0.6930+0.4870)-0.2896(0.1609+0.1287) sod-mse=0.2368(0.0721) gcn-mse=0.2465(0.0787) gcn-final-mse=0.0784(0.0928)
2020-08-03 14:09:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 14:10:06 8000-10553 loss=0.1008(0.0654+0.0354)-0.2901(0.1612+0.1288) sod-mse=0.0225(0.0722) gcn-mse=0.0275(0.0787) gcn-final-mse=0.0785(0.0929)
2020-08-03 14:13:18 9000-10553 loss=0.1673(0.0969+0.0703)-0.2884(0.1604+0.1280) sod-mse=0.0389(0.0718) gcn-mse=0.0447(0.0783) gcn-final-mse=0.0781(0.0925)
2020-08-03 14:16:30 10000-10553 loss=0.2218(0.1331+0.0887)-0.2877(0.1601+0.1276) sod-mse=0.0480(0.0716) gcn-mse=0.0652(0.0781) gcn-final-mse=0.0779(0.0924)

2020-08-03 14:18:17    0-5019 loss=0.9468(0.4882+0.4586)-0.9468(0.4882+0.4586) sod-mse=0.1641(0.1641) gcn-mse=0.1917(0.1917) gcn-final-mse=0.1835(0.1937)
2020-08-03 14:19:51 1000-5019 loss=0.0736(0.0534+0.0201)-0.3638(0.1923+0.1715) sod-mse=0.0184(0.0830) gcn-mse=0.0341(0.0900) gcn-final-mse=0.0899(0.1030)
2020-08-03 14:21:24 2000-5019 loss=0.6053(0.2932+0.3120)-0.3707(0.1952+0.1755) sod-mse=0.1073(0.0848) gcn-mse=0.1073(0.0916) gcn-final-mse=0.0916(0.1046)
2020-08-03 14:22:56 3000-5019 loss=0.0558(0.0403+0.0155)-0.3737(0.1969+0.1768) sod-mse=0.0087(0.0859) gcn-mse=0.0141(0.0927) gcn-final-mse=0.0927(0.1058)
2020-08-03 14:23:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 14:24:28 4000-5019 loss=0.1781(0.1090+0.0690)-0.3755(0.1979+0.1776) sod-mse=0.0427(0.0862) gcn-mse=0.0479(0.0932) gcn-final-mse=0.0931(0.1062)
2020-08-03 14:24:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 14:25:59 5000-5019 loss=0.8545(0.3532+0.5013)-0.3760(0.1984+0.1776) sod-mse=0.1240(0.0863) gcn-mse=0.1223(0.0934) gcn-final-mse=0.0933(0.1064)
2020-08-03 14:26:01 E: 2, Train sod-mae-score=0.0715-0.9138 gcn-mae-score=0.0781-0.8822 gcn-final-mse-score=0.0778-0.8853(0.0923/0.8853) loss=0.2877(0.1601+0.1275)
2020-08-03 14:26:01 E: 2, Test  sod-mae-score=0.0863-0.7893 gcn-mae-score=0.0934-0.7391 gcn-final-mse-score=0.0933-0.7457(0.1064/0.7457) loss=0.3758(0.1983+0.1774)

2020-08-03 14:26:01 Start Epoch 3
2020-08-03 14:26:01 Epoch:03,lr=0.0001
2020-08-03 14:26:03    0-10553 loss=0.5816(0.2968+0.2848)-0.5816(0.2968+0.2848) sod-mse=0.1280(0.1280) gcn-mse=0.1328(0.1328) gcn-final-mse=0.1326(0.1510)
2020-08-03 14:27:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 14:29:14 1000-10553 loss=0.1070(0.0702+0.0368)-0.2608(0.1484+0.1124) sod-mse=0.0306(0.0626) gcn-mse=0.0337(0.0713) gcn-final-mse=0.0707(0.0855)
2020-08-03 14:32:26 2000-10553 loss=0.1463(0.0915+0.0547)-0.2621(0.1481+0.1140) sod-mse=0.0445(0.0633) gcn-mse=0.0523(0.0710) gcn-final-mse=0.0706(0.0853)
2020-08-03 14:35:38 3000-10553 loss=0.1089(0.0684+0.0405)-0.2636(0.1487+0.1149) sod-mse=0.0226(0.0639) gcn-mse=0.0214(0.0710) gcn-final-mse=0.0706(0.0853)
2020-08-03 14:38:49 4000-10553 loss=0.0774(0.0549+0.0225)-0.2612(0.1472+0.1140) sod-mse=0.0136(0.0635) gcn-mse=0.0170(0.0704) gcn-final-mse=0.0701(0.0847)
2020-08-03 14:42:01 5000-10553 loss=0.4020(0.2375+0.1646)-0.2602(0.1466+0.1136) sod-mse=0.1233(0.0631) gcn-mse=0.1481(0.0700) gcn-final-mse=0.0696(0.0843)
2020-08-03 14:45:12 6000-10553 loss=0.2734(0.1555+0.1179)-0.2598(0.1462+0.1136) sod-mse=0.0784(0.0630) gcn-mse=0.0744(0.0695) gcn-final-mse=0.0691(0.0838)
2020-08-03 14:46:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 14:48:25 7000-10553 loss=0.2727(0.1617+0.1110)-0.2623(0.1473+0.1149) sod-mse=0.0863(0.0638) gcn-mse=0.0978(0.0703) gcn-final-mse=0.0700(0.0847)
2020-08-03 14:51:38 8000-10553 loss=0.0413(0.0299+0.0114)-0.2618(0.1471+0.1147) sod-mse=0.0063(0.0638) gcn-mse=0.0120(0.0702) gcn-final-mse=0.0698(0.0845)
2020-08-03 14:54:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 14:54:54 9000-10553 loss=0.0801(0.0536+0.0265)-0.2630(0.1476+0.1153) sod-mse=0.0186(0.0641) gcn-mse=0.0212(0.0704) gcn-final-mse=0.0701(0.0848)
2020-08-03 14:58:07 10000-10553 loss=0.4336(0.2521+0.1815)-0.2626(0.1475+0.1150) sod-mse=0.1087(0.0639) gcn-mse=0.1135(0.0703) gcn-final-mse=0.0700(0.0847)

2020-08-03 14:59:55    0-5019 loss=0.6468(0.3116+0.3353)-0.6468(0.3116+0.3353) sod-mse=0.1408(0.1408) gcn-mse=0.1348(0.1348) gcn-final-mse=0.1282(0.1419)
2020-08-03 15:01:29 1000-5019 loss=0.0792(0.0455+0.0337)-0.3748(0.1888+0.1859) sod-mse=0.0255(0.0825) gcn-mse=0.0252(0.0878) gcn-final-mse=0.0881(0.1017)
2020-08-03 15:03:03 2000-5019 loss=0.8081(0.3392+0.4689)-0.3767(0.1895+0.1871) sod-mse=0.1300(0.0829) gcn-mse=0.1199(0.0885) gcn-final-mse=0.0889(0.1023)
2020-08-03 15:04:37 3000-5019 loss=0.0580(0.0404+0.0175)-0.3801(0.1917+0.1884) sod-mse=0.0098(0.0839) gcn-mse=0.0144(0.0899) gcn-final-mse=0.0903(0.1038)
2020-08-03 15:05:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 15:06:11 4000-5019 loss=0.3098(0.1599+0.1499)-0.3797(0.1917+0.1879) sod-mse=0.0727(0.0841) gcn-mse=0.0708(0.0902) gcn-final-mse=0.0906(0.1040)
2020-08-03 15:06:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 15:07:44 5000-5019 loss=0.7662(0.3111+0.4551)-0.3809(0.1923+0.1887) sod-mse=0.0981(0.0842) gcn-mse=0.1006(0.0903) gcn-final-mse=0.0906(0.1040)
2020-08-03 15:07:45 E: 3, Train sod-mae-score=0.0639-0.9222 gcn-mae-score=0.0702-0.8912 gcn-final-mse-score=0.0699-0.8942(0.0846/0.8942) loss=0.2621(0.1473+0.1148)
2020-08-03 15:07:45 E: 3, Test  sod-mae-score=0.0841-0.7866 gcn-mae-score=0.0903-0.7173 gcn-final-mse-score=0.0906-0.7238(0.1040/0.7238) loss=0.3806(0.1922+0.1884)

2020-08-03 15:07:45 Start Epoch 4
2020-08-03 15:07:45 Epoch:04,lr=0.0001
2020-08-03 15:07:47    0-10553 loss=0.0571(0.0344+0.0228)-0.0571(0.0344+0.0228) sod-mse=0.0139(0.0139) gcn-mse=0.0140(0.0140) gcn-final-mse=0.0142(0.0203)
2020-08-03 15:11:02 1000-10553 loss=0.0944(0.0488+0.0456)-0.2383(0.1357+0.1026) sod-mse=0.0361(0.0564) gcn-mse=0.0195(0.0627) gcn-final-mse=0.0625(0.0771)
2020-08-03 15:14:16 2000-10553 loss=0.0947(0.0553+0.0394)-0.2408(0.1367+0.1041) sod-mse=0.0277(0.0571) gcn-mse=0.0338(0.0635) gcn-final-mse=0.0631(0.0778)
2020-08-03 15:17:31 3000-10553 loss=0.6001(0.3122+0.2878)-0.2400(0.1360+0.1039) sod-mse=0.1852(0.0569) gcn-mse=0.1725(0.0632) gcn-final-mse=0.0628(0.0775)
2020-08-03 15:19:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 15:20:44 4000-10553 loss=0.3652(0.2093+0.1559)-0.2373(0.1347+0.1027) sod-mse=0.0989(0.0564) gcn-mse=0.1094(0.0626) gcn-final-mse=0.0623(0.0769)
2020-08-03 15:22:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 15:23:58 5000-10553 loss=0.3021(0.1586+0.1435)-0.2379(0.1352+0.1027) sod-mse=0.0796(0.0566) gcn-mse=0.0719(0.0630) gcn-final-mse=0.0627(0.0774)
2020-08-03 15:27:12 6000-10553 loss=0.0898(0.0577+0.0320)-0.2356(0.1340+0.1016) sod-mse=0.0227(0.0559) gcn-mse=0.0199(0.0623) gcn-final-mse=0.0620(0.0767)
2020-08-03 15:30:26 7000-10553 loss=0.1597(0.0835+0.0762)-0.2374(0.1349+0.1025) sod-mse=0.0670(0.0564) gcn-mse=0.0548(0.0626) gcn-final-mse=0.0623(0.0771)
2020-08-03 15:33:38 8000-10553 loss=0.0653(0.0462+0.0191)-0.2392(0.1359+0.1033) sod-mse=0.0154(0.0569) gcn-mse=0.0274(0.0631) gcn-final-mse=0.0628(0.0775)
2020-08-03 15:34:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 15:36:52 9000-10553 loss=0.0822(0.0543+0.0280)-0.2408(0.1366+0.1041) sod-mse=0.0107(0.0574) gcn-mse=0.0163(0.0636) gcn-final-mse=0.0633(0.0780)
2020-08-03 15:40:06 10000-10553 loss=0.2414(0.1288+0.1127)-0.2400(0.1363+0.1037) sod-mse=0.0597(0.0572) gcn-mse=0.0691(0.0635) gcn-final-mse=0.0632(0.0779)

2020-08-03 15:41:55    0-5019 loss=1.1499(0.5849+0.5649)-1.1499(0.5849+0.5649) sod-mse=0.1873(0.1873) gcn-mse=0.2013(0.2013) gcn-final-mse=0.1938(0.2040)
2020-08-03 15:43:28 1000-5019 loss=0.0475(0.0353+0.0122)-0.3594(0.1895+0.1699) sod-mse=0.0108(0.0747) gcn-mse=0.0167(0.0795) gcn-final-mse=0.0796(0.0918)
2020-08-03 15:44:59 2000-5019 loss=0.6519(0.2969+0.3550)-0.3611(0.1903+0.1708) sod-mse=0.0982(0.0758) gcn-mse=0.0896(0.0806) gcn-final-mse=0.0806(0.0928)
2020-08-03 15:46:31 3000-5019 loss=0.0598(0.0414+0.0184)-0.3647(0.1923+0.1724) sod-mse=0.0107(0.0763) gcn-mse=0.0136(0.0813) gcn-final-mse=0.0813(0.0936)
2020-08-03 15:47:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 15:48:03 4000-5019 loss=0.1385(0.0887+0.0498)-0.3625(0.1914+0.1711) sod-mse=0.0296(0.0760) gcn-mse=0.0318(0.0812) gcn-final-mse=0.0811(0.0934)
2020-08-03 15:48:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 15:49:34 5000-5019 loss=0.3149(0.1633+0.1515)-0.3652(0.1930+0.1722) sod-mse=0.0790(0.0764) gcn-mse=0.0827(0.0818) gcn-final-mse=0.0817(0.0939)
2020-08-03 15:49:35 E: 4, Train sod-mae-score=0.0573-0.9298 gcn-mae-score=0.0636-0.8988 gcn-final-mse-score=0.0633-0.9017(0.0780/0.9017) loss=0.2400(0.1364+0.1037)
2020-08-03 15:49:35 E: 4, Test  sod-mae-score=0.0764-0.8039 gcn-mae-score=0.0818-0.7482 gcn-final-mse-score=0.0817-0.7546(0.0939/0.7546) loss=0.3649(0.1929+0.1720)

2020-08-03 15:49:35 Start Epoch 5
2020-08-03 15:49:35 Epoch:05,lr=0.0001
2020-08-03 15:49:37    0-10553 loss=0.0809(0.0556+0.0253)-0.0809(0.0556+0.0253) sod-mse=0.0137(0.0137) gcn-mse=0.0159(0.0159) gcn-final-mse=0.0165(0.0323)
2020-08-03 15:52:50 1000-10553 loss=0.3485(0.1922+0.1562)-0.2195(0.1261+0.0934) sod-mse=0.0962(0.0511) gcn-mse=0.0931(0.0575) gcn-final-mse=0.0568(0.0716)
2020-08-03 15:56:02 2000-10553 loss=0.4270(0.2191+0.2079)-0.2210(0.1270+0.0940) sod-mse=0.1591(0.0512) gcn-mse=0.1268(0.0576) gcn-final-mse=0.0571(0.0719)
2020-08-03 15:59:16 3000-10553 loss=0.5120(0.2777+0.2343)-0.2239(0.1286+0.0953) sod-mse=0.1463(0.0526) gcn-mse=0.1597(0.0592) gcn-final-mse=0.0587(0.0735)
2020-08-03 16:02:29 4000-10553 loss=0.1174(0.0871+0.0303)-0.2243(0.1288+0.0955) sod-mse=0.0203(0.0528) gcn-mse=0.0376(0.0593) gcn-final-mse=0.0589(0.0737)
2020-08-03 16:05:41 5000-10553 loss=0.1629(0.1020+0.0609)-0.2231(0.1281+0.0950) sod-mse=0.0487(0.0525) gcn-mse=0.0529(0.0589) gcn-final-mse=0.0584(0.0732)
2020-08-03 16:07:18 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 16:08:53 6000-10553 loss=0.0382(0.0250+0.0131)-0.2217(0.1273+0.0944) sod-mse=0.0091(0.0520) gcn-mse=0.0120(0.0583) gcn-final-mse=0.0579(0.0726)
2020-08-03 16:10:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 16:10:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 16:12:05 7000-10553 loss=0.2329(0.1483+0.0846)-0.2204(0.1266+0.0938) sod-mse=0.0525(0.0517) gcn-mse=0.0783(0.0579) gcn-final-mse=0.0575(0.0723)
2020-08-03 16:15:17 8000-10553 loss=0.0583(0.0406+0.0177)-0.2220(0.1275+0.0944) sod-mse=0.0144(0.0522) gcn-mse=0.0261(0.0584) gcn-final-mse=0.0581(0.0729)
2020-08-03 16:18:31 9000-10553 loss=0.4972(0.2784+0.2188)-0.2225(0.1277+0.0948) sod-mse=0.1065(0.0523) gcn-mse=0.1209(0.0584) gcn-final-mse=0.0581(0.0729)
2020-08-03 16:21:43 10000-10553 loss=0.1940(0.1179+0.0761)-0.2235(0.1281+0.0954) sod-mse=0.0343(0.0527) gcn-mse=0.0352(0.0586) gcn-final-mse=0.0583(0.0731)

2020-08-03 16:23:31    0-5019 loss=0.9549(0.4895+0.4653)-0.9549(0.4895+0.4653) sod-mse=0.1615(0.1615) gcn-mse=0.1562(0.1562) gcn-final-mse=0.1481(0.1594)
2020-08-03 16:25:05 1000-5019 loss=0.0685(0.0436+0.0249)-0.3587(0.1918+0.1669) sod-mse=0.0224(0.0767) gcn-mse=0.0243(0.0767) gcn-final-mse=0.0766(0.0896)
2020-08-03 16:26:38 2000-5019 loss=0.5314(0.2715+0.2599)-0.3660(0.1952+0.1708) sod-mse=0.1110(0.0778) gcn-mse=0.1017(0.0777) gcn-final-mse=0.0776(0.0905)
2020-08-03 16:28:11 3000-5019 loss=0.0507(0.0368+0.0139)-0.3662(0.1961+0.1701) sod-mse=0.0080(0.0783) gcn-mse=0.0101(0.0785) gcn-final-mse=0.0784(0.0913)
2020-08-03 16:29:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 16:29:43 4000-5019 loss=0.2344(0.1236+0.1108)-0.3662(0.1963+0.1699) sod-mse=0.0568(0.0786) gcn-mse=0.0487(0.0789) gcn-final-mse=0.0787(0.0917)
2020-08-03 16:30:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 16:31:14 5000-5019 loss=0.7011(0.3224+0.3786)-0.3665(0.1966+0.1699) sod-mse=0.1075(0.0789) gcn-mse=0.1035(0.0793) gcn-final-mse=0.0791(0.0920)
2020-08-03 16:31:16 E: 5, Train sod-mae-score=0.0524-0.9346 gcn-mae-score=0.0584-0.9038 gcn-final-mse-score=0.0581-0.9067(0.0729/0.9067) loss=0.2225(0.1276+0.0949)
2020-08-03 16:31:16 E: 5, Test  sod-mae-score=0.0789-0.8109 gcn-mae-score=0.0792-0.7564 gcn-final-mse-score=0.0790-0.7629(0.0920/0.7629) loss=0.3660(0.1964+0.1696)

2020-08-03 16:31:16 Start Epoch 6
2020-08-03 16:31:16 Epoch:06,lr=0.0001
2020-08-03 16:31:17    0-10553 loss=0.0320(0.0208+0.0112)-0.0320(0.0208+0.0112) sod-mse=0.0063(0.0063) gcn-mse=0.0095(0.0095) gcn-final-mse=0.0094(0.0143)
2020-08-03 16:34:30 1000-10553 loss=0.1307(0.0923+0.0384)-0.2129(0.1236+0.0893) sod-mse=0.0278(0.0495) gcn-mse=0.0377(0.0565) gcn-final-mse=0.0561(0.0712)
2020-08-03 16:37:42 2000-10553 loss=0.2693(0.1570+0.1124)-0.1997(0.1170+0.0828) sod-mse=0.0738(0.0455) gcn-mse=0.0888(0.0526) gcn-final-mse=0.0522(0.0674)
2020-08-03 16:39:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 16:40:55 3000-10553 loss=0.5622(0.2949+0.2672)-0.1996(0.1168+0.0828) sod-mse=0.1647(0.0454) gcn-mse=0.1635(0.0522) gcn-final-mse=0.0519(0.0669)
2020-08-03 16:44:08 4000-10553 loss=0.0885(0.0604+0.0281)-0.1979(0.1160+0.0819) sod-mse=0.0220(0.0449) gcn-mse=0.0273(0.0519) gcn-final-mse=0.0515(0.0665)
2020-08-03 16:47:20 5000-10553 loss=0.1728(0.1158+0.0570)-0.2025(0.1182+0.0843) sod-mse=0.0365(0.0463) gcn-mse=0.0481(0.0530) gcn-final-mse=0.0526(0.0676)
2020-08-03 16:50:32 6000-10553 loss=0.0427(0.0273+0.0154)-0.2025(0.1182+0.0844) sod-mse=0.0126(0.0463) gcn-mse=0.0145(0.0528) gcn-final-mse=0.0525(0.0674)
2020-08-03 16:53:18 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 16:53:42 7000-10553 loss=0.2230(0.1482+0.0748)-0.2045(0.1192+0.0853) sod-mse=0.0531(0.0467) gcn-mse=0.0843(0.0534) gcn-final-mse=0.0530(0.0679)
2020-08-03 16:56:55 8000-10553 loss=0.1991(0.1091+0.0900)-0.2039(0.1188+0.0851) sod-mse=0.0565(0.0466) gcn-mse=0.0561(0.0531) gcn-final-mse=0.0528(0.0677)
2020-08-03 16:58:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 17:00:07 9000-10553 loss=0.2713(0.1599+0.1114)-0.2044(0.1190+0.0854) sod-mse=0.0657(0.0467) gcn-mse=0.0771(0.0532) gcn-final-mse=0.0529(0.0678)
2020-08-03 17:03:19 10000-10553 loss=0.0906(0.0656+0.0250)-0.2048(0.1193+0.0855) sod-mse=0.0174(0.0468) gcn-mse=0.0262(0.0534) gcn-final-mse=0.0531(0.0680)

2020-08-03 17:05:07    0-5019 loss=0.6535(0.3583+0.2953)-0.6535(0.3583+0.2953) sod-mse=0.0942(0.0942) gcn-mse=0.1120(0.1120) gcn-final-mse=0.1041(0.1142)
2020-08-03 17:06:39 1000-5019 loss=0.0382(0.0297+0.0085)-0.3580(0.1878+0.1703) sod-mse=0.0073(0.0828) gcn-mse=0.0116(0.0855) gcn-final-mse=0.0853(0.0984)
2020-08-03 17:08:11 2000-5019 loss=0.6030(0.3078+0.2952)-0.3669(0.1918+0.1751) sod-mse=0.1073(0.0846) gcn-mse=0.1071(0.0872) gcn-final-mse=0.0870(0.1000)
2020-08-03 17:09:45 3000-5019 loss=0.0503(0.0367+0.0136)-0.3701(0.1934+0.1766) sod-mse=0.0068(0.0853) gcn-mse=0.0100(0.0878) gcn-final-mse=0.0876(0.1006)
2020-08-03 17:10:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 17:11:19 4000-5019 loss=0.2787(0.1541+0.1247)-0.3708(0.1939+0.1769) sod-mse=0.0600(0.0854) gcn-mse=0.0589(0.0880) gcn-final-mse=0.0877(0.1008)
2020-08-03 17:11:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 17:12:53 5000-5019 loss=0.4521(0.2218+0.2304)-0.3720(0.1947+0.1773) sod-mse=0.1027(0.0857) gcn-mse=0.0964(0.0885) gcn-final-mse=0.0881(0.1012)
2020-08-03 17:12:55 E: 6, Train sod-mae-score=0.0468-0.9407 gcn-mae-score=0.0534-0.9090 gcn-final-mse-score=0.0530-0.9118(0.0679/0.9118) loss=0.2046(0.1191+0.0855)
2020-08-03 17:12:55 E: 6, Test  sod-mae-score=0.0857-0.7941 gcn-mae-score=0.0885-0.7317 gcn-final-mse-score=0.0882-0.7379(0.1012/0.7379) loss=0.3719(0.1947+0.1772)

2020-08-03 17:12:55 Start Epoch 7
2020-08-03 17:12:55 Epoch:07,lr=0.0001
2020-08-03 17:12:56    0-10553 loss=0.1994(0.1345+0.0649)-0.1994(0.1345+0.0649) sod-mse=0.0374(0.0374) gcn-mse=0.0485(0.0485) gcn-final-mse=0.0508(0.0813)
2020-08-03 17:16:10 1000-10553 loss=0.1556(0.1104+0.0452)-0.1850(0.1101+0.0749) sod-mse=0.0297(0.0405) gcn-mse=0.0463(0.0486) gcn-final-mse=0.0482(0.0633)
2020-08-03 17:19:25 2000-10553 loss=0.0804(0.0569+0.0235)-0.1782(0.1061+0.0721) sod-mse=0.0173(0.0390) gcn-mse=0.0244(0.0461) gcn-final-mse=0.0456(0.0606)
2020-08-03 17:20:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 17:22:43 3000-10553 loss=0.0767(0.0485+0.0282)-0.1828(0.1084+0.0744) sod-mse=0.0165(0.0406) gcn-mse=0.0145(0.0475) gcn-final-mse=0.0470(0.0620)
2020-08-03 17:25:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 17:25:58 4000-10553 loss=0.0903(0.0641+0.0262)-0.1828(0.1086+0.0741) sod-mse=0.0144(0.0405) gcn-mse=0.0261(0.0474) gcn-final-mse=0.0469(0.0619)
2020-08-03 17:29:12 5000-10553 loss=0.1147(0.0789+0.0357)-0.1848(0.1095+0.0752) sod-mse=0.0213(0.0411) gcn-mse=0.0275(0.0478) gcn-final-mse=0.0473(0.0623)
2020-08-03 17:31:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 17:32:23 6000-10553 loss=0.3498(0.2093+0.1405)-0.1872(0.1106+0.0766) sod-mse=0.0988(0.0419) gcn-mse=0.1238(0.0483) gcn-final-mse=0.0479(0.0629)
2020-08-03 17:35:33 7000-10553 loss=0.0880(0.0610+0.0270)-0.1876(0.1108+0.0768) sod-mse=0.0169(0.0421) gcn-mse=0.0174(0.0485) gcn-final-mse=0.0481(0.0631)
2020-08-03 17:38:45 8000-10553 loss=0.2570(0.1690+0.0880)-0.1886(0.1112+0.0774) sod-mse=0.0650(0.0423) gcn-mse=0.0995(0.0486) gcn-final-mse=0.0482(0.0632)
2020-08-03 17:41:56 9000-10553 loss=0.0571(0.0423+0.0148)-0.1905(0.1121+0.0784) sod-mse=0.0097(0.0429) gcn-mse=0.0132(0.0492) gcn-final-mse=0.0487(0.0638)
2020-08-03 17:45:08 10000-10553 loss=0.2269(0.1252+0.1016)-0.1900(0.1118+0.0782) sod-mse=0.0404(0.0427) gcn-mse=0.0513(0.0490) gcn-final-mse=0.0485(0.0635)

2020-08-03 17:46:55    0-5019 loss=1.2562(0.6753+0.5809)-1.2562(0.6753+0.5809) sod-mse=0.1395(0.1395) gcn-mse=0.1604(0.1604) gcn-final-mse=0.1532(0.1635)
2020-08-03 17:48:28 1000-5019 loss=0.0420(0.0329+0.0091)-0.3633(0.1904+0.1728) sod-mse=0.0079(0.0714) gcn-mse=0.0142(0.0762) gcn-final-mse=0.0763(0.0889)
2020-08-03 17:50:00 2000-5019 loss=0.6121(0.3090+0.3031)-0.3650(0.1911+0.1738) sod-mse=0.1022(0.0723) gcn-mse=0.0997(0.0772) gcn-final-mse=0.0772(0.0898)
2020-08-03 17:51:32 3000-5019 loss=0.0499(0.0368+0.0131)-0.3696(0.1934+0.1762) sod-mse=0.0065(0.0736) gcn-mse=0.0103(0.0787) gcn-final-mse=0.0788(0.0914)
2020-08-03 17:52:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 17:53:04 4000-5019 loss=0.1347(0.0851+0.0496)-0.3682(0.1930+0.1752) sod-mse=0.0281(0.0735) gcn-mse=0.0289(0.0788) gcn-final-mse=0.0788(0.0914)
2020-08-03 17:53:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 17:54:36 5000-5019 loss=0.5845(0.2736+0.3109)-0.3695(0.1938+0.1756) sod-mse=0.1051(0.0740) gcn-mse=0.1043(0.0793) gcn-final-mse=0.0793(0.0919)
2020-08-03 17:54:38 E: 7, Train sod-mae-score=0.0429-0.9445 gcn-mae-score=0.0491-0.9134 gcn-final-mse-score=0.0487-0.9163(0.0637/0.9163) loss=0.1905(0.1120+0.0785)
2020-08-03 17:54:38 E: 7, Test  sod-mae-score=0.0739-0.8097 gcn-mae-score=0.0793-0.7531 gcn-final-mse-score=0.0792-0.7592(0.0918/0.7592) loss=0.3690(0.1936+0.1754)

2020-08-03 17:54:38 Start Epoch 8
2020-08-03 17:54:38 Epoch:08,lr=0.0001
2020-08-03 17:54:39    0-10553 loss=0.2407(0.1299+0.1108)-0.2407(0.1299+0.1108) sod-mse=0.0593(0.0593) gcn-mse=0.0559(0.0559) gcn-final-mse=0.0597(0.0835)
2020-08-03 17:57:50 1000-10553 loss=0.0659(0.0407+0.0252)-0.1705(0.1023+0.0683) sod-mse=0.0151(0.0370) gcn-mse=0.0178(0.0443) gcn-final-mse=0.0439(0.0593)
2020-08-03 18:01:02 2000-10553 loss=0.0666(0.0459+0.0207)-0.1720(0.1025+0.0695) sod-mse=0.0151(0.0376) gcn-mse=0.0239(0.0441) gcn-final-mse=0.0437(0.0588)
2020-08-03 18:04:14 3000-10553 loss=0.1485(0.0939+0.0546)-0.1762(0.1048+0.0714) sod-mse=0.0363(0.0387) gcn-mse=0.0409(0.0450) gcn-final-mse=0.0446(0.0597)
2020-08-03 18:05:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 18:07:25 4000-10553 loss=0.1829(0.1142+0.0687)-0.1778(0.1055+0.0723) sod-mse=0.0413(0.0392) gcn-mse=0.0395(0.0453) gcn-final-mse=0.0449(0.0600)
2020-08-03 18:09:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 18:10:37 5000-10553 loss=0.1042(0.0631+0.0411)-0.1771(0.1053+0.0718) sod-mse=0.0324(0.0389) gcn-mse=0.0396(0.0452) gcn-final-mse=0.0447(0.0598)
2020-08-03 18:13:46 6000-10553 loss=0.2033(0.1047+0.0987)-0.1798(0.1067+0.0731) sod-mse=0.0475(0.0396) gcn-mse=0.0463(0.0458) gcn-final-mse=0.0454(0.0605)
2020-08-03 18:16:57 7000-10553 loss=0.1186(0.0833+0.0353)-0.1818(0.1077+0.0741) sod-mse=0.0269(0.0402) gcn-mse=0.0357(0.0464) gcn-final-mse=0.0460(0.0611)
2020-08-03 18:18:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 18:20:09 8000-10553 loss=0.1080(0.0733+0.0348)-0.1825(0.1080+0.0745) sod-mse=0.0191(0.0404) gcn-mse=0.0215(0.0467) gcn-final-mse=0.0463(0.0614)
2020-08-03 18:23:20 9000-10553 loss=0.3225(0.1660+0.1566)-0.1828(0.1081+0.0747) sod-mse=0.0684(0.0406) gcn-mse=0.0602(0.0467) gcn-final-mse=0.0463(0.0614)
2020-08-03 18:26:31 10000-10553 loss=0.6179(0.3145+0.3034)-0.1844(0.1090+0.0755) sod-mse=0.1571(0.0411) gcn-mse=0.1586(0.0472) gcn-final-mse=0.0468(0.0619)

2020-08-03 18:28:16    0-5019 loss=0.5320(0.2998+0.2322)-0.5320(0.2998+0.2322) sod-mse=0.0870(0.0870) gcn-mse=0.1021(0.1021) gcn-final-mse=0.0941(0.1047)
2020-08-03 18:29:49 1000-5019 loss=0.0995(0.0617+0.0378)-0.4021(0.2071+0.1950) sod-mse=0.0320(0.0940) gcn-mse=0.0362(0.0872) gcn-final-mse=0.0869(0.0999)
2020-08-03 18:31:21 2000-5019 loss=0.5342(0.2984+0.2358)-0.4098(0.2106+0.1991) sod-mse=0.1023(0.0943) gcn-mse=0.0981(0.0877) gcn-final-mse=0.0872(0.1001)
2020-08-03 18:32:53 3000-5019 loss=0.0526(0.0390+0.0136)-0.4141(0.2126+0.2015) sod-mse=0.0070(0.0954) gcn-mse=0.0105(0.0885) gcn-final-mse=0.0881(0.1009)
2020-08-03 18:33:44 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 18:34:25 4000-5019 loss=0.1192(0.0810+0.0382)-0.4192(0.2150+0.2043) sod-mse=0.0228(0.0960) gcn-mse=0.0249(0.0891) gcn-final-mse=0.0886(0.1015)
2020-08-03 18:34:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 18:35:57 5000-5019 loss=1.0914(0.5443+0.5471)-0.4194(0.2151+0.2043) sod-mse=0.1585(0.0962) gcn-mse=0.1492(0.0894) gcn-final-mse=0.0889(0.1017)
2020-08-03 18:35:58 E: 8, Train sod-mae-score=0.0409-0.9465 gcn-mae-score=0.0471-0.9156 gcn-final-mse-score=0.0467-0.9185(0.0618/0.9185) loss=0.1839(0.1087+0.0752)
2020-08-03 18:35:58 E: 8, Test  sod-mae-score=0.0962-0.8025 gcn-mae-score=0.0894-0.7494 gcn-final-mse-score=0.0889-0.7563(0.1017/0.7563) loss=0.4192(0.2150+0.2042)

2020-08-03 18:35:58 Start Epoch 9
2020-08-03 18:35:58 Epoch:09,lr=0.0001
2020-08-03 18:36:00    0-10553 loss=0.3193(0.1788+0.1405)-0.3193(0.1788+0.1405) sod-mse=0.0677(0.0677) gcn-mse=0.0813(0.0813) gcn-final-mse=0.0798(0.1001)
2020-08-03 18:36:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 18:39:11 1000-10553 loss=0.0757(0.0644+0.0113)-0.1595(0.0970+0.0626) sod-mse=0.0063(0.0340) gcn-mse=0.0242(0.0407) gcn-final-mse=0.0402(0.0554)
2020-08-03 18:42:21 2000-10553 loss=0.0499(0.0362+0.0137)-0.1559(0.0947+0.0612) sod-mse=0.0093(0.0332) gcn-mse=0.0155(0.0395) gcn-final-mse=0.0391(0.0540)
2020-08-03 18:45:32 3000-10553 loss=0.2392(0.1410+0.0982)-0.1648(0.0993+0.0655) sod-mse=0.0679(0.0357) gcn-mse=0.0746(0.0419) gcn-final-mse=0.0414(0.0564)
2020-08-03 18:46:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 18:48:42 4000-10553 loss=0.1947(0.1228+0.0719)-0.1659(0.0999+0.0661) sod-mse=0.0506(0.0360) gcn-mse=0.0537(0.0422) gcn-final-mse=0.0418(0.0568)
2020-08-03 18:51:54 5000-10553 loss=0.3431(0.2059+0.1372)-0.1678(0.1010+0.0669) sod-mse=0.0747(0.0364) gcn-mse=0.1055(0.0428) gcn-final-mse=0.0423(0.0574)
2020-08-03 18:51:59 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 18:55:05 6000-10553 loss=0.1098(0.0741+0.0357)-0.1676(0.1009+0.0667) sod-mse=0.0255(0.0361) gcn-mse=0.0285(0.0426) gcn-final-mse=0.0421(0.0572)
2020-08-03 18:58:15 7000-10553 loss=0.1734(0.1040+0.0694)-0.1694(0.1017+0.0677) sod-mse=0.0385(0.0367) gcn-mse=0.0447(0.0430) gcn-final-mse=0.0425(0.0577)
2020-08-03 19:01:25 8000-10553 loss=0.2032(0.1338+0.0694)-0.1692(0.1016+0.0676) sod-mse=0.0468(0.0367) gcn-mse=0.0727(0.0429) gcn-final-mse=0.0425(0.0576)
2020-08-03 19:04:36 9000-10553 loss=0.1358(0.0966+0.0392)-0.1687(0.1013+0.0675) sod-mse=0.0275(0.0366) gcn-mse=0.0315(0.0428) gcn-final-mse=0.0424(0.0575)
2020-08-03 19:07:47 10000-10553 loss=0.0777(0.0531+0.0247)-0.1700(0.1019+0.0681) sod-mse=0.0169(0.0369) gcn-mse=0.0179(0.0430) gcn-final-mse=0.0426(0.0577)

2020-08-03 19:09:33    0-5019 loss=0.7399(0.3947+0.3452)-0.7399(0.3947+0.3452) sod-mse=0.1016(0.1016) gcn-mse=0.1122(0.1122) gcn-final-mse=0.1059(0.1175)
2020-08-03 19:11:06 1000-5019 loss=0.0548(0.0389+0.0159)-0.3490(0.1864+0.1626) sod-mse=0.0144(0.0808) gcn-mse=0.0200(0.0825) gcn-final-mse=0.0823(0.0956)
2020-08-03 19:12:38 2000-5019 loss=0.4986(0.2688+0.2297)-0.3587(0.1910+0.1678) sod-mse=0.0891(0.0827) gcn-mse=0.0880(0.0844) gcn-final-mse=0.0840(0.0973)
2020-08-03 19:14:09 3000-5019 loss=0.0533(0.0386+0.0147)-0.3641(0.1933+0.1708) sod-mse=0.0089(0.0834) gcn-mse=0.0118(0.0850) gcn-final-mse=0.0847(0.0979)
2020-08-03 19:15:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 19:15:41 4000-5019 loss=0.1346(0.0892+0.0454)-0.3647(0.1936+0.1710) sod-mse=0.0282(0.0833) gcn-mse=0.0320(0.0850) gcn-final-mse=0.0847(0.0979)
2020-08-03 19:16:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 19:17:12 5000-5019 loss=0.9165(0.4228+0.4937)-0.3649(0.1939+0.1710) sod-mse=0.1021(0.0835) gcn-mse=0.1018(0.0853) gcn-final-mse=0.0849(0.0981)
2020-08-03 19:17:14 E: 9, Train sod-mae-score=0.0371-0.9505 gcn-mae-score=0.0433-0.9194 gcn-final-mse-score=0.0428-0.9223(0.0579/0.9223) loss=0.1704(0.1021+0.0684)
2020-08-03 19:17:14 E: 9, Test  sod-mae-score=0.0835-0.8015 gcn-mae-score=0.0853-0.7460 gcn-final-mse-score=0.0849-0.7520(0.0981/0.7520) loss=0.3648(0.1938+0.1710)

2020-08-03 19:17:14 Start Epoch 10
2020-08-03 19:17:14 Epoch:10,lr=0.0001
2020-08-03 19:17:16    0-10553 loss=0.3833(0.1967+0.1866)-0.3833(0.1967+0.1866) sod-mse=0.1210(0.1210) gcn-mse=0.1259(0.1259) gcn-final-mse=0.1197(0.1245)
2020-08-03 19:20:27 1000-10553 loss=0.0665(0.0478+0.0186)-0.1594(0.0957+0.0637) sod-mse=0.0159(0.0348) gcn-mse=0.0230(0.0408) gcn-final-mse=0.0405(0.0550)
2020-08-03 19:22:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 19:23:37 2000-10553 loss=0.0542(0.0438+0.0104)-0.1577(0.0954+0.0623) sod-mse=0.0051(0.0339) gcn-mse=0.0098(0.0401) gcn-final-mse=0.0398(0.0546)
2020-08-03 19:26:49 3000-10553 loss=0.0435(0.0350+0.0085)-0.1553(0.0943+0.0610) sod-mse=0.0032(0.0331) gcn-mse=0.0055(0.0395) gcn-final-mse=0.0391(0.0540)
2020-08-03 19:30:00 4000-10553 loss=0.2599(0.1633+0.0966)-0.1576(0.0955+0.0622) sod-mse=0.0719(0.0336) gcn-mse=0.0806(0.0396) gcn-final-mse=0.0393(0.0543)
2020-08-03 19:33:11 5000-10553 loss=0.3536(0.1719+0.1817)-0.1587(0.0960+0.0627) sod-mse=0.0680(0.0338) gcn-mse=0.0595(0.0399) gcn-final-mse=0.0395(0.0547)
2020-08-03 19:34:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 19:36:21 6000-10553 loss=0.0606(0.0435+0.0172)-0.1604(0.0968+0.0635) sod-mse=0.0118(0.0344) gcn-mse=0.0159(0.0403) gcn-final-mse=0.0399(0.0551)
2020-08-03 19:39:33 7000-10553 loss=0.1782(0.1105+0.0677)-0.1618(0.0976+0.0642) sod-mse=0.0393(0.0348) gcn-mse=0.0429(0.0407) gcn-final-mse=0.0403(0.0555)
2020-08-03 19:42:44 8000-10553 loss=0.0457(0.0342+0.0115)-0.1628(0.0981+0.0647) sod-mse=0.0068(0.0351) gcn-mse=0.0083(0.0409) gcn-final-mse=0.0405(0.0558)
2020-08-03 19:45:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 19:45:54 9000-10553 loss=0.1270(0.0796+0.0473)-0.1628(0.0981+0.0647) sod-mse=0.0342(0.0350) gcn-mse=0.0356(0.0410) gcn-final-mse=0.0406(0.0558)
2020-08-03 19:49:05 10000-10553 loss=0.0654(0.0439+0.0214)-0.1634(0.0985+0.0649) sod-mse=0.0154(0.0351) gcn-mse=0.0169(0.0411) gcn-final-mse=0.0407(0.0559)

2020-08-03 19:50:52    0-5019 loss=0.7312(0.3677+0.3634)-0.7312(0.3677+0.3634) sod-mse=0.1131(0.1131) gcn-mse=0.1167(0.1167) gcn-final-mse=0.1090(0.1227)
2020-08-03 19:52:25 1000-5019 loss=0.0664(0.0461+0.0203)-0.3670(0.1930+0.1739) sod-mse=0.0175(0.0801) gcn-mse=0.0247(0.0831) gcn-final-mse=0.0834(0.0971)
2020-08-03 19:53:57 2000-5019 loss=0.5887(0.3015+0.2872)-0.3810(0.1998+0.1812) sod-mse=0.0952(0.0822) gcn-mse=0.0911(0.0851) gcn-final-mse=0.0853(0.0990)
2020-08-03 19:55:29 3000-5019 loss=0.0542(0.0386+0.0156)-0.3874(0.2026+0.1848) sod-mse=0.0087(0.0835) gcn-mse=0.0124(0.0863) gcn-final-mse=0.0866(0.1002)
2020-08-03 19:56:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 19:57:01 4000-5019 loss=0.1452(0.0921+0.0531)-0.3835(0.2009+0.1825) sod-mse=0.0273(0.0825) gcn-mse=0.0327(0.0856) gcn-final-mse=0.0859(0.0995)
2020-08-03 19:57:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 19:58:34 5000-5019 loss=0.4525(0.2178+0.2347)-0.3851(0.2020+0.1831) sod-mse=0.0827(0.0825) gcn-mse=0.0823(0.0858) gcn-final-mse=0.0860(0.0995)
2020-08-03 19:58:35 E:10, Train sod-mae-score=0.0353-0.9526 gcn-mae-score=0.0412-0.9215 gcn-final-mse-score=0.0407-0.9242(0.0560/0.9242) loss=0.1639(0.0986+0.0653)
2020-08-03 19:58:35 E:10, Test  sod-mae-score=0.0825-0.8017 gcn-mae-score=0.0858-0.7387 gcn-final-mse-score=0.0860-0.7440(0.0995/0.7440) loss=0.3850(0.2020+0.1830)

2020-08-03 19:58:35 Start Epoch 11
2020-08-03 19:58:35 Epoch:11,lr=0.0001
2020-08-03 19:58:37    0-10553 loss=0.0672(0.0508+0.0164)-0.0672(0.0508+0.0164) sod-mse=0.0093(0.0093) gcn-mse=0.0180(0.0180) gcn-final-mse=0.0177(0.0285)
2020-08-03 20:01:47 1000-10553 loss=0.1016(0.0803+0.0214)-0.1395(0.0866+0.0529) sod-mse=0.0106(0.0285) gcn-mse=0.0225(0.0348) gcn-final-mse=0.0343(0.0496)
2020-08-03 20:04:58 2000-10553 loss=0.1027(0.0672+0.0355)-0.1475(0.0902+0.0573) sod-mse=0.0218(0.0304) gcn-mse=0.0279(0.0363) gcn-final-mse=0.0357(0.0511)
2020-08-03 20:08:09 3000-10553 loss=0.1701(0.0993+0.0708)-0.1478(0.0906+0.0572) sod-mse=0.0504(0.0305) gcn-mse=0.0516(0.0365) gcn-final-mse=0.0360(0.0514)
2020-08-03 20:11:21 4000-10553 loss=0.0546(0.0376+0.0170)-0.1511(0.0922+0.0590) sod-mse=0.0083(0.0314) gcn-mse=0.0083(0.0372) gcn-final-mse=0.0367(0.0521)
2020-08-03 20:14:33 5000-10553 loss=0.0284(0.0226+0.0058)-0.1531(0.0931+0.0600) sod-mse=0.0032(0.0321) gcn-mse=0.0050(0.0379) gcn-final-mse=0.0374(0.0527)
2020-08-03 20:17:45 6000-10553 loss=0.1232(0.0906+0.0326)-0.1540(0.0935+0.0605) sod-mse=0.0146(0.0324) gcn-mse=0.0222(0.0381) gcn-final-mse=0.0376(0.0528)
2020-08-03 20:18:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 20:19:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 20:20:56 7000-10553 loss=0.1461(0.0899+0.0561)-0.1551(0.0941+0.0610) sod-mse=0.0330(0.0326) gcn-mse=0.0407(0.0384) gcn-final-mse=0.0380(0.0532)
2020-08-03 20:24:06 8000-10553 loss=0.3398(0.1841+0.1557)-0.1543(0.0938+0.0605) sod-mse=0.0969(0.0324) gcn-mse=0.1038(0.0383) gcn-final-mse=0.0378(0.0531)
2020-08-03 20:27:17 9000-10553 loss=0.0925(0.0661+0.0264)-0.1560(0.0947+0.0613) sod-mse=0.0188(0.0329) gcn-mse=0.0202(0.0387) gcn-final-mse=0.0382(0.0535)
2020-08-03 20:30:28 10000-10553 loss=0.1226(0.0824+0.0402)-0.1569(0.0952+0.0617) sod-mse=0.0297(0.0331) gcn-mse=0.0405(0.0390) gcn-final-mse=0.0385(0.0539)
2020-08-03 20:31:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg

2020-08-03 20:32:14    0-5019 loss=1.0139(0.5272+0.4867)-1.0139(0.5272+0.4867) sod-mse=0.1101(0.1101) gcn-mse=0.1230(0.1230) gcn-final-mse=0.1147(0.1280)
2020-08-03 20:33:47 1000-5019 loss=0.0408(0.0320+0.0087)-0.3434(0.1788+0.1646) sod-mse=0.0077(0.0663) gcn-mse=0.0135(0.0705) gcn-final-mse=0.0705(0.0840)
2020-08-03 20:35:19 2000-5019 loss=0.9366(0.4322+0.5044)-0.3430(0.1790+0.1640) sod-mse=0.0931(0.0676) gcn-mse=0.0893(0.0716) gcn-final-mse=0.0716(0.0851)
2020-08-03 20:36:52 3000-5019 loss=0.0494(0.0362+0.0132)-0.3459(0.1802+0.1657) sod-mse=0.0072(0.0679) gcn-mse=0.0098(0.0720) gcn-final-mse=0.0721(0.0855)
2020-08-03 20:37:44 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 20:38:26 4000-5019 loss=0.1279(0.0844+0.0436)-0.3438(0.1795+0.1643) sod-mse=0.0237(0.0678) gcn-mse=0.0282(0.0720) gcn-final-mse=0.0720(0.0854)
2020-08-03 20:38:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 20:40:00 5000-5019 loss=0.9395(0.4071+0.5324)-0.3443(0.1799+0.1644) sod-mse=0.0997(0.0678) gcn-mse=0.1034(0.0721) gcn-final-mse=0.0721(0.0855)
2020-08-03 20:40:02 E:11, Train sod-mae-score=0.0332-0.9555 gcn-mae-score=0.0391-0.9245 gcn-final-mse-score=0.0386-0.9272(0.0539/0.9272) loss=0.1572(0.0953+0.0619)
2020-08-03 20:40:02 E:11, Test  sod-mae-score=0.0678-0.8238 gcn-mae-score=0.0722-0.7626 gcn-final-mse-score=0.0721-0.7687(0.0855/0.7687) loss=0.3442(0.1798+0.1643)

2020-08-03 20:40:02 Start Epoch 12
2020-08-03 20:40:02 Epoch:12,lr=0.0001
2020-08-03 20:40:03    0-10553 loss=0.1498(0.0927+0.0571)-0.1498(0.0927+0.0571) sod-mse=0.0353(0.0353) gcn-mse=0.0435(0.0435) gcn-final-mse=0.0421(0.0539)
2020-08-03 20:43:16 1000-10553 loss=0.0712(0.0515+0.0197)-0.1400(0.0873+0.0528) sod-mse=0.0114(0.0282) gcn-mse=0.0114(0.0344) gcn-final-mse=0.0338(0.0493)
2020-08-03 20:44:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 20:46:27 2000-10553 loss=0.0428(0.0275+0.0152)-0.1377(0.0859+0.0518) sod-mse=0.0087(0.0275) gcn-mse=0.0102(0.0337) gcn-final-mse=0.0332(0.0487)
2020-08-03 20:49:38 3000-10553 loss=0.1612(0.1120+0.0492)-0.1396(0.0867+0.0529) sod-mse=0.0314(0.0282) gcn-mse=0.0474(0.0342) gcn-final-mse=0.0337(0.0491)
2020-08-03 20:50:01 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 20:52:50 4000-10553 loss=0.0577(0.0406+0.0171)-0.1446(0.0891+0.0555) sod-mse=0.0130(0.0296) gcn-mse=0.0140(0.0354) gcn-final-mse=0.0348(0.0502)
2020-08-03 20:56:02 5000-10553 loss=0.1476(0.1020+0.0456)-0.1477(0.0906+0.0570) sod-mse=0.0331(0.0302) gcn-mse=0.0499(0.0360) gcn-final-mse=0.0355(0.0508)
2020-08-03 20:59:16 6000-10553 loss=0.0696(0.0463+0.0234)-0.1478(0.0907+0.0571) sod-mse=0.0120(0.0303) gcn-mse=0.0148(0.0362) gcn-final-mse=0.0357(0.0510)
2020-08-03 21:02:28 7000-10553 loss=0.1191(0.0856+0.0335)-0.1465(0.0900+0.0564) sod-mse=0.0165(0.0300) gcn-mse=0.0318(0.0359) gcn-final-mse=0.0353(0.0506)
2020-08-03 21:05:40 8000-10553 loss=0.2558(0.1430+0.1127)-0.1461(0.0898+0.0563) sod-mse=0.0749(0.0299) gcn-mse=0.0802(0.0357) gcn-final-mse=0.0352(0.0505)
2020-08-03 21:06:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 21:08:50 9000-10553 loss=0.0925(0.0573+0.0352)-0.1471(0.0904+0.0567) sod-mse=0.0289(0.0302) gcn-mse=0.0357(0.0361) gcn-final-mse=0.0356(0.0509)
2020-08-03 21:12:00 10000-10553 loss=0.1861(0.1120+0.0742)-0.1490(0.0914+0.0576) sod-mse=0.0510(0.0308) gcn-mse=0.0542(0.0367) gcn-final-mse=0.0362(0.0515)

2020-08-03 21:13:47    0-5019 loss=1.0273(0.5461+0.4813)-1.0273(0.5461+0.4813) sod-mse=0.1147(0.1147) gcn-mse=0.1267(0.1267) gcn-final-mse=0.1185(0.1299)
2020-08-03 21:15:20 1000-5019 loss=0.0404(0.0319+0.0085)-0.3625(0.1883+0.1742) sod-mse=0.0075(0.0648) gcn-mse=0.0132(0.0680) gcn-final-mse=0.0681(0.0807)
2020-08-03 21:16:52 2000-5019 loss=0.8110(0.3982+0.4129)-0.3598(0.1867+0.1730) sod-mse=0.1022(0.0650) gcn-mse=0.0982(0.0682) gcn-final-mse=0.0682(0.0808)
2020-08-03 21:18:24 3000-5019 loss=0.0494(0.0362+0.0132)-0.3654(0.1894+0.1760) sod-mse=0.0074(0.0657) gcn-mse=0.0102(0.0691) gcn-final-mse=0.0691(0.0817)
2020-08-03 21:19:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 21:19:55 4000-5019 loss=0.1296(0.0889+0.0407)-0.3649(0.1894+0.1755) sod-mse=0.0225(0.0656) gcn-mse=0.0319(0.0691) gcn-final-mse=0.0691(0.0816)
2020-08-03 21:20:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 21:21:26 5000-5019 loss=0.5828(0.2772+0.3056)-0.3661(0.1901+0.1760) sod-mse=0.1066(0.0658) gcn-mse=0.1025(0.0693) gcn-final-mse=0.0692(0.0818)
2020-08-03 21:21:28 E:12, Train sod-mae-score=0.0309-0.9586 gcn-mae-score=0.0367-0.9277 gcn-final-mse-score=0.0362-0.9304(0.0516/0.9304) loss=0.1492(0.0914+0.0577)
2020-08-03 21:21:28 E:12, Test  sod-mae-score=0.0657-0.8211 gcn-mae-score=0.0693-0.7588 gcn-final-mse-score=0.0692-0.7648(0.0817/0.7648) loss=0.3659(0.1900+0.1759)

2020-08-03 21:21:28 Start Epoch 13
2020-08-03 21:21:28 Epoch:13,lr=0.0001
2020-08-03 21:21:29    0-10553 loss=0.2038(0.0952+0.1085)-0.2038(0.0952+0.1085) sod-mse=0.0221(0.0221) gcn-mse=0.0227(0.0227) gcn-final-mse=0.0246(0.0355)
2020-08-03 21:24:40 1000-10553 loss=0.0541(0.0311+0.0230)-0.1326(0.0828+0.0499) sod-mse=0.0167(0.0265) gcn-mse=0.0191(0.0323) gcn-final-mse=0.0317(0.0471)
2020-08-03 21:27:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 21:27:51 2000-10553 loss=0.1270(0.0748+0.0522)-0.1323(0.0829+0.0494) sod-mse=0.0323(0.0262) gcn-mse=0.0338(0.0321) gcn-final-mse=0.0315(0.0469)
2020-08-03 21:31:02 3000-10553 loss=0.1793(0.1050+0.0743)-0.1372(0.0854+0.0518) sod-mse=0.0488(0.0277) gcn-mse=0.0556(0.0333) gcn-final-mse=0.0328(0.0482)
2020-08-03 21:34:14 4000-10553 loss=0.1225(0.0774+0.0451)-0.1382(0.0859+0.0523) sod-mse=0.0242(0.0279) gcn-mse=0.0289(0.0335) gcn-final-mse=0.0330(0.0484)
2020-08-03 21:34:18 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 21:37:27 5000-10553 loss=0.1141(0.0725+0.0416)-0.1389(0.0864+0.0525) sod-mse=0.0228(0.0279) gcn-mse=0.0219(0.0337) gcn-final-mse=0.0331(0.0486)
2020-08-03 21:40:40 6000-10553 loss=0.0565(0.0407+0.0158)-0.1385(0.0862+0.0524) sod-mse=0.0098(0.0279) gcn-mse=0.0105(0.0336) gcn-final-mse=0.0330(0.0485)
2020-08-03 21:43:53 7000-10553 loss=0.2054(0.1337+0.0717)-0.1399(0.0868+0.0531) sod-mse=0.0578(0.0283) gcn-mse=0.1014(0.0340) gcn-final-mse=0.0334(0.0489)
2020-08-03 21:45:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 21:47:02 8000-10553 loss=0.0470(0.0363+0.0107)-0.1391(0.0865+0.0527) sod-mse=0.0076(0.0281) gcn-mse=0.0133(0.0339) gcn-final-mse=0.0333(0.0488)
2020-08-03 21:50:13 9000-10553 loss=0.1225(0.0884+0.0341)-0.1386(0.0862+0.0524) sod-mse=0.0180(0.0280) gcn-mse=0.0351(0.0337) gcn-final-mse=0.0332(0.0486)
2020-08-03 21:53:24 10000-10553 loss=0.0815(0.0514+0.0301)-0.1398(0.0867+0.0531) sod-mse=0.0217(0.0283) gcn-mse=0.0215(0.0340) gcn-final-mse=0.0335(0.0489)

2020-08-03 21:55:10    0-5019 loss=0.8487(0.4175+0.4312)-0.8487(0.4175+0.4312) sod-mse=0.0980(0.0980) gcn-mse=0.1028(0.1028) gcn-final-mse=0.0955(0.1069)
2020-08-03 21:56:43 1000-5019 loss=0.0326(0.0262+0.0064)-0.3480(0.1856+0.1623) sod-mse=0.0054(0.0674) gcn-mse=0.0074(0.0713) gcn-final-mse=0.0713(0.0841)
2020-08-03 21:58:15 2000-5019 loss=0.9451(0.4561+0.4890)-0.3590(0.1911+0.1679) sod-mse=0.1071(0.0691) gcn-mse=0.1049(0.0732) gcn-final-mse=0.0730(0.0857)
2020-08-03 21:59:46 3000-5019 loss=0.0474(0.0340+0.0134)-0.3667(0.1946+0.1720) sod-mse=0.0074(0.0701) gcn-mse=0.0078(0.0744) gcn-final-mse=0.0742(0.0868)
2020-08-03 22:00:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 22:01:18 4000-5019 loss=0.1331(0.0872+0.0460)-0.3683(0.1956+0.1728) sod-mse=0.0256(0.0702) gcn-mse=0.0312(0.0747) gcn-final-mse=0.0744(0.0870)
2020-08-03 22:01:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 22:02:49 5000-5019 loss=0.7309(0.3209+0.4100)-0.3691(0.1961+0.1731) sod-mse=0.0965(0.0702) gcn-mse=0.0956(0.0748) gcn-final-mse=0.0745(0.0871)
2020-08-03 22:02:51 E:13, Train sod-mae-score=0.0287-0.9606 gcn-mae-score=0.0343-0.9297 gcn-final-mse-score=0.0338-0.9323(0.0492/0.9323) loss=0.1409(0.0873+0.0536)
2020-08-03 22:02:51 E:13, Test  sod-mae-score=0.0702-0.8045 gcn-mae-score=0.0747-0.7432 gcn-final-mse-score=0.0744-0.7492(0.0870/0.7492) loss=0.3688(0.1959+0.1729)

2020-08-03 22:02:51 Start Epoch 14
2020-08-03 22:02:51 Epoch:14,lr=0.0001
2020-08-03 22:02:52    0-10553 loss=0.1312(0.0894+0.0418)-0.1312(0.0894+0.0418) sod-mse=0.0272(0.0272) gcn-mse=0.0256(0.0256) gcn-final-mse=0.0268(0.0546)
2020-08-03 22:06:04 1000-10553 loss=0.0822(0.0557+0.0265)-0.1401(0.0867+0.0534) sod-mse=0.0198(0.0281) gcn-mse=0.0239(0.0336) gcn-final-mse=0.0330(0.0485)
2020-08-03 22:09:14 2000-10553 loss=0.0535(0.0423+0.0112)-0.1337(0.0836+0.0501) sod-mse=0.0059(0.0266) gcn-mse=0.0087(0.0322) gcn-final-mse=0.0317(0.0472)
2020-08-03 22:12:25 3000-10553 loss=0.0843(0.0504+0.0339)-0.1312(0.0823+0.0489) sod-mse=0.0157(0.0259) gcn-mse=0.0169(0.0315) gcn-final-mse=0.0310(0.0464)
2020-08-03 22:15:35 4000-10553 loss=0.1098(0.0799+0.0299)-0.1314(0.0824+0.0490) sod-mse=0.0203(0.0259) gcn-mse=0.0297(0.0315) gcn-final-mse=0.0310(0.0464)
2020-08-03 22:17:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 22:18:45 5000-10553 loss=0.3434(0.1906+0.1529)-0.1338(0.0838+0.0501) sod-mse=0.0829(0.0266) gcn-mse=0.0797(0.0323) gcn-final-mse=0.0318(0.0473)
2020-08-03 22:20:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 22:21:57 6000-10553 loss=0.0828(0.0614+0.0214)-0.1350(0.0843+0.0507) sod-mse=0.0126(0.0270) gcn-mse=0.0145(0.0327) gcn-final-mse=0.0321(0.0476)
2020-08-03 22:22:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 22:25:08 7000-10553 loss=0.0666(0.0461+0.0206)-0.1347(0.0842+0.0505) sod-mse=0.0157(0.0269) gcn-mse=0.0209(0.0326) gcn-final-mse=0.0320(0.0475)
2020-08-03 22:28:19 8000-10553 loss=0.4184(0.2545+0.1638)-0.1355(0.0847+0.0508) sod-mse=0.1184(0.0271) gcn-mse=0.1356(0.0328) gcn-final-mse=0.0322(0.0477)
2020-08-03 22:31:31 9000-10553 loss=0.0485(0.0381+0.0104)-0.1366(0.0853+0.0514) sod-mse=0.0065(0.0274) gcn-mse=0.0097(0.0332) gcn-final-mse=0.0326(0.0481)
2020-08-03 22:34:43 10000-10553 loss=0.0762(0.0546+0.0215)-0.1371(0.0854+0.0516) sod-mse=0.0137(0.0275) gcn-mse=0.0148(0.0332) gcn-final-mse=0.0327(0.0482)

2020-08-03 22:36:30    0-5019 loss=0.9979(0.5227+0.4751)-0.9979(0.5227+0.4751) sod-mse=0.1222(0.1222) gcn-mse=0.1260(0.1260) gcn-final-mse=0.1185(0.1322)
2020-08-03 22:38:05 1000-5019 loss=0.0407(0.0311+0.0096)-0.3424(0.1806+0.1618) sod-mse=0.0084(0.0641) gcn-mse=0.0124(0.0658) gcn-final-mse=0.0657(0.0790)
2020-08-03 22:39:38 2000-5019 loss=0.9196(0.4632+0.4564)-0.3433(0.1809+0.1624) sod-mse=0.0980(0.0651) gcn-mse=0.0935(0.0666) gcn-final-mse=0.0666(0.0797)
2020-08-03 22:41:10 3000-5019 loss=0.0487(0.0351+0.0136)-0.3535(0.1856+0.1679) sod-mse=0.0073(0.0665) gcn-mse=0.0079(0.0681) gcn-final-mse=0.0681(0.0812)
2020-08-03 22:42:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 22:42:43 4000-5019 loss=0.1272(0.0824+0.0447)-0.3518(0.1850+0.1668) sod-mse=0.0239(0.0662) gcn-mse=0.0236(0.0680) gcn-final-mse=0.0679(0.0810)
2020-08-03 22:43:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 22:44:15 5000-5019 loss=0.6234(0.2637+0.3597)-0.3524(0.1854+0.1670) sod-mse=0.0935(0.0663) gcn-mse=0.0846(0.0682) gcn-final-mse=0.0681(0.0812)
2020-08-03 22:44:17 E:14, Train sod-mae-score=0.0276-0.9622 gcn-mae-score=0.0333-0.9314 gcn-final-mse-score=0.0327-0.9340(0.0482/0.9340) loss=0.1371(0.0855+0.0516)
2020-08-03 22:44:17 E:14, Test  sod-mae-score=0.0664-0.8182 gcn-mae-score=0.0682-0.7631 gcn-final-mse-score=0.0682-0.7690(0.0812/0.7690) loss=0.3524(0.1854+0.1669)

2020-08-03 22:44:17 Start Epoch 15
2020-08-03 22:44:17 Epoch:15,lr=0.0001
2020-08-03 22:44:18    0-10553 loss=0.1322(0.0821+0.0500)-0.1322(0.0821+0.0500) sod-mse=0.0341(0.0341) gcn-mse=0.0374(0.0374) gcn-final-mse=0.0361(0.0500)
2020-08-03 22:47:30 1000-10553 loss=0.0662(0.0466+0.0196)-0.1213(0.0774+0.0438) sod-mse=0.0095(0.0234) gcn-mse=0.0141(0.0292) gcn-final-mse=0.0287(0.0442)
2020-08-03 22:50:40 2000-10553 loss=0.2269(0.1581+0.0688)-0.1218(0.0779+0.0439) sod-mse=0.0483(0.0233) gcn-mse=0.0586(0.0290) gcn-final-mse=0.0284(0.0441)
2020-08-03 22:53:50 3000-10553 loss=0.1175(0.0839+0.0336)-0.1261(0.0799+0.0462) sod-mse=0.0171(0.0245) gcn-mse=0.0210(0.0301) gcn-final-mse=0.0295(0.0450)
2020-08-03 22:57:01 4000-10553 loss=0.0717(0.0549+0.0168)-0.1294(0.0815+0.0479) sod-mse=0.0116(0.0253) gcn-mse=0.0224(0.0309) gcn-final-mse=0.0303(0.0460)
2020-08-03 23:00:12 5000-10553 loss=0.1698(0.1091+0.0608)-0.1292(0.0814+0.0478) sod-mse=0.0348(0.0253) gcn-mse=0.0461(0.0309) gcn-final-mse=0.0304(0.0459)
2020-08-03 23:03:23 6000-10553 loss=0.1132(0.0782+0.0349)-0.1293(0.0814+0.0479) sod-mse=0.0176(0.0254) gcn-mse=0.0235(0.0309) gcn-final-mse=0.0304(0.0459)
2020-08-03 23:04:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 23:06:36 7000-10553 loss=0.0539(0.0416+0.0123)-0.1312(0.0823+0.0488) sod-mse=0.0065(0.0259) gcn-mse=0.0089(0.0314) gcn-final-mse=0.0309(0.0465)
2020-08-03 23:07:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 23:09:48 8000-10553 loss=0.0651(0.0492+0.0158)-0.1306(0.0820+0.0485) sod-mse=0.0106(0.0257) gcn-mse=0.0140(0.0313) gcn-final-mse=0.0307(0.0463)
2020-08-03 23:13:00 9000-10553 loss=0.0679(0.0416+0.0263)-0.1315(0.0826+0.0489) sod-mse=0.0175(0.0259) gcn-mse=0.0169(0.0316) gcn-final-mse=0.0310(0.0466)
2020-08-03 23:15:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 23:16:10 10000-10553 loss=0.0335(0.0252+0.0084)-0.1313(0.0825+0.0488) sod-mse=0.0048(0.0259) gcn-mse=0.0066(0.0315) gcn-final-mse=0.0310(0.0465)

2020-08-03 23:17:57    0-5019 loss=0.8254(0.4837+0.3417)-0.8254(0.4837+0.3417) sod-mse=0.0893(0.0893) gcn-mse=0.0990(0.0990) gcn-final-mse=0.0916(0.1051)
2020-08-03 23:19:30 1000-5019 loss=0.0492(0.0338+0.0154)-0.3227(0.1744+0.1483) sod-mse=0.0127(0.0679) gcn-mse=0.0140(0.0675) gcn-final-mse=0.0674(0.0818)
2020-08-03 23:21:03 2000-5019 loss=0.7336(0.4060+0.3276)-0.3391(0.1809+0.1582) sod-mse=0.1007(0.0696) gcn-mse=0.0953(0.0690) gcn-final-mse=0.0689(0.0833)
2020-08-03 23:22:35 3000-5019 loss=0.0470(0.0344+0.0127)-0.3464(0.1845+0.1619) sod-mse=0.0071(0.0708) gcn-mse=0.0080(0.0703) gcn-final-mse=0.0702(0.0845)
2020-08-03 23:23:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-03 23:24:07 4000-5019 loss=0.1116(0.0733+0.0383)-0.3435(0.1831+0.1604) sod-mse=0.0226(0.0706) gcn-mse=0.0209(0.0702) gcn-final-mse=0.0700(0.0843)
2020-08-03 23:24:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-03 23:25:38 5000-5019 loss=0.6397(0.2988+0.3410)-0.3428(0.1828+0.1601) sod-mse=0.1128(0.0704) gcn-mse=0.1039(0.0701) gcn-final-mse=0.0699(0.0842)
2020-08-03 23:25:39 E:15, Train sod-mae-score=0.0259-0.9639 gcn-mae-score=0.0315-0.9334 gcn-final-mse-score=0.0309-0.9361(0.0465/0.9361) loss=0.1311(0.0824+0.0487)
2020-08-03 23:25:39 E:15, Test  sod-mae-score=0.0704-0.8275 gcn-mae-score=0.0701-0.7686 gcn-final-mse-score=0.0699-0.7747(0.0842/0.7747) loss=0.3426(0.1827+0.1599)

2020-08-03 23:25:39 Start Epoch 16
2020-08-03 23:25:39 Epoch:16,lr=0.0001
2020-08-03 23:25:40    0-10553 loss=0.0377(0.0232+0.0145)-0.0377(0.0232+0.0145) sod-mse=0.0128(0.0128) gcn-mse=0.0135(0.0135) gcn-final-mse=0.0121(0.0146)
2020-08-03 23:28:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-03 23:28:50 1000-10553 loss=0.2520(0.1480+0.1040)-0.1244(0.0795+0.0450) sod-mse=0.0700(0.0237) gcn-mse=0.0834(0.0297) gcn-final-mse=0.0290(0.0448)
2020-08-03 23:32:01 2000-10553 loss=0.1557(0.0876+0.0681)-0.1216(0.0781+0.0435) sod-mse=0.0419(0.0229) gcn-mse=0.0339(0.0289) gcn-final-mse=0.0283(0.0440)
2020-08-03 23:35:11 3000-10553 loss=0.1093(0.0759+0.0334)-0.1248(0.0796+0.0452) sod-mse=0.0185(0.0239) gcn-mse=0.0166(0.0297) gcn-final-mse=0.0291(0.0447)
2020-08-03 23:38:23 4000-10553 loss=0.1096(0.0703+0.0393)-0.1256(0.0799+0.0458) sod-mse=0.0223(0.0242) gcn-mse=0.0212(0.0298) gcn-final-mse=0.0292(0.0449)
2020-08-03 23:41:33 5000-10553 loss=0.0653(0.0452+0.0201)-0.1259(0.0800+0.0459) sod-mse=0.0130(0.0243) gcn-mse=0.0158(0.0299) gcn-final-mse=0.0293(0.0450)
2020-08-03 23:44:45 6000-10553 loss=0.1336(0.0874+0.0462)-0.1248(0.0794+0.0454) sod-mse=0.0313(0.0240) gcn-mse=0.0371(0.0296) gcn-final-mse=0.0290(0.0446)
2020-08-03 23:47:56 7000-10553 loss=0.1678(0.1200+0.0478)-0.1269(0.0805+0.0464) sod-mse=0.0261(0.0246) gcn-mse=0.0286(0.0302) gcn-final-mse=0.0296(0.0452)
2020-08-03 23:50:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-03 23:51:08 8000-10553 loss=0.1673(0.1194+0.0479)-0.1269(0.0804+0.0464) sod-mse=0.0310(0.0246) gcn-mse=0.0632(0.0302) gcn-final-mse=0.0296(0.0452)
2020-08-03 23:53:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-03 23:54:19 9000-10553 loss=0.0503(0.0412+0.0091)-0.1258(0.0799+0.0459) sod-mse=0.0042(0.0243) gcn-mse=0.0081(0.0299) gcn-final-mse=0.0293(0.0450)
2020-08-03 23:57:30 10000-10553 loss=0.1416(0.0938+0.0479)-0.1266(0.0803+0.0463) sod-mse=0.0222(0.0245) gcn-mse=0.0308(0.0301) gcn-final-mse=0.0295(0.0452)

2020-08-03 23:59:17    0-5019 loss=0.6561(0.3492+0.3069)-0.6561(0.3492+0.3069) sod-mse=0.0897(0.0897) gcn-mse=0.0932(0.0932) gcn-final-mse=0.0869(0.1001)
2020-08-04 00:00:52 1000-5019 loss=0.0417(0.0329+0.0088)-0.3593(0.1876+0.1717) sod-mse=0.0076(0.0639) gcn-mse=0.0136(0.0663) gcn-final-mse=0.0663(0.0795)
2020-08-04 00:02:26 2000-5019 loss=0.9314(0.4426+0.4888)-0.3596(0.1875+0.1721) sod-mse=0.0962(0.0645) gcn-mse=0.0952(0.0668) gcn-final-mse=0.0668(0.0799)
2020-08-04 00:04:00 3000-5019 loss=0.0533(0.0382+0.0152)-0.3666(0.1909+0.1757) sod-mse=0.0078(0.0654) gcn-mse=0.0106(0.0678) gcn-final-mse=0.0678(0.0808)
2020-08-04 00:04:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 00:05:34 4000-5019 loss=0.1306(0.0897+0.0408)-0.3599(0.1882+0.1717) sod-mse=0.0205(0.0647) gcn-mse=0.0317(0.0674) gcn-final-mse=0.0674(0.0804)
2020-08-04 00:06:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 00:07:08 5000-5019 loss=1.0585(0.4325+0.6260)-0.3596(0.1884+0.1712) sod-mse=0.0887(0.0646) gcn-mse=0.0917(0.0674) gcn-final-mse=0.0673(0.0803)
2020-08-04 00:07:09 E:16, Train sod-mae-score=0.0248-0.9656 gcn-mae-score=0.0303-0.9350 gcn-final-mse-score=0.0298-0.9377(0.0454/0.9377) loss=0.1273(0.0807+0.0467)
2020-08-04 00:07:09 E:16, Test  sod-mae-score=0.0646-0.8223 gcn-mae-score=0.0674-0.7647 gcn-final-mse-score=0.0673-0.7706(0.0803/0.7706) loss=0.3594(0.1883+0.1711)

2020-08-04 00:07:09 Start Epoch 17
2020-08-04 00:07:09 Epoch:17,lr=0.0001
2020-08-04 00:07:11    0-10553 loss=0.1017(0.0754+0.0263)-0.1017(0.0754+0.0263) sod-mse=0.0141(0.0141) gcn-mse=0.0225(0.0225) gcn-final-mse=0.0202(0.0438)
2020-08-04 00:10:24 1000-10553 loss=0.0835(0.0648+0.0187)-0.1173(0.0754+0.0419) sod-mse=0.0147(0.0220) gcn-mse=0.0141(0.0272) gcn-final-mse=0.0267(0.0424)
2020-08-04 00:13:35 2000-10553 loss=0.3488(0.2081+0.1407)-0.1136(0.0735+0.0401) sod-mse=0.0758(0.0209) gcn-mse=0.0912(0.0260) gcn-final-mse=0.0255(0.0411)
2020-08-04 00:16:46 3000-10553 loss=0.1406(0.0821+0.0585)-0.1173(0.0755+0.0418) sod-mse=0.0421(0.0218) gcn-mse=0.0358(0.0270) gcn-final-mse=0.0264(0.0421)
2020-08-04 00:19:57 4000-10553 loss=0.1381(0.0882+0.0499)-0.1203(0.0769+0.0434) sod-mse=0.0311(0.0227) gcn-mse=0.0392(0.0280) gcn-final-mse=0.0274(0.0430)
2020-08-04 00:20:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 00:23:09 5000-10553 loss=0.1344(0.0892+0.0451)-0.1215(0.0775+0.0439) sod-mse=0.0288(0.0230) gcn-mse=0.0338(0.0283) gcn-final-mse=0.0277(0.0434)
2020-08-04 00:26:20 6000-10553 loss=0.2396(0.1495+0.0901)-0.1220(0.0778+0.0442) sod-mse=0.0508(0.0232) gcn-mse=0.0610(0.0285) gcn-final-mse=0.0279(0.0436)
2020-08-04 00:29:32 7000-10553 loss=0.0535(0.0407+0.0129)-0.1216(0.0776+0.0441) sod-mse=0.0062(0.0231) gcn-mse=0.0063(0.0284) gcn-final-mse=0.0278(0.0435)
2020-08-04 00:32:45 8000-10553 loss=0.0682(0.0490+0.0192)-0.1211(0.0773+0.0438) sod-mse=0.0128(0.0230) gcn-mse=0.0132(0.0283) gcn-final-mse=0.0277(0.0433)
2020-08-04 00:35:59 9000-10553 loss=0.0347(0.0247+0.0100)-0.1213(0.0774+0.0439) sod-mse=0.0054(0.0230) gcn-mse=0.0090(0.0283) gcn-final-mse=0.0277(0.0434)
2020-08-04 00:37:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 00:39:09 10000-10553 loss=0.1303(0.0910+0.0393)-0.1227(0.0781+0.0446) sod-mse=0.0281(0.0235) gcn-mse=0.0409(0.0287) gcn-final-mse=0.0281(0.0438)
2020-08-04 00:39:27 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg

2020-08-04 00:40:56    0-5019 loss=0.8579(0.4595+0.3984)-0.8579(0.4595+0.3984) sod-mse=0.0972(0.0972) gcn-mse=0.1028(0.1028) gcn-final-mse=0.0965(0.1096)
2020-08-04 00:42:29 1000-5019 loss=0.1488(0.0955+0.0533)-0.3869(0.2038+0.1831) sod-mse=0.0350(0.0710) gcn-mse=0.0387(0.0728) gcn-final-mse=0.0728(0.0863)
2020-08-04 00:44:02 2000-5019 loss=0.7534(0.3994+0.3539)-0.3935(0.2066+0.1868) sod-mse=0.0978(0.0718) gcn-mse=0.0948(0.0738) gcn-final-mse=0.0737(0.0871)
2020-08-04 00:45:34 3000-5019 loss=0.0479(0.0361+0.0117)-0.4014(0.2101+0.1913) sod-mse=0.0064(0.0728) gcn-mse=0.0089(0.0748) gcn-final-mse=0.0748(0.0881)
2020-08-04 00:46:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 00:47:07 4000-5019 loss=0.1176(0.0772+0.0403)-0.3976(0.2085+0.1891) sod-mse=0.0191(0.0725) gcn-mse=0.0211(0.0747) gcn-final-mse=0.0746(0.0879)
2020-08-04 00:47:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 00:48:39 5000-5019 loss=0.9281(0.4095+0.5186)-0.3972(0.2085+0.1887) sod-mse=0.1229(0.0726) gcn-mse=0.1189(0.0749) gcn-final-mse=0.0748(0.0880)
2020-08-04 00:48:41 E:17, Train sod-mae-score=0.0236-0.9669 gcn-mae-score=0.0288-0.9366 gcn-final-mse-score=0.0282-0.9393(0.0439/0.9393) loss=0.1231(0.0782+0.0448)
2020-08-04 00:48:41 E:17, Test  sod-mae-score=0.0726-0.8141 gcn-mae-score=0.0749-0.7571 gcn-final-mse-score=0.0748-0.7630(0.0880/0.7630) loss=0.3970(0.2084+0.1886)

2020-08-04 00:48:41 Start Epoch 18
2020-08-04 00:48:41 Epoch:18,lr=0.0001
2020-08-04 00:48:43    0-10553 loss=0.0350(0.0280+0.0070)-0.0350(0.0280+0.0070) sod-mse=0.0043(0.0043) gcn-mse=0.0080(0.0080) gcn-final-mse=0.0073(0.0154)
2020-08-04 00:51:56 1000-10553 loss=0.0695(0.0526+0.0169)-0.1112(0.0728+0.0385) sod-mse=0.0098(0.0203) gcn-mse=0.0149(0.0259) gcn-final-mse=0.0252(0.0412)
2020-08-04 00:53:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 00:55:05 2000-10553 loss=0.0573(0.0434+0.0140)-0.1189(0.0766+0.0422) sod-mse=0.0092(0.0222) gcn-mse=0.0219(0.0276) gcn-final-mse=0.0270(0.0430)
2020-08-04 00:56:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 00:58:18 3000-10553 loss=0.4282(0.2336+0.1946)-0.1180(0.0762+0.0418) sod-mse=0.1030(0.0220) gcn-mse=0.1075(0.0276) gcn-final-mse=0.0269(0.0428)
2020-08-04 01:01:29 4000-10553 loss=0.0715(0.0547+0.0168)-0.1190(0.0766+0.0424) sod-mse=0.0089(0.0223) gcn-mse=0.0155(0.0279) gcn-final-mse=0.0272(0.0431)
2020-08-04 01:04:39 5000-10553 loss=0.0172(0.0097+0.0075)-0.1218(0.0778+0.0440) sod-mse=0.0042(0.0229) gcn-mse=0.0033(0.0284) gcn-final-mse=0.0277(0.0436)
2020-08-04 01:07:51 6000-10553 loss=0.1289(0.0892+0.0397)-0.1195(0.0767+0.0428) sod-mse=0.0238(0.0223) gcn-mse=0.0252(0.0278) gcn-final-mse=0.0272(0.0430)
2020-08-04 01:11:03 7000-10553 loss=0.0838(0.0586+0.0252)-0.1197(0.0767+0.0430) sod-mse=0.0125(0.0225) gcn-mse=0.0160(0.0279) gcn-final-mse=0.0273(0.0431)
2020-08-04 01:14:15 8000-10553 loss=0.0451(0.0356+0.0095)-0.1187(0.0762+0.0425) sod-mse=0.0047(0.0223) gcn-mse=0.0081(0.0277) gcn-final-mse=0.0271(0.0428)
2020-08-04 01:17:26 9000-10553 loss=0.1841(0.1188+0.0654)-0.1201(0.0770+0.0432) sod-mse=0.0356(0.0227) gcn-mse=0.0375(0.0281) gcn-final-mse=0.0275(0.0433)
2020-08-04 01:17:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 01:20:38 10000-10553 loss=0.0874(0.0713+0.0161)-0.1207(0.0772+0.0435) sod-mse=0.0094(0.0228) gcn-mse=0.0265(0.0282) gcn-final-mse=0.0276(0.0434)

2020-08-04 01:22:25    0-5019 loss=0.7026(0.3480+0.3546)-0.7026(0.3480+0.3546) sod-mse=0.0817(0.0817) gcn-mse=0.0844(0.0844) gcn-final-mse=0.0783(0.0917)
2020-08-04 01:23:59 1000-5019 loss=0.1123(0.0665+0.0458)-0.3831(0.1957+0.1874) sod-mse=0.0203(0.0622) gcn-mse=0.0247(0.0671) gcn-final-mse=0.0670(0.0797)
2020-08-04 01:25:33 2000-5019 loss=0.6449(0.3045+0.3404)-0.4005(0.2032+0.1974) sod-mse=0.0878(0.0638) gcn-mse=0.0842(0.0687) gcn-final-mse=0.0685(0.0811)
2020-08-04 01:27:07 3000-5019 loss=0.0485(0.0370+0.0115)-0.4075(0.2061+0.2014) sod-mse=0.0055(0.0646) gcn-mse=0.0092(0.0697) gcn-final-mse=0.0696(0.0820)
2020-08-04 01:27:59 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 01:28:40 4000-5019 loss=0.1180(0.0796+0.0384)-0.4052(0.2055+0.1998) sod-mse=0.0196(0.0644) gcn-mse=0.0219(0.0696) gcn-final-mse=0.0694(0.0819)
2020-08-04 01:29:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 01:30:14 5000-5019 loss=2.0509(0.6215+1.4294)-0.4054(0.2059+0.1995) sod-mse=0.0949(0.0648) gcn-mse=0.0984(0.0700) gcn-final-mse=0.0698(0.0823)
2020-08-04 01:30:16 E:18, Train sod-mae-score=0.0228-0.9681 gcn-mae-score=0.0282-0.9374 gcn-final-mse-score=0.0276-0.9402(0.0434/0.9402) loss=0.1206(0.0772+0.0434)
2020-08-04 01:30:16 E:18, Test  sod-mae-score=0.0648-0.8246 gcn-mae-score=0.0700-0.7655 gcn-final-mse-score=0.0698-0.7719(0.0823/0.7719) loss=0.4051(0.2058+0.1993)

2020-08-04 01:30:16 Start Epoch 19
2020-08-04 01:30:16 Epoch:19,lr=0.0001
2020-08-04 01:30:17    0-10553 loss=0.1193(0.0729+0.0464)-0.1193(0.0729+0.0464) sod-mse=0.0208(0.0208) gcn-mse=0.0232(0.0232) gcn-final-mse=0.0258(0.0413)
2020-08-04 01:33:30 1000-10553 loss=0.0096(0.0066+0.0030)-0.1179(0.0759+0.0420) sod-mse=0.0024(0.0224) gcn-mse=0.0047(0.0278) gcn-final-mse=0.0273(0.0429)
2020-08-04 01:36:41 2000-10553 loss=0.0849(0.0628+0.0221)-0.1121(0.0728+0.0393) sod-mse=0.0107(0.0208) gcn-mse=0.0152(0.0260) gcn-final-mse=0.0254(0.0411)
2020-08-04 01:39:53 3000-10553 loss=0.0834(0.0635+0.0199)-0.1146(0.0740+0.0406) sod-mse=0.0115(0.0214) gcn-mse=0.0193(0.0266) gcn-final-mse=0.0259(0.0417)
2020-08-04 01:41:58 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 01:43:02 4000-10553 loss=0.0291(0.0208+0.0083)-0.1168(0.0751+0.0416) sod-mse=0.0040(0.0219) gcn-mse=0.0048(0.0271) gcn-final-mse=0.0264(0.0421)
2020-08-04 01:46:06 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 01:46:13 5000-10553 loss=0.0963(0.0730+0.0233)-0.1178(0.0756+0.0422) sod-mse=0.0121(0.0221) gcn-mse=0.0160(0.0272) gcn-final-mse=0.0265(0.0423)
2020-08-04 01:49:23 6000-10553 loss=0.0523(0.0378+0.0145)-0.1169(0.0752+0.0418) sod-mse=0.0096(0.0218) gcn-mse=0.0184(0.0269) gcn-final-mse=0.0263(0.0420)
2020-08-04 01:52:34 7000-10553 loss=0.0623(0.0462+0.0161)-0.1169(0.0752+0.0417) sod-mse=0.0092(0.0218) gcn-mse=0.0131(0.0270) gcn-final-mse=0.0263(0.0421)
2020-08-04 01:53:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 01:55:44 8000-10553 loss=0.0714(0.0518+0.0195)-0.1169(0.0752+0.0417) sod-mse=0.0112(0.0218) gcn-mse=0.0153(0.0270) gcn-final-mse=0.0263(0.0421)
2020-08-04 01:58:54 9000-10553 loss=0.1079(0.0733+0.0346)-0.1157(0.0747+0.0411) sod-mse=0.0233(0.0215) gcn-mse=0.0268(0.0267) gcn-final-mse=0.0260(0.0419)
2020-08-04 02:02:07 10000-10553 loss=0.1071(0.0770+0.0301)-0.1161(0.0748+0.0413) sod-mse=0.0135(0.0217) gcn-mse=0.0224(0.0268) gcn-final-mse=0.0262(0.0420)

2020-08-04 02:03:53    0-5019 loss=1.0248(0.4942+0.5307)-1.0248(0.4942+0.5307) sod-mse=0.1082(0.1082) gcn-mse=0.1177(0.1177) gcn-final-mse=0.1107(0.1225)
2020-08-04 02:05:28 1000-5019 loss=0.0446(0.0350+0.0096)-0.3615(0.1791+0.1824) sod-mse=0.0084(0.0595) gcn-mse=0.0156(0.0626) gcn-final-mse=0.0625(0.0758)
2020-08-04 02:07:00 2000-5019 loss=1.0265(0.4608+0.5656)-0.3655(0.1809+0.1846) sod-mse=0.0954(0.0602) gcn-mse=0.0923(0.0634) gcn-final-mse=0.0633(0.0765)
2020-08-04 02:08:32 3000-5019 loss=0.0487(0.0350+0.0137)-0.3728(0.1838+0.1890) sod-mse=0.0072(0.0609) gcn-mse=0.0082(0.0642) gcn-final-mse=0.0642(0.0773)
2020-08-04 02:09:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 02:10:04 4000-5019 loss=0.1203(0.0746+0.0457)-0.3718(0.1835+0.1883) sod-mse=0.0229(0.0607) gcn-mse=0.0210(0.0642) gcn-final-mse=0.0641(0.0772)
2020-08-04 02:10:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 02:11:36 5000-5019 loss=1.2790(0.5110+0.7680)-0.3741(0.1848+0.1893) sod-mse=0.0920(0.0609) gcn-mse=0.0969(0.0645) gcn-final-mse=0.0644(0.0775)
2020-08-04 02:11:37 E:19, Train sod-mae-score=0.0220-0.9689 gcn-mae-score=0.0271-0.9383 gcn-final-mse-score=0.0265-0.9409(0.0423/0.9409) loss=0.1172(0.0754+0.0418)
2020-08-04 02:11:37 E:19, Test  sod-mae-score=0.0609-0.8347 gcn-mae-score=0.0645-0.7766 gcn-final-mse-score=0.0644-0.7827(0.0775/0.7827) loss=0.3738(0.1847+0.1891)

2020-08-04 02:11:37 Start Epoch 20
2020-08-04 02:11:37 Epoch:20,lr=0.0000
2020-08-04 02:11:39    0-10553 loss=0.0473(0.0360+0.0113)-0.0473(0.0360+0.0113) sod-mse=0.0057(0.0057) gcn-mse=0.0102(0.0102) gcn-final-mse=0.0097(0.0215)
2020-08-04 02:14:51 1000-10553 loss=0.0907(0.0617+0.0290)-0.1060(0.0697+0.0362) sod-mse=0.0140(0.0190) gcn-mse=0.0162(0.0243) gcn-final-mse=0.0237(0.0401)
2020-08-04 02:15:50 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 02:18:03 2000-10553 loss=0.1344(0.0842+0.0502)-0.1038(0.0688+0.0350) sod-mse=0.0231(0.0183) gcn-mse=0.0285(0.0236) gcn-final-mse=0.0229(0.0393)
2020-08-04 02:21:12 3000-10553 loss=0.0474(0.0329+0.0145)-0.1027(0.0682+0.0345) sod-mse=0.0066(0.0179) gcn-mse=0.0075(0.0231) gcn-final-mse=0.0224(0.0387)
2020-08-04 02:23:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 02:24:24 4000-10553 loss=0.0522(0.0415+0.0108)-0.1006(0.0671+0.0336) sod-mse=0.0060(0.0174) gcn-mse=0.0097(0.0225) gcn-final-mse=0.0219(0.0381)
2020-08-04 02:27:33 5000-10553 loss=0.0617(0.0468+0.0149)-0.1004(0.0668+0.0335) sod-mse=0.0100(0.0173) gcn-mse=0.0134(0.0223) gcn-final-mse=0.0216(0.0379)
2020-08-04 02:28:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 02:30:43 6000-10553 loss=0.0868(0.0619+0.0249)-0.0994(0.0662+0.0331) sod-mse=0.0173(0.0171) gcn-mse=0.0237(0.0220) gcn-final-mse=0.0213(0.0375)
2020-08-04 02:33:53 7000-10553 loss=0.0589(0.0410+0.0179)-0.0990(0.0661+0.0329) sod-mse=0.0117(0.0170) gcn-mse=0.0171(0.0219) gcn-final-mse=0.0212(0.0374)
2020-08-04 02:37:04 8000-10553 loss=0.0727(0.0571+0.0156)-0.0979(0.0655+0.0323) sod-mse=0.0062(0.0167) gcn-mse=0.0116(0.0215) gcn-final-mse=0.0208(0.0370)
2020-08-04 02:40:15 9000-10553 loss=0.0686(0.0555+0.0131)-0.0973(0.0653+0.0320) sod-mse=0.0087(0.0165) gcn-mse=0.0150(0.0213) gcn-final-mse=0.0206(0.0368)
2020-08-04 02:43:26 10000-10553 loss=0.1404(0.1014+0.0390)-0.0969(0.0651+0.0318) sod-mse=0.0235(0.0164) gcn-mse=0.0359(0.0211) gcn-final-mse=0.0204(0.0367)

2020-08-04 02:45:13    0-5019 loss=0.8588(0.4858+0.3730)-0.8588(0.4858+0.3730) sod-mse=0.0824(0.0824) gcn-mse=0.0888(0.0888) gcn-final-mse=0.0821(0.0950)
2020-08-04 02:46:46 1000-5019 loss=0.0342(0.0277+0.0065)-0.3460(0.1766+0.1695) sod-mse=0.0055(0.0549) gcn-mse=0.0093(0.0587) gcn-final-mse=0.0586(0.0721)
2020-08-04 02:48:17 2000-5019 loss=1.0055(0.4895+0.5159)-0.3539(0.1796+0.1743) sod-mse=0.0951(0.0564) gcn-mse=0.0926(0.0601) gcn-final-mse=0.0599(0.0733)
2020-08-04 02:49:49 3000-5019 loss=0.0456(0.0341+0.0115)-0.3582(0.1814+0.1768) sod-mse=0.0055(0.0569) gcn-mse=0.0074(0.0607) gcn-final-mse=0.0605(0.0739)
2020-08-04 02:50:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 02:51:20 4000-5019 loss=0.1044(0.0706+0.0338)-0.3571(0.1811+0.1760) sod-mse=0.0160(0.0566) gcn-mse=0.0153(0.0606) gcn-final-mse=0.0604(0.0738)
2020-08-04 02:51:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 02:52:51 5000-5019 loss=1.4151(0.5590+0.8562)-0.3576(0.1814+0.1762) sod-mse=0.0880(0.0567) gcn-mse=0.0905(0.0608) gcn-final-mse=0.0605(0.0738)
2020-08-04 02:52:52 E:20, Train sod-mae-score=0.0163-0.9763 gcn-mae-score=0.0210-0.9468 gcn-final-mse-score=0.0203-0.9494(0.0365/0.9494) loss=0.0964(0.0648+0.0316)
2020-08-04 02:52:52 E:20, Test  sod-mae-score=0.0567-0.8415 gcn-mae-score=0.0608-0.7839 gcn-final-mse-score=0.0605-0.7898(0.0739/0.7898) loss=0.3573(0.1813+0.1760)

2020-08-04 02:52:52 Start Epoch 21
2020-08-04 02:52:52 Epoch:21,lr=0.0000
2020-08-04 02:52:54    0-10553 loss=0.0546(0.0387+0.0159)-0.0546(0.0387+0.0159) sod-mse=0.0090(0.0090) gcn-mse=0.0108(0.0108) gcn-final-mse=0.0105(0.0245)
2020-08-04 02:55:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 02:56:05 1000-10553 loss=0.0763(0.0493+0.0270)-0.0898(0.0616+0.0282) sod-mse=0.0135(0.0145) gcn-mse=0.0162(0.0189) gcn-final-mse=0.0181(0.0344)
2020-08-04 02:59:17 2000-10553 loss=0.0706(0.0434+0.0272)-0.0895(0.0612+0.0283) sod-mse=0.0184(0.0145) gcn-mse=0.0153(0.0187) gcn-final-mse=0.0180(0.0343)
2020-08-04 03:02:28 3000-10553 loss=0.0963(0.0672+0.0291)-0.0888(0.0609+0.0279) sod-mse=0.0136(0.0142) gcn-mse=0.0174(0.0185) gcn-final-mse=0.0177(0.0340)
2020-08-04 03:05:39 4000-10553 loss=0.0672(0.0470+0.0203)-0.0880(0.0605+0.0275) sod-mse=0.0135(0.0141) gcn-mse=0.0114(0.0183) gcn-final-mse=0.0176(0.0338)
2020-08-04 03:08:49 5000-10553 loss=0.1131(0.0830+0.0300)-0.0885(0.0606+0.0279) sod-mse=0.0208(0.0142) gcn-mse=0.0293(0.0184) gcn-final-mse=0.0177(0.0339)
2020-08-04 03:12:00 6000-10553 loss=0.0577(0.0423+0.0154)-0.0887(0.0607+0.0280) sod-mse=0.0078(0.0142) gcn-mse=0.0077(0.0184) gcn-final-mse=0.0177(0.0339)
2020-08-04 03:13:59 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 03:15:10 7000-10553 loss=0.0861(0.0636+0.0225)-0.0887(0.0607+0.0280) sod-mse=0.0090(0.0142) gcn-mse=0.0202(0.0184) gcn-final-mse=0.0177(0.0339)
2020-08-04 03:18:20 8000-10553 loss=0.0389(0.0268+0.0121)-0.0891(0.0609+0.0282) sod-mse=0.0067(0.0143) gcn-mse=0.0078(0.0184) gcn-final-mse=0.0177(0.0341)
2020-08-04 03:19:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 03:21:32 9000-10553 loss=0.0403(0.0281+0.0121)-0.0888(0.0608+0.0280) sod-mse=0.0068(0.0142) gcn-mse=0.0090(0.0184) gcn-final-mse=0.0177(0.0340)
2020-08-04 03:24:46 10000-10553 loss=0.0380(0.0276+0.0104)-0.0889(0.0609+0.0280) sod-mse=0.0037(0.0143) gcn-mse=0.0070(0.0184) gcn-final-mse=0.0177(0.0340)

2020-08-04 03:26:34    0-5019 loss=0.9017(0.5214+0.3803)-0.9017(0.5214+0.3803) sod-mse=0.0853(0.0853) gcn-mse=0.0913(0.0913) gcn-final-mse=0.0842(0.0978)
2020-08-04 03:28:08 1000-5019 loss=0.0334(0.0270+0.0064)-0.3485(0.1779+0.1706) sod-mse=0.0054(0.0543) gcn-mse=0.0086(0.0578) gcn-final-mse=0.0576(0.0711)
2020-08-04 03:29:42 2000-5019 loss=1.0594(0.5137+0.5457)-0.3581(0.1817+0.1765) sod-mse=0.0945(0.0560) gcn-mse=0.0922(0.0593) gcn-final-mse=0.0590(0.0725)
2020-08-04 03:31:14 3000-5019 loss=0.0457(0.0342+0.0115)-0.3622(0.1832+0.1789) sod-mse=0.0054(0.0565) gcn-mse=0.0073(0.0599) gcn-final-mse=0.0597(0.0730)
2020-08-04 03:32:05 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 03:32:46 4000-5019 loss=0.1036(0.0713+0.0323)-0.3598(0.1824+0.1774) sod-mse=0.0152(0.0560) gcn-mse=0.0155(0.0596) gcn-final-mse=0.0593(0.0727)
2020-08-04 03:33:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 03:34:18 5000-5019 loss=1.3590(0.5340+0.8251)-0.3605(0.1826+0.1779) sod-mse=0.0874(0.0560) gcn-mse=0.0894(0.0597) gcn-final-mse=0.0594(0.0728)
2020-08-04 03:34:20 E:21, Train sod-mae-score=0.0142-0.9789 gcn-mae-score=0.0183-0.9496 gcn-final-mse-score=0.0176-0.9521(0.0340/0.9521) loss=0.0888(0.0608+0.0280)
2020-08-04 03:34:20 E:21, Test  sod-mae-score=0.0560-0.8410 gcn-mae-score=0.0597-0.7857 gcn-final-mse-score=0.0594-0.7918(0.0728/0.7918) loss=0.3602(0.1825+0.1777)

2020-08-04 03:34:20 Start Epoch 22
2020-08-04 03:34:20 Epoch:22,lr=0.0000
2020-08-04 03:34:21    0-10553 loss=0.1054(0.0735+0.0319)-0.1054(0.0735+0.0319) sod-mse=0.0102(0.0102) gcn-mse=0.0157(0.0157) gcn-final-mse=0.0142(0.0331)
2020-08-04 03:37:33 1000-10553 loss=0.1202(0.0989+0.0213)-0.0849(0.0591+0.0258) sod-mse=0.0102(0.0133) gcn-mse=0.0246(0.0174) gcn-final-mse=0.0167(0.0334)
2020-08-04 03:40:44 2000-10553 loss=0.0760(0.0536+0.0224)-0.0875(0.0601+0.0274) sod-mse=0.0135(0.0138) gcn-mse=0.0211(0.0176) gcn-final-mse=0.0169(0.0335)
2020-08-04 03:43:55 3000-10553 loss=0.0687(0.0554+0.0133)-0.0858(0.0592+0.0266) sod-mse=0.0095(0.0135) gcn-mse=0.0087(0.0174) gcn-final-mse=0.0167(0.0332)
2020-08-04 03:47:06 4000-10553 loss=0.0214(0.0150+0.0065)-0.0857(0.0592+0.0266) sod-mse=0.0029(0.0135) gcn-mse=0.0044(0.0174) gcn-final-mse=0.0167(0.0332)
2020-08-04 03:50:16 5000-10553 loss=0.1807(0.1420+0.0387)-0.0861(0.0594+0.0268) sod-mse=0.0199(0.0136) gcn-mse=0.0318(0.0175) gcn-final-mse=0.0167(0.0333)
2020-08-04 03:50:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 03:53:26 6000-10553 loss=0.0559(0.0428+0.0131)-0.0863(0.0595+0.0268) sod-mse=0.0090(0.0136) gcn-mse=0.0092(0.0174) gcn-final-mse=0.0167(0.0333)
2020-08-04 03:56:35 7000-10553 loss=0.0600(0.0470+0.0130)-0.0860(0.0593+0.0267) sod-mse=0.0072(0.0135) gcn-mse=0.0153(0.0174) gcn-final-mse=0.0167(0.0332)
2020-08-04 03:59:45 8000-10553 loss=0.0272(0.0231+0.0041)-0.0863(0.0595+0.0268) sod-mse=0.0019(0.0136) gcn-mse=0.0044(0.0174) gcn-final-mse=0.0167(0.0333)
2020-08-04 04:02:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 04:02:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 04:02:53 9000-10553 loss=0.0639(0.0502+0.0137)-0.0862(0.0595+0.0268) sod-mse=0.0068(0.0136) gcn-mse=0.0147(0.0174) gcn-final-mse=0.0167(0.0333)
2020-08-04 04:06:05 10000-10553 loss=0.0499(0.0358+0.0141)-0.0860(0.0593+0.0267) sod-mse=0.0073(0.0135) gcn-mse=0.0073(0.0173) gcn-final-mse=0.0166(0.0331)

2020-08-04 04:07:51    0-5019 loss=0.9716(0.5685+0.4031)-0.9716(0.5685+0.4031) sod-mse=0.0900(0.0900) gcn-mse=0.0961(0.0961) gcn-final-mse=0.0887(0.1014)
2020-08-04 04:09:23 1000-5019 loss=0.0338(0.0275+0.0062)-0.3543(0.1801+0.1742) sod-mse=0.0052(0.0546) gcn-mse=0.0089(0.0580) gcn-final-mse=0.0578(0.0714)
2020-08-04 04:10:55 2000-5019 loss=0.9826(0.4819+0.5007)-0.3652(0.1842+0.1810) sod-mse=0.0928(0.0561) gcn-mse=0.0906(0.0595) gcn-final-mse=0.0592(0.0727)
2020-08-04 04:12:27 3000-5019 loss=0.0457(0.0341+0.0115)-0.3676(0.1851+0.1825) sod-mse=0.0053(0.0564) gcn-mse=0.0070(0.0598) gcn-final-mse=0.0596(0.0730)
2020-08-04 04:13:18 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 04:13:58 4000-5019 loss=0.1031(0.0709+0.0323)-0.3656(0.1844+0.1812) sod-mse=0.0146(0.0559) gcn-mse=0.0147(0.0596) gcn-final-mse=0.0593(0.0727)
2020-08-04 04:14:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 04:15:29 5000-5019 loss=1.3131(0.5284+0.7847)-0.3664(0.1847+0.1818) sod-mse=0.0875(0.0560) gcn-mse=0.0897(0.0597) gcn-final-mse=0.0594(0.0728)
2020-08-04 04:15:31 E:22, Train sod-mae-score=0.0135-0.9798 gcn-mae-score=0.0173-0.9509 gcn-final-mse-score=0.0166-0.9535(0.0331/0.9535) loss=0.0859(0.0593+0.0266)
2020-08-04 04:15:31 E:22, Test  sod-mae-score=0.0560-0.8408 gcn-mae-score=0.0597-0.7856 gcn-final-mse-score=0.0594-0.7917(0.0728/0.7917) loss=0.3662(0.1846+0.1816)

2020-08-04 04:15:31 Start Epoch 23
2020-08-04 04:15:31 Epoch:23,lr=0.0000
2020-08-04 04:15:32    0-10553 loss=0.0426(0.0334+0.0093)-0.0426(0.0334+0.0093) sod-mse=0.0046(0.0046) gcn-mse=0.0059(0.0059) gcn-final-mse=0.0051(0.0180)
2020-08-04 04:18:42 1000-10553 loss=0.0479(0.0363+0.0116)-0.0828(0.0576+0.0252) sod-mse=0.0087(0.0127) gcn-mse=0.0120(0.0165) gcn-final-mse=0.0157(0.0322)
2020-08-04 04:21:52 2000-10553 loss=0.0809(0.0495+0.0314)-0.0832(0.0579+0.0252) sod-mse=0.0148(0.0127) gcn-mse=0.0147(0.0164) gcn-final-mse=0.0157(0.0322)
2020-08-04 04:25:02 3000-10553 loss=0.0696(0.0509+0.0187)-0.0833(0.0579+0.0254) sod-mse=0.0123(0.0128) gcn-mse=0.0140(0.0165) gcn-final-mse=0.0158(0.0322)
2020-08-04 04:26:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 04:28:10 4000-10553 loss=0.0533(0.0407+0.0125)-0.0831(0.0578+0.0253) sod-mse=0.0070(0.0128) gcn-mse=0.0067(0.0165) gcn-final-mse=0.0158(0.0323)
2020-08-04 04:31:19 5000-10553 loss=0.1307(0.0890+0.0418)-0.0832(0.0579+0.0253) sod-mse=0.0171(0.0128) gcn-mse=0.0214(0.0165) gcn-final-mse=0.0158(0.0323)
2020-08-04 04:34:29 6000-10553 loss=0.0781(0.0555+0.0226)-0.0832(0.0579+0.0253) sod-mse=0.0146(0.0128) gcn-mse=0.0204(0.0165) gcn-final-mse=0.0158(0.0323)
2020-08-04 04:37:39 7000-10553 loss=0.0270(0.0225+0.0044)-0.0839(0.0582+0.0257) sod-mse=0.0022(0.0129) gcn-mse=0.0030(0.0166) gcn-final-mse=0.0159(0.0324)
2020-08-04 04:40:49 8000-10553 loss=0.0564(0.0467+0.0097)-0.0837(0.0581+0.0256) sod-mse=0.0046(0.0129) gcn-mse=0.0100(0.0166) gcn-final-mse=0.0159(0.0324)
2020-08-04 04:44:00 9000-10553 loss=0.0758(0.0581+0.0177)-0.0838(0.0582+0.0256) sod-mse=0.0117(0.0129) gcn-mse=0.0161(0.0167) gcn-final-mse=0.0159(0.0325)
2020-08-04 04:45:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 04:47:10 10000-10553 loss=0.1588(0.1110+0.0479)-0.0835(0.0580+0.0255) sod-mse=0.0297(0.0129) gcn-mse=0.0285(0.0166) gcn-final-mse=0.0159(0.0324)
2020-08-04 04:47:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg

2020-08-04 04:48:57    0-5019 loss=0.9540(0.5300+0.4241)-0.9540(0.5300+0.4241) sod-mse=0.0904(0.0904) gcn-mse=0.0977(0.0977) gcn-final-mse=0.0902(0.1033)
2020-08-04 04:50:31 1000-5019 loss=0.0310(0.0254+0.0056)-0.3678(0.1840+0.1838) sod-mse=0.0047(0.0526) gcn-mse=0.0070(0.0561) gcn-final-mse=0.0558(0.0692)
2020-08-04 04:52:05 2000-5019 loss=1.1545(0.5397+0.6148)-0.3762(0.1868+0.1895) sod-mse=0.0950(0.0541) gcn-mse=0.0921(0.0574) gcn-final-mse=0.0571(0.0704)
2020-08-04 04:53:39 3000-5019 loss=0.0454(0.0341+0.0113)-0.3814(0.1887+0.1927) sod-mse=0.0052(0.0545) gcn-mse=0.0071(0.0579) gcn-final-mse=0.0577(0.0708)
2020-08-04 04:54:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 04:55:12 4000-5019 loss=0.1020(0.0699+0.0321)-0.3787(0.1877+0.1910) sod-mse=0.0147(0.0540) gcn-mse=0.0139(0.0576) gcn-final-mse=0.0573(0.0705)
2020-08-04 04:55:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 04:56:45 5000-5019 loss=1.5804(0.6074+0.9730)-0.3788(0.1877+0.1911) sod-mse=0.0866(0.0540) gcn-mse=0.0893(0.0576) gcn-final-mse=0.0573(0.0704)
2020-08-04 04:56:47 E:23, Train sod-mae-score=0.0129-0.9808 gcn-mae-score=0.0166-0.9516 gcn-final-mse-score=0.0159-0.9541(0.0325/0.9541) loss=0.0835(0.0581+0.0255)
2020-08-04 04:56:47 E:23, Test  sod-mae-score=0.0540-0.8445 gcn-mae-score=0.0576-0.7889 gcn-final-mse-score=0.0573-0.7950(0.0705/0.7950) loss=0.3785(0.1876+0.1909)

2020-08-04 04:56:47 Start Epoch 24
2020-08-04 04:56:47 Epoch:24,lr=0.0000
2020-08-04 04:56:48    0-10553 loss=0.0700(0.0551+0.0148)-0.0700(0.0551+0.0148) sod-mse=0.0078(0.0078) gcn-mse=0.0092(0.0092) gcn-final-mse=0.0094(0.0340)
2020-08-04 04:59:59 1000-10553 loss=0.0682(0.0479+0.0203)-0.0807(0.0566+0.0241) sod-mse=0.0123(0.0124) gcn-mse=0.0151(0.0160) gcn-final-mse=0.0153(0.0319)
2020-08-04 05:03:09 2000-10553 loss=0.0628(0.0431+0.0196)-0.0804(0.0566+0.0238) sod-mse=0.0110(0.0121) gcn-mse=0.0130(0.0158) gcn-final-mse=0.0151(0.0318)
2020-08-04 05:06:21 3000-10553 loss=0.0555(0.0438+0.0117)-0.0804(0.0565+0.0239) sod-mse=0.0089(0.0122) gcn-mse=0.0154(0.0159) gcn-final-mse=0.0152(0.0318)
2020-08-04 05:09:32 4000-10553 loss=0.0465(0.0345+0.0120)-0.0805(0.0566+0.0239) sod-mse=0.0062(0.0122) gcn-mse=0.0075(0.0159) gcn-final-mse=0.0152(0.0318)
2020-08-04 05:12:41 5000-10553 loss=0.1044(0.0696+0.0348)-0.0811(0.0570+0.0242) sod-mse=0.0203(0.0123) gcn-mse=0.0199(0.0160) gcn-final-mse=0.0153(0.0319)
2020-08-04 05:15:51 6000-10553 loss=0.0546(0.0390+0.0156)-0.0808(0.0568+0.0240) sod-mse=0.0069(0.0122) gcn-mse=0.0114(0.0159) gcn-final-mse=0.0152(0.0318)
2020-08-04 05:19:01 7000-10553 loss=0.0612(0.0413+0.0199)-0.0812(0.0570+0.0242) sod-mse=0.0082(0.0123) gcn-mse=0.0095(0.0160) gcn-final-mse=0.0153(0.0319)
2020-08-04 05:22:13 8000-10553 loss=0.0893(0.0539+0.0355)-0.0809(0.0568+0.0241) sod-mse=0.0162(0.0122) gcn-mse=0.0198(0.0159) gcn-final-mse=0.0152(0.0318)
2020-08-04 05:24:27 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 05:25:24 9000-10553 loss=0.0413(0.0244+0.0169)-0.0813(0.0570+0.0243) sod-mse=0.0087(0.0123) gcn-mse=0.0104(0.0160) gcn-final-mse=0.0152(0.0318)
2020-08-04 05:28:35 10000-10553 loss=0.1188(0.0858+0.0330)-0.0815(0.0571+0.0244) sod-mse=0.0185(0.0123) gcn-mse=0.0273(0.0160) gcn-final-mse=0.0153(0.0319)
2020-08-04 05:29:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 05:30:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg

2020-08-04 05:30:23    0-5019 loss=0.8588(0.4975+0.3613)-0.8588(0.4975+0.3613) sod-mse=0.0871(0.0871) gcn-mse=0.0917(0.0917) gcn-final-mse=0.0845(0.0978)
2020-08-04 05:31:58 1000-5019 loss=0.0306(0.0248+0.0057)-0.3545(0.1823+0.1722) sod-mse=0.0048(0.0539) gcn-mse=0.0065(0.0566) gcn-final-mse=0.0564(0.0697)
2020-08-04 05:33:33 2000-5019 loss=1.0838(0.5377+0.5460)-0.3645(0.1858+0.1786) sod-mse=0.0955(0.0555) gcn-mse=0.0927(0.0581) gcn-final-mse=0.0578(0.0711)
2020-08-04 05:35:08 3000-5019 loss=0.0454(0.0341+0.0113)-0.3665(0.1865+0.1800) sod-mse=0.0054(0.0556) gcn-mse=0.0074(0.0583) gcn-final-mse=0.0581(0.0713)
2020-08-04 05:36:01 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 05:36:43 4000-5019 loss=0.1053(0.0722+0.0332)-0.3631(0.1852+0.1778) sod-mse=0.0155(0.0550) gcn-mse=0.0162(0.0578) gcn-final-mse=0.0576(0.0708)
2020-08-04 05:37:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 05:38:19 5000-5019 loss=1.5330(0.6061+0.9269)-0.3640(0.1856+0.1784) sod-mse=0.0871(0.0550) gcn-mse=0.0895(0.0579) gcn-final-mse=0.0576(0.0708)
2020-08-04 05:38:20 E:24, Train sod-mae-score=0.0124-0.9815 gcn-mae-score=0.0160-0.9523 gcn-final-mse-score=0.0153-0.9548(0.0319/0.9548) loss=0.0817(0.0572+0.0245)
2020-08-04 05:38:20 E:24, Test  sod-mae-score=0.0550-0.8424 gcn-mae-score=0.0579-0.7874 gcn-final-mse-score=0.0577-0.7937(0.0708/0.7937) loss=0.3638(0.1855+0.1783)

2020-08-04 05:38:20 Start Epoch 25
2020-08-04 05:38:20 Epoch:25,lr=0.0000
2020-08-04 05:38:21    0-10553 loss=0.0850(0.0669+0.0181)-0.0850(0.0669+0.0181) sod-mse=0.0128(0.0128) gcn-mse=0.0181(0.0181) gcn-final-mse=0.0150(0.0363)
2020-08-04 05:39:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 05:41:34 1000-10553 loss=0.0281(0.0225+0.0056)-0.0793(0.0561+0.0232) sod-mse=0.0028(0.0117) gcn-mse=0.0031(0.0154) gcn-final-mse=0.0146(0.0313)
2020-08-04 05:44:45 2000-10553 loss=0.0559(0.0410+0.0149)-0.0806(0.0566+0.0239) sod-mse=0.0105(0.0120) gcn-mse=0.0123(0.0156) gcn-final-mse=0.0148(0.0315)
2020-08-04 05:47:55 3000-10553 loss=0.1650(0.0962+0.0688)-0.0807(0.0566+0.0240) sod-mse=0.0326(0.0121) gcn-mse=0.0350(0.0157) gcn-final-mse=0.0150(0.0316)
2020-08-04 05:51:07 4000-10553 loss=0.0852(0.0613+0.0239)-0.0802(0.0564+0.0238) sod-mse=0.0100(0.0120) gcn-mse=0.0187(0.0156) gcn-final-mse=0.0149(0.0316)
2020-08-04 05:54:19 5000-10553 loss=0.0568(0.0444+0.0124)-0.0801(0.0564+0.0238) sod-mse=0.0056(0.0119) gcn-mse=0.0085(0.0155) gcn-final-mse=0.0148(0.0315)
2020-08-04 05:57:29 6000-10553 loss=0.0858(0.0630+0.0228)-0.0799(0.0563+0.0236) sod-mse=0.0137(0.0119) gcn-mse=0.0179(0.0155) gcn-final-mse=0.0148(0.0314)
2020-08-04 05:58:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 06:00:39 7000-10553 loss=0.0671(0.0393+0.0278)-0.0803(0.0565+0.0238) sod-mse=0.0108(0.0120) gcn-mse=0.0100(0.0156) gcn-final-mse=0.0148(0.0315)
2020-08-04 06:03:50 8000-10553 loss=0.0254(0.0201+0.0053)-0.0800(0.0563+0.0237) sod-mse=0.0034(0.0120) gcn-mse=0.0047(0.0155) gcn-final-mse=0.0148(0.0315)
2020-08-04 06:05:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 06:07:02 9000-10553 loss=0.0481(0.0339+0.0142)-0.0803(0.0564+0.0239) sod-mse=0.0069(0.0120) gcn-mse=0.0091(0.0156) gcn-final-mse=0.0148(0.0315)
2020-08-04 06:10:12 10000-10553 loss=0.0851(0.0585+0.0266)-0.0802(0.0564+0.0238) sod-mse=0.0120(0.0120) gcn-mse=0.0142(0.0155) gcn-final-mse=0.0148(0.0315)

2020-08-04 06:11:58    0-5019 loss=0.9461(0.5139+0.4322)-0.9461(0.5139+0.4322) sod-mse=0.0948(0.0948) gcn-mse=0.0985(0.0985) gcn-final-mse=0.0911(0.1043)
2020-08-04 06:13:31 1000-5019 loss=0.0300(0.0248+0.0052)-0.3753(0.1869+0.1885) sod-mse=0.0043(0.0539) gcn-mse=0.0065(0.0571) gcn-final-mse=0.0568(0.0702)
2020-08-04 06:15:04 2000-5019 loss=1.0826(0.5147+0.5679)-0.3892(0.1920+0.1972) sod-mse=0.0933(0.0557) gcn-mse=0.0911(0.0587) gcn-final-mse=0.0585(0.0717)
2020-08-04 06:16:37 3000-5019 loss=0.0448(0.0338+0.0110)-0.3916(0.1929+0.1987) sod-mse=0.0050(0.0558) gcn-mse=0.0070(0.0589) gcn-final-mse=0.0587(0.0719)
2020-08-04 06:17:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 06:18:09 4000-5019 loss=0.1045(0.0712+0.0333)-0.3880(0.1917+0.1963) sod-mse=0.0147(0.0551) gcn-mse=0.0143(0.0585) gcn-final-mse=0.0582(0.0714)
2020-08-04 06:18:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 06:19:41 5000-5019 loss=1.5487(0.5722+0.9766)-0.3885(0.1919+0.1966) sod-mse=0.0863(0.0551) gcn-mse=0.0898(0.0585) gcn-final-mse=0.0582(0.0714)
2020-08-04 06:19:42 E:25, Train sod-mae-score=0.0120-0.9820 gcn-mae-score=0.0155-0.9527 gcn-final-mse-score=0.0148-0.9552(0.0315/0.9552) loss=0.0802(0.0564+0.0238)
2020-08-04 06:19:42 E:25, Test  sod-mae-score=0.0551-0.8387 gcn-mae-score=0.0586-0.7847 gcn-final-mse-score=0.0583-0.7905(0.0714/0.7905) loss=0.3883(0.1918+0.1965)

2020-08-04 06:19:42 Start Epoch 26
2020-08-04 06:19:42 Epoch:26,lr=0.0000
2020-08-04 06:19:44    0-10553 loss=0.0546(0.0368+0.0177)-0.0546(0.0368+0.0177) sod-mse=0.0070(0.0070) gcn-mse=0.0089(0.0089) gcn-final-mse=0.0078(0.0167)
2020-08-04 06:22:57 1000-10553 loss=0.1256(0.0982+0.0273)-0.0776(0.0552+0.0224) sod-mse=0.0138(0.0113) gcn-mse=0.0239(0.0149) gcn-final-mse=0.0141(0.0308)
2020-08-04 06:26:09 2000-10553 loss=0.0359(0.0274+0.0085)-0.0789(0.0559+0.0230) sod-mse=0.0042(0.0116) gcn-mse=0.0054(0.0152) gcn-final-mse=0.0144(0.0312)
2020-08-04 06:29:19 3000-10553 loss=0.0497(0.0384+0.0113)-0.0782(0.0555+0.0227) sod-mse=0.0054(0.0115) gcn-mse=0.0072(0.0150) gcn-final-mse=0.0143(0.0311)
2020-08-04 06:32:28 4000-10553 loss=0.0626(0.0482+0.0144)-0.0789(0.0559+0.0230) sod-mse=0.0066(0.0116) gcn-mse=0.0086(0.0151) gcn-final-mse=0.0144(0.0313)
2020-08-04 06:33:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 06:35:39 5000-10553 loss=0.0643(0.0455+0.0187)-0.0790(0.0560+0.0230) sod-mse=0.0094(0.0116) gcn-mse=0.0154(0.0152) gcn-final-mse=0.0144(0.0314)
2020-08-04 06:36:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 06:38:50 6000-10553 loss=0.0565(0.0393+0.0172)-0.0788(0.0558+0.0230) sod-mse=0.0076(0.0116) gcn-mse=0.0085(0.0151) gcn-final-mse=0.0144(0.0312)
2020-08-04 06:42:00 7000-10553 loss=0.1314(0.0846+0.0468)-0.0789(0.0558+0.0231) sod-mse=0.0222(0.0116) gcn-mse=0.0269(0.0151) gcn-final-mse=0.0144(0.0312)
2020-08-04 06:45:09 8000-10553 loss=0.0327(0.0247+0.0080)-0.0789(0.0558+0.0232) sod-mse=0.0036(0.0116) gcn-mse=0.0042(0.0151) gcn-final-mse=0.0144(0.0312)
2020-08-04 06:48:18 9000-10553 loss=0.0407(0.0317+0.0090)-0.0788(0.0558+0.0231) sod-mse=0.0049(0.0116) gcn-mse=0.0093(0.0151) gcn-final-mse=0.0144(0.0311)
2020-08-04 06:49:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 06:51:29 10000-10553 loss=0.0469(0.0389+0.0080)-0.0788(0.0557+0.0231) sod-mse=0.0030(0.0116) gcn-mse=0.0066(0.0151) gcn-final-mse=0.0144(0.0311)

2020-08-04 06:53:15    0-5019 loss=0.8417(0.4506+0.3911)-0.8417(0.4506+0.3911) sod-mse=0.0891(0.0891) gcn-mse=0.0924(0.0924) gcn-final-mse=0.0854(0.0985)
2020-08-04 06:54:49 1000-5019 loss=0.0288(0.0238+0.0050)-0.3769(0.1884+0.1884) sod-mse=0.0041(0.0535) gcn-mse=0.0054(0.0567) gcn-final-mse=0.0565(0.0699)
2020-08-04 06:56:21 2000-5019 loss=1.1150(0.5322+0.5828)-0.3884(0.1925+0.1959) sod-mse=0.0951(0.0549) gcn-mse=0.0923(0.0581) gcn-final-mse=0.0578(0.0711)
2020-08-04 06:57:54 3000-5019 loss=0.0443(0.0335+0.0108)-0.3901(0.1930+0.1971) sod-mse=0.0049(0.0549) gcn-mse=0.0067(0.0582) gcn-final-mse=0.0580(0.0712)
2020-08-04 06:58:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 06:59:26 4000-5019 loss=0.1023(0.0702+0.0321)-0.3859(0.1915+0.1944) sod-mse=0.0144(0.0543) gcn-mse=0.0144(0.0577) gcn-final-mse=0.0574(0.0707)
2020-08-04 06:59:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 07:00:57 5000-5019 loss=1.5703(0.6021+0.9682)-0.3868(0.1918+0.1949) sod-mse=0.0857(0.0543) gcn-mse=0.0878(0.0578) gcn-final-mse=0.0575(0.0707)
2020-08-04 07:00:58 E:26, Train sod-mae-score=0.0116-0.9827 gcn-mae-score=0.0151-0.9537 gcn-final-mse-score=0.0144-0.9562(0.0311/0.9562) loss=0.0788(0.0557+0.0231)
2020-08-04 07:00:58 E:26, Test  sod-mae-score=0.0543-0.8404 gcn-mae-score=0.0578-0.7854 gcn-final-mse-score=0.0575-0.7913(0.0708/0.7913) loss=0.3865(0.1917+0.1948)

2020-08-04 07:00:58 Start Epoch 27
2020-08-04 07:00:58 Epoch:27,lr=0.0000
2020-08-04 07:01:00    0-10553 loss=0.0824(0.0496+0.0328)-0.0824(0.0496+0.0328) sod-mse=0.0163(0.0163) gcn-mse=0.0152(0.0152) gcn-final-mse=0.0160(0.0310)
2020-08-04 07:04:11 1000-10553 loss=0.0560(0.0383+0.0177)-0.0764(0.0545+0.0218) sod-mse=0.0089(0.0109) gcn-mse=0.0103(0.0144) gcn-final-mse=0.0137(0.0305)
2020-08-04 07:07:22 2000-10553 loss=0.0789(0.0567+0.0222)-0.0786(0.0557+0.0229) sod-mse=0.0149(0.0114) gcn-mse=0.0134(0.0148) gcn-final-mse=0.0140(0.0309)
2020-08-04 07:08:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 07:10:33 3000-10553 loss=0.0816(0.0634+0.0182)-0.0787(0.0558+0.0229) sod-mse=0.0128(0.0114) gcn-mse=0.0212(0.0148) gcn-final-mse=0.0141(0.0310)
2020-08-04 07:13:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 07:13:43 4000-10553 loss=0.1049(0.0789+0.0260)-0.0788(0.0559+0.0229) sod-mse=0.0181(0.0114) gcn-mse=0.0204(0.0149) gcn-final-mse=0.0141(0.0311)
2020-08-04 07:16:54 5000-10553 loss=0.0457(0.0312+0.0145)-0.0788(0.0558+0.0230) sod-mse=0.0082(0.0115) gcn-mse=0.0091(0.0149) gcn-final-mse=0.0142(0.0311)
2020-08-04 07:20:05 6000-10553 loss=0.0945(0.0585+0.0360)-0.0787(0.0557+0.0229) sod-mse=0.0176(0.0114) gcn-mse=0.0175(0.0149) gcn-final-mse=0.0142(0.0311)
2020-08-04 07:23:15 7000-10553 loss=0.1488(0.0922+0.0566)-0.0780(0.0554+0.0226) sod-mse=0.0244(0.0113) gcn-mse=0.0274(0.0148) gcn-final-mse=0.0141(0.0309)
2020-08-04 07:26:26 8000-10553 loss=0.1091(0.0661+0.0430)-0.0781(0.0554+0.0226) sod-mse=0.0227(0.0113) gcn-mse=0.0226(0.0148) gcn-final-mse=0.0141(0.0310)
2020-08-04 07:29:36 9000-10553 loss=0.0323(0.0251+0.0072)-0.0778(0.0553+0.0225) sod-mse=0.0037(0.0113) gcn-mse=0.0044(0.0148) gcn-final-mse=0.0140(0.0309)
2020-08-04 07:30:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 07:32:47 10000-10553 loss=0.0843(0.0683+0.0160)-0.0776(0.0552+0.0225) sod-mse=0.0083(0.0112) gcn-mse=0.0162(0.0147) gcn-final-mse=0.0140(0.0308)

2020-08-04 07:34:33    0-5019 loss=0.8792(0.5039+0.3753)-0.8792(0.5039+0.3753) sod-mse=0.0912(0.0912) gcn-mse=0.0956(0.0956) gcn-final-mse=0.0882(0.1014)
2020-08-04 07:36:06 1000-5019 loss=0.0295(0.0240+0.0055)-0.3783(0.1930+0.1853) sod-mse=0.0046(0.0564) gcn-mse=0.0056(0.0579) gcn-final-mse=0.0577(0.0712)
2020-08-04 07:37:39 2000-5019 loss=0.9596(0.4907+0.4690)-0.3905(0.1975+0.1930) sod-mse=0.0909(0.0578) gcn-mse=0.0903(0.0592) gcn-final-mse=0.0589(0.0723)
2020-08-04 07:39:11 3000-5019 loss=0.0445(0.0337+0.0108)-0.3936(0.1987+0.1949) sod-mse=0.0051(0.0580) gcn-mse=0.0070(0.0595) gcn-final-mse=0.0593(0.0726)
2020-08-04 07:40:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 07:40:46 4000-5019 loss=0.1013(0.0700+0.0313)-0.3919(0.1981+0.1937) sod-mse=0.0148(0.0574) gcn-mse=0.0140(0.0592) gcn-final-mse=0.0589(0.0723)
2020-08-04 07:41:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 07:42:21 5000-5019 loss=1.3222(0.5496+0.7726)-0.3927(0.1985+0.1942) sod-mse=0.0871(0.0575) gcn-mse=0.0898(0.0593) gcn-final-mse=0.0590(0.0723)
2020-08-04 07:42:22 E:27, Train sod-mae-score=0.0113-0.9829 gcn-mae-score=0.0147-0.9540 gcn-final-mse-score=0.0140-0.9565(0.0308/0.9565) loss=0.0777(0.0552+0.0226)
2020-08-04 07:42:22 E:27, Test  sod-mae-score=0.0575-0.8341 gcn-mae-score=0.0593-0.7820 gcn-final-mse-score=0.0590-0.7880(0.0723/0.7880) loss=0.3925(0.1984+0.1941)

2020-08-04 07:42:22 Start Epoch 28
2020-08-04 07:42:22 Epoch:28,lr=0.0000
2020-08-04 07:42:24    0-10553 loss=0.0830(0.0666+0.0164)-0.0830(0.0666+0.0164) sod-mse=0.0070(0.0070) gcn-mse=0.0138(0.0138) gcn-final-mse=0.0116(0.0333)
2020-08-04 07:44:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 07:45:36 1000-10553 loss=0.1754(0.1090+0.0664)-0.0747(0.0534+0.0213) sod-mse=0.0320(0.0109) gcn-mse=0.0243(0.0142) gcn-final-mse=0.0134(0.0300)
2020-08-04 07:48:46 2000-10553 loss=0.0611(0.0463+0.0148)-0.0755(0.0540+0.0215) sod-mse=0.0098(0.0109) gcn-mse=0.0098(0.0142) gcn-final-mse=0.0135(0.0303)
2020-08-04 07:51:57 3000-10553 loss=0.1015(0.0620+0.0395)-0.0755(0.0540+0.0215) sod-mse=0.0204(0.0109) gcn-mse=0.0236(0.0142) gcn-final-mse=0.0135(0.0304)
2020-08-04 07:55:09 4000-10553 loss=0.0581(0.0434+0.0147)-0.0759(0.0542+0.0216) sod-mse=0.0082(0.0109) gcn-mse=0.0093(0.0142) gcn-final-mse=0.0135(0.0304)
2020-08-04 07:58:20 5000-10553 loss=0.0686(0.0460+0.0226)-0.0761(0.0543+0.0218) sod-mse=0.0130(0.0109) gcn-mse=0.0126(0.0142) gcn-final-mse=0.0135(0.0304)
2020-08-04 08:00:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 08:01:31 6000-10553 loss=0.1236(0.0842+0.0394)-0.0763(0.0544+0.0219) sod-mse=0.0164(0.0110) gcn-mse=0.0254(0.0143) gcn-final-mse=0.0136(0.0305)
2020-08-04 08:01:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 08:04:42 7000-10553 loss=0.0578(0.0464+0.0114)-0.0764(0.0545+0.0219) sod-mse=0.0055(0.0110) gcn-mse=0.0119(0.0143) gcn-final-mse=0.0136(0.0305)
2020-08-04 08:07:52 8000-10553 loss=0.0560(0.0417+0.0143)-0.0763(0.0544+0.0219) sod-mse=0.0069(0.0110) gcn-mse=0.0071(0.0143) gcn-final-mse=0.0136(0.0305)
2020-08-04 08:11:02 9000-10553 loss=0.0464(0.0362+0.0101)-0.0768(0.0547+0.0221) sod-mse=0.0066(0.0111) gcn-mse=0.0061(0.0144) gcn-final-mse=0.0137(0.0306)
2020-08-04 08:14:14 10000-10553 loss=0.0796(0.0608+0.0189)-0.0767(0.0546+0.0221) sod-mse=0.0114(0.0111) gcn-mse=0.0173(0.0144) gcn-final-mse=0.0137(0.0305)

2020-08-04 08:16:01    0-5019 loss=0.9360(0.5091+0.4269)-0.9360(0.5091+0.4269) sod-mse=0.0978(0.0978) gcn-mse=0.1008(0.1008) gcn-final-mse=0.0935(0.1069)
2020-08-04 08:17:35 1000-5019 loss=0.0302(0.0248+0.0054)-0.3786(0.1891+0.1895) sod-mse=0.0045(0.0536) gcn-mse=0.0066(0.0563) gcn-final-mse=0.0561(0.0696)
2020-08-04 08:19:09 2000-5019 loss=1.1316(0.5503+0.5812)-0.3882(0.1925+0.1957) sod-mse=0.0932(0.0548) gcn-mse=0.0931(0.0575) gcn-final-mse=0.0572(0.0706)
2020-08-04 08:20:44 3000-5019 loss=0.0441(0.0334+0.0107)-0.3921(0.1940+0.1981) sod-mse=0.0049(0.0551) gcn-mse=0.0067(0.0578) gcn-final-mse=0.0576(0.0710)
2020-08-04 08:21:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 08:22:19 4000-5019 loss=0.1020(0.0694+0.0326)-0.3884(0.1927+0.1957) sod-mse=0.0150(0.0545) gcn-mse=0.0139(0.0574) gcn-final-mse=0.0572(0.0706)
2020-08-04 08:22:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 08:23:53 5000-5019 loss=1.8377(0.6554+1.1823)-0.3880(0.1925+0.1955) sod-mse=0.0863(0.0544) gcn-mse=0.0893(0.0574) gcn-final-mse=0.0571(0.0704)
2020-08-04 08:23:54 E:28, Train sod-mae-score=0.0110-0.9833 gcn-mae-score=0.0144-0.9544 gcn-final-mse-score=0.0136-0.9569(0.0305/0.9569) loss=0.0767(0.0546+0.0221)
2020-08-04 08:23:54 E:28, Test  sod-mae-score=0.0544-0.8419 gcn-mae-score=0.0574-0.7881 gcn-final-mse-score=0.0571-0.7942(0.0705/0.7942) loss=0.3878(0.1924+0.1953)

2020-08-04 08:23:54 Start Epoch 29
2020-08-04 08:23:54 Epoch:29,lr=0.0000
2020-08-04 08:23:56    0-10553 loss=0.0361(0.0259+0.0103)-0.0361(0.0259+0.0103) sod-mse=0.0051(0.0051) gcn-mse=0.0088(0.0088) gcn-final-mse=0.0086(0.0146)
2020-08-04 08:27:07 1000-10553 loss=0.0523(0.0402+0.0122)-0.0757(0.0542+0.0215) sod-mse=0.0092(0.0108) gcn-mse=0.0150(0.0142) gcn-final-mse=0.0134(0.0304)
2020-08-04 08:30:18 2000-10553 loss=0.0787(0.0572+0.0216)-0.0753(0.0539+0.0213) sod-mse=0.0106(0.0107) gcn-mse=0.0132(0.0140) gcn-final-mse=0.0133(0.0302)
2020-08-04 08:30:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 08:33:30 3000-10553 loss=0.0696(0.0460+0.0236)-0.0755(0.0539+0.0216) sod-mse=0.0116(0.0108) gcn-mse=0.0135(0.0141) gcn-final-mse=0.0133(0.0301)
2020-08-04 08:36:41 4000-10553 loss=0.1052(0.0782+0.0270)-0.0762(0.0544+0.0218) sod-mse=0.0133(0.0109) gcn-mse=0.0194(0.0143) gcn-final-mse=0.0135(0.0304)
2020-08-04 08:37:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 08:39:51 5000-10553 loss=0.0388(0.0300+0.0088)-0.0761(0.0544+0.0218) sod-mse=0.0051(0.0109) gcn-mse=0.0092(0.0142) gcn-final-mse=0.0134(0.0303)
2020-08-04 08:43:04 6000-10553 loss=0.2042(0.1220+0.0822)-0.0760(0.0543+0.0218) sod-mse=0.0346(0.0108) gcn-mse=0.0371(0.0141) gcn-final-mse=0.0134(0.0302)
2020-08-04 08:46:15 7000-10553 loss=0.0903(0.0627+0.0276)-0.0759(0.0542+0.0217) sod-mse=0.0128(0.0108) gcn-mse=0.0224(0.0141) gcn-final-mse=0.0134(0.0302)
2020-08-04 08:48:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 08:49:26 8000-10553 loss=0.1171(0.0674+0.0498)-0.0760(0.0543+0.0217) sod-mse=0.0191(0.0108) gcn-mse=0.0155(0.0141) gcn-final-mse=0.0134(0.0303)
2020-08-04 08:52:37 9000-10553 loss=0.0381(0.0281+0.0101)-0.0758(0.0542+0.0216) sod-mse=0.0035(0.0108) gcn-mse=0.0052(0.0141) gcn-final-mse=0.0133(0.0303)
2020-08-04 08:55:47 10000-10553 loss=0.0591(0.0438+0.0153)-0.0757(0.0541+0.0215) sod-mse=0.0103(0.0108) gcn-mse=0.0097(0.0141) gcn-final-mse=0.0133(0.0303)

2020-08-04 08:57:32    0-5019 loss=0.8982(0.4846+0.4136)-0.8982(0.4846+0.4136) sod-mse=0.0888(0.0888) gcn-mse=0.0932(0.0932) gcn-final-mse=0.0859(0.0990)
2020-08-04 08:59:07 1000-5019 loss=0.0292(0.0243+0.0049)-0.3895(0.1904+0.1992) sod-mse=0.0040(0.0537) gcn-mse=0.0058(0.0568) gcn-final-mse=0.0565(0.0701)
2020-08-04 09:00:43 2000-5019 loss=1.1975(0.5614+0.6361)-0.4010(0.1943+0.2067) sod-mse=0.0949(0.0549) gcn-mse=0.0932(0.0579) gcn-final-mse=0.0577(0.0711)
2020-08-04 09:02:19 3000-5019 loss=0.0443(0.0335+0.0108)-0.4041(0.1953+0.2087) sod-mse=0.0048(0.0550) gcn-mse=0.0071(0.0582) gcn-final-mse=0.0580(0.0713)
2020-08-04 09:03:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 09:03:54 4000-5019 loss=0.1026(0.0696+0.0330)-0.4007(0.1942+0.2065) sod-mse=0.0149(0.0545) gcn-mse=0.0141(0.0578) gcn-final-mse=0.0575(0.0709)
2020-08-04 09:04:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 09:05:29 5000-5019 loss=1.6973(0.6256+1.0717)-0.4009(0.1943+0.2067) sod-mse=0.0857(0.0544) gcn-mse=0.0888(0.0578) gcn-final-mse=0.0575(0.0708)
2020-08-04 09:05:31 E:29, Train sod-mae-score=0.0107-0.9837 gcn-mae-score=0.0141-0.9550 gcn-final-mse-score=0.0133-0.9575(0.0303/0.9575) loss=0.0756(0.0541+0.0215)
2020-08-04 09:05:31 E:29, Test  sod-mae-score=0.0545-0.8391 gcn-mae-score=0.0579-0.7852 gcn-final-mse-score=0.0575-0.7913(0.0709/0.7913) loss=0.4008(0.1942+0.2066)

Process finished with exit code 0
