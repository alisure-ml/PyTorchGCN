/home/ubuntu/anaconda3/envs/alisure36torch/bin/python /mnt/4T/ALISURE/GCN/PyTorchGCN/MyGCN/SPERunner_1_PYG_CONV_Fast_SOD_SGPU_E2E_BS1_MoreConv.py
2020-08-12 10:36:58 name:E2E2-Pretrain_BS1-MoreConv-1-C2PC2PC3C3C3_False_False_lr0001 epochs:40 ckpt:./ckpt2/dgl/1_PYG_CONV_Fast-SOD_BAS/E2E2-Pretrain_BS1-MoreConv-1-C2PC2PC3C3C3_False_False_lr0001 sp size:4 down_ratio:4 workers:16 gpu:1 has_mask:False has_residual:True is_normalize:True has_bn:True improved:True concat:True is_sgd:False weight_decay:0.0

2020-08-12 10:36:58 Cuda available with GPU: GeForce GTX 1080
2020-08-12 10:37:06 Total param: 37303488 lr_s=[[0, 0.0001], [20, 1e-05], [30, 1e-06]] Optimizer=Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.0001
    weight_decay: 0.0
)
2020-08-12 10:37:06 MyGCNNet(
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
    (conv_sod_gcn2): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_sod_gcn1): ConvBlock(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (relu): ReLU(inplace=True)
    )
    (conv_conv4): ConvBlock(
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
    (cat_conv4): ConvBlock(
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
2020-08-12 10:37:06 The number of parameters: 37303488
2020-08-12 10:37:07 Load Model: ./ckpt2/dgl/1_PYG_CONV_Fast-ImageNet/1_4_4_MoreConv/epoch_14.pkl

2020-08-12 10:37:07 Start Epoch 0
2020-08-12 10:37:07 Epoch:00,lr=0.0001
2020-08-12 10:37:09    0-10553 loss=1.4170(0.7203+0.6967)-1.4170(0.7203+0.6967) sod-mse=0.4996(0.4996) gcn-mse=0.4907(0.4907) gcn-final-mse=0.4928(0.5031)
2020-08-12 10:41:55 1000-10553 loss=0.9593(0.5503+0.4090)-0.6586(0.3219+0.3367) sod-mse=0.1819(0.2070) gcn-mse=0.2273(0.1881) gcn-final-mse=0.1884(0.2021)
2020-08-12 10:46:57 2000-10553 loss=0.5290(0.2421+0.2869)-0.5809(0.2890+0.2919) sod-mse=0.1941(0.1752) gcn-mse=0.1406(0.1649) gcn-final-mse=0.1651(0.1788)
2020-08-12 10:51:58 3000-10553 loss=0.2257(0.1321+0.0936)-0.5455(0.2743+0.2713) sod-mse=0.0502(0.1616) gcn-mse=0.0640(0.1540) gcn-final-mse=0.1542(0.1679)
2020-08-12 10:52:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 10:57:01 4000-10553 loss=0.2121(0.1186+0.0934)-0.5142(0.2613+0.2529) sod-mse=0.0543(0.1496) gcn-mse=0.0476(0.1452) gcn-final-mse=0.1453(0.1591)
2020-08-12 10:58:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 11:02:02 5000-10553 loss=0.2511(0.1413+0.1098)-0.4924(0.2520+0.2404) sod-mse=0.0584(0.1416) gcn-mse=0.0652(0.1388) gcn-final-mse=0.1389(0.1526)
2020-08-12 11:07:02 6000-10553 loss=0.9862(0.6080+0.3782)-0.4796(0.2470+0.2326) sod-mse=0.2339(0.1364) gcn-mse=0.2524(0.1350) gcn-final-mse=0.1351(0.1489)
2020-08-12 11:09:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 11:12:02 7000-10553 loss=0.1883(0.0770+0.1113)-0.4681(0.2423+0.2258) sod-mse=0.0963(0.1321) gcn-mse=0.0541(0.1319) gcn-final-mse=0.1319(0.1458)
2020-08-12 11:16:58 8000-10553 loss=0.3094(0.1652+0.1443)-0.4568(0.2377+0.2191) sod-mse=0.1093(0.1280) gcn-mse=0.0967(0.1288) gcn-final-mse=0.1288(0.1428)
2020-08-12 11:22:00 9000-10553 loss=0.2605(0.1509+0.1096)-0.4492(0.2345+0.2147) sod-mse=0.0791(0.1251) gcn-mse=0.0827(0.1263) gcn-final-mse=0.1263(0.1403)
2020-08-12 11:26:58 10000-10553 loss=0.3326(0.1850+0.1476)-0.4429(0.2317+0.2111) sod-mse=0.0953(0.1227) gcn-mse=0.0986(0.1244) gcn-final-mse=0.1244(0.1384)

2020-08-12 11:29:43    0-5019 loss=0.8114(0.4448+0.3666)-0.8114(0.4448+0.3666) sod-mse=0.1496(0.1496) gcn-mse=0.1631(0.1631) gcn-final-mse=0.1554(0.1656)
2020-08-12 11:32:11 1000-5019 loss=0.1192(0.0699+0.0493)-0.4393(0.2346+0.2047) sod-mse=0.0443(0.1201) gcn-mse=0.0492(0.1207) gcn-final-mse=0.1210(0.1347)
2020-08-12 11:34:33 2000-5019 loss=0.5589(0.2845+0.2744)-0.4491(0.2386+0.2105) sod-mse=0.1771(0.1218) gcn-mse=0.1619(0.1221) gcn-final-mse=0.1225(0.1361)
2020-08-12 11:36:56 3000-5019 loss=0.0806(0.0516+0.0290)-0.4525(0.2402+0.2123) sod-mse=0.0170(0.1233) gcn-mse=0.0233(0.1234) gcn-final-mse=0.1238(0.1374)
2020-08-12 11:38:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 11:39:23 4000-5019 loss=0.2371(0.1287+0.1084)-0.4519(0.2400+0.2120) sod-mse=0.0737(0.1235) gcn-mse=0.0663(0.1237) gcn-final-mse=0.1240(0.1376)
2020-08-12 11:40:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 11:41:48 5000-5019 loss=0.6876(0.3268+0.3608)-0.4521(0.2403+0.2119) sod-mse=0.1432(0.1236) gcn-mse=0.1348(0.1239) gcn-final-mse=0.1242(0.1377)
2020-08-12 11:41:50 E: 0, Train sod-mae-score=0.1214-0.8571 gcn-mae-score=0.1232-0.8387 gcn-final-mse-score=0.1232-0.8422(0.1373/0.8422) loss=0.4389(0.2299+0.2090)
2020-08-12 11:41:50 E: 0, Test  sod-mae-score=0.1235-0.7419 gcn-mae-score=0.1238-0.6887 gcn-final-mse-score=0.1241-0.6953(0.1377/0.6953) loss=0.4517(0.2400+0.2116)

2020-08-12 11:41:50 Start Epoch 1
2020-08-12 11:41:50 Epoch:01,lr=0.0001
2020-08-12 11:41:53    0-10553 loss=0.2699(0.1511+0.1188)-0.2699(0.1511+0.1188) sod-mse=0.0711(0.0711) gcn-mse=0.0699(0.0699) gcn-final-mse=0.0726(0.0894)
2020-08-12 11:46:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 11:46:51 1000-10553 loss=0.2977(0.1580+0.1396)-0.3469(0.1882+0.1587) sod-mse=0.0349(0.0901) gcn-mse=0.0479(0.0966) gcn-final-mse=0.0967(0.1106)
2020-08-12 11:51:53 2000-10553 loss=0.1846(0.1139+0.0707)-0.3466(0.1880+0.1586) sod-mse=0.0454(0.0897) gcn-mse=0.0477(0.0959) gcn-final-mse=0.0959(0.1100)
2020-08-12 11:56:51 3000-10553 loss=0.4976(0.2821+0.2155)-0.3533(0.1915+0.1617) sod-mse=0.1500(0.0916) gcn-mse=0.1488(0.0975) gcn-final-mse=0.0974(0.1114)
2020-08-12 12:01:56 4000-10553 loss=0.3034(0.1618+0.1416)-0.3493(0.1895+0.1598) sod-mse=0.1036(0.0906) gcn-mse=0.0993(0.0965) gcn-final-mse=0.0963(0.1104)
2020-08-12 12:02:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 12:06:56 5000-10553 loss=0.0948(0.0479+0.0469)-0.3475(0.1888+0.1587) sod-mse=0.0399(0.0901) gcn-mse=0.0321(0.0961) gcn-final-mse=0.0959(0.1100)
2020-08-12 12:11:56 6000-10553 loss=0.9519(0.4836+0.4683)-0.3457(0.1878+0.1579) sod-mse=0.2415(0.0895) gcn-mse=0.2421(0.0954) gcn-final-mse=0.0952(0.1094)
2020-08-12 12:16:58 7000-10553 loss=0.6158(0.3536+0.2622)-0.3418(0.1859+0.1558) sod-mse=0.1587(0.0882) gcn-mse=0.1922(0.0943) gcn-final-mse=0.0941(0.1083)
2020-08-12 12:21:56 8000-10553 loss=0.9621(0.4542+0.5079)-0.3400(0.1853+0.1547) sod-mse=0.3073(0.0877) gcn-mse=0.2716(0.0939) gcn-final-mse=0.0937(0.1079)
2020-08-12 12:26:54 9000-10553 loss=0.0622(0.0426+0.0196)-0.3384(0.1846+0.1538) sod-mse=0.0169(0.0871) gcn-mse=0.0226(0.0934) gcn-final-mse=0.0931(0.1074)
2020-08-12 12:27:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 12:31:54 10000-10553 loss=0.9237(0.5037+0.4201)-0.3370(0.1838+0.1532) sod-mse=0.2474(0.0866) gcn-mse=0.2354(0.0928) gcn-final-mse=0.0925(0.1068)

2020-08-12 12:34:41    0-5019 loss=0.9939(0.5651+0.4289)-0.9939(0.5651+0.4289) sod-mse=0.1311(0.1311) gcn-mse=0.1743(0.1743) gcn-final-mse=0.1629(0.1723)
2020-08-12 12:37:04 1000-5019 loss=0.0970(0.0582+0.0387)-0.4637(0.2389+0.2248) sod-mse=0.0348(0.1155) gcn-mse=0.0381(0.1161) gcn-final-mse=0.1164(0.1300)
2020-08-12 12:39:25 2000-5019 loss=0.5029(0.2724+0.2305)-0.4746(0.2435+0.2311) sod-mse=0.1350(0.1181) gcn-mse=0.1359(0.1185) gcn-final-mse=0.1187(0.1322)
2020-08-12 12:41:49 3000-5019 loss=0.0685(0.0475+0.0210)-0.4863(0.2487+0.2376) sod-mse=0.0108(0.1206) gcn-mse=0.0192(0.1206) gcn-final-mse=0.1209(0.1344)
2020-08-12 12:43:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 12:44:13 4000-5019 loss=0.2433(0.1337+0.1095)-0.4847(0.2482+0.2365) sod-mse=0.0642(0.1205) gcn-mse=0.0626(0.1207) gcn-final-mse=0.1210(0.1345)
2020-08-12 12:44:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 12:46:35 5000-5019 loss=0.5926(0.2726+0.3200)-0.4823(0.2474+0.2349) sod-mse=0.1326(0.1203) gcn-mse=0.1244(0.1206) gcn-final-mse=0.1209(0.1343)
2020-08-12 12:46:37 E: 1, Train sod-mae-score=0.0864-0.9009 gcn-mae-score=0.0925-0.8701 gcn-final-mse-score=0.0923-0.8733(0.1066/0.8733) loss=0.3365(0.1835+0.1529)
2020-08-12 12:46:37 E: 1, Test  sod-mae-score=0.1203-0.7604 gcn-mae-score=0.1206-0.7034 gcn-final-mse-score=0.1208-0.7099(0.1343/0.7099) loss=0.4821(0.2473+0.2348)

2020-08-12 12:46:37 Start Epoch 2
2020-08-12 12:46:37 Epoch:02,lr=0.0001
2020-08-12 12:46:40    0-10553 loss=0.2331(0.1438+0.0892)-0.2331(0.1438+0.0892) sod-mse=0.0653(0.0653) gcn-mse=0.0828(0.0828) gcn-final-mse=0.0794(0.0956)
2020-08-12 12:51:36 1000-10553 loss=0.2327(0.1234+0.1093)-0.2988(0.1640+0.1349) sod-mse=0.0636(0.0751) gcn-mse=0.0600(0.0812) gcn-final-mse=0.0810(0.0951)
2020-08-12 12:56:34 2000-10553 loss=0.0806(0.0630+0.0176)-0.2962(0.1640+0.1323) sod-mse=0.0149(0.0742) gcn-mse=0.0500(0.0810) gcn-final-mse=0.0808(0.0950)
2020-08-12 12:58:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 13:01:33 3000-10553 loss=0.1066(0.0705+0.0361)-0.2927(0.1623+0.1304) sod-mse=0.0263(0.0735) gcn-mse=0.0337(0.0801) gcn-final-mse=0.0798(0.0941)
2020-08-12 13:06:31 4000-10553 loss=0.2809(0.1602+0.1207)-0.2936(0.1628+0.1308) sod-mse=0.0912(0.0734) gcn-mse=0.0879(0.0802) gcn-final-mse=0.0799(0.0943)
2020-08-12 13:11:29 5000-10553 loss=0.1807(0.1097+0.0710)-0.2958(0.1639+0.1319) sod-mse=0.0511(0.0742) gcn-mse=0.0595(0.0810) gcn-final-mse=0.0807(0.0950)
2020-08-12 13:16:28 6000-10553 loss=0.2542(0.1386+0.1156)-0.2971(0.1646+0.1325) sod-mse=0.0771(0.0745) gcn-mse=0.0742(0.0813) gcn-final-mse=0.0810(0.0953)
2020-08-12 13:21:30 7000-10553 loss=0.0795(0.0556+0.0239)-0.2991(0.1655+0.1336) sod-mse=0.0142(0.0752) gcn-mse=0.0250(0.0817) gcn-final-mse=0.0814(0.0959)
2020-08-12 13:25:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 13:26:33 8000-10553 loss=0.4943(0.2488+0.2456)-0.2979(0.1651+0.1328) sod-mse=0.1662(0.0747) gcn-mse=0.1584(0.0814) gcn-final-mse=0.0811(0.0956)
2020-08-12 13:31:32 9000-10553 loss=0.1023(0.0722+0.0301)-0.2973(0.1647+0.1326) sod-mse=0.0213(0.0745) gcn-mse=0.0343(0.0811) gcn-final-mse=0.0808(0.0953)
2020-08-12 13:33:58 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 13:36:35 10000-10553 loss=0.4086(0.2015+0.2071)-0.2961(0.1641+0.1320) sod-mse=0.1365(0.0742) gcn-mse=0.1230(0.0807) gcn-final-mse=0.0804(0.0949)

2020-08-12 13:39:23    0-5019 loss=1.3731(0.6626+0.7105)-1.3731(0.6626+0.7105) sod-mse=0.2288(0.2288) gcn-mse=0.2198(0.2198) gcn-final-mse=0.2126(0.2205)
2020-08-12 13:41:54 1000-5019 loss=0.0994(0.0659+0.0335)-0.3880(0.2057+0.1823) sod-mse=0.0296(0.0973) gcn-mse=0.0436(0.1048) gcn-final-mse=0.1052(0.1184)
2020-08-12 13:44:19 2000-5019 loss=0.6232(0.3012+0.3221)-0.3897(0.2064+0.1833) sod-mse=0.1154(0.0979) gcn-mse=0.1159(0.1052) gcn-final-mse=0.1057(0.1188)
2020-08-12 13:46:47 3000-5019 loss=0.0611(0.0429+0.0182)-0.3915(0.2075+0.1840) sod-mse=0.0117(0.0984) gcn-mse=0.0166(0.1059) gcn-final-mse=0.1064(0.1196)
2020-08-12 13:48:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 13:49:16 4000-5019 loss=0.2639(0.1543+0.1096)-0.3922(0.2080+0.1842) sod-mse=0.0671(0.0986) gcn-mse=0.0760(0.1062) gcn-final-mse=0.1067(0.1198)
2020-08-12 13:49:58 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 13:51:41 5000-5019 loss=0.4665(0.2299+0.2366)-0.3938(0.2090+0.1848) sod-mse=0.1137(0.0989) gcn-mse=0.1234(0.1067) gcn-final-mse=0.1071(0.1202)
2020-08-12 13:51:43 E: 2, Train sod-mae-score=0.0741-0.9123 gcn-mae-score=0.0806-0.8806 gcn-final-mse-score=0.0804-0.8836(0.0948/0.8836) loss=0.2963(0.1642+0.1321)
2020-08-12 13:51:43 E: 2, Test  sod-mae-score=0.0988-0.7677 gcn-mae-score=0.1066-0.6967 gcn-final-mse-score=0.1070-0.7027(0.1201/0.7027) loss=0.3935(0.2089+0.1846)

2020-08-12 13:51:43 Start Epoch 3
2020-08-12 13:51:43 Epoch:03,lr=0.0001
2020-08-12 13:51:45    0-10553 loss=0.4292(0.2361+0.1932)-0.4292(0.2361+0.1932) sod-mse=0.1406(0.1406) gcn-mse=0.1480(0.1480) gcn-final-mse=0.1483(0.1616)
2020-08-12 13:56:54 1000-10553 loss=0.2320(0.1195+0.1125)-0.2685(0.1501+0.1184) sod-mse=0.0631(0.0661) gcn-mse=0.0600(0.0725) gcn-final-mse=0.0722(0.0867)
2020-08-12 14:01:56 2000-10553 loss=0.1054(0.0560+0.0494)-0.2654(0.1491+0.1164) sod-mse=0.0262(0.0649) gcn-mse=0.0281(0.0719) gcn-final-mse=0.0715(0.0861)
2020-08-12 14:06:56 3000-10553 loss=0.1602(0.1010+0.0592)-0.2684(0.1504+0.1180) sod-mse=0.0370(0.0659) gcn-mse=0.0507(0.0724) gcn-final-mse=0.0722(0.0867)
2020-08-12 14:11:57 4000-10553 loss=0.3575(0.2116+0.1459)-0.2684(0.1503+0.1181) sod-mse=0.0988(0.0660) gcn-mse=0.1159(0.0726) gcn-final-mse=0.0723(0.0868)
2020-08-12 14:16:56 5000-10553 loss=0.2186(0.1344+0.0842)-0.2665(0.1492+0.1173) sod-mse=0.0615(0.0655) gcn-mse=0.0659(0.0718) gcn-final-mse=0.0715(0.0861)
2020-08-12 14:22:01 6000-10553 loss=0.2542(0.1439+0.1103)-0.2681(0.1500+0.1181) sod-mse=0.0494(0.0658) gcn-mse=0.0496(0.0720) gcn-final-mse=0.0718(0.0863)
2020-08-12 14:23:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 14:24:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 14:26:59 7000-10553 loss=0.0980(0.0773+0.0207)-0.2669(0.1494+0.1175) sod-mse=0.0118(0.0655) gcn-mse=0.0291(0.0718) gcn-final-mse=0.0716(0.0862)
2020-08-12 14:31:59 8000-10553 loss=0.1002(0.0751+0.0251)-0.2682(0.1501+0.1181) sod-mse=0.0200(0.0658) gcn-mse=0.0396(0.0721) gcn-final-mse=0.0719(0.0864)
2020-08-12 14:37:02 9000-10553 loss=0.2930(0.1778+0.1152)-0.2692(0.1506+0.1186) sod-mse=0.0720(0.0660) gcn-mse=0.1059(0.0722) gcn-final-mse=0.0720(0.0866)
2020-08-12 14:42:03 10000-10553 loss=0.6057(0.3171+0.2887)-0.2696(0.1509+0.1187) sod-mse=0.2009(0.0661) gcn-mse=0.1969(0.0723) gcn-final-mse=0.0721(0.0867)
2020-08-12 14:43:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg

2020-08-12 14:44:52    0-5019 loss=1.1030(0.6138+0.4892)-1.1030(0.6138+0.4892) sod-mse=0.1631(0.1631) gcn-mse=0.1859(0.1859) gcn-final-mse=0.1788(0.1863)
2020-08-12 14:47:19 1000-5019 loss=0.0625(0.0421+0.0205)-0.3863(0.2068+0.1795) sod-mse=0.0188(0.0944) gcn-mse=0.0233(0.0991) gcn-final-mse=0.0992(0.1127)
2020-08-12 14:49:46 2000-5019 loss=0.6081(0.3043+0.3038)-0.3961(0.2110+0.1851) sod-mse=0.1146(0.0964) gcn-mse=0.1214(0.1009) gcn-final-mse=0.1011(0.1146)
2020-08-12 14:52:14 3000-5019 loss=0.0582(0.0415+0.0166)-0.4022(0.2139+0.1883) sod-mse=0.0092(0.0981) gcn-mse=0.0131(0.1025) gcn-final-mse=0.1028(0.1162)
2020-08-12 14:53:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 14:54:39 4000-5019 loss=0.1553(0.1007+0.0547)-0.4018(0.2138+0.1879) sod-mse=0.0321(0.0982) gcn-mse=0.0428(0.1027) gcn-final-mse=0.1030(0.1164)
2020-08-12 14:55:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 14:57:05 5000-5019 loss=0.5161(0.2595+0.2567)-0.4014(0.2137+0.1876) sod-mse=0.1139(0.0982) gcn-mse=0.1109(0.1028) gcn-final-mse=0.1030(0.1164)
2020-08-12 14:57:07 E: 3, Train sod-mae-score=0.0659-0.9207 gcn-mae-score=0.0721-0.8899 gcn-final-mse-score=0.0718-0.8928(0.0864/0.8928) loss=0.2689(0.1505+0.1184)
2020-08-12 14:57:07 E: 3, Test  sod-mae-score=0.0982-0.7801 gcn-mae-score=0.1028-0.7230 gcn-final-mse-score=0.1030-0.7298(0.1164/0.7298) loss=0.4012(0.2136+0.1875)

2020-08-12 14:57:07 Start Epoch 4
2020-08-12 14:57:07 Epoch:04,lr=0.0001
2020-08-12 14:57:10    0-10553 loss=0.3421(0.1951+0.1469)-0.3421(0.1951+0.1469) sod-mse=0.1036(0.1036) gcn-mse=0.1123(0.1123) gcn-final-mse=0.1152(0.1323)
2020-08-12 15:02:13 1000-10553 loss=0.3902(0.2233+0.1668)-0.2435(0.1387+0.1048) sod-mse=0.1026(0.0581) gcn-mse=0.1216(0.0653) gcn-final-mse=0.0649(0.0796)
2020-08-12 15:03:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 15:07:16 2000-10553 loss=0.3359(0.1941+0.1418)-0.2430(0.1385+0.1045) sod-mse=0.0943(0.0579) gcn-mse=0.1340(0.0650) gcn-final-mse=0.0647(0.0796)
2020-08-12 15:12:18 3000-10553 loss=0.0693(0.0521+0.0172)-0.2511(0.1420+0.1091) sod-mse=0.0103(0.0606) gcn-mse=0.0171(0.0670) gcn-final-mse=0.0667(0.0815)
2020-08-12 15:15:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 15:17:24 4000-10553 loss=0.4202(0.2480+0.1722)-0.2518(0.1424+0.1094) sod-mse=0.1126(0.0608) gcn-mse=0.1306(0.0674) gcn-final-mse=0.0671(0.0818)
2020-08-12 15:22:26 5000-10553 loss=0.1009(0.0676+0.0333)-0.2518(0.1425+0.1093) sod-mse=0.0218(0.0609) gcn-mse=0.0312(0.0676) gcn-final-mse=0.0674(0.0820)
2020-08-12 15:27:25 6000-10553 loss=0.2274(0.1464+0.0810)-0.2498(0.1415+0.1083) sod-mse=0.0581(0.0602) gcn-mse=0.0660(0.0670) gcn-final-mse=0.0667(0.0814)
2020-08-12 15:32:27 7000-10553 loss=0.6955(0.3316+0.3639)-0.2469(0.1400+0.1068) sod-mse=0.1573(0.0593) gcn-mse=0.1434(0.0661) gcn-final-mse=0.0657(0.0804)
2020-08-12 15:37:33 8000-10553 loss=0.0801(0.0586+0.0215)-0.2456(0.1393+0.1063) sod-mse=0.0130(0.0590) gcn-mse=0.0184(0.0656) gcn-final-mse=0.0653(0.0800)
2020-08-12 15:39:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 15:42:32 9000-10553 loss=0.1384(0.0903+0.0481)-0.2475(0.1402+0.1073) sod-mse=0.0310(0.0596) gcn-mse=0.0400(0.0660) gcn-final-mse=0.0657(0.0804)
2020-08-12 15:47:36 10000-10553 loss=0.1559(0.0939+0.0621)-0.2461(0.1394+0.1067) sod-mse=0.0422(0.0592) gcn-mse=0.0582(0.0656) gcn-final-mse=0.0653(0.0800)

2020-08-12 15:50:22    0-5019 loss=0.6217(0.3171+0.3047)-0.6217(0.3171+0.3047) sod-mse=0.1086(0.1086) gcn-mse=0.1132(0.1132) gcn-final-mse=0.1068(0.1213)
2020-08-12 15:52:54 1000-5019 loss=0.0630(0.0419+0.0211)-0.4082(0.2148+0.1933) sod-mse=0.0191(0.1056) gcn-mse=0.0234(0.1087) gcn-final-mse=0.1088(0.1235)
2020-08-12 15:55:19 2000-5019 loss=0.4245(0.2386+0.1859)-0.4120(0.2166+0.1954) sod-mse=0.1048(0.1065) gcn-mse=0.1078(0.1096) gcn-final-mse=0.1095(0.1242)
2020-08-12 15:57:45 3000-5019 loss=0.0574(0.0411+0.0163)-0.4184(0.2193+0.1991) sod-mse=0.0103(0.1081) gcn-mse=0.0150(0.1108) gcn-final-mse=0.1108(0.1255)
2020-08-12 15:59:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 16:00:13 4000-5019 loss=0.1510(0.1000+0.0510)-0.4178(0.2191+0.1987) sod-mse=0.0362(0.1081) gcn-mse=0.0476(0.1110) gcn-final-mse=0.1109(0.1256)
2020-08-12 16:00:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 16:02:37 5000-5019 loss=0.7126(0.3440+0.3686)-0.4178(0.2193+0.1985) sod-mse=0.1363(0.1083) gcn-mse=0.1317(0.1113) gcn-final-mse=0.1112(0.1258)
2020-08-12 16:02:39 E: 4, Train sod-mae-score=0.0590-0.9268 gcn-mae-score=0.0653-0.8959 gcn-final-mse-score=0.0650-0.8988(0.0797/0.8988) loss=0.2453(0.1390+0.1063)
2020-08-12 16:02:39 E: 4, Test  sod-mae-score=0.1083-0.7704 gcn-mae-score=0.1112-0.7054 gcn-final-mse-score=0.1111-0.7119(0.1258/0.7119) loss=0.4175(0.2191+0.1983)

2020-08-12 16:02:39 Start Epoch 5
2020-08-12 16:02:39 Epoch:05,lr=0.0001
2020-08-12 16:02:42    0-10553 loss=0.6202(0.3335+0.2868)-0.6202(0.3335+0.2868) sod-mse=0.1420(0.1420) gcn-mse=0.1537(0.1537) gcn-final-mse=0.1503(0.1684)
2020-08-12 16:07:44 1000-10553 loss=0.2723(0.1576+0.1148)-0.2222(0.1281+0.0941) sod-mse=0.0575(0.0519) gcn-mse=0.0717(0.0592) gcn-final-mse=0.0587(0.0734)
2020-08-12 16:12:49 2000-10553 loss=0.2407(0.1403+0.1004)-0.2194(0.1266+0.0928) sod-mse=0.0588(0.0510) gcn-mse=0.0600(0.0579) gcn-final-mse=0.0575(0.0722)
2020-08-12 16:17:50 3000-10553 loss=0.1589(0.1072+0.0517)-0.2245(0.1288+0.0956) sod-mse=0.0339(0.0528) gcn-mse=0.0378(0.0592) gcn-final-mse=0.0589(0.0736)
2020-08-12 16:22:49 4000-10553 loss=1.2910(0.6221+0.6690)-0.2233(0.1283+0.0950) sod-mse=0.2096(0.0524) gcn-mse=0.2241(0.0589) gcn-final-mse=0.0586(0.0733)
2020-08-12 16:27:50 5000-10553 loss=0.8151(0.4997+0.3154)-0.2254(0.1292+0.0962) sod-mse=0.1762(0.0530) gcn-mse=0.1881(0.0594) gcn-final-mse=0.0591(0.0738)
2020-08-12 16:30:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 16:32:52 6000-10553 loss=0.1753(0.0953+0.0800)-0.2261(0.1295+0.0965) sod-mse=0.0519(0.0532) gcn-mse=0.0490(0.0595) gcn-final-mse=0.0592(0.0741)
2020-08-12 16:37:49 7000-10553 loss=0.2134(0.1386+0.0749)-0.2277(0.1304+0.0973) sod-mse=0.0478(0.0537) gcn-mse=0.0693(0.0600) gcn-final-mse=0.0597(0.0746)
2020-08-12 16:39:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 16:42:49 8000-10553 loss=0.0650(0.0505+0.0145)-0.2274(0.1302+0.0972) sod-mse=0.0116(0.0535) gcn-mse=0.0201(0.0598) gcn-final-mse=0.0595(0.0744)
2020-08-12 16:47:48 9000-10553 loss=0.0699(0.0476+0.0223)-0.2288(0.1308+0.0979) sod-mse=0.0168(0.0540) gcn-mse=0.0187(0.0601) gcn-final-mse=0.0599(0.0747)
2020-08-12 16:52:47 10000-10553 loss=0.0782(0.0553+0.0229)-0.2292(0.1310+0.0982) sod-mse=0.0159(0.0541) gcn-mse=0.0154(0.0603) gcn-final-mse=0.0599(0.0748)
2020-08-12 16:53:43 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg

2020-08-12 16:55:36    0-5019 loss=1.7269(0.8360+0.8909)-1.7269(0.8360+0.8909) sod-mse=0.1999(0.1999) gcn-mse=0.2086(0.2086) gcn-final-mse=0.2014(0.2097)
2020-08-12 16:58:04 1000-5019 loss=0.0519(0.0398+0.0121)-0.3456(0.1819+0.1637) sod-mse=0.0108(0.0743) gcn-mse=0.0212(0.0802) gcn-final-mse=0.0802(0.0933)
2020-08-12 17:00:28 2000-5019 loss=0.5842(0.2753+0.3089)-0.3530(0.1851+0.1679) sod-mse=0.0930(0.0756) gcn-mse=0.0897(0.0816) gcn-final-mse=0.0816(0.0947)
2020-08-12 17:02:54 3000-5019 loss=0.0529(0.0376+0.0153)-0.3587(0.1876+0.1711) sod-mse=0.0079(0.0766) gcn-mse=0.0108(0.0827) gcn-final-mse=0.0827(0.0957)
2020-08-12 17:04:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 17:05:22 4000-5019 loss=0.3270(0.1875+0.1395)-0.3560(0.1867+0.1693) sod-mse=0.0662(0.0762) gcn-mse=0.0736(0.0824) gcn-final-mse=0.0824(0.0954)
2020-08-12 17:06:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 17:07:45 5000-5019 loss=0.4238(0.2058+0.2180)-0.3571(0.1874+0.1697) sod-mse=0.0808(0.0765) gcn-mse=0.0800(0.0827) gcn-final-mse=0.0827(0.0957)
2020-08-12 17:07:48 E: 5, Train sod-mae-score=0.0541-0.9332 gcn-mae-score=0.0602-0.9023 gcn-final-mse-score=0.0599-0.9051(0.0747/0.9051) loss=0.2289(0.1308+0.0980)
2020-08-12 17:07:48 E: 5, Test  sod-mae-score=0.0765-0.8049 gcn-mae-score=0.0827-0.7481 gcn-final-mse-score=0.0827-0.7543(0.0957/0.7543) loss=0.3568(0.1873+0.1695)

2020-08-12 17:07:48 Start Epoch 6
2020-08-12 17:07:48 Epoch:06,lr=0.0001
2020-08-12 17:07:50    0-10553 loss=0.0837(0.0605+0.0232)-0.0837(0.0605+0.0232) sod-mse=0.0180(0.0180) gcn-mse=0.0233(0.0233) gcn-final-mse=0.0212(0.0352)
2020-08-12 17:11:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 17:12:50 1000-10553 loss=0.4843(0.2872+0.1971)-0.2081(0.1214+0.0867) sod-mse=0.1160(0.0476) gcn-mse=0.1392(0.0546) gcn-final-mse=0.0542(0.0694)
2020-08-12 17:17:53 2000-10553 loss=0.7504(0.3822+0.3682)-0.2119(0.1226+0.0893) sod-mse=0.1717(0.0491) gcn-mse=0.1871(0.0553) gcn-final-mse=0.0549(0.0701)
2020-08-12 17:22:54 3000-10553 loss=0.0701(0.0432+0.0269)-0.2127(0.1229+0.0898) sod-mse=0.0155(0.0494) gcn-mse=0.0151(0.0555) gcn-final-mse=0.0551(0.0702)
2020-08-12 17:23:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 17:27:54 4000-10553 loss=0.3251(0.1594+0.1657)-0.2134(0.1230+0.0904) sod-mse=0.0769(0.0496) gcn-mse=0.0747(0.0556) gcn-final-mse=0.0552(0.0702)
2020-08-12 17:32:53 5000-10553 loss=0.4122(0.2408+0.1714)-0.2102(0.1215+0.0887) sod-mse=0.0995(0.0486) gcn-mse=0.1329(0.0547) gcn-final-mse=0.0543(0.0694)
2020-08-12 17:37:56 6000-10553 loss=0.0667(0.0485+0.0183)-0.2101(0.1214+0.0887) sod-mse=0.0105(0.0486) gcn-mse=0.0145(0.0546) gcn-final-mse=0.0542(0.0691)
2020-08-12 17:41:44 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 17:42:49 7000-10553 loss=0.6577(0.2990+0.3586)-0.2138(0.1231+0.0907) sod-mse=0.1084(0.0497) gcn-mse=0.1232(0.0555) gcn-final-mse=0.0551(0.0701)
2020-08-12 17:47:49 8000-10553 loss=0.1318(0.0946+0.0372)-0.2137(0.1230+0.0907) sod-mse=0.0293(0.0496) gcn-mse=0.0589(0.0554) gcn-final-mse=0.0550(0.0700)
2020-08-12 17:52:47 9000-10553 loss=0.3066(0.1803+0.1263)-0.2139(0.1232+0.0907) sod-mse=0.0576(0.0496) gcn-mse=0.0639(0.0554) gcn-final-mse=0.0551(0.0701)
2020-08-12 17:57:48 10000-10553 loss=0.1102(0.0717+0.0384)-0.2146(0.1234+0.0912) sod-mse=0.0272(0.0498) gcn-mse=0.0335(0.0556) gcn-final-mse=0.0552(0.0702)

2020-08-12 18:00:35    0-5019 loss=1.3061(0.7243+0.5818)-1.3061(0.7243+0.5818) sod-mse=0.1781(0.1781) gcn-mse=0.1782(0.1782) gcn-final-mse=0.1717(0.1814)
2020-08-12 18:02:59 1000-5019 loss=0.0671(0.0486+0.0185)-0.3780(0.2044+0.1736) sod-mse=0.0169(0.0911) gcn-mse=0.0301(0.0995) gcn-final-mse=0.0994(0.1135)
2020-08-12 18:05:28 2000-5019 loss=0.5521(0.2723+0.2798)-0.3853(0.2083+0.1770) sod-mse=0.1098(0.0927) gcn-mse=0.1100(0.1013) gcn-final-mse=0.1011(0.1152)
2020-08-12 18:07:55 3000-5019 loss=0.0543(0.0378+0.0164)-0.3909(0.2110+0.1799) sod-mse=0.0099(0.0941) gcn-mse=0.0116(0.1028) gcn-final-mse=0.1027(0.1168)
2020-08-12 18:09:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 18:10:23 4000-5019 loss=0.1644(0.1008+0.0636)-0.3917(0.2115+0.1801) sod-mse=0.0366(0.0943) gcn-mse=0.0379(0.1032) gcn-final-mse=0.1031(0.1172)
2020-08-12 18:11:05 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 18:12:49 5000-5019 loss=0.5757(0.2759+0.2999)-0.3897(0.2108+0.1789) sod-mse=0.1197(0.0940) gcn-mse=0.1140(0.1030) gcn-final-mse=0.1029(0.1170)
2020-08-12 18:12:51 E: 6, Train sod-mae-score=0.0500-0.9371 gcn-mae-score=0.0558-0.9064 gcn-final-mse-score=0.0554-0.9092(0.0704/0.9092) loss=0.2155(0.1239+0.0916)
2020-08-12 18:12:51 E: 6, Test  sod-mae-score=0.0940-0.7738 gcn-mae-score=0.1030-0.7177 gcn-final-mse-score=0.1029-0.7231(0.1170/0.7231) loss=0.3896(0.2107+0.1789)

2020-08-12 18:12:51 Start Epoch 7
2020-08-12 18:12:51 Epoch:07,lr=0.0001
2020-08-12 18:12:54    0-10553 loss=0.2028(0.1053+0.0975)-0.2028(0.1053+0.0975) sod-mse=0.0518(0.0518) gcn-mse=0.0486(0.0486) gcn-final-mse=0.0486(0.0598)
2020-08-12 18:17:56 1000-10553 loss=0.2783(0.1548+0.1235)-0.1851(0.1089+0.0762) sod-mse=0.0733(0.0414) gcn-mse=0.0823(0.0479) gcn-final-mse=0.0476(0.0629)
2020-08-12 18:18:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 18:22:59 2000-10553 loss=0.3211(0.1663+0.1548)-0.1970(0.1150+0.0820) sod-mse=0.1087(0.0442) gcn-mse=0.1244(0.0505) gcn-final-mse=0.0501(0.0653)
2020-08-12 18:27:59 3000-10553 loss=0.1426(0.0798+0.0628)-0.1971(0.1156+0.0816) sod-mse=0.0309(0.0441) gcn-mse=0.0339(0.0511) gcn-final-mse=0.0506(0.0658)
2020-08-12 18:32:59 4000-10553 loss=0.4488(0.2567+0.1921)-0.1963(0.1149+0.0814) sod-mse=0.1318(0.0440) gcn-mse=0.1460(0.0505) gcn-final-mse=0.0501(0.0653)
2020-08-12 18:33:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 18:37:59 5000-10553 loss=0.3769(0.2080+0.1689)-0.1984(0.1160+0.0824) sod-mse=0.0443(0.0446) gcn-mse=0.0537(0.0511) gcn-final-mse=0.0507(0.0659)
2020-08-12 18:39:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 18:42:21 6000-10553 loss=0.1629(0.0839+0.0791)-0.1993(0.1162+0.0831) sod-mse=0.0375(0.0451) gcn-mse=0.0379(0.0514) gcn-final-mse=0.0510(0.0660)
2020-08-12 18:46:35 7000-10553 loss=0.1085(0.0728+0.0357)-0.1990(0.1161+0.0830) sod-mse=0.0206(0.0451) gcn-mse=0.0218(0.0513) gcn-final-mse=0.0509(0.0660)
2020-08-12 18:50:42 8000-10553 loss=0.1156(0.0708+0.0448)-0.1974(0.1152+0.0821) sod-mse=0.0262(0.0445) gcn-mse=0.0300(0.0508) gcn-final-mse=0.0504(0.0655)
2020-08-12 18:54:49 9000-10553 loss=0.2588(0.1236+0.1352)-0.1984(0.1157+0.0826) sod-mse=0.0477(0.0449) gcn-mse=0.0495(0.0511) gcn-final-mse=0.0508(0.0658)
2020-08-12 18:58:57 10000-10553 loss=0.0682(0.0449+0.0233)-0.2000(0.1165+0.0835) sod-mse=0.0146(0.0454) gcn-mse=0.0153(0.0516) gcn-final-mse=0.0512(0.0662)

2020-08-12 19:01:15    0-5019 loss=0.8368(0.4473+0.3895)-0.8368(0.4473+0.3895) sod-mse=0.1101(0.1101) gcn-mse=0.1236(0.1236) gcn-final-mse=0.1164(0.1275)
2020-08-12 19:03:09 1000-5019 loss=0.0462(0.0354+0.0108)-0.3416(0.1811+0.1604) sod-mse=0.0096(0.0729) gcn-mse=0.0167(0.0765) gcn-final-mse=0.0765(0.0898)
2020-08-12 19:05:02 2000-5019 loss=0.6748(0.3351+0.3397)-0.3470(0.1838+0.1632) sod-mse=0.1060(0.0742) gcn-mse=0.1021(0.0780) gcn-final-mse=0.0780(0.0912)
2020-08-12 19:06:55 3000-5019 loss=0.0504(0.0371+0.0133)-0.3525(0.1862+0.1662) sod-mse=0.0071(0.0752) gcn-mse=0.0096(0.0791) gcn-final-mse=0.0791(0.0923)
2020-08-12 19:07:58 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 19:08:49 4000-5019 loss=0.1323(0.0857+0.0466)-0.3524(0.1863+0.1662) sod-mse=0.0276(0.0752) gcn-mse=0.0289(0.0791) gcn-final-mse=0.0791(0.0923)
2020-08-12 19:09:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 19:10:41 5000-5019 loss=0.4225(0.2007+0.2218)-0.3517(0.1862+0.1655) sod-mse=0.0940(0.0753) gcn-mse=0.0858(0.0793) gcn-final-mse=0.0792(0.0924)
2020-08-12 19:10:43 E: 7, Train sod-mae-score=0.0456-0.9427 gcn-mae-score=0.0517-0.9119 gcn-final-mse-score=0.0514-0.9146(0.0664/0.9146) loss=0.2005(0.1168+0.0837)
2020-08-12 19:10:43 E: 7, Test  sod-mae-score=0.0752-0.8069 gcn-mae-score=0.0793-0.7558 gcn-final-mse-score=0.0792-0.7619(0.0924/0.7619) loss=0.3515(0.1861+0.1654)

2020-08-12 19:10:43 Start Epoch 8
2020-08-12 19:10:43 Epoch:08,lr=0.0001
2020-08-12 19:10:45    0-10553 loss=0.0478(0.0375+0.0103)-0.0478(0.0375+0.0103) sod-mse=0.0062(0.0062) gcn-mse=0.0093(0.0093) gcn-final-mse=0.0079(0.0206)
2020-08-12 19:11:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 19:14:52 1000-10553 loss=0.0620(0.0412+0.0208)-0.1790(0.1057+0.0733) sod-mse=0.0118(0.0401) gcn-mse=0.0162(0.0456) gcn-final-mse=0.0452(0.0603)
2020-08-12 19:18:58 2000-10553 loss=0.1915(0.1124+0.0792)-0.1845(0.1088+0.0757) sod-mse=0.0480(0.0414) gcn-mse=0.0528(0.0475) gcn-final-mse=0.0470(0.0620)
2020-08-12 19:23:05 3000-10553 loss=0.0533(0.0380+0.0153)-0.1919(0.1124+0.0795) sod-mse=0.0094(0.0430) gcn-mse=0.0140(0.0490) gcn-final-mse=0.0485(0.0636)
2020-08-12 19:27:10 4000-10553 loss=0.2183(0.1053+0.1130)-0.1893(0.1113+0.0780) sod-mse=0.0470(0.0423) gcn-mse=0.0437(0.0484) gcn-final-mse=0.0480(0.0632)
2020-08-12 19:28:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 19:31:16 5000-10553 loss=0.0856(0.0558+0.0298)-0.1911(0.1122+0.0790) sod-mse=0.0198(0.0429) gcn-mse=0.0219(0.0489) gcn-final-mse=0.0484(0.0636)
2020-08-12 19:35:21 6000-10553 loss=0.0587(0.0439+0.0148)-0.1937(0.1133+0.0804) sod-mse=0.0088(0.0436) gcn-mse=0.0105(0.0494) gcn-final-mse=0.0490(0.0642)
2020-08-12 19:39:26 7000-10553 loss=0.2150(0.1358+0.0792)-0.1912(0.1122+0.0790) sod-mse=0.0524(0.0429) gcn-mse=0.0643(0.0489) gcn-final-mse=0.0485(0.0637)
2020-08-12 19:39:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 19:43:31 8000-10553 loss=0.0368(0.0285+0.0083)-0.1913(0.1122+0.0791) sod-mse=0.0047(0.0430) gcn-mse=0.0061(0.0488) gcn-final-mse=0.0484(0.0636)
2020-08-12 19:47:37 9000-10553 loss=0.0852(0.0593+0.0260)-0.1920(0.1125+0.0796) sod-mse=0.0187(0.0432) gcn-mse=0.0256(0.0490) gcn-final-mse=0.0486(0.0638)
2020-08-12 19:51:44 10000-10553 loss=0.0816(0.0522+0.0294)-0.1911(0.1120+0.0791) sod-mse=0.0184(0.0429) gcn-mse=0.0167(0.0488) gcn-final-mse=0.0483(0.0635)

2020-08-12 19:54:01    0-5019 loss=1.1382(0.5331+0.6052)-1.1382(0.5331+0.6052) sod-mse=0.1285(0.1285) gcn-mse=0.1328(0.1328) gcn-final-mse=0.1265(0.1390)
2020-08-12 19:55:54 1000-5019 loss=0.0407(0.0325+0.0082)-0.3451(0.1790+0.1661) sod-mse=0.0069(0.0699) gcn-mse=0.0148(0.0781) gcn-final-mse=0.0780(0.0916)
2020-08-12 19:57:45 2000-5019 loss=0.6275(0.2735+0.3540)-0.3515(0.1823+0.1692) sod-mse=0.1022(0.0718) gcn-mse=0.0974(0.0803) gcn-final-mse=0.0802(0.0937)
2020-08-12 19:59:37 3000-5019 loss=0.0507(0.0374+0.0133)-0.3546(0.1837+0.1708) sod-mse=0.0069(0.0724) gcn-mse=0.0109(0.0811) gcn-final-mse=0.0811(0.0946)
2020-08-12 20:00:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 20:01:28 4000-5019 loss=0.2294(0.1394+0.0900)-0.3542(0.1842+0.1700) sod-mse=0.0544(0.0725) gcn-mse=0.0679(0.0815) gcn-final-mse=0.0814(0.0949)
2020-08-12 20:02:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 20:03:19 5000-5019 loss=0.7306(0.2942+0.4363)-0.3533(0.1839+0.1694) sod-mse=0.0933(0.0725) gcn-mse=0.0914(0.0815) gcn-final-mse=0.0814(0.0949)
2020-08-12 20:03:21 E: 8, Train sod-mae-score=0.0430-0.9443 gcn-mae-score=0.0488-0.9139 gcn-final-mse-score=0.0484-0.9167(0.0636/0.9167) loss=0.1913(0.1120+0.0793)
2020-08-12 20:03:21 E: 8, Test  sod-mae-score=0.0725-0.8052 gcn-mae-score=0.0815-0.7493 gcn-final-mse-score=0.0814-0.7546(0.0949/0.7546) loss=0.3532(0.1839+0.1693)

2020-08-12 20:03:21 Start Epoch 9
2020-08-12 20:03:21 Epoch:09,lr=0.0001
2020-08-12 20:03:23    0-10553 loss=0.2149(0.1292+0.0857)-0.2149(0.1292+0.0857) sod-mse=0.0565(0.0565) gcn-mse=0.0622(0.0622) gcn-final-mse=0.0663(0.0854)
2020-08-12 20:07:29 1000-10553 loss=0.2131(0.1057+0.1073)-0.1676(0.0998+0.0677) sod-mse=0.0570(0.0360) gcn-mse=0.0642(0.0417) gcn-final-mse=0.0413(0.0565)
2020-08-12 20:11:35 2000-10553 loss=0.5092(0.2722+0.2370)-0.1647(0.0986+0.0661) sod-mse=0.1358(0.0355) gcn-mse=0.1462(0.0413) gcn-final-mse=0.0409(0.0562)
2020-08-12 20:14:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 20:15:43 3000-10553 loss=0.3654(0.2435+0.1219)-0.1736(0.1030+0.0706) sod-mse=0.0858(0.0380) gcn-mse=0.1084(0.0436) gcn-final-mse=0.0432(0.0585)
2020-08-12 20:19:49 4000-10553 loss=0.2374(0.1318+0.1056)-0.1733(0.1028+0.0705) sod-mse=0.0693(0.0380) gcn-mse=0.0759(0.0435) gcn-final-mse=0.0431(0.0584)
2020-08-12 20:23:57 5000-10553 loss=0.0523(0.0421+0.0103)-0.1756(0.1040+0.0716) sod-mse=0.0048(0.0386) gcn-mse=0.0109(0.0442) gcn-final-mse=0.0438(0.0590)
2020-08-12 20:28:04 6000-10553 loss=0.0506(0.0367+0.0139)-0.1767(0.1047+0.0720) sod-mse=0.0090(0.0388) gcn-mse=0.0126(0.0445) gcn-final-mse=0.0441(0.0593)
2020-08-12 20:29:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 20:32:11 7000-10553 loss=0.1194(0.0878+0.0317)-0.1781(0.1055+0.0726) sod-mse=0.0219(0.0392) gcn-mse=0.0254(0.0450) gcn-final-mse=0.0446(0.0598)
2020-08-12 20:36:18 8000-10553 loss=0.0364(0.0259+0.0105)-0.1791(0.1059+0.0731) sod-mse=0.0063(0.0395) gcn-mse=0.0083(0.0453) gcn-final-mse=0.0449(0.0601)
2020-08-12 20:40:26 9000-10553 loss=0.0427(0.0329+0.0097)-0.1804(0.1066+0.0738) sod-mse=0.0048(0.0399) gcn-mse=0.0057(0.0456) gcn-final-mse=0.0452(0.0604)
2020-08-12 20:41:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 20:44:32 10000-10553 loss=0.1650(0.0950+0.0699)-0.1830(0.1079+0.0751) sod-mse=0.0496(0.0407) gcn-mse=0.0628(0.0464) gcn-final-mse=0.0459(0.0612)

2020-08-12 20:46:50    0-5019 loss=1.1987(0.6936+0.5052)-1.1987(0.6936+0.5052) sod-mse=0.1587(0.1587) gcn-mse=0.1690(0.1690) gcn-final-mse=0.1627(0.1745)
2020-08-12 20:48:44 1000-5019 loss=0.0591(0.0399+0.0192)-0.3235(0.1744+0.1491) sod-mse=0.0176(0.0740) gcn-mse=0.0213(0.0731) gcn-final-mse=0.0731(0.0870)
2020-08-12 20:50:36 2000-5019 loss=0.7677(0.3846+0.3831)-0.3219(0.1738+0.1482) sod-mse=0.1069(0.0748) gcn-mse=0.0984(0.0740) gcn-final-mse=0.0739(0.0878)
2020-08-12 20:52:28 3000-5019 loss=0.0487(0.0338+0.0149)-0.3251(0.1752+0.1499) sod-mse=0.0089(0.0756) gcn-mse=0.0073(0.0749) gcn-final-mse=0.0748(0.0888)
2020-08-12 20:53:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 20:54:21 4000-5019 loss=0.1769(0.1097+0.0672)-0.3230(0.1743+0.1486) sod-mse=0.0440(0.0753) gcn-mse=0.0496(0.0748) gcn-final-mse=0.0747(0.0886)
2020-08-12 20:54:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 20:56:12 5000-5019 loss=0.5078(0.2482+0.2596)-0.3233(0.1746+0.1487) sod-mse=0.1021(0.0755) gcn-mse=0.0923(0.0750) gcn-final-mse=0.0749(0.0888)
2020-08-12 20:56:14 E: 9, Train sod-mae-score=0.0407-0.9473 gcn-mae-score=0.0464-0.9168 gcn-final-mse-score=0.0459-0.9196(0.0611/0.9196) loss=0.1832(0.1080+0.0752)
2020-08-12 20:56:14 E: 9, Test  sod-mae-score=0.0755-0.8150 gcn-mae-score=0.0750-0.7610 gcn-final-mse-score=0.0749-0.7673(0.0888/0.7673) loss=0.3231(0.1745+0.1486)

2020-08-12 20:56:14 Start Epoch 10
2020-08-12 20:56:14 Epoch:10,lr=0.0001
2020-08-12 20:56:16    0-10553 loss=0.0480(0.0321+0.0159)-0.0480(0.0321+0.0159) sod-mse=0.0134(0.0134) gcn-mse=0.0196(0.0196) gcn-final-mse=0.0179(0.0239)
2020-08-12 21:00:22 1000-10553 loss=0.2403(0.1471+0.0932)-0.1525(0.0924+0.0601) sod-mse=0.0494(0.0320) gcn-mse=0.0562(0.0383) gcn-final-mse=0.0380(0.0528)
2020-08-12 21:04:28 2000-10553 loss=0.0744(0.0555+0.0189)-0.1546(0.0939+0.0607) sod-mse=0.0148(0.0324) gcn-mse=0.0155(0.0387) gcn-final-mse=0.0383(0.0533)
2020-08-12 21:08:34 3000-10553 loss=0.0529(0.0336+0.0193)-0.1581(0.0955+0.0626) sod-mse=0.0122(0.0337) gcn-mse=0.0142(0.0396) gcn-final-mse=0.0392(0.0544)
2020-08-12 21:12:40 4000-10553 loss=0.1803(0.1102+0.0701)-0.1618(0.0973+0.0645) sod-mse=0.0366(0.0347) gcn-mse=0.0385(0.0404) gcn-final-mse=0.0400(0.0552)
2020-08-12 21:16:46 5000-10553 loss=0.1304(0.1039+0.0266)-0.1643(0.0986+0.0657) sod-mse=0.0142(0.0355) gcn-mse=0.0251(0.0411) gcn-final-mse=0.0407(0.0560)
2020-08-12 21:20:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 21:20:51 6000-10553 loss=0.0503(0.0385+0.0118)-0.1662(0.0996+0.0666) sod-mse=0.0061(0.0359) gcn-mse=0.0109(0.0416) gcn-final-mse=0.0412(0.0565)
2020-08-12 21:24:57 7000-10553 loss=0.2420(0.1203+0.1217)-0.1671(0.1000+0.0671) sod-mse=0.0421(0.0361) gcn-mse=0.0463(0.0418) gcn-final-mse=0.0414(0.0567)
2020-08-12 21:27:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 21:27:44 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 21:29:03 8000-10553 loss=0.0259(0.0189+0.0069)-0.1679(0.1005+0.0675) sod-mse=0.0034(0.0363) gcn-mse=0.0056(0.0421) gcn-final-mse=0.0417(0.0570)
2020-08-12 21:33:08 9000-10553 loss=1.1516(0.5132+0.6384)-0.1678(0.1004+0.0674) sod-mse=0.1670(0.0363) gcn-mse=0.1678(0.0420) gcn-final-mse=0.0416(0.0569)
2020-08-12 21:37:16 10000-10553 loss=0.1451(0.1004+0.0448)-0.1688(0.1009+0.0679) sod-mse=0.0255(0.0366) gcn-mse=0.0356(0.0423) gcn-final-mse=0.0419(0.0572)

2020-08-12 21:39:33    0-5019 loss=1.1266(0.5644+0.5622)-1.1266(0.5644+0.5622) sod-mse=0.1531(0.1531) gcn-mse=0.1464(0.1464) gcn-final-mse=0.1395(0.1511)
2020-08-12 21:41:26 1000-5019 loss=0.0668(0.0480+0.0187)-0.3251(0.1678+0.1573) sod-mse=0.0170(0.0669) gcn-mse=0.0284(0.0710) gcn-final-mse=0.0711(0.0850)
2020-08-12 21:43:18 2000-5019 loss=0.7713(0.3374+0.4339)-0.3290(0.1700+0.1590) sod-mse=0.1061(0.0683) gcn-mse=0.1034(0.0726) gcn-final-mse=0.0727(0.0865)
2020-08-12 21:45:10 3000-5019 loss=0.0492(0.0361+0.0131)-0.3320(0.1713+0.1606) sod-mse=0.0066(0.0689) gcn-mse=0.0093(0.0732) gcn-final-mse=0.0734(0.0872)
2020-08-12 21:46:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 21:47:01 4000-5019 loss=0.1766(0.1053+0.0713)-0.3290(0.1703+0.1587) sod-mse=0.0413(0.0685) gcn-mse=0.0459(0.0730) gcn-final-mse=0.0730(0.0868)
2020-08-12 21:47:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 21:48:53 5000-5019 loss=0.4413(0.2086+0.2327)-0.3289(0.1705+0.1584) sod-mse=0.0845(0.0687) gcn-mse=0.0787(0.0732) gcn-final-mse=0.0732(0.0870)
2020-08-12 21:48:54 E:10, Train sod-mae-score=0.0368-0.9509 gcn-mae-score=0.0424-0.9207 gcn-final-mse-score=0.0420-0.9234(0.0573/0.9234) loss=0.1693(0.1011+0.0682)
2020-08-12 21:48:54 E:10, Test  sod-mae-score=0.0686-0.8201 gcn-mae-score=0.0732-0.7606 gcn-final-mse-score=0.0732-0.7667(0.0870/0.7667) loss=0.3286(0.1704+0.1582)

2020-08-12 21:48:54 Start Epoch 11
2020-08-12 21:48:54 Epoch:11,lr=0.0001
2020-08-12 21:48:56    0-10553 loss=0.1536(0.0888+0.0648)-0.1536(0.0888+0.0648) sod-mse=0.0344(0.0344) gcn-mse=0.0269(0.0269) gcn-final-mse=0.0260(0.0444)
2020-08-12 21:53:02 1000-10553 loss=0.1937(0.1065+0.0872)-0.1564(0.0951+0.0613) sod-mse=0.0360(0.0332) gcn-mse=0.0443(0.0391) gcn-final-mse=0.0385(0.0537)
2020-08-12 21:53:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 21:57:10 2000-10553 loss=0.1427(0.0935+0.0492)-0.1593(0.0962+0.0632) sod-mse=0.0359(0.0342) gcn-mse=0.0333(0.0395) gcn-final-mse=0.0390(0.0542)
2020-08-12 22:01:17 3000-10553 loss=0.0512(0.0371+0.0141)-0.1584(0.0959+0.0626) sod-mse=0.0075(0.0339) gcn-mse=0.0081(0.0396) gcn-final-mse=0.0391(0.0543)
2020-08-12 22:05:25 4000-10553 loss=0.0641(0.0494+0.0147)-0.1558(0.0945+0.0612) sod-mse=0.0068(0.0332) gcn-mse=0.0089(0.0388) gcn-final-mse=0.0383(0.0535)
2020-08-12 22:09:31 5000-10553 loss=0.0717(0.0492+0.0225)-0.1586(0.0958+0.0628) sod-mse=0.0118(0.0338) gcn-mse=0.0105(0.0395) gcn-final-mse=0.0390(0.0542)
2020-08-12 22:13:40 6000-10553 loss=0.0662(0.0469+0.0193)-0.1612(0.0972+0.0640) sod-mse=0.0100(0.0345) gcn-mse=0.0201(0.0402) gcn-final-mse=0.0397(0.0549)
2020-08-12 22:16:06 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 22:17:47 7000-10553 loss=0.3101(0.1442+0.1659)-0.1603(0.0968+0.0636) sod-mse=0.0743(0.0343) gcn-mse=0.0748(0.0400) gcn-final-mse=0.0394(0.0547)
2020-08-12 22:21:51 8000-10553 loss=0.1703(0.1089+0.0614)-0.1608(0.0970+0.0638) sod-mse=0.0373(0.0344) gcn-mse=0.0476(0.0401) gcn-final-mse=0.0396(0.0549)
2020-08-12 22:25:57 9000-10553 loss=0.1305(0.0843+0.0462)-0.1616(0.0974+0.0642) sod-mse=0.0299(0.0346) gcn-mse=0.0364(0.0402) gcn-final-mse=0.0397(0.0550)
2020-08-12 22:30:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 22:30:05 10000-10553 loss=0.0634(0.0383+0.0251)-0.1616(0.0974+0.0643) sod-mse=0.0167(0.0346) gcn-mse=0.0149(0.0402) gcn-final-mse=0.0397(0.0550)

2020-08-12 22:32:20    0-5019 loss=1.3109(0.7183+0.5926)-1.3109(0.7183+0.5926) sod-mse=0.1706(0.1706) gcn-mse=0.1674(0.1674) gcn-final-mse=0.1610(0.1749)
2020-08-12 22:34:13 1000-5019 loss=0.0555(0.0369+0.0186)-0.3528(0.1839+0.1689) sod-mse=0.0166(0.0693) gcn-mse=0.0179(0.0697) gcn-final-mse=0.0699(0.0831)
2020-08-12 22:36:05 2000-5019 loss=0.9445(0.4487+0.4959)-0.3565(0.1855+0.1710) sod-mse=0.1110(0.0702) gcn-mse=0.1057(0.0708) gcn-final-mse=0.0709(0.0841)
2020-08-12 22:37:58 3000-5019 loss=0.0494(0.0362+0.0132)-0.3600(0.1872+0.1728) sod-mse=0.0070(0.0707) gcn-mse=0.0101(0.0714) gcn-final-mse=0.0716(0.0848)
2020-08-12 22:38:59 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 22:39:49 4000-5019 loss=0.1462(0.0920+0.0542)-0.3561(0.1857+0.1704) sod-mse=0.0323(0.0704) gcn-mse=0.0359(0.0712) gcn-final-mse=0.0714(0.0845)
2020-08-12 22:40:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 22:41:40 5000-5019 loss=0.8638(0.3716+0.4922)-0.3549(0.1854+0.1695) sod-mse=0.0891(0.0705) gcn-mse=0.0882(0.0714) gcn-final-mse=0.0715(0.0846)
2020-08-12 22:41:42 E:11, Train sod-mae-score=0.0345-0.9537 gcn-mae-score=0.0401-0.9225 gcn-final-mse-score=0.0396-0.9253(0.0550/0.9253) loss=0.1613(0.0972+0.0641)
2020-08-12 22:41:42 E:11, Test  sod-mae-score=0.0705-0.8229 gcn-mae-score=0.0714-0.7633 gcn-final-mse-score=0.0715-0.7692(0.0846/0.7692) loss=0.3546(0.1853+0.1694)

2020-08-12 22:41:42 Start Epoch 12
2020-08-12 22:41:42 Epoch:12,lr=0.0001
2020-08-12 22:41:43    0-10553 loss=0.1185(0.0691+0.0495)-0.1185(0.0691+0.0495) sod-mse=0.0348(0.0348) gcn-mse=0.0347(0.0347) gcn-final-mse=0.0342(0.0469)
2020-08-12 22:45:51 1000-10553 loss=0.1930(0.1093+0.0837)-0.1525(0.0926+0.0599) sod-mse=0.0478(0.0319) gcn-mse=0.0521(0.0375) gcn-final-mse=0.0370(0.0524)
2020-08-12 22:46:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-12 22:49:59 2000-10553 loss=0.0961(0.0623+0.0338)-0.1485(0.0907+0.0578) sod-mse=0.0252(0.0311) gcn-mse=0.0421(0.0367) gcn-final-mse=0.0362(0.0515)
2020-08-12 22:54:05 3000-10553 loss=0.0444(0.0289+0.0155)-0.1490(0.0910+0.0580) sod-mse=0.0126(0.0310) gcn-mse=0.0161(0.0364) gcn-final-mse=0.0359(0.0513)
2020-08-12 22:55:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-12 22:58:11 4000-10553 loss=0.0194(0.0130+0.0064)-0.1484(0.0908+0.0576) sod-mse=0.0046(0.0308) gcn-mse=0.0058(0.0363) gcn-final-mse=0.0357(0.0512)
2020-08-12 23:02:18 5000-10553 loss=0.0628(0.0441+0.0187)-0.1504(0.0918+0.0586) sod-mse=0.0102(0.0314) gcn-mse=0.0162(0.0368) gcn-final-mse=0.0363(0.0518)
2020-08-12 23:06:24 6000-10553 loss=0.1385(0.0896+0.0490)-0.1514(0.0922+0.0591) sod-mse=0.0226(0.0317) gcn-mse=0.0256(0.0372) gcn-final-mse=0.0366(0.0521)
2020-08-12 23:10:30 7000-10553 loss=0.0653(0.0496+0.0157)-0.1530(0.0931+0.0599) sod-mse=0.0076(0.0322) gcn-mse=0.0146(0.0377) gcn-final-mse=0.0372(0.0526)
2020-08-12 23:14:37 8000-10553 loss=0.2490(0.1596+0.0894)-0.1533(0.0932+0.0601) sod-mse=0.0643(0.0322) gcn-mse=0.0911(0.0378) gcn-final-mse=0.0372(0.0527)
2020-08-12 23:18:43 9000-10553 loss=0.1530(0.1173+0.0357)-0.1548(0.0939+0.0608) sod-mse=0.0225(0.0326) gcn-mse=0.0414(0.0381) gcn-final-mse=0.0376(0.0530)
2020-08-12 23:21:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 23:22:48 10000-10553 loss=0.0764(0.0570+0.0194)-0.1551(0.0940+0.0610) sod-mse=0.0109(0.0327) gcn-mse=0.0201(0.0382) gcn-final-mse=0.0377(0.0531)

2020-08-12 23:25:04    0-5019 loss=0.6922(0.3956+0.2966)-0.6922(0.3956+0.2966) sod-mse=0.1066(0.1066) gcn-mse=0.1173(0.1173) gcn-final-mse=0.1114(0.1243)
2020-08-12 23:26:57 1000-5019 loss=0.0442(0.0317+0.0125)-0.3303(0.1792+0.1511) sod-mse=0.0112(0.0733) gcn-mse=0.0123(0.0722) gcn-final-mse=0.0721(0.0856)
2020-08-12 23:28:48 2000-5019 loss=0.5097(0.2705+0.2393)-0.3373(0.1820+0.1553) sod-mse=0.0938(0.0748) gcn-mse=0.0865(0.0737) gcn-final-mse=0.0737(0.0871)
2020-08-12 23:30:40 3000-5019 loss=0.0479(0.0352+0.0126)-0.3378(0.1820+0.1558) sod-mse=0.0065(0.0752) gcn-mse=0.0082(0.0742) gcn-final-mse=0.0741(0.0876)
2020-08-12 23:31:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-12 23:32:31 4000-5019 loss=0.1348(0.0847+0.0501)-0.3346(0.1807+0.1539) sod-mse=0.0276(0.0747) gcn-mse=0.0276(0.0738) gcn-final-mse=0.0737(0.0872)
2020-08-12 23:33:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-12 23:34:21 5000-5019 loss=1.0312(0.4756+0.5556)-0.3325(0.1799+0.1525) sod-mse=0.1234(0.0745) gcn-mse=0.1245(0.0737) gcn-final-mse=0.0736(0.0870)
2020-08-12 23:34:23 E:12, Train sod-mae-score=0.0327-0.9561 gcn-mae-score=0.0381-0.9258 gcn-final-mse-score=0.0376-0.9285(0.0531/0.9285) loss=0.1550(0.0940+0.0610)
2020-08-12 23:34:23 E:12, Test  sod-mae-score=0.0745-0.8154 gcn-mae-score=0.0737-0.7595 gcn-final-mse-score=0.0736-0.7654(0.0870/0.7654) loss=0.3323(0.1798+0.1524)

2020-08-12 23:34:23 Start Epoch 13
2020-08-12 23:34:23 Epoch:13,lr=0.0001
2020-08-12 23:34:25    0-10553 loss=0.0632(0.0465+0.0167)-0.0632(0.0465+0.0167) sod-mse=0.0097(0.0097) gcn-mse=0.0128(0.0128) gcn-final-mse=0.0119(0.0247)
2020-08-12 23:38:31 1000-10553 loss=0.0887(0.0564+0.0322)-0.1459(0.0893+0.0566) sod-mse=0.0177(0.0303) gcn-mse=0.0201(0.0358) gcn-final-mse=0.0353(0.0508)
2020-08-12 23:42:37 2000-10553 loss=0.0655(0.0474+0.0181)-0.1456(0.0886+0.0570) sod-mse=0.0118(0.0304) gcn-mse=0.0125(0.0354) gcn-final-mse=0.0349(0.0502)
2020-08-12 23:46:42 3000-10553 loss=0.1436(0.0889+0.0547)-0.1474(0.0899+0.0575) sod-mse=0.0271(0.0310) gcn-mse=0.0290(0.0362) gcn-final-mse=0.0357(0.0511)
2020-08-12 23:50:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-12 23:50:48 4000-10553 loss=0.1465(0.1038+0.0426)-0.1455(0.0890+0.0565) sod-mse=0.0291(0.0304) gcn-mse=0.0459(0.0356) gcn-final-mse=0.0351(0.0506)
2020-08-12 23:54:54 5000-10553 loss=0.1620(0.1102+0.0518)-0.1483(0.0906+0.0577) sod-mse=0.0403(0.0310) gcn-mse=0.0612(0.0364) gcn-final-mse=0.0359(0.0515)
2020-08-12 23:58:59 6000-10553 loss=0.0559(0.0407+0.0152)-0.1485(0.0908+0.0577) sod-mse=0.0086(0.0311) gcn-mse=0.0127(0.0364) gcn-final-mse=0.0359(0.0515)
2020-08-12 23:59:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 00:03:04 7000-10553 loss=0.1961(0.1097+0.0864)-0.1492(0.0911+0.0581) sod-mse=0.0437(0.0313) gcn-mse=0.0438(0.0366) gcn-final-mse=0.0361(0.0517)
2020-08-13 00:07:09 8000-10553 loss=0.1013(0.0757+0.0256)-0.1488(0.0908+0.0579) sod-mse=0.0212(0.0311) gcn-mse=0.0260(0.0364) gcn-final-mse=0.0359(0.0515)
2020-08-13 00:07:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 00:11:15 9000-10553 loss=0.0704(0.0575+0.0129)-0.1477(0.0903+0.0573) sod-mse=0.0075(0.0308) gcn-mse=0.0231(0.0361) gcn-final-mse=0.0356(0.0512)
2020-08-13 00:15:19 10000-10553 loss=0.1310(0.0843+0.0467)-0.1472(0.0901+0.0571) sod-mse=0.0305(0.0307) gcn-mse=0.0408(0.0360) gcn-final-mse=0.0355(0.0510)

2020-08-13 00:17:36    0-5019 loss=0.8743(0.5144+0.3599)-0.8743(0.5144+0.3599) sod-mse=0.0888(0.0888) gcn-mse=0.0987(0.0987) gcn-final-mse=0.0912(0.1026)
2020-08-13 00:19:29 1000-5019 loss=0.0580(0.0406+0.0173)-0.3339(0.1761+0.1578) sod-mse=0.0150(0.0668) gcn-mse=0.0197(0.0703) gcn-final-mse=0.0703(0.0836)
2020-08-13 00:21:21 2000-5019 loss=0.6601(0.3135+0.3466)-0.3439(0.1808+0.1631) sod-mse=0.1023(0.0689) gcn-mse=0.0936(0.0725) gcn-final-mse=0.0725(0.0857)
2020-08-13 00:23:12 3000-5019 loss=0.0530(0.0370+0.0160)-0.3521(0.1841+0.1679) sod-mse=0.0085(0.0697) gcn-mse=0.0104(0.0734) gcn-final-mse=0.0735(0.0867)
2020-08-13 00:24:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 00:25:04 4000-5019 loss=0.1461(0.0946+0.0515)-0.3492(0.1831+0.1662) sod-mse=0.0311(0.0697) gcn-mse=0.0350(0.0736) gcn-final-mse=0.0735(0.0867)
2020-08-13 00:25:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 00:26:55 5000-5019 loss=0.8967(0.4106+0.4861)-0.3494(0.1833+0.1661) sod-mse=0.1109(0.0699) gcn-mse=0.1053(0.0737) gcn-final-mse=0.0736(0.0869)
2020-08-13 00:26:57 E:13, Train sod-mae-score=0.0306-0.9577 gcn-mae-score=0.0359-0.9272 gcn-final-mse-score=0.0353-0.9298(0.0509/0.9298) loss=0.1469(0.0899+0.0569)
2020-08-13 00:26:57 E:13, Test  sod-mae-score=0.0698-0.8062 gcn-mae-score=0.0737-0.7479 gcn-final-mse-score=0.0736-0.7535(0.0868/0.7535) loss=0.3492(0.1832+0.1660)

2020-08-13 00:26:57 Start Epoch 14
2020-08-13 00:26:57 Epoch:14,lr=0.0001
2020-08-13 00:26:59    0-10553 loss=0.1231(0.0790+0.0441)-0.1231(0.0790+0.0441) sod-mse=0.0283(0.0283) gcn-mse=0.0314(0.0314) gcn-final-mse=0.0317(0.0439)
2020-08-13 00:29:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 00:31:04 1000-10553 loss=0.2614(0.1433+0.1181)-0.1370(0.0847+0.0523) sod-mse=0.0823(0.0280) gcn-mse=0.0826(0.0332) gcn-final-mse=0.0326(0.0481)
2020-08-13 00:33:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 00:35:09 2000-10553 loss=0.1030(0.0613+0.0417)-0.1397(0.0861+0.0537) sod-mse=0.0290(0.0287) gcn-mse=0.0289(0.0338) gcn-final-mse=0.0333(0.0487)
2020-08-13 00:35:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 00:39:15 3000-10553 loss=0.3710(0.2123+0.1587)-0.1438(0.0882+0.0556) sod-mse=0.0499(0.0297) gcn-mse=0.0544(0.0351) gcn-final-mse=0.0346(0.0498)
2020-08-13 00:43:19 4000-10553 loss=0.0921(0.0598+0.0323)-0.1461(0.0896+0.0565) sod-mse=0.0152(0.0303) gcn-mse=0.0217(0.0357) gcn-final-mse=0.0352(0.0506)
2020-08-13 00:47:23 5000-10553 loss=0.0812(0.0535+0.0277)-0.1427(0.0879+0.0548) sod-mse=0.0135(0.0294) gcn-mse=0.0129(0.0348) gcn-final-mse=0.0343(0.0498)
2020-08-13 00:51:28 6000-10553 loss=0.1493(0.1005+0.0489)-0.1417(0.0875+0.0542) sod-mse=0.0376(0.0290) gcn-mse=0.0509(0.0345) gcn-final-mse=0.0339(0.0495)
2020-08-13 00:55:33 7000-10553 loss=0.1460(0.0952+0.0507)-0.1431(0.0881+0.0550) sod-mse=0.0267(0.0294) gcn-mse=0.0256(0.0346) gcn-final-mse=0.0341(0.0496)
2020-08-13 00:59:36 8000-10553 loss=0.2925(0.1734+0.1191)-0.1439(0.0884+0.0555) sod-mse=0.0952(0.0296) gcn-mse=0.1167(0.0348) gcn-final-mse=0.0343(0.0499)
2020-08-13 01:03:42 9000-10553 loss=0.0628(0.0426+0.0202)-0.1442(0.0885+0.0557) sod-mse=0.0095(0.0297) gcn-mse=0.0113(0.0348) gcn-final-mse=0.0343(0.0499)
2020-08-13 01:07:47 10000-10553 loss=0.0966(0.0642+0.0324)-0.1453(0.0891+0.0562) sod-mse=0.0186(0.0299) gcn-mse=0.0183(0.0351) gcn-final-mse=0.0346(0.0502)

2020-08-13 01:10:04    0-5019 loss=1.0479(0.6128+0.4351)-1.0479(0.6128+0.4351) sod-mse=0.0927(0.0927) gcn-mse=0.1036(0.1036) gcn-final-mse=0.0964(0.1091)
2020-08-13 01:11:55 1000-5019 loss=0.0716(0.0506+0.0210)-0.3325(0.1747+0.1579) sod-mse=0.0172(0.0627) gcn-mse=0.0263(0.0657) gcn-final-mse=0.0657(0.0792)
2020-08-13 01:13:46 2000-5019 loss=0.6398(0.3345+0.3053)-0.3374(0.1762+0.1612) sod-mse=0.0911(0.0636) gcn-mse=0.0903(0.0666) gcn-final-mse=0.0666(0.0800)
2020-08-13 01:15:37 3000-5019 loss=0.0456(0.0338+0.0119)-0.3423(0.1783+0.1640) sod-mse=0.0056(0.0644) gcn-mse=0.0075(0.0676) gcn-final-mse=0.0675(0.0809)
2020-08-13 01:16:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 01:17:28 4000-5019 loss=0.1284(0.0895+0.0390)-0.3405(0.1778+0.1628) sod-mse=0.0219(0.0643) gcn-mse=0.0329(0.0676) gcn-final-mse=0.0675(0.0809)
2020-08-13 01:18:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 01:19:18 5000-5019 loss=0.8222(0.3680+0.4541)-0.3434(0.1793+0.1641) sod-mse=0.1186(0.0647) gcn-mse=0.1140(0.0681) gcn-final-mse=0.0680(0.0814)
2020-08-13 01:19:20 E:14, Train sod-mae-score=0.0300-0.9592 gcn-mae-score=0.0351-0.9290 gcn-final-mse-score=0.0345-0.9318(0.0502/0.9318) loss=0.1454(0.0892+0.0562)
2020-08-13 01:19:20 E:14, Test  sod-mae-score=0.0647-0.8219 gcn-mae-score=0.0681-0.7693 gcn-final-mse-score=0.0680-0.7751(0.0814/0.7751) loss=0.3432(0.1792+0.1640)

2020-08-13 01:19:20 Start Epoch 15
2020-08-13 01:19:20 Epoch:15,lr=0.0001
2020-08-13 01:19:21    0-10553 loss=0.0846(0.0537+0.0310)-0.0846(0.0537+0.0310) sod-mse=0.0217(0.0217) gcn-mse=0.0268(0.0268) gcn-final-mse=0.0272(0.0364)
2020-08-13 01:23:27 1000-10553 loss=0.1001(0.0527+0.0474)-0.1375(0.0854+0.0521) sod-mse=0.0289(0.0280) gcn-mse=0.0270(0.0333) gcn-final-mse=0.0327(0.0483)
2020-08-13 01:27:31 2000-10553 loss=0.0915(0.0704+0.0212)-0.1349(0.0839+0.0510) sod-mse=0.0111(0.0273) gcn-mse=0.0180(0.0324) gcn-final-mse=0.0319(0.0474)
2020-08-13 01:29:05 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 01:31:35 3000-10553 loss=0.1050(0.0709+0.0341)-0.1363(0.0847+0.0517) sod-mse=0.0210(0.0275) gcn-mse=0.0308(0.0326) gcn-final-mse=0.0321(0.0477)
2020-08-13 01:35:42 4000-10553 loss=0.1358(0.0947+0.0411)-0.1351(0.0841+0.0511) sod-mse=0.0295(0.0271) gcn-mse=0.0544(0.0322) gcn-final-mse=0.0317(0.0474)
2020-08-13 01:39:49 5000-10553 loss=0.0869(0.0632+0.0237)-0.1363(0.0846+0.0517) sod-mse=0.0171(0.0275) gcn-mse=0.0245(0.0325) gcn-final-mse=0.0319(0.0477)
2020-08-13 01:43:55 6000-10553 loss=0.0915(0.0638+0.0278)-0.1360(0.0844+0.0516) sod-mse=0.0190(0.0274) gcn-mse=0.0279(0.0324) gcn-final-mse=0.0319(0.0476)
2020-08-13 01:48:01 7000-10553 loss=0.1241(0.0783+0.0458)-0.1365(0.0847+0.0518) sod-mse=0.0239(0.0276) gcn-mse=0.0367(0.0327) gcn-final-mse=0.0321(0.0478)
2020-08-13 01:49:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 01:52:07 8000-10553 loss=0.1191(0.0646+0.0546)-0.1372(0.0850+0.0522) sod-mse=0.0244(0.0278) gcn-mse=0.0289(0.0329) gcn-final-mse=0.0323(0.0479)
2020-08-13 01:56:14 9000-10553 loss=0.0510(0.0369+0.0141)-0.1376(0.0853+0.0523) sod-mse=0.0078(0.0279) gcn-mse=0.0152(0.0330) gcn-final-mse=0.0324(0.0481)
2020-08-13 02:00:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 02:00:21 10000-10553 loss=0.0954(0.0725+0.0229)-0.1377(0.0853+0.0524) sod-mse=0.0182(0.0279) gcn-mse=0.0275(0.0330) gcn-final-mse=0.0325(0.0481)

2020-08-13 02:02:38    0-5019 loss=0.6472(0.3331+0.3141)-0.6472(0.3331+0.3141) sod-mse=0.0926(0.0926) gcn-mse=0.0989(0.0989) gcn-final-mse=0.0925(0.1062)
2020-08-13 02:04:31 1000-5019 loss=0.0644(0.0485+0.0159)-0.3500(0.1829+0.1671) sod-mse=0.0144(0.0653) gcn-mse=0.0284(0.0686) gcn-final-mse=0.0685(0.0823)
2020-08-13 02:06:22 2000-5019 loss=0.6410(0.3264+0.3146)-0.3669(0.1908+0.1761) sod-mse=0.0908(0.0676) gcn-mse=0.0864(0.0713) gcn-final-mse=0.0711(0.0848)
2020-08-13 02:08:14 3000-5019 loss=0.0459(0.0342+0.0117)-0.3674(0.1905+0.1770) sod-mse=0.0056(0.0680) gcn-mse=0.0074(0.0717) gcn-final-mse=0.0716(0.0852)
2020-08-13 02:09:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 02:10:05 4000-5019 loss=0.1531(0.1041+0.0490)-0.3601(0.1873+0.1728) sod-mse=0.0291(0.0671) gcn-mse=0.0370(0.0709) gcn-final-mse=0.0707(0.0844)
2020-08-13 02:10:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 02:11:56 5000-5019 loss=1.1837(0.5091+0.6746)-0.3590(0.1870+0.1720) sod-mse=0.1081(0.0672) gcn-mse=0.1045(0.0710) gcn-final-mse=0.0708(0.0845)
2020-08-13 02:11:58 E:15, Train sod-mae-score=0.0279-0.9617 gcn-mae-score=0.0330-0.9313 gcn-final-mse-score=0.0325-0.9340(0.0481/0.9340) loss=0.1378(0.0854+0.0524)
2020-08-13 02:11:58 E:15, Test  sod-mae-score=0.0672-0.8268 gcn-mae-score=0.0710-0.7732 gcn-final-mse-score=0.0708-0.7794(0.0845/0.7794) loss=0.3586(0.1869+0.1717)

2020-08-13 02:11:58 Start Epoch 16
2020-08-13 02:11:58 Epoch:16,lr=0.0001
2020-08-13 02:12:00    0-10553 loss=0.0763(0.0466+0.0296)-0.0763(0.0466+0.0296) sod-mse=0.0179(0.0179) gcn-mse=0.0203(0.0203) gcn-final-mse=0.0255(0.0371)
2020-08-13 02:16:06 1000-10553 loss=0.1599(0.0909+0.0689)-0.1242(0.0788+0.0454) sod-mse=0.0430(0.0238) gcn-mse=0.0449(0.0294) gcn-final-mse=0.0288(0.0446)
2020-08-13 02:20:12 2000-10553 loss=0.0878(0.0553+0.0326)-0.1310(0.0820+0.0490) sod-mse=0.0212(0.0255) gcn-mse=0.0223(0.0308) gcn-final-mse=0.0302(0.0459)
2020-08-13 02:21:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 02:22:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 02:24:21 3000-10553 loss=0.0545(0.0380+0.0164)-0.1327(0.0827+0.0500) sod-mse=0.0104(0.0261) gcn-mse=0.0137(0.0313) gcn-final-mse=0.0307(0.0464)
2020-08-13 02:28:27 4000-10553 loss=0.0728(0.0502+0.0226)-0.1320(0.0824+0.0496) sod-mse=0.0086(0.0260) gcn-mse=0.0132(0.0311) gcn-final-mse=0.0305(0.0463)
2020-08-13 02:32:33 5000-10553 loss=0.0884(0.0513+0.0371)-0.1318(0.0823+0.0495) sod-mse=0.0175(0.0259) gcn-mse=0.0160(0.0311) gcn-final-mse=0.0305(0.0462)
2020-08-13 02:36:39 6000-10553 loss=0.1982(0.1266+0.0716)-0.1335(0.0832+0.0503) sod-mse=0.0449(0.0264) gcn-mse=0.0527(0.0315) gcn-final-mse=0.0309(0.0466)
2020-08-13 02:40:45 7000-10553 loss=0.2039(0.1240+0.0798)-0.1350(0.0840+0.0511) sod-mse=0.0500(0.0269) gcn-mse=0.0568(0.0320) gcn-final-mse=0.0314(0.0471)
2020-08-13 02:44:50 8000-10553 loss=0.0792(0.0515+0.0277)-0.1343(0.0836+0.0507) sod-mse=0.0152(0.0267) gcn-mse=0.0154(0.0319) gcn-final-mse=0.0313(0.0470)
2020-08-13 02:48:54 9000-10553 loss=0.0684(0.0534+0.0150)-0.1337(0.0834+0.0503) sod-mse=0.0079(0.0266) gcn-mse=0.0116(0.0317) gcn-final-mse=0.0311(0.0469)
2020-08-13 02:53:01 10000-10553 loss=0.0862(0.0692+0.0170)-0.1344(0.0837+0.0507) sod-mse=0.0110(0.0268) gcn-mse=0.0224(0.0319) gcn-final-mse=0.0313(0.0470)
2020-08-13 02:53:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg

2020-08-13 02:55:18    0-5019 loss=0.8877(0.5482+0.3395)-0.8877(0.5482+0.3395) sod-mse=0.0926(0.0926) gcn-mse=0.1155(0.1155) gcn-final-mse=0.1085(0.1212)
2020-08-13 02:57:10 1000-5019 loss=0.0518(0.0368+0.0150)-0.3706(0.1938+0.1768) sod-mse=0.0130(0.0699) gcn-mse=0.0169(0.0701) gcn-final-mse=0.0699(0.0833)
2020-08-13 02:59:02 2000-5019 loss=0.6521(0.3585+0.2936)-0.3835(0.1993+0.1842) sod-mse=0.0929(0.0720) gcn-mse=0.0912(0.0721) gcn-final-mse=0.0719(0.0852)
2020-08-13 03:00:53 3000-5019 loss=0.0492(0.0369+0.0123)-0.3892(0.2015+0.1877) sod-mse=0.0058(0.0730) gcn-mse=0.0092(0.0731) gcn-final-mse=0.0730(0.0862)
2020-08-13 03:01:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 03:02:45 4000-5019 loss=0.1201(0.0816+0.0385)-0.3833(0.1988+0.1845) sod-mse=0.0227(0.0722) gcn-mse=0.0273(0.0724) gcn-final-mse=0.0723(0.0855)
2020-08-13 03:03:18 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 03:04:36 5000-5019 loss=0.7354(0.3552+0.3802)-0.3816(0.1982+0.1834) sod-mse=0.1353(0.0719) gcn-mse=0.1206(0.0722) gcn-final-mse=0.0721(0.0852)
2020-08-13 03:04:38 E:16, Train sod-mae-score=0.0270-0.9630 gcn-mae-score=0.0321-0.9331 gcn-final-mse-score=0.0315-0.9357(0.0472/0.9357) loss=0.1351(0.0841+0.0510)
2020-08-13 03:04:38 E:16, Test  sod-mae-score=0.0718-0.8188 gcn-mae-score=0.0722-0.7648 gcn-final-mse-score=0.0720-0.7709(0.0852/0.7709) loss=0.3811(0.1980+0.1831)

2020-08-13 03:04:38 Start Epoch 17
2020-08-13 03:04:38 Epoch:17,lr=0.0001
2020-08-13 03:04:40    0-10553 loss=0.0303(0.0181+0.0121)-0.0303(0.0181+0.0121) sod-mse=0.0070(0.0070) gcn-mse=0.0075(0.0075) gcn-final-mse=0.0072(0.0113)
2020-08-13 03:08:46 1000-10553 loss=0.1150(0.0720+0.0430)-0.1251(0.0786+0.0465) sod-mse=0.0305(0.0249) gcn-mse=0.0378(0.0294) gcn-final-mse=0.0290(0.0447)
2020-08-13 03:08:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 03:12:52 2000-10553 loss=0.0912(0.0617+0.0295)-0.1272(0.0800+0.0473) sod-mse=0.0145(0.0252) gcn-mse=0.0163(0.0298) gcn-final-mse=0.0292(0.0450)
2020-08-13 03:16:58 3000-10553 loss=0.1909(0.1085+0.0824)-0.1273(0.0799+0.0474) sod-mse=0.0536(0.0251) gcn-mse=0.0542(0.0298) gcn-final-mse=0.0292(0.0449)
2020-08-13 03:21:04 4000-10553 loss=0.0772(0.0575+0.0197)-0.1263(0.0794+0.0468) sod-mse=0.0099(0.0247) gcn-mse=0.0147(0.0295) gcn-final-mse=0.0289(0.0447)
2020-08-13 03:24:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 03:25:09 5000-10553 loss=0.0661(0.0461+0.0200)-0.1262(0.0796+0.0466) sod-mse=0.0115(0.0246) gcn-mse=0.0170(0.0296) gcn-final-mse=0.0290(0.0448)
2020-08-13 03:29:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 03:29:16 6000-10553 loss=0.0742(0.0611+0.0131)-0.1271(0.0801+0.0470) sod-mse=0.0099(0.0248) gcn-mse=0.0089(0.0297) gcn-final-mse=0.0291(0.0450)
2020-08-13 03:33:20 7000-10553 loss=0.0732(0.0507+0.0225)-0.1258(0.0794+0.0464) sod-mse=0.0157(0.0245) gcn-mse=0.0239(0.0295) gcn-final-mse=0.0289(0.0447)
2020-08-13 03:37:26 8000-10553 loss=0.1028(0.0681+0.0347)-0.1261(0.0796+0.0465) sod-mse=0.0177(0.0246) gcn-mse=0.0294(0.0295) gcn-final-mse=0.0289(0.0448)
2020-08-13 03:41:33 9000-10553 loss=0.1106(0.0637+0.0469)-0.1278(0.0804+0.0474) sod-mse=0.0333(0.0250) gcn-mse=0.0277(0.0300) gcn-final-mse=0.0294(0.0452)
2020-08-13 03:45:38 10000-10553 loss=0.4736(0.2766+0.1971)-0.1289(0.0809+0.0480) sod-mse=0.1041(0.0254) gcn-mse=0.1150(0.0302) gcn-final-mse=0.0297(0.0455)

2020-08-13 03:47:54    0-5019 loss=1.2914(0.8009+0.4906)-1.2914(0.8009+0.4906) sod-mse=0.1384(0.1384) gcn-mse=0.1512(0.1512) gcn-final-mse=0.1453(0.1594)
2020-08-13 03:49:46 1000-5019 loss=0.0305(0.0245+0.0060)-0.3174(0.1736+0.1438) sod-mse=0.0050(0.0626) gcn-mse=0.0061(0.0625) gcn-final-mse=0.0624(0.0758)
2020-08-13 03:51:37 2000-5019 loss=0.7642(0.4051+0.3591)-0.3311(0.1802+0.1509) sod-mse=0.0964(0.0652) gcn-mse=0.0924(0.0652) gcn-final-mse=0.0650(0.0782)
2020-08-13 03:53:28 3000-5019 loss=0.0472(0.0356+0.0116)-0.3369(0.1828+0.1541) sod-mse=0.0060(0.0662) gcn-mse=0.0091(0.0662) gcn-final-mse=0.0661(0.0793)
2020-08-13 03:54:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 03:55:20 4000-5019 loss=0.1073(0.0745+0.0328)-0.3336(0.1813+0.1523) sod-mse=0.0176(0.0658) gcn-mse=0.0183(0.0659) gcn-final-mse=0.0658(0.0790)
2020-08-13 03:55:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 03:57:10 5000-5019 loss=0.5953(0.2832+0.3120)-0.3339(0.1815+0.1524) sod-mse=0.1165(0.0658) gcn-mse=0.1009(0.0659) gcn-final-mse=0.0658(0.0789)
2020-08-13 03:57:12 E:17, Train sod-mae-score=0.0254-0.9648 gcn-mae-score=0.0303-0.9352 gcn-final-mse-score=0.0297-0.9379(0.0455/0.9379) loss=0.1290(0.0810+0.0480)
2020-08-13 03:57:12 E:17, Test  sod-mae-score=0.0658-0.8267 gcn-mae-score=0.0659-0.7711 gcn-final-mse-score=0.0658-0.7770(0.0789/0.7770) loss=0.3338(0.1814+0.1523)

2020-08-13 03:57:12 Start Epoch 18
2020-08-13 03:57:12 Epoch:18,lr=0.0001
2020-08-13 03:57:14    0-10553 loss=0.0944(0.0673+0.0271)-0.0944(0.0673+0.0271) sod-mse=0.0193(0.0193) gcn-mse=0.0218(0.0218) gcn-final-mse=0.0194(0.0376)
2020-08-13 04:01:21 1000-10553 loss=0.0856(0.0579+0.0277)-0.1122(0.0721+0.0400) sod-mse=0.0176(0.0210) gcn-mse=0.0239(0.0258) gcn-final-mse=0.0251(0.0404)
2020-08-13 04:05:27 2000-10553 loss=0.0928(0.0612+0.0316)-0.1165(0.0746+0.0419) sod-mse=0.0211(0.0222) gcn-mse=0.0195(0.0273) gcn-final-mse=0.0266(0.0420)
2020-08-13 04:09:33 3000-10553 loss=0.0491(0.0341+0.0151)-0.1174(0.0751+0.0422) sod-mse=0.0077(0.0224) gcn-mse=0.0087(0.0273) gcn-final-mse=0.0267(0.0424)
2020-08-13 04:13:38 4000-10553 loss=0.1124(0.0687+0.0437)-0.1167(0.0748+0.0419) sod-mse=0.0212(0.0222) gcn-mse=0.0220(0.0271) gcn-final-mse=0.0265(0.0421)
2020-08-13 04:15:50 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 04:17:43 5000-10553 loss=0.0710(0.0545+0.0166)-0.1213(0.0771+0.0442) sod-mse=0.0114(0.0234) gcn-mse=0.0135(0.0282) gcn-final-mse=0.0275(0.0433)
2020-08-13 04:18:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 04:21:50 6000-10553 loss=0.0422(0.0300+0.0121)-0.1231(0.0780+0.0451) sod-mse=0.0054(0.0240) gcn-mse=0.0083(0.0287) gcn-final-mse=0.0281(0.0439)
2020-08-13 04:25:53 7000-10553 loss=0.0735(0.0567+0.0167)-0.1235(0.0781+0.0453) sod-mse=0.0125(0.0241) gcn-mse=0.0195(0.0287) gcn-final-mse=0.0281(0.0440)
2020-08-13 04:29:59 8000-10553 loss=0.4780(0.2588+0.2192)-0.1236(0.0782+0.0454) sod-mse=0.0871(0.0241) gcn-mse=0.0870(0.0287) gcn-final-mse=0.0281(0.0440)
2020-08-13 04:34:04 9000-10553 loss=0.0754(0.0581+0.0173)-0.1253(0.0790+0.0463) sod-mse=0.0098(0.0245) gcn-mse=0.0182(0.0292) gcn-final-mse=0.0286(0.0445)
2020-08-13 04:35:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 04:38:09 10000-10553 loss=0.0691(0.0480+0.0211)-0.1252(0.0790+0.0462) sod-mse=0.0091(0.0244) gcn-mse=0.0147(0.0291) gcn-final-mse=0.0285(0.0445)

2020-08-13 04:40:26    0-5019 loss=1.3568(0.7044+0.6524)-1.3568(0.7044+0.6524) sod-mse=0.1374(0.1374) gcn-mse=0.1468(0.1468) gcn-final-mse=0.1404(0.1534)
2020-08-13 04:42:20 1000-5019 loss=0.0397(0.0332+0.0065)-0.3746(0.1857+0.1889) sod-mse=0.0054(0.0586) gcn-mse=0.0129(0.0619) gcn-final-mse=0.0618(0.0749)
2020-08-13 04:44:11 2000-5019 loss=1.0591(0.4848+0.5743)-0.3806(0.1875+0.1931) sod-mse=0.1021(0.0600) gcn-mse=0.0995(0.0631) gcn-final-mse=0.0631(0.0761)
2020-08-13 04:46:02 3000-5019 loss=0.0475(0.0355+0.0120)-0.3822(0.1879+0.1943) sod-mse=0.0059(0.0604) gcn-mse=0.0083(0.0636) gcn-final-mse=0.0637(0.0766)
2020-08-13 04:47:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 04:47:53 4000-5019 loss=0.1146(0.0754+0.0391)-0.3792(0.1870+0.1922) sod-mse=0.0174(0.0602) gcn-mse=0.0183(0.0636) gcn-final-mse=0.0636(0.0765)
2020-08-13 04:48:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 04:49:44 5000-5019 loss=0.9156(0.3721+0.5435)-0.3782(0.1868+0.1914) sod-mse=0.1218(0.0603) gcn-mse=0.1123(0.0637) gcn-final-mse=0.0636(0.0765)
2020-08-13 04:49:46 E:18, Train sod-mae-score=0.0243-0.9660 gcn-mae-score=0.0290-0.9362 gcn-final-mse-score=0.0284-0.9389(0.0443/0.9389) loss=0.1246(0.0787+0.0459)
2020-08-13 04:49:46 E:18, Test  sod-mae-score=0.0602-0.8279 gcn-mae-score=0.0637-0.7709 gcn-final-mse-score=0.0636-0.7768(0.0765/0.7768) loss=0.3779(0.1867+0.1912)

2020-08-13 04:49:46 Start Epoch 19
2020-08-13 04:49:46 Epoch:19,lr=0.0001
2020-08-13 04:49:48    0-10553 loss=0.1693(0.1195+0.0498)-0.1693(0.1195+0.0498) sod-mse=0.0268(0.0268) gcn-mse=0.0314(0.0314) gcn-final-mse=0.0277(0.0549)
2020-08-13 04:53:54 1000-10553 loss=0.1720(0.1078+0.0642)-0.1132(0.0737+0.0396) sod-mse=0.0491(0.0208) gcn-mse=0.0584(0.0261) gcn-final-mse=0.0253(0.0410)
2020-08-13 04:58:00 2000-10553 loss=0.1744(0.1155+0.0590)-0.1164(0.0751+0.0413) sod-mse=0.0307(0.0217) gcn-mse=0.0362(0.0266) gcn-final-mse=0.0258(0.0418)
2020-08-13 05:02:07 3000-10553 loss=0.0662(0.0515+0.0147)-0.1156(0.0745+0.0410) sod-mse=0.0072(0.0216) gcn-mse=0.0085(0.0264) gcn-final-mse=0.0257(0.0416)
2020-08-13 05:03:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 05:06:13 4000-10553 loss=0.0513(0.0352+0.0161)-0.1157(0.0745+0.0411) sod-mse=0.0073(0.0215) gcn-mse=0.0080(0.0264) gcn-final-mse=0.0257(0.0417)
2020-08-13 05:10:19 5000-10553 loss=0.0492(0.0334+0.0158)-0.1190(0.0760+0.0429) sod-mse=0.0118(0.0225) gcn-mse=0.0135(0.0271) gcn-final-mse=0.0264(0.0424)
2020-08-13 05:14:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 05:14:25 6000-10553 loss=0.0976(0.0620+0.0356)-0.1205(0.0767+0.0437) sod-mse=0.0265(0.0230) gcn-mse=0.0243(0.0277) gcn-final-mse=0.0270(0.0430)
2020-08-13 05:15:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 05:18:32 7000-10553 loss=0.2976(0.1854+0.1122)-0.1217(0.0773+0.0444) sod-mse=0.0598(0.0234) gcn-mse=0.0914(0.0281) gcn-final-mse=0.0274(0.0434)
2020-08-13 05:22:38 8000-10553 loss=0.0635(0.0408+0.0228)-0.1216(0.0773+0.0444) sod-mse=0.0162(0.0234) gcn-mse=0.0215(0.0281) gcn-final-mse=0.0274(0.0434)
2020-08-13 05:26:42 9000-10553 loss=0.1094(0.0770+0.0323)-0.1216(0.0773+0.0443) sod-mse=0.0173(0.0234) gcn-mse=0.0211(0.0280) gcn-final-mse=0.0274(0.0434)
2020-08-13 05:30:48 10000-10553 loss=0.0757(0.0466+0.0291)-0.1222(0.0776+0.0446) sod-mse=0.0139(0.0235) gcn-mse=0.0140(0.0282) gcn-final-mse=0.0276(0.0436)

2020-08-13 05:33:06    0-5019 loss=0.8359(0.4997+0.3362)-0.8359(0.4997+0.3362) sod-mse=0.0836(0.0836) gcn-mse=0.0978(0.0978) gcn-final-mse=0.0907(0.1034)
2020-08-13 05:34:58 1000-5019 loss=0.0545(0.0412+0.0133)-0.3933(0.1987+0.1946) sod-mse=0.0117(0.0706) gcn-mse=0.0204(0.0730) gcn-final-mse=0.0730(0.0865)
2020-08-13 05:36:50 2000-5019 loss=0.7324(0.3767+0.3557)-0.3989(0.1997+0.1991) sod-mse=0.1063(0.0716) gcn-mse=0.0959(0.0739) gcn-final-mse=0.0739(0.0873)
2020-08-13 05:38:41 3000-5019 loss=0.0467(0.0352+0.0115)-0.4139(0.2053+0.2086) sod-mse=0.0052(0.0733) gcn-mse=0.0084(0.0755) gcn-final-mse=0.0756(0.0889)
2020-08-13 05:39:43 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 05:40:32 4000-5019 loss=0.1209(0.0824+0.0385)-0.4101(0.2038+0.2063) sod-mse=0.0197(0.0729) gcn-mse=0.0262(0.0752) gcn-final-mse=0.0753(0.0886)
2020-08-13 05:41:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 05:42:22 5000-5019 loss=0.5838(0.2927+0.2911)-0.4091(0.2035+0.2056) sod-mse=0.1170(0.0730) gcn-mse=0.1118(0.0754) gcn-final-mse=0.0754(0.0887)
2020-08-13 05:42:24 E:19, Train sod-mae-score=0.0235-0.9674 gcn-mae-score=0.0282-0.9373 gcn-final-mse-score=0.0276-0.9399(0.0435/0.9399) loss=0.1222(0.0776+0.0446)
2020-08-13 05:42:24 E:19, Test  sod-mae-score=0.0730-0.8043 gcn-mae-score=0.0754-0.7452 gcn-final-mse-score=0.0754-0.7509(0.0887/0.7509) loss=0.4091(0.2035+0.2057)

2020-08-13 05:42:24 Start Epoch 20
2020-08-13 05:42:24 Epoch:20,lr=0.0000
2020-08-13 05:42:26    0-10553 loss=0.1255(0.0749+0.0506)-0.1255(0.0749+0.0506) sod-mse=0.0245(0.0245) gcn-mse=0.0269(0.0269) gcn-final-mse=0.0243(0.0394)
2020-08-13 05:46:31 1000-10553 loss=0.0752(0.0540+0.0211)-0.1117(0.0724+0.0392) sod-mse=0.0140(0.0203) gcn-mse=0.0184(0.0256) gcn-final-mse=0.0249(0.0411)
2020-08-13 05:50:35 2000-10553 loss=0.1809(0.1079+0.0731)-0.1073(0.0703+0.0370) sod-mse=0.0249(0.0193) gcn-mse=0.0402(0.0244) gcn-final-mse=0.0238(0.0398)
2020-08-13 05:54:41 3000-10553 loss=0.1194(0.0809+0.0385)-0.1053(0.0691+0.0362) sod-mse=0.0179(0.0188) gcn-mse=0.0281(0.0237) gcn-final-mse=0.0230(0.0391)
2020-08-13 05:58:47 4000-10553 loss=0.1190(0.0804+0.0386)-0.1041(0.0684+0.0357) sod-mse=0.0207(0.0185) gcn-mse=0.0240(0.0232) gcn-final-mse=0.0226(0.0386)
2020-08-13 06:02:52 5000-10553 loss=0.0496(0.0314+0.0183)-0.1030(0.0679+0.0352) sod-mse=0.0095(0.0182) gcn-mse=0.0078(0.0229) gcn-final-mse=0.0222(0.0382)
2020-08-13 06:03:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 06:06:57 6000-10553 loss=0.0452(0.0348+0.0104)-0.1023(0.0676+0.0347) sod-mse=0.0056(0.0180) gcn-mse=0.0079(0.0226) gcn-final-mse=0.0220(0.0381)
2020-08-13 06:10:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 06:11:01 7000-10553 loss=0.0994(0.0632+0.0362)-0.1016(0.0673+0.0343) sod-mse=0.0163(0.0178) gcn-mse=0.0174(0.0224) gcn-final-mse=0.0217(0.0379)
2020-08-13 06:15:06 8000-10553 loss=0.0609(0.0397+0.0211)-0.1006(0.0668+0.0338) sod-mse=0.0142(0.0175) gcn-mse=0.0171(0.0221) gcn-final-mse=0.0214(0.0376)
2020-08-13 06:19:13 9000-10553 loss=0.0721(0.0577+0.0145)-0.1002(0.0666+0.0336) sod-mse=0.0078(0.0174) gcn-mse=0.0141(0.0219) gcn-final-mse=0.0212(0.0375)
2020-08-13 06:19:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 06:23:17 10000-10553 loss=0.0574(0.0452+0.0122)-0.0995(0.0662+0.0333) sod-mse=0.0070(0.0172) gcn-mse=0.0131(0.0216) gcn-final-mse=0.0210(0.0372)

2020-08-13 06:25:34    0-5019 loss=0.9168(0.5260+0.3908)-0.9168(0.5260+0.3908) sod-mse=0.0942(0.0942) gcn-mse=0.1047(0.1047) gcn-final-mse=0.0974(0.1106)
2020-08-13 06:27:27 1000-5019 loss=0.0341(0.0283+0.0058)-0.3486(0.1752+0.1734) sod-mse=0.0049(0.0546) gcn-mse=0.0093(0.0580) gcn-final-mse=0.0578(0.0713)
2020-08-13 06:29:20 2000-5019 loss=0.9395(0.4398+0.4997)-0.3538(0.1766+0.1772) sod-mse=0.0987(0.0563) gcn-mse=0.0976(0.0596) gcn-final-mse=0.0594(0.0728)
2020-08-13 06:31:11 3000-5019 loss=0.0468(0.0349+0.0119)-0.3549(0.1767+0.1782) sod-mse=0.0056(0.0567) gcn-mse=0.0083(0.0600) gcn-final-mse=0.0599(0.0733)
2020-08-13 06:32:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 06:33:03 4000-5019 loss=0.1197(0.0815+0.0382)-0.3506(0.1753+0.1754) sod-mse=0.0171(0.0561) gcn-mse=0.0240(0.0596) gcn-final-mse=0.0594(0.0728)
2020-08-13 06:33:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 06:34:53 5000-5019 loss=1.2332(0.5159+0.7173)-0.3489(0.1747+0.1743) sod-mse=0.0898(0.0561) gcn-mse=0.0917(0.0596) gcn-final-mse=0.0594(0.0728)
2020-08-13 06:34:55 E:20, Train sod-mae-score=0.0170-0.9749 gcn-mae-score=0.0215-0.9453 gcn-final-mse-score=0.0208-0.9479(0.0371/0.9479) loss=0.0990(0.0660+0.0331)
2020-08-13 06:34:55 E:20, Test  sod-mae-score=0.0561-0.8342 gcn-mae-score=0.0596-0.7789 gcn-final-mse-score=0.0594-0.7840(0.0728/0.7840) loss=0.3489(0.1746+0.1742)

2020-08-13 06:34:55 Start Epoch 21
2020-08-13 06:34:55 Epoch:21,lr=0.0000
2020-08-13 06:34:57    0-10553 loss=0.1341(0.0832+0.0510)-0.1341(0.0832+0.0510) sod-mse=0.0300(0.0300) gcn-mse=0.0344(0.0344) gcn-final-mse=0.0346(0.0513)
2020-08-13 06:39:01 1000-10553 loss=0.0774(0.0640+0.0133)-0.0886(0.0607+0.0280) sod-mse=0.0067(0.0145) gcn-mse=0.0225(0.0186) gcn-final-mse=0.0179(0.0343)
2020-08-13 06:43:05 2000-10553 loss=0.0642(0.0479+0.0163)-0.0932(0.0629+0.0303) sod-mse=0.0108(0.0154) gcn-mse=0.0189(0.0194) gcn-final-mse=0.0187(0.0353)
2020-08-13 06:47:11 3000-10553 loss=0.0489(0.0354+0.0134)-0.0940(0.0633+0.0307) sod-mse=0.0081(0.0156) gcn-mse=0.0105(0.0195) gcn-final-mse=0.0188(0.0353)
2020-08-13 06:51:17 4000-10553 loss=0.0929(0.0682+0.0247)-0.0947(0.0637+0.0310) sod-mse=0.0135(0.0158) gcn-mse=0.0198(0.0197) gcn-final-mse=0.0189(0.0355)
2020-08-13 06:55:22 5000-10553 loss=0.1405(0.0883+0.0522)-0.0935(0.0630+0.0305) sod-mse=0.0258(0.0156) gcn-mse=0.0269(0.0194) gcn-final-mse=0.0187(0.0352)
2020-08-13 06:59:28 6000-10553 loss=0.1102(0.0729+0.0373)-0.0927(0.0625+0.0301) sod-mse=0.0203(0.0154) gcn-mse=0.0247(0.0192) gcn-final-mse=0.0185(0.0350)
2020-08-13 07:00:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 07:03:36 7000-10553 loss=0.1393(0.0877+0.0516)-0.0927(0.0626+0.0301) sod-mse=0.0265(0.0154) gcn-mse=0.0305(0.0192) gcn-final-mse=0.0185(0.0350)
2020-08-13 07:07:42 8000-10553 loss=0.0531(0.0390+0.0141)-0.0927(0.0626+0.0301) sod-mse=0.0086(0.0154) gcn-mse=0.0110(0.0191) gcn-final-mse=0.0184(0.0350)
2020-08-13 07:09:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 07:11:48 9000-10553 loss=0.2621(0.1486+0.1135)-0.0924(0.0624+0.0300) sod-mse=0.0448(0.0153) gcn-mse=0.0503(0.0190) gcn-final-mse=0.0183(0.0349)
2020-08-13 07:13:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 07:15:54 10000-10553 loss=0.0766(0.0487+0.0279)-0.0919(0.0621+0.0298) sod-mse=0.0169(0.0152) gcn-mse=0.0199(0.0189) gcn-final-mse=0.0182(0.0347)

2020-08-13 07:18:11    0-5019 loss=0.9817(0.5810+0.4007)-0.9817(0.5810+0.4007) sod-mse=0.0939(0.0939) gcn-mse=0.1047(0.1047) gcn-final-mse=0.0970(0.1094)
2020-08-13 07:20:03 1000-5019 loss=0.0329(0.0273+0.0057)-0.3448(0.1768+0.1680) sod-mse=0.0047(0.0542) gcn-mse=0.0082(0.0565) gcn-final-mse=0.0564(0.0700)
2020-08-13 07:21:56 2000-5019 loss=1.0328(0.4909+0.5419)-0.3504(0.1783+0.1721) sod-mse=0.0998(0.0558) gcn-mse=0.0988(0.0581) gcn-final-mse=0.0579(0.0714)
2020-08-13 07:23:47 3000-5019 loss=0.0467(0.0346+0.0121)-0.3504(0.1781+0.1722) sod-mse=0.0058(0.0559) gcn-mse=0.0079(0.0583) gcn-final-mse=0.0581(0.0716)
2020-08-13 07:24:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 07:25:39 4000-5019 loss=0.1137(0.0778+0.0359)-0.3460(0.1765+0.1695) sod-mse=0.0170(0.0553) gcn-mse=0.0212(0.0579) gcn-final-mse=0.0577(0.0712)
2020-08-13 07:26:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 07:27:31 5000-5019 loss=1.1917(0.5148+0.6769)-0.3444(0.1760+0.1684) sod-mse=0.0923(0.0553) gcn-mse=0.0943(0.0579) gcn-final-mse=0.0576(0.0711)
2020-08-13 07:27:32 E:21, Train sod-mae-score=0.0152-0.9777 gcn-mae-score=0.0189-0.9487 gcn-final-mse-score=0.0182-0.9513(0.0347/0.9513) loss=0.0917(0.0620+0.0297)
2020-08-13 07:27:32 E:21, Test  sod-mae-score=0.0553-0.8381 gcn-mae-score=0.0579-0.7840 gcn-final-mse-score=0.0577-0.7895(0.0711/0.7895) loss=0.3443(0.1760+0.1683)

2020-08-13 07:27:32 Start Epoch 22
2020-08-13 07:27:32 Epoch:22,lr=0.0000
2020-08-13 07:27:34    0-10553 loss=0.0579(0.0405+0.0174)-0.0579(0.0405+0.0174) sod-mse=0.0081(0.0081) gcn-mse=0.0075(0.0075) gcn-final-mse=0.0067(0.0197)
2020-08-13 07:31:41 1000-10553 loss=0.1359(0.0759+0.0600)-0.0914(0.0615+0.0299) sod-mse=0.0316(0.0143) gcn-mse=0.0320(0.0179) gcn-final-mse=0.0173(0.0337)
2020-08-13 07:35:48 2000-10553 loss=0.0438(0.0332+0.0106)-0.0914(0.0614+0.0299) sod-mse=0.0047(0.0148) gcn-mse=0.0062(0.0183) gcn-final-mse=0.0176(0.0340)
2020-08-13 07:39:54 3000-10553 loss=0.0927(0.0631+0.0297)-0.0913(0.0617+0.0296) sod-mse=0.0162(0.0148) gcn-mse=0.0188(0.0183) gcn-final-mse=0.0176(0.0342)
2020-08-13 07:44:00 4000-10553 loss=0.0783(0.0577+0.0205)-0.0901(0.0611+0.0291) sod-mse=0.0117(0.0146) gcn-mse=0.0156(0.0181) gcn-final-mse=0.0173(0.0339)
2020-08-13 07:48:04 5000-10553 loss=0.2398(0.1373+0.1025)-0.0890(0.0606+0.0284) sod-mse=0.0313(0.0143) gcn-mse=0.0295(0.0179) gcn-final-mse=0.0171(0.0337)
2020-08-13 07:52:09 6000-10553 loss=0.1334(0.0749+0.0585)-0.0890(0.0606+0.0284) sod-mse=0.0302(0.0143) gcn-mse=0.0230(0.0179) gcn-final-mse=0.0171(0.0337)
2020-08-13 07:56:16 7000-10553 loss=0.0632(0.0468+0.0164)-0.0888(0.0606+0.0283) sod-mse=0.0090(0.0143) gcn-mse=0.0090(0.0179) gcn-final-mse=0.0171(0.0337)
2020-08-13 07:59:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 08:00:21 8000-10553 loss=0.1696(0.1046+0.0650)-0.0886(0.0605+0.0281) sod-mse=0.0328(0.0143) gcn-mse=0.0363(0.0178) gcn-final-mse=0.0171(0.0337)
2020-08-13 08:03:58 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 08:04:26 9000-10553 loss=0.1404(0.0935+0.0469)-0.0883(0.0603+0.0280) sod-mse=0.0245(0.0142) gcn-mse=0.0364(0.0178) gcn-final-mse=0.0170(0.0336)
2020-08-13 08:08:30 10000-10553 loss=0.1431(0.0853+0.0577)-0.0881(0.0602+0.0280) sod-mse=0.0223(0.0142) gcn-mse=0.0240(0.0177) gcn-final-mse=0.0170(0.0336)
2020-08-13 08:10:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg

2020-08-13 08:10:47    0-5019 loss=1.0079(0.5997+0.4082)-1.0079(0.5997+0.4082) sod-mse=0.0994(0.0994) gcn-mse=0.1089(0.1089) gcn-final-mse=0.1010(0.1133)
2020-08-13 08:12:38 1000-5019 loss=0.0326(0.0270+0.0056)-0.3371(0.1733+0.1638) sod-mse=0.0047(0.0529) gcn-mse=0.0078(0.0553) gcn-final-mse=0.0552(0.0690)
2020-08-13 08:14:28 2000-5019 loss=0.9450(0.4700+0.4750)-0.3446(0.1756+0.1690) sod-mse=0.1008(0.0547) gcn-mse=0.0999(0.0571) gcn-final-mse=0.0569(0.0706)
2020-08-13 08:16:19 3000-5019 loss=0.0457(0.0342+0.0115)-0.3435(0.1749+0.1686) sod-mse=0.0053(0.0548) gcn-mse=0.0076(0.0573) gcn-final-mse=0.0571(0.0708)
2020-08-13 08:17:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 08:18:11 4000-5019 loss=0.1138(0.0783+0.0354)-0.3392(0.1731+0.1661) sod-mse=0.0162(0.0542) gcn-mse=0.0214(0.0569) gcn-final-mse=0.0567(0.0704)
2020-08-13 08:18:43 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 08:20:02 5000-5019 loss=1.2059(0.5388+0.6672)-0.3394(0.1734+0.1660) sod-mse=0.0914(0.0544) gcn-mse=0.0948(0.0570) gcn-final-mse=0.0567(0.0704)
2020-08-13 08:20:03 E:22, Train sod-mae-score=0.0142-0.9790 gcn-mae-score=0.0178-0.9497 gcn-final-mse-score=0.0170-0.9523(0.0336/0.9523) loss=0.0883(0.0603+0.0280)
2020-08-13 08:20:03 E:22, Test  sod-mae-score=0.0544-0.8399 gcn-mae-score=0.0570-0.7839 gcn-final-mse-score=0.0568-0.7893(0.0704/0.7893) loss=0.3393(0.1734+0.1660)

2020-08-13 08:20:03 Start Epoch 23
2020-08-13 08:20:03 Epoch:23,lr=0.0000
2020-08-13 08:20:05    0-10553 loss=0.0974(0.0720+0.0254)-0.0974(0.0720+0.0254) sod-mse=0.0184(0.0184) gcn-mse=0.0277(0.0277) gcn-final-mse=0.0272(0.0504)
2020-08-13 08:24:11 1000-10553 loss=0.0598(0.0407+0.0191)-0.0844(0.0584+0.0260) sod-mse=0.0141(0.0132) gcn-mse=0.0207(0.0167) gcn-final-mse=0.0160(0.0327)
2020-08-13 08:28:16 2000-10553 loss=0.0851(0.0662+0.0190)-0.0848(0.0586+0.0262) sod-mse=0.0107(0.0133) gcn-mse=0.0209(0.0169) gcn-final-mse=0.0161(0.0329)
2020-08-13 08:32:20 3000-10553 loss=0.0682(0.0405+0.0277)-0.0848(0.0585+0.0262) sod-mse=0.0148(0.0134) gcn-mse=0.0131(0.0169) gcn-final-mse=0.0162(0.0328)
2020-08-13 08:36:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 08:36:24 4000-10553 loss=0.0814(0.0570+0.0244)-0.0855(0.0590+0.0266) sod-mse=0.0108(0.0135) gcn-mse=0.0135(0.0170) gcn-final-mse=0.0163(0.0330)
2020-08-13 08:40:28 5000-10553 loss=0.1095(0.0772+0.0323)-0.0858(0.0591+0.0268) sod-mse=0.0137(0.0136) gcn-mse=0.0181(0.0171) gcn-final-mse=0.0164(0.0330)
2020-08-13 08:44:33 6000-10553 loss=0.0392(0.0314+0.0078)-0.0855(0.0589+0.0266) sod-mse=0.0040(0.0135) gcn-mse=0.0047(0.0170) gcn-final-mse=0.0163(0.0329)
2020-08-13 08:48:36 7000-10553 loss=0.0552(0.0372+0.0179)-0.0853(0.0588+0.0265) sod-mse=0.0101(0.0134) gcn-mse=0.0107(0.0169) gcn-final-mse=0.0162(0.0328)
2020-08-13 08:52:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 08:52:40 8000-10553 loss=0.0478(0.0327+0.0151)-0.0853(0.0588+0.0265) sod-mse=0.0088(0.0134) gcn-mse=0.0086(0.0169) gcn-final-mse=0.0161(0.0328)
2020-08-13 08:56:47 9000-10553 loss=0.1312(0.0997+0.0315)-0.0854(0.0588+0.0266) sod-mse=0.0191(0.0135) gcn-mse=0.0355(0.0169) gcn-final-mse=0.0161(0.0329)
2020-08-13 09:00:53 10000-10553 loss=0.0767(0.0585+0.0182)-0.0853(0.0588+0.0265) sod-mse=0.0099(0.0135) gcn-mse=0.0139(0.0169) gcn-final-mse=0.0162(0.0329)
2020-08-13 09:02:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg

2020-08-13 09:03:09    0-5019 loss=1.0253(0.6070+0.4184)-1.0253(0.6070+0.4184) sod-mse=0.0994(0.0994) gcn-mse=0.1095(0.1095) gcn-final-mse=0.1015(0.1133)
2020-08-13 09:05:01 1000-5019 loss=0.0330(0.0274+0.0056)-0.3521(0.1783+0.1738) sod-mse=0.0047(0.0527) gcn-mse=0.0083(0.0552) gcn-final-mse=0.0550(0.0687)
2020-08-13 09:06:53 2000-5019 loss=0.9882(0.4683+0.5199)-0.3605(0.1807+0.1799) sod-mse=0.0991(0.0546) gcn-mse=0.0976(0.0569) gcn-final-mse=0.0567(0.0703)
2020-08-13 09:08:45 3000-5019 loss=0.0457(0.0341+0.0116)-0.3591(0.1799+0.1792) sod-mse=0.0053(0.0547) gcn-mse=0.0076(0.0572) gcn-final-mse=0.0570(0.0705)
2020-08-13 09:09:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 09:10:36 4000-5019 loss=0.1151(0.0797+0.0354)-0.3551(0.1785+0.1766) sod-mse=0.0166(0.0543) gcn-mse=0.0228(0.0569) gcn-final-mse=0.0567(0.0703)
2020-08-13 09:11:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 09:12:27 5000-5019 loss=1.2332(0.5139+0.7193)-0.3546(0.1785+0.1760) sod-mse=0.0945(0.0543) gcn-mse=0.0981(0.0570) gcn-final-mse=0.0567(0.0702)
2020-08-13 09:12:29 E:23, Train sod-mae-score=0.0135-0.9798 gcn-mae-score=0.0169-0.9509 gcn-final-mse-score=0.0162-0.9534(0.0329/0.9534) loss=0.0854(0.0588+0.0266)
2020-08-13 09:12:29 E:23, Test  sod-mae-score=0.0543-0.8399 gcn-mae-score=0.0570-0.7869 gcn-final-mse-score=0.0567-0.7926(0.0702/0.7926) loss=0.3544(0.1785+0.1759)

2020-08-13 09:12:29 Start Epoch 24
2020-08-13 09:12:29 Epoch:24,lr=0.0000
2020-08-13 09:12:31    0-10553 loss=0.0627(0.0516+0.0111)-0.0627(0.0516+0.0111) sod-mse=0.0072(0.0072) gcn-mse=0.0095(0.0095) gcn-final-mse=0.0094(0.0294)
2020-08-13 09:16:39 1000-10553 loss=0.1014(0.0708+0.0306)-0.0823(0.0573+0.0250) sod-mse=0.0167(0.0127) gcn-mse=0.0202(0.0162) gcn-final-mse=0.0155(0.0321)
2020-08-13 09:20:45 2000-10553 loss=0.0483(0.0343+0.0140)-0.0815(0.0569+0.0246) sod-mse=0.0073(0.0125) gcn-mse=0.0087(0.0159) gcn-final-mse=0.0152(0.0319)
2020-08-13 09:24:52 3000-10553 loss=0.1149(0.0839+0.0310)-0.0830(0.0577+0.0253) sod-mse=0.0220(0.0127) gcn-mse=0.0216(0.0162) gcn-final-mse=0.0154(0.0322)
2020-08-13 09:28:59 4000-10553 loss=0.0690(0.0540+0.0149)-0.0830(0.0577+0.0253) sod-mse=0.0076(0.0128) gcn-mse=0.0097(0.0162) gcn-final-mse=0.0155(0.0322)
2020-08-13 09:33:06 5000-10553 loss=0.1390(0.0934+0.0456)-0.0833(0.0578+0.0255) sod-mse=0.0225(0.0129) gcn-mse=0.0340(0.0163) gcn-final-mse=0.0155(0.0323)
2020-08-13 09:37:15 6000-10553 loss=0.1323(0.0936+0.0387)-0.0834(0.0579+0.0255) sod-mse=0.0179(0.0129) gcn-mse=0.0205(0.0164) gcn-final-mse=0.0156(0.0323)
2020-08-13 09:38:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 09:41:23 7000-10553 loss=0.0737(0.0525+0.0212)-0.0831(0.0578+0.0253) sod-mse=0.0128(0.0128) gcn-mse=0.0189(0.0163) gcn-final-mse=0.0155(0.0323)
2020-08-13 09:44:27 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 09:45:32 8000-10553 loss=0.1423(0.0983+0.0440)-0.0830(0.0577+0.0253) sod-mse=0.0199(0.0128) gcn-mse=0.0401(0.0162) gcn-final-mse=0.0155(0.0322)
2020-08-13 09:47:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 09:49:39 9000-10553 loss=0.0995(0.0660+0.0336)-0.0831(0.0578+0.0253) sod-mse=0.0175(0.0128) gcn-mse=0.0247(0.0162) gcn-final-mse=0.0155(0.0323)
2020-08-13 09:53:45 10000-10553 loss=0.0754(0.0511+0.0243)-0.0831(0.0577+0.0254) sod-mse=0.0129(0.0129) gcn-mse=0.0215(0.0162) gcn-final-mse=0.0155(0.0323)

2020-08-13 09:56:04    0-5019 loss=1.0006(0.5966+0.4039)-1.0006(0.5966+0.4039) sod-mse=0.0969(0.0969) gcn-mse=0.1057(0.1057) gcn-final-mse=0.0974(0.1094)
2020-08-13 09:57:57 1000-5019 loss=0.0302(0.0248+0.0054)-0.3530(0.1785+0.1745) sod-mse=0.0044(0.0523) gcn-mse=0.0058(0.0545) gcn-final-mse=0.0543(0.0679)
2020-08-13 09:59:51 2000-5019 loss=1.0527(0.5130+0.5397)-0.3628(0.1815+0.1812) sod-mse=0.0991(0.0542) gcn-mse=0.0987(0.0564) gcn-final-mse=0.0561(0.0695)
2020-08-13 10:01:44 3000-5019 loss=0.0456(0.0341+0.0115)-0.3594(0.1799+0.1795) sod-mse=0.0054(0.0542) gcn-mse=0.0074(0.0564) gcn-final-mse=0.0562(0.0695)
2020-08-13 10:02:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 10:03:37 4000-5019 loss=0.1191(0.0832+0.0359)-0.3549(0.1781+0.1768) sod-mse=0.0169(0.0537) gcn-mse=0.0256(0.0561) gcn-final-mse=0.0558(0.0692)
2020-08-13 10:04:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 10:05:30 5000-5019 loss=1.4441(0.6313+0.8128)-0.3544(0.1781+0.1763) sod-mse=0.0885(0.0537) gcn-mse=0.0929(0.0561) gcn-final-mse=0.0558(0.0692)
2020-08-13 10:05:32 E:24, Train sod-mae-score=0.0129-0.9806 gcn-mae-score=0.0162-0.9510 gcn-final-mse-score=0.0155-0.9536(0.0323/0.9536) loss=0.0831(0.0577+0.0254)
2020-08-13 10:05:32 E:24, Test  sod-mae-score=0.0537-0.8404 gcn-mae-score=0.0561-0.7877 gcn-final-mse-score=0.0558-0.7933(0.0692/0.7933) loss=0.3544(0.1781+0.1763)

2020-08-13 10:05:32 Start Epoch 25
2020-08-13 10:05:32 Epoch:25,lr=0.0000
2020-08-13 10:05:34    0-10553 loss=0.0894(0.0607+0.0287)-0.0894(0.0607+0.0287) sod-mse=0.0199(0.0199) gcn-mse=0.0212(0.0212) gcn-final-mse=0.0222(0.0402)
2020-08-13 10:09:39 1000-10553 loss=0.1215(0.0907+0.0307)-0.0805(0.0568+0.0237) sod-mse=0.0133(0.0122) gcn-mse=0.0190(0.0157) gcn-final-mse=0.0149(0.0318)
2020-08-13 10:13:44 2000-10553 loss=0.0887(0.0684+0.0202)-0.0823(0.0575+0.0247) sod-mse=0.0113(0.0126) gcn-mse=0.0153(0.0159) gcn-final-mse=0.0151(0.0322)
2020-08-13 10:16:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 10:17:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 10:17:50 3000-10553 loss=0.0907(0.0540+0.0367)-0.0811(0.0567+0.0244) sod-mse=0.0206(0.0124) gcn-mse=0.0194(0.0157) gcn-final-mse=0.0149(0.0318)
2020-08-13 10:21:55 4000-10553 loss=0.0634(0.0458+0.0176)-0.0808(0.0565+0.0243) sod-mse=0.0098(0.0124) gcn-mse=0.0116(0.0156) gcn-final-mse=0.0149(0.0317)
2020-08-13 10:26:03 5000-10553 loss=0.0211(0.0157+0.0053)-0.0816(0.0569+0.0247) sod-mse=0.0031(0.0125) gcn-mse=0.0061(0.0158) gcn-final-mse=0.0150(0.0318)
2020-08-13 10:30:09 6000-10553 loss=0.0438(0.0341+0.0097)-0.0818(0.0570+0.0248) sod-mse=0.0044(0.0126) gcn-mse=0.0053(0.0158) gcn-final-mse=0.0150(0.0319)
2020-08-13 10:32:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 10:34:14 7000-10553 loss=0.0332(0.0249+0.0083)-0.0818(0.0570+0.0248) sod-mse=0.0057(0.0126) gcn-mse=0.0070(0.0158) gcn-final-mse=0.0150(0.0319)
2020-08-13 10:38:19 8000-10553 loss=0.0868(0.0612+0.0256)-0.0817(0.0570+0.0248) sod-mse=0.0115(0.0125) gcn-mse=0.0215(0.0158) gcn-final-mse=0.0150(0.0318)
2020-08-13 10:42:23 9000-10553 loss=0.0694(0.0506+0.0189)-0.0814(0.0568+0.0246) sod-mse=0.0129(0.0124) gcn-mse=0.0159(0.0157) gcn-final-mse=0.0149(0.0318)
2020-08-13 10:46:27 10000-10553 loss=0.1039(0.0648+0.0391)-0.0816(0.0568+0.0247) sod-mse=0.0139(0.0125) gcn-mse=0.0256(0.0157) gcn-final-mse=0.0150(0.0318)

2020-08-13 10:48:44    0-5019 loss=1.0635(0.6326+0.4309)-1.0635(0.6326+0.4309) sod-mse=0.1000(0.1000) gcn-mse=0.1079(0.1079) gcn-final-mse=0.1001(0.1125)
2020-08-13 10:50:36 1000-5019 loss=0.0302(0.0252+0.0050)-0.3595(0.1805+0.1790) sod-mse=0.0040(0.0514) gcn-mse=0.0063(0.0544) gcn-final-mse=0.0542(0.0678)
2020-08-13 10:52:26 2000-5019 loss=1.0379(0.5022+0.5357)-0.3685(0.1828+0.1857) sod-mse=0.0961(0.0535) gcn-mse=0.0964(0.0563) gcn-final-mse=0.0560(0.0695)
2020-08-13 10:54:17 3000-5019 loss=0.0458(0.0339+0.0119)-0.3684(0.1825+0.1859) sod-mse=0.0054(0.0538) gcn-mse=0.0073(0.0566) gcn-final-mse=0.0564(0.0698)
2020-08-13 10:55:18 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 10:56:07 4000-5019 loss=0.1125(0.0781+0.0344)-0.3637(0.1809+0.1827) sod-mse=0.0155(0.0533) gcn-mse=0.0209(0.0563) gcn-final-mse=0.0560(0.0695)
2020-08-13 10:56:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 10:57:57 5000-5019 loss=1.2617(0.5364+0.7253)-0.3632(0.1810+0.1822) sod-mse=0.0878(0.0534) gcn-mse=0.0917(0.0563) gcn-final-mse=0.0561(0.0695)
2020-08-13 10:57:59 E:25, Train sod-mae-score=0.0125-0.9813 gcn-mae-score=0.0157-0.9525 gcn-final-mse-score=0.0150-0.9550(0.0318/0.9550) loss=0.0816(0.0569+0.0247)
2020-08-13 10:57:59 E:25, Test  sod-mae-score=0.0534-0.8412 gcn-mae-score=0.0563-0.7871 gcn-final-mse-score=0.0561-0.7927(0.0695/0.7927) loss=0.3631(0.1809+0.1821)

2020-08-13 10:57:59 Start Epoch 26
2020-08-13 10:57:59 Epoch:26,lr=0.0000
2020-08-13 10:58:01    0-10553 loss=0.0773(0.0603+0.0170)-0.0773(0.0603+0.0170) sod-mse=0.0071(0.0071) gcn-mse=0.0122(0.0122) gcn-final-mse=0.0108(0.0273)
2020-08-13 11:02:05 1000-10553 loss=0.0694(0.0451+0.0243)-0.0822(0.0577+0.0245) sod-mse=0.0163(0.0125) gcn-mse=0.0200(0.0159) gcn-final-mse=0.0152(0.0324)
2020-08-13 11:06:10 2000-10553 loss=0.1346(0.0720+0.0626)-0.0796(0.0559+0.0237) sod-mse=0.0221(0.0121) gcn-mse=0.0216(0.0154) gcn-final-mse=0.0146(0.0315)
2020-08-13 11:10:15 3000-10553 loss=0.0484(0.0320+0.0164)-0.0787(0.0554+0.0232) sod-mse=0.0080(0.0119) gcn-mse=0.0072(0.0151) gcn-final-mse=0.0143(0.0312)
2020-08-13 11:13:05 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 11:14:21 4000-10553 loss=0.0646(0.0504+0.0142)-0.0786(0.0554+0.0232) sod-mse=0.0062(0.0118) gcn-mse=0.0097(0.0150) gcn-final-mse=0.0143(0.0311)
2020-08-13 11:18:26 5000-10553 loss=0.0452(0.0346+0.0106)-0.0793(0.0558+0.0235) sod-mse=0.0052(0.0119) gcn-mse=0.0068(0.0151) gcn-final-mse=0.0144(0.0313)
2020-08-13 11:22:31 6000-10553 loss=0.0922(0.0677+0.0244)-0.0802(0.0562+0.0241) sod-mse=0.0152(0.0121) gcn-mse=0.0199(0.0153) gcn-final-mse=0.0145(0.0314)
2020-08-13 11:26:36 7000-10553 loss=0.0967(0.0731+0.0236)-0.0801(0.0561+0.0240) sod-mse=0.0145(0.0121) gcn-mse=0.0181(0.0152) gcn-final-mse=0.0144(0.0313)
2020-08-13 11:29:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 11:30:40 8000-10553 loss=0.0726(0.0557+0.0169)-0.0799(0.0560+0.0239) sod-mse=0.0086(0.0120) gcn-mse=0.0143(0.0152) gcn-final-mse=0.0144(0.0313)
2020-08-13 11:34:43 9000-10553 loss=0.1042(0.0665+0.0377)-0.0800(0.0561+0.0239) sod-mse=0.0187(0.0120) gcn-mse=0.0214(0.0152) gcn-final-mse=0.0144(0.0313)
2020-08-13 11:34:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 11:38:50 10000-10553 loss=0.0662(0.0468+0.0195)-0.0799(0.0560+0.0239) sod-mse=0.0103(0.0120) gcn-mse=0.0132(0.0152) gcn-final-mse=0.0144(0.0313)

2020-08-13 11:41:08    0-5019 loss=1.0303(0.6261+0.4043)-1.0303(0.6261+0.4043) sod-mse=0.0957(0.0957) gcn-mse=0.1014(0.1014) gcn-final-mse=0.0939(0.1067)
2020-08-13 11:43:00 1000-5019 loss=0.0321(0.0264+0.0057)-0.3502(0.1786+0.1716) sod-mse=0.0047(0.0525) gcn-mse=0.0076(0.0551) gcn-final-mse=0.0548(0.0687)
2020-08-13 11:44:53 2000-5019 loss=0.9844(0.4879+0.4965)-0.3630(0.1828+0.1802) sod-mse=0.0980(0.0549) gcn-mse=0.0967(0.0572) gcn-final-mse=0.0570(0.0707)
2020-08-13 11:46:45 3000-5019 loss=0.0451(0.0337+0.0114)-0.3625(0.1821+0.1804) sod-mse=0.0052(0.0551) gcn-mse=0.0069(0.0574) gcn-final-mse=0.0572(0.0709)
2020-08-13 11:47:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 11:48:38 4000-5019 loss=0.1086(0.0757+0.0329)-0.3576(0.1801+0.1775) sod-mse=0.0149(0.0546) gcn-mse=0.0180(0.0570) gcn-final-mse=0.0567(0.0705)
2020-08-13 11:49:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 11:50:31 5000-5019 loss=1.1302(0.5068+0.6234)-0.3578(0.1804+0.1774) sod-mse=0.0969(0.0546) gcn-mse=0.0985(0.0571) gcn-final-mse=0.0568(0.0705)
2020-08-13 11:50:33 E:26, Train sod-mae-score=0.0121-0.9818 gcn-mae-score=0.0153-0.9530 gcn-final-mse-score=0.0145-0.9554(0.0314/0.9554) loss=0.0804(0.0563+0.0241)
2020-08-13 11:50:33 E:26, Test  sod-mae-score=0.0546-0.8384 gcn-mae-score=0.0571-0.7823 gcn-final-mse-score=0.0568-0.7877(0.0705/0.7877) loss=0.3578(0.1804+0.1774)

2020-08-13 11:50:33 Start Epoch 27
2020-08-13 11:50:33 Epoch:27,lr=0.0000
2020-08-13 11:50:34    0-10553 loss=0.0990(0.0605+0.0385)-0.0990(0.0605+0.0385) sod-mse=0.0229(0.0229) gcn-mse=0.0214(0.0214) gcn-final-mse=0.0216(0.0375)
2020-08-13 11:54:41 1000-10553 loss=0.0528(0.0404+0.0123)-0.0798(0.0561+0.0237) sod-mse=0.0068(0.0119) gcn-mse=0.0077(0.0150) gcn-final-mse=0.0143(0.0314)
2020-08-13 11:58:47 2000-10553 loss=0.1439(0.0830+0.0608)-0.0783(0.0552+0.0230) sod-mse=0.0245(0.0117) gcn-mse=0.0252(0.0147) gcn-final-mse=0.0140(0.0310)
2020-08-13 12:00:44 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 12:02:56 3000-10553 loss=0.0849(0.0558+0.0291)-0.0785(0.0554+0.0231) sod-mse=0.0153(0.0117) gcn-mse=0.0152(0.0148) gcn-final-mse=0.0140(0.0310)
2020-08-13 12:07:02 4000-10553 loss=0.0206(0.0160+0.0046)-0.0783(0.0553+0.0230) sod-mse=0.0037(0.0116) gcn-mse=0.0053(0.0148) gcn-final-mse=0.0140(0.0310)
2020-08-13 12:11:09 5000-10553 loss=0.0963(0.0725+0.0238)-0.0793(0.0559+0.0234) sod-mse=0.0132(0.0117) gcn-mse=0.0185(0.0149) gcn-final-mse=0.0141(0.0312)
2020-08-13 12:15:16 6000-10553 loss=0.0778(0.0645+0.0134)-0.0792(0.0558+0.0234) sod-mse=0.0096(0.0117) gcn-mse=0.0180(0.0149) gcn-final-mse=0.0141(0.0311)
2020-08-13 12:19:24 7000-10553 loss=0.1036(0.0774+0.0262)-0.0792(0.0558+0.0234) sod-mse=0.0185(0.0118) gcn-mse=0.0182(0.0149) gcn-final-mse=0.0141(0.0311)
2020-08-13 12:19:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 12:21:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 12:23:31 8000-10553 loss=0.0744(0.0522+0.0222)-0.0794(0.0559+0.0236) sod-mse=0.0121(0.0118) gcn-mse=0.0189(0.0150) gcn-final-mse=0.0142(0.0312)
2020-08-13 12:27:37 9000-10553 loss=0.0618(0.0444+0.0175)-0.0797(0.0560+0.0237) sod-mse=0.0092(0.0118) gcn-mse=0.0126(0.0150) gcn-final-mse=0.0142(0.0313)
2020-08-13 12:31:46 10000-10553 loss=0.1364(0.0907+0.0457)-0.0794(0.0558+0.0236) sod-mse=0.0175(0.0118) gcn-mse=0.0229(0.0149) gcn-final-mse=0.0141(0.0311)

2020-08-13 12:34:05    0-5019 loss=1.1110(0.6485+0.4624)-1.1110(0.6485+0.4624) sod-mse=0.1029(0.1029) gcn-mse=0.1092(0.1092) gcn-final-mse=0.1012(0.1128)
2020-08-13 12:35:57 1000-5019 loss=0.0302(0.0252+0.0050)-0.3658(0.1817+0.1842) sod-mse=0.0041(0.0514) gcn-mse=0.0062(0.0540) gcn-final-mse=0.0538(0.0674)
2020-08-13 12:37:49 2000-5019 loss=1.0796(0.5096+0.5700)-0.3740(0.1843+0.1897) sod-mse=0.0955(0.0534) gcn-mse=0.0968(0.0559) gcn-final-mse=0.0556(0.0691)
2020-08-13 12:39:41 3000-5019 loss=0.0457(0.0341+0.0115)-0.3724(0.1834+0.1890) sod-mse=0.0053(0.0534) gcn-mse=0.0072(0.0559) gcn-final-mse=0.0557(0.0692)
2020-08-13 12:40:43 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 12:41:33 4000-5019 loss=0.1083(0.0759+0.0324)-0.3680(0.1819+0.1861) sod-mse=0.0150(0.0530) gcn-mse=0.0182(0.0557) gcn-final-mse=0.0554(0.0689)
2020-08-13 12:42:06 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 12:43:26 5000-5019 loss=1.5776(0.6242+0.9534)-0.3677(0.1820+0.1857) sod-mse=0.0889(0.0530) gcn-mse=0.0928(0.0557) gcn-final-mse=0.0554(0.0688)
2020-08-13 12:43:28 E:27, Train sod-mae-score=0.0117-0.9824 gcn-mae-score=0.0149-0.9534 gcn-final-mse-score=0.0141-0.9559(0.0311/0.9559) loss=0.0791(0.0556+0.0235)
2020-08-13 12:43:28 E:27, Test  sod-mae-score=0.0530-0.8410 gcn-mae-score=0.0557-0.7876 gcn-final-mse-score=0.0554-0.7932(0.0688/0.7932) loss=0.3676(0.1820+0.1856)

2020-08-13 12:43:28 Start Epoch 28
2020-08-13 12:43:28 Epoch:28,lr=0.0000
2020-08-13 12:43:29    0-10553 loss=0.0366(0.0310+0.0055)-0.0366(0.0310+0.0055) sod-mse=0.0027(0.0027) gcn-mse=0.0078(0.0078) gcn-final-mse=0.0067(0.0177)
2020-08-13 12:47:36 1000-10553 loss=0.0794(0.0589+0.0205)-0.0770(0.0549+0.0221) sod-mse=0.0100(0.0111) gcn-mse=0.0070(0.0142) gcn-final-mse=0.0134(0.0307)
2020-08-13 12:51:43 2000-10553 loss=0.0605(0.0508+0.0098)-0.0762(0.0543+0.0220) sod-mse=0.0044(0.0110) gcn-mse=0.0109(0.0141) gcn-final-mse=0.0134(0.0304)
2020-08-13 12:55:48 3000-10553 loss=0.0452(0.0354+0.0097)-0.0773(0.0549+0.0225) sod-mse=0.0074(0.0112) gcn-mse=0.0140(0.0143) gcn-final-mse=0.0136(0.0306)
2020-08-13 12:59:54 4000-10553 loss=0.1360(0.0909+0.0451)-0.0767(0.0545+0.0222) sod-mse=0.0224(0.0111) gcn-mse=0.0255(0.0142) gcn-final-mse=0.0135(0.0304)
2020-08-13 13:03:59 5000-10553 loss=0.0684(0.0501+0.0183)-0.0770(0.0547+0.0223) sod-mse=0.0084(0.0112) gcn-mse=0.0078(0.0143) gcn-final-mse=0.0135(0.0305)
2020-08-13 13:08:04 6000-10553 loss=0.0500(0.0390+0.0110)-0.0774(0.0548+0.0226) sod-mse=0.0051(0.0113) gcn-mse=0.0105(0.0144) gcn-final-mse=0.0136(0.0305)
2020-08-13 13:12:09 7000-10553 loss=0.0715(0.0467+0.0248)-0.0774(0.0548+0.0226) sod-mse=0.0131(0.0113) gcn-mse=0.0175(0.0144) gcn-final-mse=0.0136(0.0306)
2020-08-13 13:16:13 8000-10553 loss=0.0437(0.0358+0.0079)-0.0778(0.0550+0.0228) sod-mse=0.0034(0.0114) gcn-mse=0.0062(0.0145) gcn-final-mse=0.0137(0.0307)
2020-08-13 13:17:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 13:18:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 13:18:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 13:20:18 9000-10553 loss=0.0726(0.0561+0.0165)-0.0778(0.0551+0.0228) sod-mse=0.0081(0.0114) gcn-mse=0.0147(0.0145) gcn-final-mse=0.0137(0.0307)
2020-08-13 13:24:23 10000-10553 loss=0.0779(0.0614+0.0165)-0.0777(0.0550+0.0227) sod-mse=0.0080(0.0114) gcn-mse=0.0151(0.0145) gcn-final-mse=0.0137(0.0307)

2020-08-13 13:26:38    0-5019 loss=1.1357(0.6573+0.4784)-1.1357(0.6573+0.4784) sod-mse=0.1064(0.1064) gcn-mse=0.1104(0.1104) gcn-final-mse=0.1027(0.1147)
2020-08-13 13:28:31 1000-5019 loss=0.0293(0.0244+0.0049)-0.3746(0.1854+0.1892) sod-mse=0.0039(0.0514) gcn-mse=0.0056(0.0541) gcn-final-mse=0.0539(0.0675)
2020-08-13 13:30:23 2000-5019 loss=1.0997(0.5182+0.5815)-0.3850(0.1884+0.1966) sod-mse=0.0970(0.0535) gcn-mse=0.0971(0.0560) gcn-final-mse=0.0558(0.0692)
2020-08-13 13:32:15 3000-5019 loss=0.0454(0.0339+0.0115)-0.3834(0.1872+0.1962) sod-mse=0.0052(0.0537) gcn-mse=0.0071(0.0562) gcn-final-mse=0.0560(0.0695)
2020-08-13 13:33:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 13:34:06 4000-5019 loss=0.1095(0.0768+0.0327)-0.3774(0.1849+0.1925) sod-mse=0.0149(0.0532) gcn-mse=0.0181(0.0558) gcn-final-mse=0.0555(0.0690)
2020-08-13 13:34:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 13:35:56 5000-5019 loss=1.5269(0.6339+0.8930)-0.3778(0.1853+0.1925) sod-mse=0.0872(0.0533) gcn-mse=0.0913(0.0558) gcn-final-mse=0.0555(0.0690)
2020-08-13 13:35:58 E:28, Train sod-mae-score=0.0114-0.9828 gcn-mae-score=0.0145-0.9537 gcn-final-mse-score=0.0137-0.9563(0.0307/0.9563) loss=0.0778(0.0550+0.0228)
2020-08-13 13:35:58 E:28, Test  sod-mae-score=0.0533-0.8388 gcn-mae-score=0.0559-0.7873 gcn-final-mse-score=0.0556-0.7930(0.0690/0.7930) loss=0.3778(0.1853+0.1925)

2020-08-13 13:35:58 Start Epoch 29
2020-08-13 13:35:58 Epoch:29,lr=0.0000
2020-08-13 13:35:59    0-10553 loss=0.0478(0.0369+0.0109)-0.0478(0.0369+0.0109) sod-mse=0.0044(0.0044) gcn-mse=0.0095(0.0095) gcn-final-mse=0.0099(0.0265)
2020-08-13 13:40:05 1000-10553 loss=0.0471(0.0362+0.0109)-0.0741(0.0527+0.0214) sod-mse=0.0076(0.0108) gcn-mse=0.0093(0.0136) gcn-final-mse=0.0129(0.0297)
2020-08-13 13:41:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 13:44:12 2000-10553 loss=0.0619(0.0476+0.0143)-0.0753(0.0536+0.0217) sod-mse=0.0074(0.0110) gcn-mse=0.0073(0.0138) gcn-final-mse=0.0131(0.0301)
2020-08-13 13:48:16 3000-10553 loss=0.0503(0.0408+0.0095)-0.0752(0.0536+0.0217) sod-mse=0.0045(0.0109) gcn-mse=0.0074(0.0138) gcn-final-mse=0.0131(0.0302)
2020-08-13 13:52:21 4000-10553 loss=0.0535(0.0424+0.0111)-0.0752(0.0535+0.0216) sod-mse=0.0060(0.0109) gcn-mse=0.0075(0.0139) gcn-final-mse=0.0131(0.0301)
2020-08-13 13:52:50 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 13:56:26 5000-10553 loss=0.0642(0.0499+0.0143)-0.0758(0.0539+0.0219) sod-mse=0.0068(0.0110) gcn-mse=0.0092(0.0140) gcn-final-mse=0.0132(0.0302)
2020-08-13 14:00:32 6000-10553 loss=0.0641(0.0477+0.0164)-0.0766(0.0544+0.0222) sod-mse=0.0086(0.0112) gcn-mse=0.0115(0.0141) gcn-final-mse=0.0134(0.0305)
2020-08-13 14:04:36 7000-10553 loss=0.0481(0.0313+0.0168)-0.0771(0.0546+0.0226) sod-mse=0.0111(0.0112) gcn-mse=0.0114(0.0142) gcn-final-mse=0.0134(0.0305)
2020-08-13 14:08:41 8000-10553 loss=0.0358(0.0302+0.0056)-0.0774(0.0547+0.0227) sod-mse=0.0026(0.0113) gcn-mse=0.0045(0.0142) gcn-final-mse=0.0135(0.0306)
2020-08-13 14:12:45 9000-10553 loss=0.0169(0.0128+0.0041)-0.0775(0.0548+0.0227) sod-mse=0.0028(0.0113) gcn-mse=0.0067(0.0143) gcn-final-mse=0.0135(0.0306)
2020-08-13 14:15:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 14:16:51 10000-10553 loss=0.0656(0.0474+0.0183)-0.0774(0.0547+0.0226) sod-mse=0.0106(0.0113) gcn-mse=0.0146(0.0142) gcn-final-mse=0.0135(0.0306)

2020-08-13 14:19:08    0-5019 loss=1.1392(0.6759+0.4633)-1.1392(0.6759+0.4633) sod-mse=0.1014(0.1014) gcn-mse=0.1062(0.1062) gcn-final-mse=0.0981(0.1098)
2020-08-13 14:21:00 1000-5019 loss=0.0296(0.0248+0.0048)-0.3795(0.1869+0.1926) sod-mse=0.0039(0.0513) gcn-mse=0.0056(0.0543) gcn-final-mse=0.0541(0.0678)
2020-08-13 14:22:50 2000-5019 loss=1.0994(0.5241+0.5753)-0.3887(0.1894+0.1993) sod-mse=0.0957(0.0534) gcn-mse=0.0978(0.0561) gcn-final-mse=0.0559(0.0695)
2020-08-13 14:24:41 3000-5019 loss=0.0449(0.0336+0.0114)-0.3866(0.1882+0.1984) sod-mse=0.0051(0.0535) gcn-mse=0.0067(0.0562) gcn-final-mse=0.0560(0.0696)
2020-08-13 14:25:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 14:26:31 4000-5019 loss=0.1083(0.0755+0.0328)-0.3806(0.1860+0.1947) sod-mse=0.0145(0.0530) gcn-mse=0.0174(0.0559) gcn-final-mse=0.0556(0.0692)
2020-08-13 14:27:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 14:28:21 5000-5019 loss=1.4482(0.5877+0.8605)-0.3807(0.1862+0.1945) sod-mse=0.0946(0.0530) gcn-mse=0.0978(0.0559) gcn-final-mse=0.0556(0.0691)
2020-08-13 14:28:23 E:29, Train sod-mae-score=0.0112-0.9832 gcn-mae-score=0.0142-0.9543 gcn-final-mse-score=0.0134-0.9568(0.0305/0.9568) loss=0.0772(0.0547+0.0226)
2020-08-13 14:28:23 E:29, Test  sod-mae-score=0.0530-0.8394 gcn-mae-score=0.0559-0.7853 gcn-final-mse-score=0.0556-0.7908(0.0692/0.7908) loss=0.3806(0.1862+0.1944)

2020-08-13 14:28:23 Start Epoch 30
2020-08-13 14:28:23 Epoch:30,lr=0.0000
2020-08-13 14:28:24    0-10553 loss=0.0622(0.0494+0.0128)-0.0622(0.0494+0.0128) sod-mse=0.0086(0.0086) gcn-mse=0.0117(0.0117) gcn-final-mse=0.0112(0.0305)
2020-08-13 14:32:29 1000-10553 loss=0.0654(0.0530+0.0124)-0.0753(0.0536+0.0217) sod-mse=0.0061(0.0110) gcn-mse=0.0079(0.0140) gcn-final-mse=0.0132(0.0304)
2020-08-13 14:36:34 2000-10553 loss=0.0719(0.0549+0.0170)-0.0758(0.0540+0.0219) sod-mse=0.0078(0.0109) gcn-mse=0.0081(0.0138) gcn-final-mse=0.0131(0.0304)
2020-08-13 14:40:39 3000-10553 loss=0.0735(0.0491+0.0244)-0.0755(0.0538+0.0217) sod-mse=0.0122(0.0109) gcn-mse=0.0131(0.0137) gcn-final-mse=0.0130(0.0302)
2020-08-13 14:41:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 14:44:43 4000-10553 loss=0.0333(0.0244+0.0090)-0.0752(0.0537+0.0215) sod-mse=0.0055(0.0108) gcn-mse=0.0071(0.0137) gcn-final-mse=0.0129(0.0302)
2020-08-13 14:48:47 5000-10553 loss=0.0987(0.0643+0.0344)-0.0755(0.0538+0.0217) sod-mse=0.0195(0.0108) gcn-mse=0.0214(0.0137) gcn-final-mse=0.0130(0.0302)
2020-08-13 14:52:52 6000-10553 loss=0.0825(0.0619+0.0205)-0.0755(0.0539+0.0216) sod-mse=0.0122(0.0108) gcn-mse=0.0173(0.0138) gcn-final-mse=0.0130(0.0303)
2020-08-13 14:56:57 7000-10553 loss=0.0433(0.0354+0.0079)-0.0756(0.0539+0.0217) sod-mse=0.0043(0.0108) gcn-mse=0.0074(0.0137) gcn-final-mse=0.0130(0.0302)
2020-08-13 15:01:02 8000-10553 loss=0.0440(0.0331+0.0110)-0.0758(0.0540+0.0218) sod-mse=0.0055(0.0108) gcn-mse=0.0115(0.0138) gcn-final-mse=0.0130(0.0302)
2020-08-13 15:03:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 15:05:08 9000-10553 loss=0.0964(0.0652+0.0313)-0.0757(0.0540+0.0218) sod-mse=0.0165(0.0108) gcn-mse=0.0185(0.0138) gcn-final-mse=0.0130(0.0302)
2020-08-13 15:06:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 15:09:13 10000-10553 loss=0.0347(0.0257+0.0090)-0.0756(0.0539+0.0217) sod-mse=0.0053(0.0108) gcn-mse=0.0054(0.0138) gcn-final-mse=0.0130(0.0302)

2020-08-13 15:11:30    0-5019 loss=1.1362(0.6691+0.4672)-1.1362(0.6691+0.4672) sod-mse=0.1015(0.1015) gcn-mse=0.1058(0.1058) gcn-final-mse=0.0978(0.1096)
2020-08-13 15:13:22 1000-5019 loss=0.0290(0.0243+0.0047)-0.3818(0.1868+0.1949) sod-mse=0.0038(0.0509) gcn-mse=0.0052(0.0537) gcn-final-mse=0.0535(0.0671)
2020-08-13 15:15:12 2000-5019 loss=1.1103(0.5222+0.5881)-0.3913(0.1896+0.2017) sod-mse=0.0958(0.0530) gcn-mse=0.0966(0.0556) gcn-final-mse=0.0553(0.0688)
2020-08-13 15:17:02 3000-5019 loss=0.0450(0.0337+0.0114)-0.3894(0.1885+0.2009) sod-mse=0.0052(0.0531) gcn-mse=0.0069(0.0557) gcn-final-mse=0.0555(0.0689)
2020-08-13 15:18:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 15:18:53 4000-5019 loss=0.1075(0.0750+0.0325)-0.3833(0.1862+0.1971) sod-mse=0.0145(0.0526) gcn-mse=0.0171(0.0553) gcn-final-mse=0.0551(0.0685)
2020-08-13 15:19:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 15:20:43 5000-5019 loss=1.6217(0.6265+0.9952)-0.3832(0.1864+0.1968) sod-mse=0.0893(0.0526) gcn-mse=0.0945(0.0553) gcn-final-mse=0.0550(0.0684)
2020-08-13 15:20:45 E:30, Train sod-mae-score=0.0108-0.9835 gcn-mae-score=0.0137-0.9547 gcn-final-mse-score=0.0130-0.9571(0.0301/0.9571) loss=0.0755(0.0538+0.0217)
2020-08-13 15:20:45 E:30, Test  sod-mae-score=0.0526-0.8403 gcn-mae-score=0.0553-0.7880 gcn-final-mse-score=0.0550-0.7937(0.0685/0.7937) loss=0.3832(0.1864+0.1968)

2020-08-13 15:20:45 Start Epoch 31
2020-08-13 15:20:45 Epoch:31,lr=0.0000
2020-08-13 15:20:46    0-10553 loss=0.2556(0.1641+0.0915)-0.2556(0.1641+0.0915) sod-mse=0.0474(0.0474) gcn-mse=0.0521(0.0521) gcn-final-mse=0.0544(0.0898)
2020-08-13 15:21:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 15:22:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 15:24:50 1000-10553 loss=0.0634(0.0504+0.0130)-0.0740(0.0527+0.0213) sod-mse=0.0065(0.0108) gcn-mse=0.0087(0.0137) gcn-final-mse=0.0129(0.0297)
2020-08-13 15:28:53 2000-10553 loss=0.0635(0.0477+0.0158)-0.0745(0.0532+0.0213) sod-mse=0.0105(0.0108) gcn-mse=0.0098(0.0137) gcn-final-mse=0.0129(0.0299)
2020-08-13 15:32:57 3000-10553 loss=0.1120(0.0710+0.0410)-0.0743(0.0531+0.0212) sod-mse=0.0190(0.0107) gcn-mse=0.0172(0.0135) gcn-final-mse=0.0127(0.0298)
2020-08-13 15:37:02 4000-10553 loss=0.0628(0.0412+0.0217)-0.0748(0.0534+0.0214) sod-mse=0.0111(0.0107) gcn-mse=0.0114(0.0136) gcn-final-mse=0.0128(0.0300)
2020-08-13 15:37:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 15:41:06 5000-10553 loss=0.0165(0.0125+0.0040)-0.0747(0.0534+0.0213) sod-mse=0.0017(0.0107) gcn-mse=0.0031(0.0136) gcn-final-mse=0.0128(0.0300)
2020-08-13 15:45:12 6000-10553 loss=0.1310(0.0744+0.0566)-0.0749(0.0535+0.0214) sod-mse=0.0150(0.0107) gcn-mse=0.0201(0.0136) gcn-final-mse=0.0128(0.0299)
2020-08-13 15:49:19 7000-10553 loss=0.0521(0.0348+0.0172)-0.0752(0.0537+0.0215) sod-mse=0.0083(0.0107) gcn-mse=0.0087(0.0137) gcn-final-mse=0.0129(0.0300)
2020-08-13 15:53:25 8000-10553 loss=0.0591(0.0430+0.0161)-0.0750(0.0536+0.0214) sod-mse=0.0075(0.0107) gcn-mse=0.0106(0.0136) gcn-final-mse=0.0129(0.0300)
2020-08-13 15:57:31 9000-10553 loss=0.0822(0.0623+0.0199)-0.0753(0.0537+0.0216) sod-mse=0.0101(0.0107) gcn-mse=0.0136(0.0137) gcn-final-mse=0.0129(0.0300)
2020-08-13 16:01:35 10000-10553 loss=0.0299(0.0244+0.0055)-0.0753(0.0537+0.0216) sod-mse=0.0025(0.0107) gcn-mse=0.0051(0.0136) gcn-final-mse=0.0129(0.0300)

2020-08-13 16:03:52    0-5019 loss=1.1541(0.6763+0.4777)-1.1541(0.6763+0.4777) sod-mse=0.1033(0.1033) gcn-mse=0.1073(0.1073) gcn-final-mse=0.0994(0.1112)
2020-08-13 16:05:45 1000-5019 loss=0.0288(0.0241+0.0047)-0.3862(0.1877+0.1985) sod-mse=0.0037(0.0506) gcn-mse=0.0051(0.0535) gcn-final-mse=0.0532(0.0668)
2020-08-13 16:07:37 2000-5019 loss=1.1416(0.5315+0.6102)-0.3956(0.1904+0.2052) sod-mse=0.0966(0.0527) gcn-mse=0.0972(0.0553) gcn-final-mse=0.0550(0.0684)
2020-08-13 16:09:29 3000-5019 loss=0.0451(0.0337+0.0114)-0.3941(0.1894+0.2047) sod-mse=0.0052(0.0528) gcn-mse=0.0069(0.0554) gcn-final-mse=0.0552(0.0685)
2020-08-13 16:10:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 16:11:21 4000-5019 loss=0.1073(0.0751+0.0322)-0.3882(0.1872+0.2009) sod-mse=0.0144(0.0523) gcn-mse=0.0169(0.0550) gcn-final-mse=0.0548(0.0682)
2020-08-13 16:11:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 16:13:12 5000-5019 loss=1.7575(0.6425+1.1150)-0.3880(0.1874+0.2006) sod-mse=0.0876(0.0523) gcn-mse=0.0924(0.0551) gcn-final-mse=0.0547(0.0681)
2020-08-13 16:13:14 E:31, Train sod-mae-score=0.0107-0.9837 gcn-mae-score=0.0136-0.9549 gcn-final-mse-score=0.0129-0.9574(0.0300/0.9574) loss=0.0752(0.0537+0.0215)
2020-08-13 16:13:14 E:31, Test  sod-mae-score=0.0523-0.8412 gcn-mae-score=0.0551-0.7888 gcn-final-mse-score=0.0548-0.7945(0.0681/0.7945) loss=0.3880(0.1874+0.2005)

2020-08-13 16:13:14 Start Epoch 32
2020-08-13 16:13:14 Epoch:32,lr=0.0000
2020-08-13 16:13:15    0-10553 loss=0.0432(0.0336+0.0097)-0.0432(0.0336+0.0097) sod-mse=0.0047(0.0047) gcn-mse=0.0053(0.0053) gcn-final-mse=0.0043(0.0183)
2020-08-13 16:17:22 1000-10553 loss=0.0969(0.0744+0.0225)-0.0760(0.0544+0.0217) sod-mse=0.0104(0.0109) gcn-mse=0.0180(0.0139) gcn-final-mse=0.0131(0.0306)
2020-08-13 16:21:26 2000-10553 loss=0.0631(0.0459+0.0172)-0.0748(0.0535+0.0213) sod-mse=0.0094(0.0107) gcn-mse=0.0111(0.0135) gcn-final-mse=0.0127(0.0300)
2020-08-13 16:25:30 3000-10553 loss=0.0621(0.0455+0.0166)-0.0748(0.0536+0.0213) sod-mse=0.0115(0.0107) gcn-mse=0.0126(0.0136) gcn-final-mse=0.0128(0.0300)
2020-08-13 16:29:37 4000-10553 loss=0.0599(0.0436+0.0163)-0.0746(0.0534+0.0212) sod-mse=0.0100(0.0106) gcn-mse=0.0113(0.0135) gcn-final-mse=0.0127(0.0299)
2020-08-13 16:33:43 5000-10553 loss=0.0973(0.0660+0.0313)-0.0748(0.0535+0.0213) sod-mse=0.0149(0.0107) gcn-mse=0.0144(0.0136) gcn-final-mse=0.0128(0.0300)
2020-08-13 16:36:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 16:37:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 16:37:49 6000-10553 loss=0.1048(0.0816+0.0231)-0.0748(0.0535+0.0213) sod-mse=0.0128(0.0107) gcn-mse=0.0176(0.0136) gcn-final-mse=0.0128(0.0300)
2020-08-13 16:41:54 7000-10553 loss=0.0845(0.0588+0.0257)-0.0744(0.0532+0.0212) sod-mse=0.0142(0.0106) gcn-mse=0.0152(0.0135) gcn-final-mse=0.0127(0.0299)
2020-08-13 16:46:00 8000-10553 loss=0.0947(0.0624+0.0323)-0.0745(0.0533+0.0212) sod-mse=0.0195(0.0106) gcn-mse=0.0198(0.0135) gcn-final-mse=0.0127(0.0299)
2020-08-13 16:50:06 9000-10553 loss=0.0619(0.0512+0.0108)-0.0748(0.0535+0.0212) sod-mse=0.0046(0.0106) gcn-mse=0.0095(0.0135) gcn-final-mse=0.0127(0.0300)
2020-08-13 16:54:11 10000-10553 loss=0.0864(0.0468+0.0396)-0.0746(0.0534+0.0212) sod-mse=0.0151(0.0106) gcn-mse=0.0173(0.0135) gcn-final-mse=0.0127(0.0299)
2020-08-13 16:54:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg

2020-08-13 16:56:29    0-5019 loss=1.1504(0.6732+0.4771)-1.1504(0.6732+0.4771) sod-mse=0.1029(0.1029) gcn-mse=0.1065(0.1065) gcn-final-mse=0.0986(0.1105)
2020-08-13 16:58:22 1000-5019 loss=0.0288(0.0242+0.0046)-0.3866(0.1874+0.1992) sod-mse=0.0037(0.0506) gcn-mse=0.0053(0.0536) gcn-final-mse=0.0533(0.0669)
2020-08-13 17:00:16 2000-5019 loss=1.1370(0.5302+0.6069)-0.3964(0.1903+0.2062) sod-mse=0.0958(0.0527) gcn-mse=0.0968(0.0554) gcn-final-mse=0.0551(0.0686)
2020-08-13 17:02:11 3000-5019 loss=0.0453(0.0338+0.0114)-0.3949(0.1892+0.2057) sod-mse=0.0052(0.0528) gcn-mse=0.0070(0.0555) gcn-final-mse=0.0553(0.0687)
2020-08-13 17:03:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 17:04:06 4000-5019 loss=0.1081(0.0756+0.0325)-0.3891(0.1871+0.2020) sod-mse=0.0145(0.0524) gcn-mse=0.0175(0.0552) gcn-final-mse=0.0549(0.0684)
2020-08-13 17:04:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 17:06:00 5000-5019 loss=1.8020(0.6473+1.1547)-0.3892(0.1873+0.2018) sod-mse=0.0877(0.0524) gcn-mse=0.0927(0.0552) gcn-final-mse=0.0549(0.0683)
2020-08-13 17:06:02 E:32, Train sod-mae-score=0.0106-0.9840 gcn-mae-score=0.0135-0.9549 gcn-final-mse-score=0.0127-0.9574(0.0299/0.9574) loss=0.0746(0.0534+0.0212)
2020-08-13 17:06:02 E:32, Test  sod-mae-score=0.0524-0.8408 gcn-mae-score=0.0552-0.7887 gcn-final-mse-score=0.0549-0.7944(0.0683/0.7944) loss=0.3891(0.1873+0.2018)

2020-08-13 17:06:02 Start Epoch 33
2020-08-13 17:06:02 Epoch:33,lr=0.0000
2020-08-13 17:06:04    0-10553 loss=0.0433(0.0310+0.0123)-0.0433(0.0310+0.0123) sod-mse=0.0058(0.0058) gcn-mse=0.0060(0.0060) gcn-final-mse=0.0061(0.0191)
2020-08-13 17:10:09 1000-10553 loss=0.0811(0.0626+0.0185)-0.0770(0.0542+0.0228) sod-mse=0.0092(0.0109) gcn-mse=0.0160(0.0137) gcn-final-mse=0.0129(0.0301)
2020-08-13 17:14:14 2000-10553 loss=0.0952(0.0733+0.0218)-0.0763(0.0543+0.0221) sod-mse=0.0093(0.0108) gcn-mse=0.0123(0.0136) gcn-final-mse=0.0128(0.0301)
2020-08-13 17:18:21 3000-10553 loss=0.0856(0.0655+0.0201)-0.0749(0.0534+0.0215) sod-mse=0.0099(0.0106) gcn-mse=0.0164(0.0134) gcn-final-mse=0.0126(0.0298)
2020-08-13 17:22:26 4000-10553 loss=0.0768(0.0517+0.0251)-0.0750(0.0535+0.0215) sod-mse=0.0115(0.0106) gcn-mse=0.0143(0.0135) gcn-final-mse=0.0127(0.0299)
2020-08-13 17:22:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 17:26:32 5000-10553 loss=0.0514(0.0416+0.0098)-0.0746(0.0533+0.0213) sod-mse=0.0038(0.0106) gcn-mse=0.0077(0.0134) gcn-final-mse=0.0127(0.0298)
2020-08-13 17:30:38 6000-10553 loss=0.0708(0.0471+0.0237)-0.0749(0.0535+0.0214) sod-mse=0.0111(0.0106) gcn-mse=0.0132(0.0135) gcn-final-mse=0.0127(0.0299)
2020-08-13 17:34:44 7000-10553 loss=0.0595(0.0374+0.0221)-0.0749(0.0535+0.0213) sod-mse=0.0131(0.0106) gcn-mse=0.0138(0.0135) gcn-final-mse=0.0127(0.0299)
2020-08-13 17:38:50 8000-10553 loss=0.0697(0.0473+0.0224)-0.0747(0.0534+0.0212) sod-mse=0.0101(0.0106) gcn-mse=0.0128(0.0135) gcn-final-mse=0.0127(0.0299)
2020-08-13 17:42:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 17:42:55 9000-10553 loss=0.0378(0.0277+0.0102)-0.0745(0.0534+0.0212) sod-mse=0.0041(0.0106) gcn-mse=0.0045(0.0135) gcn-final-mse=0.0127(0.0299)
2020-08-13 17:47:02 10000-10553 loss=0.0485(0.0401+0.0084)-0.0744(0.0533+0.0211) sod-mse=0.0038(0.0105) gcn-mse=0.0082(0.0134) gcn-final-mse=0.0127(0.0299)
2020-08-13 17:48:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg

2020-08-13 17:49:19    0-5019 loss=1.1806(0.6879+0.4927)-1.1806(0.6879+0.4927) sod-mse=0.1047(0.1047) gcn-mse=0.1084(0.1084) gcn-final-mse=0.1005(0.1124)
2020-08-13 17:51:12 1000-5019 loss=0.0289(0.0243+0.0046)-0.3902(0.1880+0.2022) sod-mse=0.0037(0.0505) gcn-mse=0.0052(0.0534) gcn-final-mse=0.0531(0.0667)
2020-08-13 17:53:03 2000-5019 loss=1.1631(0.5382+0.6250)-0.3993(0.1906+0.2086) sod-mse=0.0962(0.0526) gcn-mse=0.0977(0.0552) gcn-final-mse=0.0549(0.0683)
2020-08-13 17:54:53 3000-5019 loss=0.0451(0.0338+0.0113)-0.3979(0.1896+0.2083) sod-mse=0.0052(0.0527) gcn-mse=0.0069(0.0553) gcn-final-mse=0.0551(0.0685)
2020-08-13 17:55:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 17:56:44 4000-5019 loss=0.1081(0.0756+0.0325)-0.3920(0.1875+0.2045) sod-mse=0.0144(0.0522) gcn-mse=0.0172(0.0550) gcn-final-mse=0.0547(0.0681)
2020-08-13 17:57:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 17:58:35 5000-5019 loss=1.9521(0.6484+1.3037)-0.3922(0.1878+0.2044) sod-mse=0.0868(0.0522) gcn-mse=0.0916(0.0550) gcn-final-mse=0.0547(0.0681)
2020-08-13 17:58:37 E:33, Train sod-mae-score=0.0106-0.9839 gcn-mae-score=0.0135-0.9551 gcn-final-mse-score=0.0127-0.9577(0.0299/0.9577) loss=0.0745(0.0533+0.0212)
2020-08-13 17:58:37 E:33, Test  sod-mae-score=0.0522-0.8406 gcn-mae-score=0.0550-0.7884 gcn-final-mse-score=0.0547-0.7941(0.0681/0.7941) loss=0.3922(0.1878+0.2043)

2020-08-13 17:58:37 Start Epoch 34
2020-08-13 17:58:37 Epoch:34,lr=0.0000
2020-08-13 17:58:38    0-10553 loss=0.0966(0.0748+0.0218)-0.0966(0.0748+0.0218) sod-mse=0.0133(0.0133) gcn-mse=0.0261(0.0261) gcn-final-mse=0.0247(0.0446)
2020-08-13 18:02:44 1000-10553 loss=0.1060(0.0786+0.0274)-0.0726(0.0524+0.0203) sod-mse=0.0179(0.0101) gcn-mse=0.0129(0.0130) gcn-final-mse=0.0123(0.0296)
2020-08-13 18:04:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 18:06:49 2000-10553 loss=0.0462(0.0344+0.0117)-0.0731(0.0526+0.0205) sod-mse=0.0062(0.0102) gcn-mse=0.0095(0.0131) gcn-final-mse=0.0123(0.0295)
2020-08-13 18:08:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 18:10:55 3000-10553 loss=0.1063(0.0827+0.0235)-0.0727(0.0523+0.0204) sod-mse=0.0149(0.0102) gcn-mse=0.0221(0.0130) gcn-final-mse=0.0123(0.0294)
2020-08-13 18:12:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 18:15:00 4000-10553 loss=0.0627(0.0488+0.0139)-0.0737(0.0529+0.0208) sod-mse=0.0060(0.0104) gcn-mse=0.0092(0.0132) gcn-final-mse=0.0125(0.0298)
2020-08-13 18:19:05 5000-10553 loss=0.1084(0.0775+0.0308)-0.0740(0.0530+0.0210) sod-mse=0.0082(0.0104) gcn-mse=0.0118(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 18:23:10 6000-10553 loss=0.0089(0.0067+0.0022)-0.0740(0.0531+0.0209) sod-mse=0.0010(0.0104) gcn-mse=0.0012(0.0133) gcn-final-mse=0.0125(0.0298)
2020-08-13 18:27:15 7000-10553 loss=0.0505(0.0372+0.0133)-0.0740(0.0531+0.0209) sod-mse=0.0068(0.0104) gcn-mse=0.0074(0.0133) gcn-final-mse=0.0125(0.0298)
2020-08-13 18:31:21 8000-10553 loss=0.1014(0.0669+0.0346)-0.0746(0.0534+0.0212) sod-mse=0.0152(0.0105) gcn-mse=0.0164(0.0134) gcn-final-mse=0.0126(0.0299)
2020-08-13 18:35:26 9000-10553 loss=0.0447(0.0360+0.0087)-0.0743(0.0532+0.0211) sod-mse=0.0037(0.0105) gcn-mse=0.0062(0.0134) gcn-final-mse=0.0126(0.0298)
2020-08-13 18:39:32 10000-10553 loss=0.0935(0.0724+0.0211)-0.0743(0.0532+0.0211) sod-mse=0.0099(0.0105) gcn-mse=0.0111(0.0133) gcn-final-mse=0.0126(0.0298)

2020-08-13 18:41:49    0-5019 loss=1.1612(0.6814+0.4798)-1.1612(0.6814+0.4798) sod-mse=0.1027(0.1027) gcn-mse=0.1069(0.1069) gcn-final-mse=0.0989(0.1110)
2020-08-13 18:43:42 1000-5019 loss=0.0288(0.0242+0.0046)-0.3900(0.1880+0.2020) sod-mse=0.0037(0.0504) gcn-mse=0.0052(0.0533) gcn-final-mse=0.0530(0.0667)
2020-08-13 18:45:34 2000-5019 loss=1.1420(0.5313+0.6107)-0.3997(0.1908+0.2089) sod-mse=0.0955(0.0526) gcn-mse=0.0967(0.0551) gcn-final-mse=0.0549(0.0683)
2020-08-13 18:47:26 3000-5019 loss=0.0451(0.0337+0.0113)-0.3983(0.1898+0.2085) sod-mse=0.0052(0.0526) gcn-mse=0.0069(0.0552) gcn-final-mse=0.0550(0.0685)
2020-08-13 18:48:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 18:49:18 4000-5019 loss=0.1076(0.0755+0.0321)-0.3924(0.1876+0.2048) sod-mse=0.0144(0.0522) gcn-mse=0.0171(0.0549) gcn-final-mse=0.0546(0.0681)
2020-08-13 18:49:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 18:51:10 5000-5019 loss=1.9831(0.6546+1.3285)-0.3926(0.1879+0.2047) sod-mse=0.0871(0.0522) gcn-mse=0.0921(0.0549) gcn-final-mse=0.0546(0.0681)
2020-08-13 18:51:12 E:34, Train sod-mae-score=0.0105-0.9841 gcn-mae-score=0.0134-0.9554 gcn-final-mse-score=0.0126-0.9578(0.0298/0.9578) loss=0.0743(0.0532+0.0211)
2020-08-13 18:51:12 E:34, Test  sod-mae-score=0.0522-0.8410 gcn-mae-score=0.0550-0.7889 gcn-final-mse-score=0.0546-0.7946(0.0681/0.7946) loss=0.3925(0.1879+0.2046)

2020-08-13 18:51:12 Start Epoch 35
2020-08-13 18:51:12 Epoch:35,lr=0.0000
2020-08-13 18:51:13    0-10553 loss=0.1802(0.1115+0.0687)-0.1802(0.1115+0.0687) sod-mse=0.0240(0.0240) gcn-mse=0.0288(0.0288) gcn-final-mse=0.0261(0.0516)
2020-08-13 18:55:21 1000-10553 loss=0.0427(0.0298+0.0130)-0.0765(0.0540+0.0225) sod-mse=0.0064(0.0109) gcn-mse=0.0087(0.0136) gcn-final-mse=0.0129(0.0298)
2020-08-13 18:59:29 2000-10553 loss=0.0664(0.0502+0.0162)-0.0744(0.0531+0.0213) sod-mse=0.0073(0.0106) gcn-mse=0.0118(0.0134) gcn-final-mse=0.0125(0.0294)
2020-08-13 19:01:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 19:03:37 3000-10553 loss=0.0359(0.0274+0.0085)-0.0738(0.0529+0.0210) sod-mse=0.0043(0.0104) gcn-mse=0.0061(0.0133) gcn-final-mse=0.0124(0.0294)
2020-08-13 19:07:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 19:07:45 4000-10553 loss=0.0960(0.0593+0.0366)-0.0740(0.0530+0.0210) sod-mse=0.0171(0.0105) gcn-mse=0.0159(0.0133) gcn-final-mse=0.0125(0.0295)
2020-08-13 19:11:54 5000-10553 loss=0.0481(0.0388+0.0093)-0.0742(0.0532+0.0210) sod-mse=0.0046(0.0105) gcn-mse=0.0055(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 19:16:02 6000-10553 loss=0.0344(0.0260+0.0084)-0.0739(0.0530+0.0209) sod-mse=0.0044(0.0104) gcn-mse=0.0032(0.0133) gcn-final-mse=0.0125(0.0296)
2020-08-13 19:19:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 19:20:11 7000-10553 loss=0.0385(0.0307+0.0077)-0.0740(0.0531+0.0210) sod-mse=0.0059(0.0104) gcn-mse=0.0072(0.0133) gcn-final-mse=0.0125(0.0296)
2020-08-13 19:24:19 8000-10553 loss=0.0868(0.0567+0.0301)-0.0740(0.0531+0.0210) sod-mse=0.0147(0.0105) gcn-mse=0.0252(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 19:28:29 9000-10553 loss=0.0532(0.0391+0.0141)-0.0738(0.0530+0.0208) sod-mse=0.0064(0.0104) gcn-mse=0.0089(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 19:32:38 10000-10553 loss=0.0706(0.0449+0.0258)-0.0739(0.0531+0.0209) sod-mse=0.0105(0.0104) gcn-mse=0.0152(0.0133) gcn-final-mse=0.0125(0.0297)

2020-08-13 19:34:57    0-5019 loss=1.1757(0.6864+0.4893)-1.1757(0.6864+0.4893) sod-mse=0.1036(0.1036) gcn-mse=0.1074(0.1074) gcn-final-mse=0.0993(0.1113)
2020-08-13 19:36:50 1000-5019 loss=0.0287(0.0241+0.0046)-0.3932(0.1884+0.2048) sod-mse=0.0037(0.0505) gcn-mse=0.0051(0.0534) gcn-final-mse=0.0531(0.0668)
2020-08-13 19:38:43 2000-5019 loss=1.1665(0.5377+0.6288)-0.4020(0.1909+0.2111) sod-mse=0.0959(0.0526) gcn-mse=0.0972(0.0552) gcn-final-mse=0.0549(0.0684)
2020-08-13 19:40:35 3000-5019 loss=0.0451(0.0338+0.0114)-0.4006(0.1899+0.2107) sod-mse=0.0052(0.0527) gcn-mse=0.0069(0.0553) gcn-final-mse=0.0551(0.0685)
2020-08-13 19:41:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 19:42:26 4000-5019 loss=0.1081(0.0758+0.0323)-0.3947(0.1878+0.2069) sod-mse=0.0144(0.0523) gcn-mse=0.0173(0.0550) gcn-final-mse=0.0547(0.0682)
2020-08-13 19:42:59 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 19:44:17 5000-5019 loss=1.9879(0.6475+1.3404)-0.3950(0.1882+0.2068) sod-mse=0.0872(0.0523) gcn-mse=0.0921(0.0550) gcn-final-mse=0.0547(0.0682)
2020-08-13 19:44:19 E:35, Train sod-mae-score=0.0104-0.9841 gcn-mae-score=0.0133-0.9552 gcn-final-mse-score=0.0126-0.9577(0.0298/0.9577) loss=0.0741(0.0532+0.0209)
2020-08-13 19:44:19 E:35, Test  sod-mae-score=0.0523-0.8399 gcn-mae-score=0.0550-0.7879 gcn-final-mse-score=0.0547-0.7936(0.0682/0.7936) loss=0.3949(0.1882+0.2068)

2020-08-13 19:44:19 Start Epoch 36
2020-08-13 19:44:19 Epoch:36,lr=0.0000
2020-08-13 19:44:21    0-10553 loss=0.0890(0.0636+0.0254)-0.0890(0.0636+0.0254) sod-mse=0.0125(0.0125) gcn-mse=0.0168(0.0168) gcn-final-mse=0.0161(0.0378)
2020-08-13 19:48:25 1000-10553 loss=0.0372(0.0297+0.0075)-0.0739(0.0531+0.0208) sod-mse=0.0055(0.0102) gcn-mse=0.0067(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 19:52:31 2000-10553 loss=0.1062(0.0716+0.0346)-0.0735(0.0528+0.0207) sod-mse=0.0186(0.0102) gcn-mse=0.0219(0.0132) gcn-final-mse=0.0124(0.0296)
2020-08-13 19:56:35 3000-10553 loss=0.0643(0.0464+0.0179)-0.0743(0.0533+0.0210) sod-mse=0.0086(0.0104) gcn-mse=0.0101(0.0133) gcn-final-mse=0.0125(0.0298)
2020-08-13 20:00:39 4000-10553 loss=0.1227(0.0876+0.0351)-0.0747(0.0535+0.0212) sod-mse=0.0224(0.0105) gcn-mse=0.0274(0.0134) gcn-final-mse=0.0126(0.0299)
2020-08-13 20:04:43 5000-10553 loss=0.3050(0.2058+0.0992)-0.0745(0.0534+0.0211) sod-mse=0.0536(0.0105) gcn-mse=0.0618(0.0134) gcn-final-mse=0.0126(0.0299)
2020-08-13 20:07:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 20:08:49 6000-10553 loss=0.1045(0.0750+0.0295)-0.0741(0.0532+0.0209) sod-mse=0.0170(0.0104) gcn-mse=0.0163(0.0133) gcn-final-mse=0.0125(0.0298)
2020-08-13 20:12:54 7000-10553 loss=0.0250(0.0199+0.0051)-0.0740(0.0531+0.0209) sod-mse=0.0020(0.0104) gcn-mse=0.0024(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 20:15:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 20:16:59 8000-10553 loss=0.1101(0.0733+0.0369)-0.0741(0.0532+0.0209) sod-mse=0.0130(0.0104) gcn-mse=0.0173(0.0133) gcn-final-mse=0.0125(0.0298)
2020-08-13 20:21:03 9000-10553 loss=0.1000(0.0628+0.0372)-0.0740(0.0531+0.0209) sod-mse=0.0193(0.0104) gcn-mse=0.0197(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 20:24:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 20:25:08 10000-10553 loss=0.0425(0.0289+0.0136)-0.0741(0.0532+0.0209) sod-mse=0.0084(0.0104) gcn-mse=0.0131(0.0133) gcn-final-mse=0.0125(0.0298)

2020-08-13 20:27:24    0-5019 loss=1.1685(0.6790+0.4895)-1.1685(0.6790+0.4895) sod-mse=0.1027(0.1027) gcn-mse=0.1069(0.1069) gcn-final-mse=0.0989(0.1108)
2020-08-13 20:29:17 1000-5019 loss=0.0285(0.0240+0.0046)-0.3990(0.1897+0.2093) sod-mse=0.0036(0.0503) gcn-mse=0.0050(0.0532) gcn-final-mse=0.0530(0.0665)
2020-08-13 20:31:09 2000-5019 loss=1.1718(0.5380+0.6338)-0.4081(0.1923+0.2158) sod-mse=0.0956(0.0524) gcn-mse=0.0966(0.0550) gcn-final-mse=0.0548(0.0681)
2020-08-13 20:33:01 3000-5019 loss=0.0452(0.0338+0.0114)-0.4067(0.1912+0.2155) sod-mse=0.0052(0.0525) gcn-mse=0.0070(0.0552) gcn-final-mse=0.0549(0.0683)
2020-08-13 20:34:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 20:34:53 4000-5019 loss=0.1078(0.0755+0.0323)-0.4007(0.1891+0.2116) sod-mse=0.0143(0.0520) gcn-mse=0.0170(0.0548) gcn-final-mse=0.0546(0.0679)
2020-08-13 20:35:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 20:36:44 5000-5019 loss=2.1257(0.6598+1.4658)-0.4008(0.1894+0.2114) sod-mse=0.0871(0.0520) gcn-mse=0.0921(0.0549) gcn-final-mse=0.0545(0.0679)
2020-08-13 20:36:46 E:36, Train sod-mae-score=0.0104-0.9841 gcn-mae-score=0.0133-0.9552 gcn-final-mse-score=0.0125-0.9577(0.0298/0.9577) loss=0.0740(0.0531+0.0209)
2020-08-13 20:36:46 E:36, Test  sod-mae-score=0.0520-0.8402 gcn-mae-score=0.0549-0.7884 gcn-final-mse-score=0.0545-0.7941(0.0679/0.7941) loss=0.4008(0.1894+0.2114)

2020-08-13 20:36:46 Start Epoch 37
2020-08-13 20:36:46 Epoch:37,lr=0.0000
2020-08-13 20:36:47    0-10553 loss=0.0775(0.0513+0.0262)-0.0775(0.0513+0.0262) sod-mse=0.0118(0.0118) gcn-mse=0.0135(0.0135) gcn-final-mse=0.0128(0.0259)
2020-08-13 20:40:54 1000-10553 loss=0.0572(0.0397+0.0174)-0.0736(0.0530+0.0206) sod-mse=0.0094(0.0103) gcn-mse=0.0116(0.0134) gcn-final-mse=0.0126(0.0299)
2020-08-13 20:44:59 2000-10553 loss=0.0322(0.0235+0.0086)-0.0734(0.0527+0.0207) sod-mse=0.0045(0.0103) gcn-mse=0.0055(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 20:49:04 3000-10553 loss=0.0377(0.0274+0.0103)-0.0742(0.0531+0.0210) sod-mse=0.0045(0.0104) gcn-mse=0.0071(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 20:53:09 4000-10553 loss=0.0278(0.0202+0.0075)-0.0735(0.0528+0.0207) sod-mse=0.0035(0.0103) gcn-mse=0.0045(0.0132) gcn-final-mse=0.0124(0.0296)
2020-08-13 20:55:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 20:57:14 5000-10553 loss=0.0435(0.0357+0.0078)-0.0736(0.0529+0.0207) sod-mse=0.0032(0.0103) gcn-mse=0.0062(0.0132) gcn-final-mse=0.0125(0.0296)
2020-08-13 21:01:19 6000-10553 loss=0.0811(0.0577+0.0234)-0.0737(0.0529+0.0208) sod-mse=0.0114(0.0104) gcn-mse=0.0136(0.0133) gcn-final-mse=0.0125(0.0296)
2020-08-13 21:04:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 21:05:24 7000-10553 loss=0.0928(0.0751+0.0177)-0.0738(0.0530+0.0208) sod-mse=0.0131(0.0104) gcn-mse=0.0176(0.0132) gcn-final-mse=0.0125(0.0297)
2020-08-13 21:09:30 8000-10553 loss=0.0879(0.0643+0.0236)-0.0740(0.0531+0.0209) sod-mse=0.0111(0.0104) gcn-mse=0.0174(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 21:09:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 21:13:36 9000-10553 loss=0.2257(0.1298+0.0959)-0.0739(0.0531+0.0208) sod-mse=0.0264(0.0104) gcn-mse=0.0241(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 21:17:42 10000-10553 loss=0.0708(0.0495+0.0214)-0.0738(0.0530+0.0208) sod-mse=0.0076(0.0104) gcn-mse=0.0114(0.0132) gcn-final-mse=0.0125(0.0297)

2020-08-13 21:19:57    0-5019 loss=1.1586(0.6769+0.4817)-1.1586(0.6769+0.4817) sod-mse=0.1025(0.1025) gcn-mse=0.1069(0.1069) gcn-final-mse=0.0989(0.1107)
2020-08-13 21:21:49 1000-5019 loss=0.0285(0.0239+0.0046)-0.3975(0.1900+0.2075) sod-mse=0.0037(0.0504) gcn-mse=0.0050(0.0532) gcn-final-mse=0.0529(0.0665)
2020-08-13 21:23:39 2000-5019 loss=1.1813(0.5455+0.6359)-0.4063(0.1923+0.2139) sod-mse=0.0960(0.0524) gcn-mse=0.0969(0.0549) gcn-final-mse=0.0547(0.0680)
2020-08-13 21:25:29 3000-5019 loss=0.0452(0.0339+0.0114)-0.4050(0.1913+0.2137) sod-mse=0.0052(0.0525) gcn-mse=0.0070(0.0551) gcn-final-mse=0.0548(0.0682)
2020-08-13 21:26:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 21:27:19 4000-5019 loss=0.1085(0.0762+0.0323)-0.3989(0.1891+0.2097) sod-mse=0.0144(0.0520) gcn-mse=0.0175(0.0547) gcn-final-mse=0.0545(0.0678)
2020-08-13 21:27:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 21:29:09 5000-5019 loss=2.0836(0.6639+1.4197)-0.3991(0.1894+0.2097) sod-mse=0.0864(0.0521) gcn-mse=0.0910(0.0548) gcn-final-mse=0.0544(0.0678)
2020-08-13 21:29:11 E:37, Train sod-mae-score=0.0104-0.9842 gcn-mae-score=0.0132-0.9553 gcn-final-mse-score=0.0125-0.9579(0.0297/0.9579) loss=0.0738(0.0530+0.0208)
2020-08-13 21:29:11 E:37, Test  sod-mae-score=0.0521-0.8404 gcn-mae-score=0.0548-0.7885 gcn-final-mse-score=0.0545-0.7942(0.0678/0.7942) loss=0.3991(0.1894+0.2097)

2020-08-13 21:29:11 Start Epoch 38
2020-08-13 21:29:11 Epoch:38,lr=0.0000
2020-08-13 21:29:12    0-10553 loss=0.0468(0.0350+0.0118)-0.0468(0.0350+0.0118) sod-mse=0.0051(0.0051) gcn-mse=0.0052(0.0052) gcn-final-mse=0.0066(0.0286)
2020-08-13 21:31:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 21:33:16 1000-10553 loss=0.1720(0.1204+0.0516)-0.0752(0.0539+0.0212) sod-mse=0.0236(0.0107) gcn-mse=0.0318(0.0135) gcn-final-mse=0.0127(0.0304)
2020-08-13 21:33:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 21:37:20 2000-10553 loss=0.0765(0.0492+0.0273)-0.0735(0.0529+0.0206) sod-mse=0.0146(0.0103) gcn-mse=0.0188(0.0131) gcn-final-mse=0.0124(0.0297)
2020-08-13 21:41:22 3000-10553 loss=0.0640(0.0473+0.0167)-0.0734(0.0529+0.0205) sod-mse=0.0078(0.0103) gcn-mse=0.0102(0.0132) gcn-final-mse=0.0124(0.0297)
2020-08-13 21:45:25 4000-10553 loss=0.0883(0.0654+0.0229)-0.0744(0.0533+0.0211) sod-mse=0.0134(0.0104) gcn-mse=0.0182(0.0133) gcn-final-mse=0.0125(0.0298)
2020-08-13 21:49:29 5000-10553 loss=0.1064(0.0805+0.0258)-0.0740(0.0531+0.0209) sod-mse=0.0125(0.0104) gcn-mse=0.0126(0.0133) gcn-final-mse=0.0125(0.0297)
2020-08-13 21:52:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 21:53:34 6000-10553 loss=0.1386(0.1018+0.0368)-0.0742(0.0533+0.0209) sod-mse=0.0192(0.0104) gcn-mse=0.0250(0.0133) gcn-final-mse=0.0125(0.0299)
2020-08-13 21:57:39 7000-10553 loss=0.0700(0.0557+0.0143)-0.0741(0.0532+0.0209) sod-mse=0.0064(0.0104) gcn-mse=0.0097(0.0133) gcn-final-mse=0.0125(0.0298)
2020-08-13 22:01:47 8000-10553 loss=0.0740(0.0565+0.0176)-0.0740(0.0531+0.0209) sod-mse=0.0092(0.0104) gcn-mse=0.0108(0.0132) gcn-final-mse=0.0124(0.0298)
2020-08-13 22:05:56 9000-10553 loss=0.0820(0.0625+0.0195)-0.0741(0.0532+0.0209) sod-mse=0.0093(0.0104) gcn-mse=0.0145(0.0132) gcn-final-mse=0.0124(0.0298)
2020-08-13 22:10:03 10000-10553 loss=0.0604(0.0490+0.0114)-0.0738(0.0530+0.0208) sod-mse=0.0086(0.0104) gcn-mse=0.0163(0.0132) gcn-final-mse=0.0124(0.0297)

2020-08-13 22:12:22    0-5019 loss=1.1734(0.6837+0.4896)-1.1734(0.6837+0.4896) sod-mse=0.1024(0.1024) gcn-mse=0.1067(0.1067) gcn-final-mse=0.0986(0.1104)
2020-08-13 22:14:15 1000-5019 loss=0.0283(0.0238+0.0045)-0.4001(0.1900+0.2101) sod-mse=0.0036(0.0501) gcn-mse=0.0048(0.0530) gcn-final-mse=0.0528(0.0662)
2020-08-13 22:16:07 2000-5019 loss=1.1941(0.5472+0.6470)-0.4091(0.1924+0.2166) sod-mse=0.0961(0.0522) gcn-mse=0.0970(0.0547) gcn-final-mse=0.0545(0.0678)
2020-08-13 22:17:57 3000-5019 loss=0.0451(0.0338+0.0113)-0.4078(0.1914+0.2164) sod-mse=0.0052(0.0522) gcn-mse=0.0070(0.0549) gcn-final-mse=0.0546(0.0680)
2020-08-13 22:18:59 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 22:19:48 4000-5019 loss=0.1083(0.0758+0.0324)-0.4018(0.1893+0.2124) sod-mse=0.0144(0.0518) gcn-mse=0.0170(0.0545) gcn-final-mse=0.0543(0.0676)
2020-08-13 22:20:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 22:21:38 5000-5019 loss=2.1689(0.6706+1.4983)-0.4019(0.1896+0.2123) sod-mse=0.0865(0.0518) gcn-mse=0.0914(0.0546) gcn-final-mse=0.0542(0.0676)
2020-08-13 22:21:40 E:38, Train sod-mae-score=0.0103-0.9842 gcn-mae-score=0.0132-0.9554 gcn-final-mse-score=0.0124-0.9579(0.0297/0.9579) loss=0.0737(0.0530+0.0208)
2020-08-13 22:21:40 E:38, Test  sod-mae-score=0.0518-0.8406 gcn-mae-score=0.0546-0.7885 gcn-final-mse-score=0.0543-0.7942(0.0676/0.7942) loss=0.4019(0.1896+0.2123)

2020-08-13 22:21:40 Start Epoch 39
2020-08-13 22:21:40 Epoch:39,lr=0.0000
2020-08-13 22:21:42    0-10553 loss=0.1613(0.1186+0.0427)-0.1613(0.1186+0.0427) sod-mse=0.0224(0.0224) gcn-mse=0.0236(0.0236) gcn-final-mse=0.0255(0.0724)
2020-08-13 22:25:47 1000-10553 loss=0.0702(0.0591+0.0110)-0.0745(0.0536+0.0209) sod-mse=0.0054(0.0104) gcn-mse=0.0087(0.0132) gcn-final-mse=0.0124(0.0299)
2020-08-13 22:29:53 2000-10553 loss=0.0345(0.0269+0.0076)-0.0740(0.0530+0.0210) sod-mse=0.0042(0.0104) gcn-mse=0.0078(0.0133) gcn-final-mse=0.0124(0.0296)
2020-08-13 22:33:58 3000-10553 loss=0.1332(0.1070+0.0262)-0.0738(0.0529+0.0209) sod-mse=0.0178(0.0104) gcn-mse=0.0266(0.0132) gcn-final-mse=0.0124(0.0296)
2020-08-13 22:38:03 4000-10553 loss=0.0421(0.0305+0.0115)-0.0735(0.0528+0.0207) sod-mse=0.0054(0.0103) gcn-mse=0.0073(0.0131) gcn-final-mse=0.0123(0.0295)
2020-08-13 22:40:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-13 22:42:10 5000-10553 loss=0.0400(0.0312+0.0088)-0.0732(0.0526+0.0206) sod-mse=0.0063(0.0102) gcn-mse=0.0126(0.0131) gcn-final-mse=0.0123(0.0295)
2020-08-13 22:42:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-13 22:46:15 6000-10553 loss=0.0533(0.0391+0.0143)-0.0736(0.0529+0.0207) sod-mse=0.0061(0.0103) gcn-mse=0.0083(0.0131) gcn-final-mse=0.0123(0.0296)
2020-08-13 22:50:19 7000-10553 loss=0.1051(0.0671+0.0380)-0.0734(0.0528+0.0206) sod-mse=0.0191(0.0103) gcn-mse=0.0216(0.0131) gcn-final-mse=0.0123(0.0296)
2020-08-13 22:54:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-13 22:54:24 8000-10553 loss=0.0434(0.0339+0.0096)-0.0735(0.0528+0.0207) sod-mse=0.0045(0.0103) gcn-mse=0.0056(0.0131) gcn-final-mse=0.0123(0.0295)
2020-08-13 22:58:29 9000-10553 loss=0.0192(0.0158+0.0035)-0.0734(0.0528+0.0206) sod-mse=0.0027(0.0103) gcn-mse=0.0034(0.0131) gcn-final-mse=0.0123(0.0295)
2020-08-13 23:02:33 10000-10553 loss=0.1038(0.0680+0.0358)-0.0735(0.0528+0.0206) sod-mse=0.0189(0.0103) gcn-mse=0.0181(0.0131) gcn-final-mse=0.0123(0.0296)

2020-08-13 23:04:49    0-5019 loss=1.1976(0.6952+0.5023)-1.1976(0.6952+0.5023) sod-mse=0.1040(0.1040) gcn-mse=0.1078(0.1078) gcn-final-mse=0.0998(0.1116)
2020-08-13 23:06:41 1000-5019 loss=0.0283(0.0237+0.0045)-0.4021(0.1908+0.2112) sod-mse=0.0036(0.0503) gcn-mse=0.0047(0.0532) gcn-final-mse=0.0529(0.0664)
2020-08-13 23:08:32 2000-5019 loss=1.2152(0.5559+0.6593)-0.4112(0.1933+0.2179) sod-mse=0.0963(0.0523) gcn-mse=0.0973(0.0549) gcn-final-mse=0.0546(0.0680)
2020-08-13 23:10:23 3000-5019 loss=0.0452(0.0338+0.0114)-0.4098(0.1922+0.2176) sod-mse=0.0052(0.0524) gcn-mse=0.0069(0.0550) gcn-final-mse=0.0548(0.0681)
2020-08-13 23:11:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-13 23:12:14 4000-5019 loss=0.1086(0.0762+0.0324)-0.4037(0.1901+0.2136) sod-mse=0.0144(0.0519) gcn-mse=0.0173(0.0547) gcn-final-mse=0.0544(0.0678)
2020-08-13 23:12:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-13 23:14:04 5000-5019 loss=2.1582(0.6709+1.4874)-0.4039(0.1904+0.2135) sod-mse=0.0857(0.0519) gcn-mse=0.0902(0.0547) gcn-final-mse=0.0544(0.0677)
2020-08-13 23:14:06 E:39, Train sod-mae-score=0.0103-0.9843 gcn-mae-score=0.0131-0.9553 gcn-final-mse-score=0.0124-0.9578(0.0296/0.9578) loss=0.0735(0.0528+0.0206)
2020-08-13 23:14:06 E:39, Test  sod-mae-score=0.0519-0.8398 gcn-mae-score=0.0547-0.7882 gcn-final-mse-score=0.0544-0.7939(0.0677/0.7939) loss=0.4038(0.1904+0.2134)

Process finished with exit code 0
