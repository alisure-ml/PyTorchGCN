/home/ubuntu/anaconda3/envs/alisure36torch/bin/python /mnt/4T/ALISURE/GCN/PyTorchGCN/MyGCN/SPERunner_1_PYG_CONV_Fast_SOD_SGPU_E2E_BS1_MoreConv.py
2020-08-04 09:56:05 name:E2E2-BS1-MoreConv-1-C2PC2PC3C3C3_False_False_lr0001 epochs:50 ckpt:./ckpt2/dgl/1_PYG_CONV_Fast-SOD_BAS/E2E2-BS1-MoreConv-1-C2PC2PC3C3C3_False_False_lr0001 sp size:4 down_ratio:4 workers:16 gpu:1 has_mask:False has_residual:True is_normalize:True has_bn:True improved:True concat:True is_sgd:False weight_decay:0.0

2020-08-04 09:56:05 Cuda available with GPU: GeForce GTX 1080
2020-08-04 09:56:11 Total param: 37303488 lr_s=[[0, 0.0001], [20, 1e-05], [35, 1e-06]] Optimizer=Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    eps: 1e-08
    lr: 0.0001
    weight_decay: 0.0
)
2020-08-04 09:56:11 MyGCNNet(
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
2020-08-04 09:56:11 The number of parameters: 37303488

2020-08-04 09:56:11 Start Epoch 0
2020-08-04 09:56:11 Epoch:00,lr=0.0001
2020-08-04 09:56:13    0-10553 loss=1.3594(0.6665+0.6930)-1.3594(0.6665+0.6930) sod-mse=0.4968(0.4968) gcn-mse=0.4625(0.4625) gcn-final-mse=0.4625(0.4762)
2020-08-04 10:00:15 1000-10553 loss=0.3633(0.1622+0.2011)-0.6523(0.3216+0.3306) sod-mse=0.1709(0.2045) gcn-mse=0.1223(0.1888) gcn-final-mse=0.1891(0.2028)
2020-08-04 10:04:22 2000-10553 loss=0.1867(0.1154+0.0714)-0.5704(0.2859+0.2845) sod-mse=0.0599(0.1713) gcn-mse=0.0813(0.1630) gcn-final-mse=0.1632(0.1769)
2020-08-04 10:08:28 3000-10553 loss=0.3345(0.1782+0.1563)-0.5357(0.2722+0.2635) sod-mse=0.1227(0.1571) gcn-mse=0.1062(0.1527) gcn-final-mse=0.1528(0.1667)
2020-08-04 10:12:35 4000-10553 loss=0.1191(0.0798+0.0393)-0.5083(0.2605+0.2478) sod-mse=0.0324(0.1468) gcn-mse=0.0548(0.1449) gcn-final-mse=0.1450(0.1588)
2020-08-04 10:16:42 5000-10553 loss=0.1844(0.1140+0.0703)-0.4896(0.2525+0.2371) sod-mse=0.0525(0.1396) gcn-mse=0.0716(0.1392) gcn-final-mse=0.1393(0.1531)
2020-08-04 10:19:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 10:20:46 6000-10553 loss=0.1722(0.0899+0.0823)-0.4761(0.2467+0.2293) sod-mse=0.0525(0.1345) gcn-mse=0.0509(0.1351) gcn-final-mse=0.1352(0.1490)
2020-08-04 10:24:52 7000-10553 loss=0.1252(0.0861+0.0391)-0.4659(0.2423+0.2236) sod-mse=0.0322(0.1308) gcn-mse=0.0613(0.1318) gcn-final-mse=0.1319(0.1457)
2020-08-04 10:28:57 8000-10553 loss=0.8043(0.4195+0.3849)-0.4573(0.2388+0.2185) sod-mse=0.2266(0.1275) gcn-mse=0.2004(0.1293) gcn-final-mse=0.1294(0.1433)
2020-08-04 10:33:04 9000-10553 loss=0.2164(0.1127+0.1037)-0.4491(0.2352+0.2139) sod-mse=0.0637(0.1246) gcn-mse=0.0674(0.1269) gcn-final-mse=0.1269(0.1409)
2020-08-04 10:36:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 10:37:09 10000-10553 loss=0.2279(0.1299+0.0980)-0.4411(0.2316+0.2095) sod-mse=0.0754(0.1218) gcn-mse=0.0720(0.1245) gcn-final-mse=0.1245(0.1385)
2020-08-04 10:38:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg

2020-08-04 10:39:25    0-5019 loss=1.2769(0.6167+0.6602)-1.2769(0.6167+0.6602) sod-mse=0.2316(0.2316) gcn-mse=0.2441(0.2441) gcn-final-mse=0.2348(0.2437)
2020-08-04 10:41:18 1000-5019 loss=0.0892(0.0629+0.0263)-0.4265(0.2234+0.2031) sod-mse=0.0235(0.1002) gcn-mse=0.0424(0.1129) gcn-final-mse=0.1133(0.1264)
2020-08-04 10:43:09 2000-5019 loss=0.6742(0.2896+0.3845)-0.4349(0.2267+0.2082) sod-mse=0.1387(0.1023) gcn-mse=0.1338(0.1146) gcn-final-mse=0.1151(0.1282)
2020-08-04 10:45:01 3000-5019 loss=0.0709(0.0499+0.0210)-0.4401(0.2292+0.2108) sod-mse=0.0116(0.1038) gcn-mse=0.0231(0.1160) gcn-final-mse=0.1165(0.1296)
2020-08-04 10:46:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 10:46:54 4000-5019 loss=0.2496(0.1405+0.1091)-0.4391(0.2290+0.2101) sod-mse=0.0658(0.1038) gcn-mse=0.0716(0.1161) gcn-final-mse=0.1166(0.1297)
2020-08-04 10:47:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 10:48:50 5000-5019 loss=0.6527(0.2813+0.3714)-0.4388(0.2290+0.2098) sod-mse=0.1334(0.1038) gcn-mse=0.1384(0.1163) gcn-final-mse=0.1167(0.1298)
2020-08-04 10:48:51 E: 0, Train sod-mae-score=0.1205-0.8565 gcn-mae-score=0.1234-0.8381 gcn-final-mse-score=0.1233-0.8416(0.1373/0.8416) loss=0.4374(0.2299+0.2075)
2020-08-04 10:48:51 E: 0, Test  sod-mae-score=0.1038-0.7453 gcn-mae-score=0.1162-0.7025 gcn-final-mse-score=0.1167-0.7089(0.1297/0.7089) loss=0.4385(0.2289+0.2096)

2020-08-04 10:48:51 Start Epoch 1
2020-08-04 10:48:51 Epoch:01,lr=0.0001
2020-08-04 10:48:53    0-10553 loss=0.0901(0.0615+0.0287)-0.0901(0.0615+0.0287) sod-mse=0.0177(0.0177) gcn-mse=0.0354(0.0354) gcn-final-mse=0.0358(0.0438)
2020-08-04 10:50:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 10:53:04 1000-10553 loss=0.1789(0.1208+0.0580)-0.3532(0.1911+0.1621) sod-mse=0.0410(0.0907) gcn-mse=0.0689(0.0971) gcn-final-mse=0.0971(0.1110)
2020-08-04 10:57:12 2000-10553 loss=0.2407(0.1333+0.1074)-0.3475(0.1889+0.1586) sod-mse=0.0796(0.0896) gcn-mse=0.0752(0.0961) gcn-final-mse=0.0959(0.1099)
2020-08-04 11:01:20 3000-10553 loss=0.6368(0.3220+0.3149)-0.3417(0.1863+0.1554) sod-mse=0.2150(0.0880) gcn-mse=0.1918(0.0946) gcn-final-mse=0.0944(0.1085)
2020-08-04 11:05:26 4000-10553 loss=0.3819(0.2002+0.1817)-0.3388(0.1848+0.1540) sod-mse=0.0779(0.0873) gcn-mse=0.0938(0.0938) gcn-final-mse=0.0937(0.1078)
2020-08-04 11:08:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 11:08:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 11:09:32 5000-10553 loss=1.1612(0.5840+0.5772)-0.3374(0.1840+0.1534) sod-mse=0.2050(0.0869) gcn-mse=0.1988(0.0931) gcn-final-mse=0.0930(0.1071)
2020-08-04 11:13:38 6000-10553 loss=0.4458(0.2358+0.2100)-0.3373(0.1842+0.1532) sod-mse=0.1034(0.0869) gcn-mse=0.1092(0.0934) gcn-final-mse=0.0932(0.1073)
2020-08-04 11:17:43 7000-10553 loss=0.1887(0.1297+0.0590)-0.3403(0.1856+0.1547) sod-mse=0.0401(0.0878) gcn-mse=0.0586(0.0940) gcn-final-mse=0.0938(0.1080)
2020-08-04 11:21:47 8000-10553 loss=0.2178(0.1202+0.0976)-0.3385(0.1847+0.1538) sod-mse=0.0680(0.0872) gcn-mse=0.0762(0.0935) gcn-final-mse=0.0933(0.1075)
2020-08-04 11:25:52 9000-10553 loss=0.0445(0.0305+0.0140)-0.3372(0.1839+0.1532) sod-mse=0.0113(0.0868) gcn-mse=0.0139(0.0930) gcn-final-mse=0.0928(0.1070)
2020-08-04 11:29:57 10000-10553 loss=0.1750(0.1134+0.0617)-0.3353(0.1830+0.1523) sod-mse=0.0435(0.0862) gcn-mse=0.0568(0.0924) gcn-final-mse=0.0922(0.1064)

2020-08-04 11:32:12    0-5019 loss=0.7001(0.3615+0.3385)-0.7001(0.3615+0.3385) sod-mse=0.1218(0.1218) gcn-mse=0.1354(0.1354) gcn-final-mse=0.1277(0.1385)
2020-08-04 11:34:05 1000-5019 loss=0.0748(0.0499+0.0249)-0.4692(0.2413+0.2279) sod-mse=0.0226(0.1110) gcn-mse=0.0312(0.1168) gcn-final-mse=0.1170(0.1305)
2020-08-04 11:35:57 2000-5019 loss=0.5986(0.3084+0.2902)-0.4809(0.2465+0.2344) sod-mse=0.1135(0.1129) gcn-mse=0.1177(0.1188) gcn-final-mse=0.1189(0.1323)
2020-08-04 11:37:50 3000-5019 loss=0.0640(0.0464+0.0177)-0.4884(0.2497+0.2387) sod-mse=0.0102(0.1143) gcn-mse=0.0186(0.1201) gcn-final-mse=0.1202(0.1337)
2020-08-04 11:38:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 11:39:44 4000-5019 loss=0.1861(0.1178+0.0683)-0.4894(0.2502+0.2392) sod-mse=0.0456(0.1145) gcn-mse=0.0569(0.1204) gcn-final-mse=0.1205(0.1339)
2020-08-04 11:40:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 11:41:37 5000-5019 loss=0.5847(0.2633+0.3215)-0.4910(0.2512+0.2398) sod-mse=0.1387(0.1147) gcn-mse=0.1298(0.1206) gcn-final-mse=0.1206(0.1340)
2020-08-04 11:41:39 E: 1, Train sod-mae-score=0.0859-0.9008 gcn-mae-score=0.0919-0.8700 gcn-final-mse-score=0.0918-0.8730(0.1060/0.8730) loss=0.3340(0.1823+0.1517)
2020-08-04 11:41:39 E: 1, Test  sod-mae-score=0.1147-0.7474 gcn-mae-score=0.1206-0.6937 gcn-final-mse-score=0.1206-0.7010(0.1340/0.7010) loss=0.4907(0.2511+0.2397)

2020-08-04 11:41:39 Start Epoch 2
2020-08-04 11:41:39 Epoch:02,lr=0.0001
2020-08-04 11:41:40    0-10553 loss=0.2941(0.1696+0.1245)-0.2941(0.1696+0.1245) sod-mse=0.0631(0.0631) gcn-mse=0.0751(0.0751) gcn-final-mse=0.0778(0.0958)
2020-08-04 11:45:49 1000-10553 loss=1.7251(0.8646+0.8605)-0.2995(0.1657+0.1339) sod-mse=0.3398(0.0741) gcn-mse=0.3379(0.0805) gcn-final-mse=0.0802(0.0945)
2020-08-04 11:49:57 2000-10553 loss=0.7287(0.3786+0.3501)-0.3028(0.1677+0.1350) sod-mse=0.1402(0.0754) gcn-mse=0.1470(0.0822) gcn-final-mse=0.0818(0.0964)
2020-08-04 11:54:06 3000-10553 loss=0.3437(0.1874+0.1563)-0.3001(0.1657+0.1343) sod-mse=0.1250(0.0748) gcn-mse=0.1254(0.0810) gcn-final-mse=0.0807(0.0953)
2020-08-04 11:58:15 4000-10553 loss=0.2283(0.1382+0.0900)-0.2990(0.1654+0.1336) sod-mse=0.0547(0.0743) gcn-mse=0.0731(0.0808) gcn-final-mse=0.0804(0.0949)
2020-08-04 12:02:21 5000-10553 loss=0.1200(0.0790+0.0410)-0.2986(0.1651+0.1336) sod-mse=0.0294(0.0745) gcn-mse=0.0344(0.0809) gcn-final-mse=0.0805(0.0950)
2020-08-04 12:04:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 12:05:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 12:06:24 6000-10553 loss=0.1809(0.1331+0.0478)-0.2986(0.1652+0.1333) sod-mse=0.0347(0.0745) gcn-mse=0.0796(0.0809) gcn-final-mse=0.0805(0.0950)
2020-08-04 12:10:29 7000-10553 loss=0.5368(0.2992+0.2376)-0.2958(0.1638+0.1320) sod-mse=0.1230(0.0737) gcn-mse=0.1178(0.0802) gcn-final-mse=0.0799(0.0943)
2020-08-04 12:14:33 8000-10553 loss=0.2557(0.1469+0.1088)-0.2965(0.1642+0.1322) sod-mse=0.0820(0.0739) gcn-mse=0.0951(0.0804) gcn-final-mse=0.0801(0.0945)
2020-08-04 12:16:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 12:18:36 9000-10553 loss=0.3271(0.1958+0.1313)-0.2960(0.1639+0.1321) sod-mse=0.0727(0.0738) gcn-mse=0.0829(0.0801) gcn-final-mse=0.0798(0.0943)
2020-08-04 12:22:39 10000-10553 loss=0.0836(0.0505+0.0332)-0.2943(0.1631+0.1313) sod-mse=0.0256(0.0733) gcn-mse=0.0224(0.0796) gcn-final-mse=0.0793(0.0938)

2020-08-04 12:24:56    0-5019 loss=0.7162(0.3849+0.3313)-0.7162(0.3849+0.3313) sod-mse=0.1679(0.1679) gcn-mse=0.1455(0.1455) gcn-final-mse=0.1391(0.1529)
2020-08-04 12:26:47 1000-5019 loss=0.2632(0.1401+0.1231)-0.3492(0.1869+0.1623) sod-mse=0.0920(0.1045) gcn-mse=0.0881(0.0988) gcn-final-mse=0.0990(0.1135)
2020-08-04 12:28:39 2000-5019 loss=0.6364(0.3373+0.2992)-0.3517(0.1883+0.1634) sod-mse=0.1203(0.1054) gcn-mse=0.1119(0.0998) gcn-final-mse=0.1000(0.1144)
2020-08-04 12:30:30 3000-5019 loss=0.0697(0.0453+0.0244)-0.3567(0.1911+0.1656) sod-mse=0.0180(0.1067) gcn-mse=0.0194(0.1012) gcn-final-mse=0.1014(0.1158)
2020-08-04 12:31:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 12:32:21 4000-5019 loss=0.1996(0.1128+0.0868)-0.3580(0.1919+0.1661) sod-mse=0.0603(0.1071) gcn-mse=0.0521(0.1017) gcn-final-mse=0.1018(0.1163)
2020-08-04 12:32:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 12:34:12 5000-5019 loss=0.5992(0.2874+0.3118)-0.3593(0.1928+0.1666) sod-mse=0.1440(0.1074) gcn-mse=0.1282(0.1021) gcn-final-mse=0.1022(0.1167)
2020-08-04 12:34:14 E: 2, Train sod-mae-score=0.0733-0.9137 gcn-mae-score=0.0796-0.8828 gcn-final-mse-score=0.0793-0.8858(0.0937/0.8858) loss=0.2942(0.1629+0.1313)
2020-08-04 12:34:14 E: 2, Test  sod-mae-score=0.1074-0.7883 gcn-mae-score=0.1021-0.7213 gcn-final-mse-score=0.1022-0.7280(0.1167/0.7280) loss=0.3592(0.1927+0.1665)

2020-08-04 12:34:14 Start Epoch 3
2020-08-04 12:34:14 Epoch:03,lr=0.0001
2020-08-04 12:34:15    0-10553 loss=0.2042(0.1275+0.0767)-0.2042(0.1275+0.0767) sod-mse=0.0639(0.0639) gcn-mse=0.0727(0.0727) gcn-final-mse=0.0704(0.0869)
2020-08-04 12:38:21 1000-10553 loss=0.0727(0.0493+0.0234)-0.2590(0.1464+0.1126) sod-mse=0.0185(0.0637) gcn-mse=0.0175(0.0710) gcn-final-mse=0.0705(0.0849)
2020-08-04 12:42:25 2000-10553 loss=0.1911(0.1138+0.0773)-0.2603(0.1462+0.1142) sod-mse=0.0561(0.0638) gcn-mse=0.0585(0.0701) gcn-final-mse=0.0698(0.0841)
2020-08-04 12:44:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 12:46:29 3000-10553 loss=0.5821(0.2876+0.2945)-0.2578(0.1450+0.1128) sod-mse=0.1299(0.0630) gcn-mse=0.1405(0.0694) gcn-final-mse=0.0692(0.0836)
2020-08-04 12:50:34 4000-10553 loss=1.0121(0.6563+0.3558)-0.2621(0.1471+0.1150) sod-mse=0.1530(0.0640) gcn-mse=0.2247(0.0705) gcn-final-mse=0.0703(0.0847)
2020-08-04 12:54:37 5000-10553 loss=0.2245(0.1382+0.0862)-0.2632(0.1477+0.1155) sod-mse=0.0632(0.0643) gcn-mse=0.0720(0.0708) gcn-final-mse=0.0706(0.0850)
2020-08-04 12:58:41 6000-10553 loss=0.1189(0.0855+0.0334)-0.2645(0.1484+0.1161) sod-mse=0.0243(0.0647) gcn-mse=0.0495(0.0709) gcn-final-mse=0.0707(0.0852)
2020-08-04 13:02:45 7000-10553 loss=0.2716(0.1567+0.1149)-0.2655(0.1489+0.1166) sod-mse=0.0655(0.0648) gcn-mse=0.0809(0.0711) gcn-final-mse=0.0708(0.0853)
2020-08-04 13:06:48 8000-10553 loss=0.2364(0.1254+0.1110)-0.2622(0.1473+0.1149) sod-mse=0.0882(0.0638) gcn-mse=0.0789(0.0702) gcn-final-mse=0.0699(0.0845)
2020-08-04 13:10:52 9000-10553 loss=0.1053(0.0576+0.0477)-0.2628(0.1476+0.1152) sod-mse=0.0251(0.0641) gcn-mse=0.0186(0.0704) gcn-final-mse=0.0701(0.0847)
2020-08-04 13:14:55 10000-10553 loss=0.1240(0.0701+0.0540)-0.2645(0.1484+0.1161) sod-mse=0.0373(0.0645) gcn-mse=0.0229(0.0708) gcn-final-mse=0.0705(0.0851)
2020-08-04 13:15:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 13:15:50 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg

2020-08-04 13:17:13    0-5019 loss=0.7373(0.4517+0.2856)-0.7373(0.4517+0.2856) sod-mse=0.1107(0.1107) gcn-mse=0.1253(0.1253) gcn-final-mse=0.1179(0.1293)
2020-08-04 13:19:05 1000-5019 loss=0.0731(0.0430+0.0301)-0.3782(0.1969+0.1813) sod-mse=0.0281(0.1069) gcn-mse=0.0242(0.0968) gcn-final-mse=0.0968(0.1105)
2020-08-04 13:20:56 2000-5019 loss=0.4880(0.2675+0.2204)-0.3851(0.1996+0.1855) sod-mse=0.1189(0.1089) gcn-mse=0.1051(0.0985) gcn-final-mse=0.0984(0.1120)
2020-08-04 13:22:47 3000-5019 loss=0.0605(0.0418+0.0188)-0.3912(0.2024+0.1888) sod-mse=0.0113(0.1102) gcn-mse=0.0150(0.0997) gcn-final-mse=0.0997(0.1133)
2020-08-04 13:23:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 13:24:38 4000-5019 loss=0.1710(0.1064+0.0645)-0.3915(0.2025+0.1890) sod-mse=0.0416(0.1104) gcn-mse=0.0446(0.1000) gcn-final-mse=0.0999(0.1135)
2020-08-04 13:25:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 13:26:29 5000-5019 loss=0.5897(0.2841+0.3056)-0.3910(0.2023+0.1887) sod-mse=0.1634(0.1105) gcn-mse=0.1383(0.1001) gcn-final-mse=0.1000(0.1136)
2020-08-04 13:26:31 E: 3, Train sod-mae-score=0.0646-0.9225 gcn-mae-score=0.0708-0.8919 gcn-final-mse-score=0.0705-0.8949(0.0851/0.8949) loss=0.2647(0.1485+0.1162)
2020-08-04 13:26:31 E: 3, Test  sod-mae-score=0.1105-0.7920 gcn-mae-score=0.1001-0.7284 gcn-final-mse-score=0.1000-0.7346(0.1135/0.7346) loss=0.3908(0.2022+0.1886)

2020-08-04 13:26:31 Start Epoch 4
2020-08-04 13:26:31 Epoch:04,lr=0.0001
2020-08-04 13:26:33    0-10553 loss=0.1781(0.0923+0.0858)-0.1781(0.0923+0.0858) sod-mse=0.0589(0.0589) gcn-mse=0.0522(0.0522) gcn-final-mse=0.0531(0.0597)
2020-08-04 13:30:38 1000-10553 loss=0.0860(0.0624+0.0235)-0.2400(0.1363+0.1037) sod-mse=0.0142(0.0574) gcn-mse=0.0178(0.0647) gcn-final-mse=0.0644(0.0793)
2020-08-04 13:34:43 2000-10553 loss=0.2217(0.1306+0.0911)-0.2456(0.1393+0.1063) sod-mse=0.0611(0.0590) gcn-mse=0.0768(0.0662) gcn-final-mse=0.0659(0.0808)
2020-08-04 13:37:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 13:37:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 13:38:47 3000-10553 loss=0.2141(0.1303+0.0838)-0.2466(0.1396+0.1070) sod-mse=0.0600(0.0593) gcn-mse=0.0604(0.0662) gcn-final-mse=0.0659(0.0806)
2020-08-04 13:42:53 4000-10553 loss=0.1256(0.0889+0.0367)-0.2398(0.1361+0.1036) sod-mse=0.0272(0.0573) gcn-mse=0.0396(0.0642) gcn-final-mse=0.0639(0.0785)
2020-08-04 13:46:56 5000-10553 loss=0.0577(0.0458+0.0120)-0.2402(0.1365+0.1037) sod-mse=0.0070(0.0574) gcn-mse=0.0125(0.0642) gcn-final-mse=0.0639(0.0786)
2020-08-04 13:51:00 6000-10553 loss=0.2636(0.1537+0.1099)-0.2423(0.1375+0.1048) sod-mse=0.0781(0.0580) gcn-mse=0.0938(0.0646) gcn-final-mse=0.0643(0.0791)
2020-08-04 13:55:04 7000-10553 loss=0.9773(0.5009+0.4764)-0.2413(0.1369+0.1044) sod-mse=0.1829(0.0576) gcn-mse=0.1810(0.0641) gcn-final-mse=0.0638(0.0785)
2020-08-04 13:57:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 13:59:06 8000-10553 loss=0.1849(0.1258+0.0591)-0.2405(0.1365+0.1040) sod-mse=0.0422(0.0575) gcn-mse=0.0558(0.0639) gcn-final-mse=0.0636(0.0783)
2020-08-04 14:03:08 9000-10553 loss=0.6556(0.3439+0.3117)-0.2413(0.1368+0.1044) sod-mse=0.1429(0.0577) gcn-mse=0.1587(0.0640) gcn-final-mse=0.0637(0.0784)
2020-08-04 14:07:11 10000-10553 loss=0.6467(0.3354+0.3113)-0.2425(0.1375+0.1050) sod-mse=0.1834(0.0580) gcn-mse=0.2067(0.0643) gcn-final-mse=0.0640(0.0787)

2020-08-04 14:09:27    0-5019 loss=0.6945(0.3835+0.3111)-0.6945(0.3835+0.3111) sod-mse=0.1067(0.1067) gcn-mse=0.1161(0.1161) gcn-final-mse=0.1079(0.1201)
2020-08-04 14:11:20 1000-5019 loss=0.0947(0.0583+0.0365)-0.3814(0.2024+0.1790) sod-mse=0.0328(0.1007) gcn-mse=0.0369(0.1025) gcn-final-mse=0.1025(0.1166)
2020-08-04 14:13:12 2000-5019 loss=0.6739(0.3527+0.3212)-0.3893(0.2059+0.1834) sod-mse=0.1202(0.1025) gcn-mse=0.1161(0.1043) gcn-final-mse=0.1043(0.1184)
2020-08-04 14:15:04 3000-5019 loss=0.0544(0.0378+0.0166)-0.3970(0.2093+0.1877) sod-mse=0.0090(0.1046) gcn-mse=0.0109(0.1062) gcn-final-mse=0.1062(0.1202)
2020-08-04 14:16:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 14:16:56 4000-5019 loss=0.1728(0.1103+0.0625)-0.3965(0.2091+0.1874) sod-mse=0.0357(0.1045) gcn-mse=0.0436(0.1063) gcn-final-mse=0.1063(0.1203)
2020-08-04 14:17:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 14:18:47 5000-5019 loss=0.5721(0.2779+0.2942)-0.3970(0.2096+0.1874) sod-mse=0.1381(0.1047) gcn-mse=0.1251(0.1066) gcn-final-mse=0.1066(0.1206)
2020-08-04 14:18:49 E: 4, Train sod-mae-score=0.0582-0.9292 gcn-mae-score=0.0644-0.8979 gcn-final-mse-score=0.0641-0.9008(0.0788/0.9008) loss=0.2433(0.1378+0.1055)
2020-08-04 14:18:49 E: 4, Test  sod-mae-score=0.1047-0.7799 gcn-mae-score=0.1066-0.7273 gcn-final-mse-score=0.1066-0.7335(0.1206/0.7335) loss=0.3971(0.2096+0.1874)

2020-08-04 14:18:49 Start Epoch 5
2020-08-04 14:18:49 Epoch:05,lr=0.0001
2020-08-04 14:18:50    0-10553 loss=0.1759(0.1023+0.0736)-0.1759(0.1023+0.0736) sod-mse=0.0427(0.0427) gcn-mse=0.0503(0.0503) gcn-final-mse=0.0500(0.0659)
2020-08-04 14:22:55 1000-10553 loss=0.1359(0.0863+0.0496)-0.2184(0.1263+0.0921) sod-mse=0.0290(0.0504) gcn-mse=0.0339(0.0578) gcn-final-mse=0.0576(0.0723)
2020-08-04 14:26:59 2000-10553 loss=0.1256(0.0868+0.0388)-0.2201(0.1270+0.0931) sod-mse=0.0280(0.0510) gcn-mse=0.0500(0.0583) gcn-final-mse=0.0580(0.0728)
2020-08-04 14:31:04 3000-10553 loss=0.4380(0.2514+0.1866)-0.2270(0.1300+0.0970) sod-mse=0.1017(0.0534) gcn-mse=0.1379(0.0601) gcn-final-mse=0.0598(0.0746)
2020-08-04 14:35:07 4000-10553 loss=0.3073(0.2026+0.1047)-0.2246(0.1290+0.0957) sod-mse=0.0632(0.0526) gcn-mse=0.0955(0.0594) gcn-final-mse=0.0590(0.0738)
2020-08-04 14:36:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 14:39:11 5000-10553 loss=0.8975(0.5045+0.3930)-0.2252(0.1292+0.0960) sod-mse=0.2161(0.0527) gcn-mse=0.2370(0.0595) gcn-final-mse=0.0591(0.0739)
2020-08-04 14:43:15 6000-10553 loss=0.1532(0.0989+0.0542)-0.2270(0.1300+0.0969) sod-mse=0.0319(0.0532) gcn-mse=0.0344(0.0597) gcn-final-mse=0.0594(0.0742)
2020-08-04 14:47:18 7000-10553 loss=0.1620(0.1168+0.0452)-0.2262(0.1295+0.0967) sod-mse=0.0303(0.0530) gcn-mse=0.0502(0.0595) gcn-final-mse=0.0591(0.0740)
2020-08-04 14:51:22 8000-10553 loss=0.0302(0.0216+0.0086)-0.2261(0.1295+0.0967) sod-mse=0.0067(0.0530) gcn-mse=0.0106(0.0594) gcn-final-mse=0.0590(0.0738)
2020-08-04 14:55:24 9000-10553 loss=0.3996(0.2304+0.1692)-0.2272(0.1299+0.0973) sod-mse=0.0811(0.0533) gcn-mse=0.0831(0.0596) gcn-final-mse=0.0592(0.0740)
2020-08-04 14:55:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 14:57:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 14:59:28 10000-10553 loss=0.4768(0.2290+0.2478)-0.2273(0.1300+0.0973) sod-mse=0.1117(0.0534) gcn-mse=0.0983(0.0596) gcn-final-mse=0.0593(0.0741)

2020-08-04 15:01:44    0-5019 loss=0.9377(0.4849+0.4528)-0.9377(0.4849+0.4528) sod-mse=0.1111(0.1111) gcn-mse=0.1201(0.1201) gcn-final-mse=0.1112(0.1224)
2020-08-04 15:03:37 1000-5019 loss=0.0429(0.0340+0.0089)-0.3523(0.1825+0.1699) sod-mse=0.0078(0.0741) gcn-mse=0.0158(0.0798) gcn-final-mse=0.0799(0.0929)
2020-08-04 15:05:29 2000-5019 loss=0.7799(0.3390+0.4409)-0.3584(0.1855+0.1729) sod-mse=0.1123(0.0752) gcn-mse=0.1037(0.0811) gcn-final-mse=0.0812(0.0941)
2020-08-04 15:07:22 3000-5019 loss=0.0508(0.0363+0.0145)-0.3626(0.1870+0.1755) sod-mse=0.0076(0.0760) gcn-mse=0.0100(0.0818) gcn-final-mse=0.0820(0.0948)
2020-08-04 15:08:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 15:09:16 4000-5019 loss=0.1582(0.0946+0.0636)-0.3626(0.1873+0.1753) sod-mse=0.0336(0.0760) gcn-mse=0.0375(0.0819) gcn-final-mse=0.0820(0.0949)
2020-08-04 15:09:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 15:11:08 5000-5019 loss=0.4748(0.2199+0.2549)-0.3629(0.1877+0.1753) sod-mse=0.1023(0.0761) gcn-mse=0.0981(0.0821) gcn-final-mse=0.0821(0.0950)
2020-08-04 15:11:10 E: 5, Train sod-mae-score=0.0533-0.9336 gcn-mae-score=0.0595-0.9025 gcn-final-mse-score=0.0591-0.9054(0.0740/0.9054) loss=0.2268(0.1297+0.0971)
2020-08-04 15:11:10 E: 5, Test  sod-mae-score=0.0761-0.7954 gcn-mae-score=0.0821-0.7381 gcn-final-mse-score=0.0821-0.7437(0.0950/0.7437) loss=0.3627(0.1876+0.1751)

2020-08-04 15:11:10 Start Epoch 6
2020-08-04 15:11:10 Epoch:06,lr=0.0001
2020-08-04 15:11:11    0-10553 loss=0.0280(0.0232+0.0048)-0.0280(0.0232+0.0048) sod-mse=0.0027(0.0027) gcn-mse=0.0076(0.0076) gcn-final-mse=0.0071(0.0140)
2020-08-04 15:11:59 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 15:15:16 1000-10553 loss=0.2202(0.1347+0.0854)-0.2107(0.1218+0.0889) sod-mse=0.0585(0.0490) gcn-mse=0.0676(0.0561) gcn-final-mse=0.0557(0.0703)
2020-08-04 15:16:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 15:19:20 2000-10553 loss=0.1811(0.1081+0.0730)-0.2142(0.1234+0.0908) sod-mse=0.0486(0.0497) gcn-mse=0.0551(0.0560) gcn-final-mse=0.0555(0.0705)
2020-08-04 15:23:24 3000-10553 loss=0.0514(0.0324+0.0190)-0.2116(0.1217+0.0899) sod-mse=0.0124(0.0491) gcn-mse=0.0153(0.0549) gcn-final-mse=0.0545(0.0694)
2020-08-04 15:27:28 4000-10553 loss=0.1700(0.0883+0.0817)-0.2099(0.1209+0.0890) sod-mse=0.0284(0.0486) gcn-mse=0.0294(0.0544) gcn-final-mse=0.0540(0.0689)
2020-08-04 15:31:33 5000-10553 loss=0.3447(0.1954+0.1493)-0.2132(0.1227+0.0906) sod-mse=0.1106(0.0495) gcn-mse=0.1182(0.0553) gcn-final-mse=0.0549(0.0698)
2020-08-04 15:32:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 15:35:37 6000-10553 loss=0.2162(0.1174+0.0988)-0.2152(0.1236+0.0916) sod-mse=0.0619(0.0500) gcn-mse=0.0665(0.0557) gcn-final-mse=0.0553(0.0702)
2020-08-04 15:39:43 7000-10553 loss=0.1010(0.0593+0.0417)-0.2139(0.1231+0.0909) sod-mse=0.0335(0.0496) gcn-mse=0.0311(0.0555) gcn-final-mse=0.0551(0.0700)
2020-08-04 15:43:47 8000-10553 loss=0.1426(0.0889+0.0538)-0.2132(0.1227+0.0905) sod-mse=0.0385(0.0494) gcn-mse=0.0376(0.0553) gcn-final-mse=0.0550(0.0698)
2020-08-04 15:47:50 9000-10553 loss=0.8416(0.4305+0.4111)-0.2131(0.1227+0.0904) sod-mse=0.1708(0.0493) gcn-mse=0.1785(0.0552) gcn-final-mse=0.0548(0.0697)
2020-08-04 15:51:53 10000-10553 loss=0.1101(0.0787+0.0314)-0.2128(0.1226+0.0902) sod-mse=0.0248(0.0492) gcn-mse=0.0364(0.0551) gcn-final-mse=0.0548(0.0697)

2020-08-04 15:54:09    0-5019 loss=0.8646(0.5156+0.3490)-0.8646(0.5156+0.3490) sod-mse=0.1241(0.1241) gcn-mse=0.1545(0.1545) gcn-final-mse=0.1474(0.1573)
2020-08-04 15:56:01 1000-5019 loss=0.0548(0.0348+0.0200)-0.3220(0.1717+0.1503) sod-mse=0.0183(0.0813) gcn-mse=0.0166(0.0772) gcn-final-mse=0.0772(0.0906)
2020-08-04 15:57:53 2000-5019 loss=0.5578(0.2965+0.2612)-0.3254(0.1728+0.1526) sod-mse=0.1020(0.0828) gcn-mse=0.0942(0.0785) gcn-final-mse=0.0785(0.0919)
2020-08-04 15:59:44 3000-5019 loss=0.0506(0.0356+0.0150)-0.3316(0.1755+0.1561) sod-mse=0.0093(0.0844) gcn-mse=0.0092(0.0800) gcn-final-mse=0.0801(0.0934)
2020-08-04 16:00:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 16:01:35 4000-5019 loss=0.1430(0.0898+0.0531)-0.3302(0.1750+0.1551) sod-mse=0.0329(0.0842) gcn-mse=0.0318(0.0800) gcn-final-mse=0.0800(0.0933)
2020-08-04 16:02:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 16:03:25 5000-5019 loss=0.5207(0.2616+0.2591)-0.3298(0.1749+0.1548) sod-mse=0.1050(0.0843) gcn-mse=0.0939(0.0801) gcn-final-mse=0.0801(0.0935)
2020-08-04 16:03:27 E: 6, Train sod-mae-score=0.0492-0.9385 gcn-mae-score=0.0551-0.9075 gcn-final-mse-score=0.0547-0.9103(0.0696/0.9103) loss=0.2127(0.1225+0.0901)
2020-08-04 16:03:27 E: 6, Test  sod-mae-score=0.0842-0.8041 gcn-mae-score=0.0801-0.7535 gcn-final-mse-score=0.0801-0.7596(0.0934/0.7596) loss=0.3295(0.1748+0.1547)

2020-08-04 16:03:27 Start Epoch 7
2020-08-04 16:03:27 Epoch:07,lr=0.0001
2020-08-04 16:03:28    0-10553 loss=1.0597(0.5194+0.5403)-1.0597(0.5194+0.5403) sod-mse=0.1525(0.1525) gcn-mse=0.1352(0.1352) gcn-final-mse=0.1488(0.1543)
2020-08-04 16:07:33 1000-10553 loss=0.0846(0.0537+0.0308)-0.1946(0.1135+0.0811) sod-mse=0.0140(0.0441) gcn-mse=0.0152(0.0506) gcn-final-mse=0.0502(0.0652)
2020-08-04 16:11:37 2000-10553 loss=0.0586(0.0471+0.0115)-0.2024(0.1177+0.0847) sod-mse=0.0063(0.0464) gcn-mse=0.0154(0.0530) gcn-final-mse=0.0526(0.0676)
2020-08-04 16:15:41 3000-10553 loss=0.1689(0.1063+0.0627)-0.1981(0.1156+0.0825) sod-mse=0.0495(0.0451) gcn-mse=0.0587(0.0516) gcn-final-mse=0.0513(0.0662)
2020-08-04 16:19:44 4000-10553 loss=0.1766(0.1052+0.0714)-0.1966(0.1147+0.0818) sod-mse=0.0468(0.0447) gcn-mse=0.0573(0.0511) gcn-final-mse=0.0507(0.0657)
2020-08-04 16:23:48 5000-10553 loss=0.2059(0.1115+0.0943)-0.1974(0.1152+0.0822) sod-mse=0.0478(0.0449) gcn-mse=0.0438(0.0511) gcn-final-mse=0.0507(0.0657)
2020-08-04 16:27:51 6000-10553 loss=0.1110(0.0750+0.0360)-0.1967(0.1148+0.0819) sod-mse=0.0246(0.0447) gcn-mse=0.0250(0.0509) gcn-final-mse=0.0505(0.0655)
2020-08-04 16:30:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 16:31:55 7000-10553 loss=0.1614(0.0985+0.0629)-0.1986(0.1157+0.0829) sod-mse=0.0413(0.0453) gcn-mse=0.0499(0.0513) gcn-final-mse=0.0509(0.0660)
2020-08-04 16:35:58 8000-10553 loss=0.1370(0.0812+0.0558)-0.1972(0.1149+0.0823) sod-mse=0.0381(0.0450) gcn-mse=0.0452(0.0509) gcn-final-mse=0.0506(0.0656)
2020-08-04 16:39:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 16:40:02 9000-10553 loss=0.2277(0.1298+0.0979)-0.1975(0.1151+0.0824) sod-mse=0.0506(0.0451) gcn-mse=0.0573(0.0510) gcn-final-mse=0.0506(0.0657)
2020-08-04 16:43:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 16:44:07 10000-10553 loss=0.1163(0.0930+0.0233)-0.1989(0.1158+0.0831) sod-mse=0.0123(0.0454) gcn-mse=0.0226(0.0512) gcn-final-mse=0.0508(0.0659)

2020-08-04 16:46:24    0-5019 loss=0.8098(0.4917+0.3181)-0.8098(0.4917+0.3181) sod-mse=0.1296(0.1296) gcn-mse=0.1185(0.1185) gcn-final-mse=0.1110(0.1260)
2020-08-04 16:48:16 1000-5019 loss=0.0853(0.0434+0.0419)-0.3196(0.1709+0.1487) sod-mse=0.0392(0.0970) gcn-mse=0.0239(0.0797) gcn-final-mse=0.0800(0.0941)
2020-08-04 16:50:07 2000-5019 loss=0.7108(0.4074+0.3034)-0.3183(0.1700+0.1484) sod-mse=0.1247(0.0970) gcn-mse=0.1049(0.0796) gcn-final-mse=0.0798(0.0940)
2020-08-04 16:51:58 3000-5019 loss=0.0604(0.0389+0.0215)-0.3237(0.1726+0.1511) sod-mse=0.0155(0.0983) gcn-mse=0.0129(0.0809) gcn-final-mse=0.0811(0.0952)
2020-08-04 16:53:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 16:53:50 4000-5019 loss=0.1565(0.0919+0.0646)-0.3250(0.1733+0.1517) sod-mse=0.0466(0.0986) gcn-mse=0.0364(0.0812) gcn-final-mse=0.0814(0.0955)
2020-08-04 16:54:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 16:55:40 5000-5019 loss=0.5126(0.2652+0.2474)-0.3259(0.1740+0.1519) sod-mse=0.1358(0.0988) gcn-mse=0.1143(0.0815) gcn-final-mse=0.0817(0.0957)
2020-08-04 16:55:42 E: 7, Train sod-mae-score=0.0454-0.9418 gcn-mae-score=0.0513-0.9116 gcn-final-mse-score=0.0509-0.9145(0.0659/0.9145) loss=0.1992(0.1159+0.0833)
2020-08-04 16:55:42 E: 7, Test  sod-mae-score=0.0988-0.8181 gcn-mae-score=0.0815-0.7594 gcn-final-mse-score=0.0817-0.7656(0.0958/0.7656) loss=0.3258(0.1739+0.1519)

2020-08-04 16:55:42 Start Epoch 8
2020-08-04 16:55:42 Epoch:08,lr=0.0001
2020-08-04 16:55:44    0-10553 loss=0.4614(0.2523+0.2091)-0.4614(0.2523+0.2091) sod-mse=0.1423(0.1423) gcn-mse=0.1272(0.1272) gcn-final-mse=0.1327(0.1522)
2020-08-04 16:59:48 1000-10553 loss=0.1681(0.1087+0.0594)-0.1739(0.1034+0.0706) sod-mse=0.0404(0.0385) gcn-mse=0.0579(0.0446) gcn-final-mse=0.0442(0.0593)
2020-08-04 17:00:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 17:03:51 2000-10553 loss=0.0691(0.0449+0.0242)-0.1856(0.1092+0.0764) sod-mse=0.0159(0.0413) gcn-mse=0.0265(0.0474) gcn-final-mse=0.0470(0.0622)
2020-08-04 17:07:56 3000-10553 loss=0.2480(0.1458+0.1022)-0.1834(0.1080+0.0754) sod-mse=0.0361(0.0408) gcn-mse=0.0535(0.0468) gcn-final-mse=0.0464(0.0615)
2020-08-04 17:10:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 17:12:00 4000-10553 loss=0.0708(0.0530+0.0178)-0.1841(0.1084+0.0756) sod-mse=0.0096(0.0409) gcn-mse=0.0162(0.0470) gcn-final-mse=0.0466(0.0618)
2020-08-04 17:16:05 5000-10553 loss=0.2357(0.1256+0.1101)-0.1877(0.1102+0.0775) sod-mse=0.0516(0.0421) gcn-mse=0.0496(0.0480) gcn-final-mse=0.0476(0.0627)
2020-08-04 17:16:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 17:20:08 6000-10553 loss=0.5070(0.2822+0.2248)-0.1871(0.1098+0.0773) sod-mse=0.1193(0.0420) gcn-mse=0.1315(0.0478) gcn-final-mse=0.0474(0.0625)
2020-08-04 17:24:11 7000-10553 loss=0.2906(0.1722+0.1183)-0.1880(0.1103+0.0776) sod-mse=0.0835(0.0422) gcn-mse=0.0881(0.0480) gcn-final-mse=0.0476(0.0627)
2020-08-04 17:28:14 8000-10553 loss=0.0363(0.0260+0.0102)-0.1876(0.1102+0.0774) sod-mse=0.0056(0.0421) gcn-mse=0.0094(0.0479) gcn-final-mse=0.0475(0.0626)
2020-08-04 17:32:18 9000-10553 loss=0.2183(0.1400+0.0783)-0.1881(0.1105+0.0777) sod-mse=0.0547(0.0423) gcn-mse=0.0685(0.0481) gcn-final-mse=0.0477(0.0628)
2020-08-04 17:36:22 10000-10553 loss=0.1841(0.1153+0.0687)-0.1889(0.1109+0.0780) sod-mse=0.0420(0.0424) gcn-mse=0.0565(0.0483) gcn-final-mse=0.0479(0.0630)

2020-08-04 17:38:38    0-5019 loss=1.1570(0.5608+0.5963)-1.1570(0.5608+0.5963) sod-mse=0.1176(0.1176) gcn-mse=0.1255(0.1255) gcn-final-mse=0.1199(0.1312)
2020-08-04 17:40:30 1000-5019 loss=0.0391(0.0321+0.0070)-0.3798(0.1825+0.1974) sod-mse=0.0060(0.0658) gcn-mse=0.0136(0.0723) gcn-final-mse=0.0724(0.0846)
2020-08-04 17:42:21 2000-5019 loss=1.2116(0.4506+0.7611)-0.3916(0.1856+0.2060) sod-mse=0.1063(0.0672) gcn-mse=0.1067(0.0736) gcn-final-mse=0.0737(0.0859)
2020-08-04 17:44:11 3000-5019 loss=0.0547(0.0402+0.0146)-0.3965(0.1875+0.2089) sod-mse=0.0070(0.0681) gcn-mse=0.0126(0.0746) gcn-final-mse=0.0747(0.0868)
2020-08-04 17:45:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 17:46:02 4000-5019 loss=0.1489(0.0902+0.0588)-0.3910(0.1861+0.2049) sod-mse=0.0297(0.0676) gcn-mse=0.0314(0.0742) gcn-final-mse=0.0743(0.0864)
2020-08-04 17:46:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 17:47:52 5000-5019 loss=0.7571(0.3129+0.4442)-0.3902(0.1860+0.2042) sod-mse=0.1254(0.0674) gcn-mse=0.1186(0.0742) gcn-final-mse=0.0742(0.0863)
2020-08-04 17:47:54 E: 8, Train sod-mae-score=0.0427-0.9453 gcn-mae-score=0.0485-0.9145 gcn-final-mse-score=0.0481-0.9172(0.0632/0.9172) loss=0.1896(0.1112+0.0784)
2020-08-04 17:47:54 E: 8, Test  sod-mae-score=0.0674-0.8148 gcn-mae-score=0.0742-0.7599 gcn-final-mse-score=0.0742-0.7661(0.0863/0.7661) loss=0.3897(0.1858+0.2039)

2020-08-04 17:47:54 Start Epoch 9
2020-08-04 17:47:54 Epoch:09,lr=0.0001
2020-08-04 17:47:55    0-10553 loss=0.0397(0.0326+0.0071)-0.0397(0.0326+0.0071) sod-mse=0.0036(0.0036) gcn-mse=0.0081(0.0081) gcn-final-mse=0.0090(0.0240)
2020-08-04 17:52:00 1000-10553 loss=0.0634(0.0472+0.0162)-0.1572(0.0954+0.0617) sod-mse=0.0113(0.0337) gcn-mse=0.0165(0.0405) gcn-final-mse=0.0401(0.0551)
2020-08-04 17:56:04 2000-10553 loss=0.3219(0.1685+0.1535)-0.1661(0.0995+0.0665) sod-mse=0.0516(0.0362) gcn-mse=0.0487(0.0422) gcn-final-mse=0.0418(0.0571)
2020-08-04 18:00:09 3000-10553 loss=0.1329(0.0964+0.0365)-0.1670(0.1001+0.0669) sod-mse=0.0167(0.0363) gcn-mse=0.0395(0.0423) gcn-final-mse=0.0419(0.0571)
2020-08-04 18:04:12 4000-10553 loss=0.2014(0.1104+0.0909)-0.1724(0.1028+0.0696) sod-mse=0.0557(0.0380) gcn-mse=0.0540(0.0439) gcn-final-mse=0.0435(0.0588)
2020-08-04 18:07:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 18:08:16 5000-10553 loss=0.0332(0.0258+0.0074)-0.1722(0.1028+0.0694) sod-mse=0.0037(0.0378) gcn-mse=0.0059(0.0437) gcn-final-mse=0.0433(0.0586)
2020-08-04 18:12:19 6000-10553 loss=0.1553(0.0990+0.0563)-0.1731(0.1032+0.0699) sod-mse=0.0289(0.0381) gcn-mse=0.0379(0.0439) gcn-final-mse=0.0435(0.0588)
2020-08-04 18:16:23 7000-10553 loss=0.1463(0.0873+0.0590)-0.1731(0.1032+0.0699) sod-mse=0.0454(0.0381) gcn-mse=0.0532(0.0439) gcn-final-mse=0.0434(0.0588)
2020-08-04 18:18:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 18:20:26 8000-10553 loss=0.1341(0.0852+0.0489)-0.1745(0.1039+0.0706) sod-mse=0.0325(0.0384) gcn-mse=0.0321(0.0441) gcn-final-mse=0.0437(0.0590)
2020-08-04 18:22:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 18:24:29 9000-10553 loss=0.1989(0.1247+0.0742)-0.1747(0.1038+0.0709) sod-mse=0.0513(0.0385) gcn-mse=0.0577(0.0441) gcn-final-mse=0.0437(0.0590)
2020-08-04 18:28:33 10000-10553 loss=0.0867(0.0673+0.0194)-0.1757(0.1043+0.0714) sod-mse=0.0132(0.0387) gcn-mse=0.0222(0.0444) gcn-final-mse=0.0439(0.0592)

2020-08-04 18:30:50    0-5019 loss=0.9158(0.5315+0.3844)-0.9158(0.5315+0.3844) sod-mse=0.1136(0.1136) gcn-mse=0.1321(0.1321) gcn-final-mse=0.1244(0.1344)
2020-08-04 18:32:42 1000-5019 loss=0.0845(0.0563+0.0281)-0.3545(0.1898+0.1647) sod-mse=0.0257(0.0854) gcn-mse=0.0356(0.0838) gcn-final-mse=0.0841(0.0980)
2020-08-04 18:34:32 2000-5019 loss=0.3720(0.2016+0.1704)-0.3555(0.1899+0.1656) sod-mse=0.0868(0.0862) gcn-mse=0.0782(0.0844) gcn-final-mse=0.0846(0.0985)
2020-08-04 18:36:23 3000-5019 loss=0.0547(0.0396+0.0151)-0.3628(0.1929+0.1698) sod-mse=0.0093(0.0878) gcn-mse=0.0135(0.0859) gcn-final-mse=0.0861(0.1000)
2020-08-04 18:37:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 18:38:14 4000-5019 loss=0.1229(0.0787+0.0442)-0.3622(0.1929+0.1693) sod-mse=0.0259(0.0875) gcn-mse=0.0244(0.0857) gcn-final-mse=0.0859(0.0997)
2020-08-04 18:38:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 18:40:05 5000-5019 loss=0.2815(0.1583+0.1232)-0.3618(0.1928+0.1690) sod-mse=0.0863(0.0876) gcn-mse=0.0880(0.0859) gcn-final-mse=0.0861(0.0998)
2020-08-04 18:40:07 E: 9, Train sod-mae-score=0.0391-0.9488 gcn-mae-score=0.0447-0.9179 gcn-final-mse-score=0.0442-0.9207(0.0595/0.9207) loss=0.1769(0.1049+0.0720)
2020-08-04 18:40:07 E: 9, Test  sod-mae-score=0.0876-0.8004 gcn-mae-score=0.0860-0.7435 gcn-final-mse-score=0.0861-0.7495(0.0999/0.7495) loss=0.3619(0.1928+0.1691)

2020-08-04 18:40:07 Start Epoch 10
2020-08-04 18:40:07 Epoch:10,lr=0.0001
2020-08-04 18:40:08    0-10553 loss=0.2064(0.1153+0.0911)-0.2064(0.1153+0.0911) sod-mse=0.0634(0.0634) gcn-mse=0.0551(0.0551) gcn-final-mse=0.0622(0.0883)
2020-08-04 18:44:12 1000-10553 loss=0.1225(0.0759+0.0466)-0.1682(0.1008+0.0674) sod-mse=0.0263(0.0367) gcn-mse=0.0232(0.0430) gcn-final-mse=0.0425(0.0579)
2020-08-04 18:45:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 18:48:16 2000-10553 loss=0.0502(0.0380+0.0122)-0.1699(0.1015+0.0684) sod-mse=0.0064(0.0372) gcn-mse=0.0112(0.0433) gcn-final-mse=0.0428(0.0580)
2020-08-04 18:52:20 3000-10553 loss=0.2497(0.1206+0.1291)-0.1661(0.0995+0.0666) sod-mse=0.0509(0.0360) gcn-mse=0.0538(0.0420) gcn-final-mse=0.0414(0.0567)
2020-08-04 18:56:24 4000-10553 loss=0.2344(0.1293+0.1051)-0.1645(0.0988+0.0657) sod-mse=0.0584(0.0355) gcn-mse=0.0615(0.0415) gcn-final-mse=0.0410(0.0563)
2020-08-04 19:00:29 5000-10553 loss=0.0592(0.0450+0.0142)-0.1639(0.0985+0.0654) sod-mse=0.0084(0.0353) gcn-mse=0.0096(0.0412) gcn-final-mse=0.0407(0.0561)
2020-08-04 19:04:32 6000-10553 loss=0.1479(0.0991+0.0488)-0.1662(0.0995+0.0667) sod-mse=0.0320(0.0360) gcn-mse=0.0345(0.0417) gcn-final-mse=0.0412(0.0566)
2020-08-04 19:08:36 7000-10553 loss=0.2820(0.1638+0.1182)-0.1673(0.1002+0.0671) sod-mse=0.0649(0.0363) gcn-mse=0.0781(0.0420) gcn-final-mse=0.0416(0.0570)
2020-08-04 19:12:40 8000-10553 loss=0.3008(0.1932+0.1076)-0.1686(0.1008+0.0679) sod-mse=0.0634(0.0366) gcn-mse=0.0773(0.0423) gcn-final-mse=0.0419(0.0572)
2020-08-04 19:14:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 19:16:44 9000-10553 loss=0.0926(0.0566+0.0360)-0.1694(0.1011+0.0683) sod-mse=0.0253(0.0369) gcn-mse=0.0346(0.0425) gcn-final-mse=0.0420(0.0574)
2020-08-04 19:20:46 10000-10553 loss=0.2476(0.1583+0.0893)-0.1691(0.1009+0.0682) sod-mse=0.0512(0.0369) gcn-mse=0.0722(0.0424) gcn-final-mse=0.0419(0.0573)
2020-08-04 19:21:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg

2020-08-04 19:23:03    0-5019 loss=0.7665(0.3986+0.3679)-0.7665(0.3986+0.3679) sod-mse=0.0925(0.0925) gcn-mse=0.1005(0.1005) gcn-final-mse=0.0937(0.1045)
2020-08-04 19:24:54 1000-5019 loss=0.0559(0.0412+0.0147)-0.3495(0.1837+0.1658) sod-mse=0.0127(0.0687) gcn-mse=0.0202(0.0711) gcn-final-mse=0.0712(0.0838)
2020-08-04 19:26:46 2000-5019 loss=0.8142(0.4043+0.4099)-0.3554(0.1862+0.1692) sod-mse=0.1001(0.0697) gcn-mse=0.0981(0.0722) gcn-final-mse=0.0723(0.0848)
2020-08-04 19:28:37 3000-5019 loss=0.0532(0.0390+0.0142)-0.3562(0.1866+0.1697) sod-mse=0.0075(0.0702) gcn-mse=0.0114(0.0730) gcn-final-mse=0.0731(0.0855)
2020-08-04 19:29:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 19:30:30 4000-5019 loss=0.1971(0.1179+0.0792)-0.3528(0.1852+0.1676) sod-mse=0.0442(0.0699) gcn-mse=0.0495(0.0728) gcn-final-mse=0.0728(0.0853)
2020-08-04 19:31:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 19:32:22 5000-5019 loss=1.0381(0.4503+0.5879)-0.3527(0.1852+0.1675) sod-mse=0.0981(0.0699) gcn-mse=0.1007(0.0728) gcn-final-mse=0.0729(0.0853)
2020-08-04 19:32:24 E:10, Train sod-mae-score=0.0368-0.9514 gcn-mae-score=0.0423-0.9213 gcn-final-mse-score=0.0418-0.9240(0.0572/0.9240) loss=0.1689(0.1008+0.0681)
2020-08-04 19:32:24 E:10, Test  sod-mae-score=0.0699-0.8217 gcn-mae-score=0.0728-0.7656 gcn-final-mse-score=0.0728-0.7716(0.0852/0.7716) loss=0.3523(0.1850+0.1672)

2020-08-04 19:32:24 Start Epoch 11
2020-08-04 19:32:24 Epoch:11,lr=0.0001
2020-08-04 19:32:26    0-10553 loss=0.1358(0.0817+0.0540)-0.1358(0.0817+0.0540) sod-mse=0.0374(0.0374) gcn-mse=0.0311(0.0311) gcn-final-mse=0.0376(0.0545)
2020-08-04 19:36:29 1000-10553 loss=0.1214(0.0787+0.0427)-0.1556(0.0942+0.0614) sod-mse=0.0237(0.0331) gcn-mse=0.0323(0.0386) gcn-final-mse=0.0381(0.0538)
2020-08-04 19:38:06 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 19:40:34 2000-10553 loss=0.1851(0.1036+0.0815)-0.1529(0.0928+0.0601) sod-mse=0.0299(0.0323) gcn-mse=0.0360(0.0378) gcn-final-mse=0.0373(0.0527)
2020-08-04 19:44:37 3000-10553 loss=1.0875(0.5386+0.5490)-0.1601(0.0963+0.0637) sod-mse=0.2664(0.0341) gcn-mse=0.2555(0.0395) gcn-final-mse=0.0390(0.0545)
2020-08-04 19:48:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 19:48:40 4000-10553 loss=0.1395(0.0926+0.0468)-0.1594(0.0961+0.0633) sod-mse=0.0220(0.0340) gcn-mse=0.0256(0.0394) gcn-final-mse=0.0389(0.0544)
2020-08-04 19:52:44 5000-10553 loss=0.2756(0.1503+0.1253)-0.1593(0.0961+0.0632) sod-mse=0.0456(0.0340) gcn-mse=0.0485(0.0394) gcn-final-mse=0.0389(0.0544)
2020-08-04 19:56:47 6000-10553 loss=0.2736(0.1766+0.0970)-0.1588(0.0959+0.0629) sod-mse=0.0506(0.0338) gcn-mse=0.0644(0.0394) gcn-final-mse=0.0389(0.0544)
2020-08-04 20:00:50 7000-10553 loss=0.1271(0.0821+0.0450)-0.1611(0.0970+0.0640) sod-mse=0.0279(0.0346) gcn-mse=0.0328(0.0400) gcn-final-mse=0.0395(0.0550)
2020-08-04 20:04:54 8000-10553 loss=0.0776(0.0452+0.0324)-0.1618(0.0974+0.0644) sod-mse=0.0255(0.0347) gcn-mse=0.0260(0.0401) gcn-final-mse=0.0396(0.0551)
2020-08-04 20:08:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 20:09:00 9000-10553 loss=0.1431(0.0849+0.0581)-0.1616(0.0973+0.0643) sod-mse=0.0354(0.0346) gcn-mse=0.0360(0.0401) gcn-final-mse=0.0396(0.0550)
2020-08-04 20:13:04 10000-10553 loss=0.2591(0.1420+0.1170)-0.1627(0.0978+0.0649) sod-mse=0.0287(0.0350) gcn-mse=0.0258(0.0404) gcn-final-mse=0.0399(0.0553)

2020-08-04 20:15:18    0-5019 loss=0.9416(0.5180+0.4236)-0.9416(0.5180+0.4236) sod-mse=0.0941(0.0941) gcn-mse=0.1002(0.1002) gcn-final-mse=0.0931(0.1032)
2020-08-04 20:17:10 1000-5019 loss=0.0490(0.0374+0.0116)-0.4040(0.2143+0.1897) sod-mse=0.0103(0.0879) gcn-mse=0.0183(0.0936) gcn-final-mse=0.0933(0.1064)
2020-08-04 20:19:02 2000-5019 loss=0.5985(0.3143+0.2842)-0.4049(0.2148+0.1900) sod-mse=0.0986(0.0879) gcn-mse=0.0983(0.0937) gcn-final-mse=0.0934(0.1064)
2020-08-04 20:20:53 3000-5019 loss=0.0497(0.0365+0.0133)-0.4104(0.2172+0.1931) sod-mse=0.0069(0.0891) gcn-mse=0.0094(0.0949) gcn-final-mse=0.0946(0.1077)
2020-08-04 20:21:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 20:22:43 4000-5019 loss=0.1373(0.0905+0.0468)-0.4097(0.2171+0.1926) sod-mse=0.0233(0.0892) gcn-mse=0.0260(0.0953) gcn-final-mse=0.0949(0.1080)
2020-08-04 20:23:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 20:24:33 5000-5019 loss=0.4821(0.2332+0.2490)-0.4088(0.2167+0.1920) sod-mse=0.1150(0.0892) gcn-mse=0.1039(0.0954) gcn-final-mse=0.0951(0.1081)
2020-08-04 20:24:35 E:11, Train sod-mae-score=0.0351-0.9537 gcn-mae-score=0.0405-0.9236 gcn-final-mse-score=0.0400-0.9263(0.0554/0.9263) loss=0.1630(0.0979+0.0651)
2020-08-04 20:24:35 E:11, Test  sod-mae-score=0.0893-0.7840 gcn-mae-score=0.0955-0.7273 gcn-final-mse-score=0.0951-0.7335(0.1082/0.7335) loss=0.4090(0.2168+0.1922)

2020-08-04 20:24:35 Start Epoch 12
2020-08-04 20:24:35 Epoch:12,lr=0.0001
2020-08-04 20:24:37    0-10553 loss=0.1904(0.1233+0.0671)-0.1904(0.1233+0.0671) sod-mse=0.0506(0.0506) gcn-mse=0.0751(0.0751) gcn-final-mse=0.0771(0.0864)
2020-08-04 20:28:43 1000-10553 loss=0.4819(0.2444+0.2376)-0.1377(0.0856+0.0521) sod-mse=0.1369(0.0275) gcn-mse=0.1221(0.0337) gcn-final-mse=0.0330(0.0483)
2020-08-04 20:30:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 20:32:46 2000-10553 loss=0.1562(0.1121+0.0441)-0.1422(0.0878+0.0544) sod-mse=0.0286(0.0291) gcn-mse=0.0439(0.0351) gcn-final-mse=0.0345(0.0498)
2020-08-04 20:36:51 3000-10553 loss=0.1311(0.0843+0.0468)-0.1454(0.0893+0.0561) sod-mse=0.0289(0.0301) gcn-mse=0.0400(0.0358) gcn-final-mse=0.0352(0.0506)
2020-08-04 20:40:54 4000-10553 loss=0.0132(0.0096+0.0036)-0.1451(0.0893+0.0558) sod-mse=0.0019(0.0300) gcn-mse=0.0035(0.0357) gcn-final-mse=0.0351(0.0506)
2020-08-04 20:44:59 5000-10553 loss=0.0525(0.0409+0.0117)-0.1465(0.0899+0.0566) sod-mse=0.0104(0.0304) gcn-mse=0.0234(0.0360) gcn-final-mse=0.0354(0.0509)
2020-08-04 20:49:02 6000-10553 loss=0.1234(0.0871+0.0363)-0.1471(0.0902+0.0570) sod-mse=0.0266(0.0306) gcn-mse=0.0405(0.0361) gcn-final-mse=0.0356(0.0511)
2020-08-04 20:52:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 20:53:07 7000-10553 loss=0.0404(0.0281+0.0123)-0.1487(0.0909+0.0578) sod-mse=0.0096(0.0310) gcn-mse=0.0115(0.0365) gcn-final-mse=0.0360(0.0514)
2020-08-04 20:56:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 20:57:09 8000-10553 loss=0.1629(0.0902+0.0727)-0.1499(0.0914+0.0585) sod-mse=0.0311(0.0313) gcn-mse=0.0320(0.0367) gcn-final-mse=0.0361(0.0516)
2020-08-04 21:01:12 9000-10553 loss=0.2547(0.1291+0.1255)-0.1512(0.0920+0.0592) sod-mse=0.0704(0.0317) gcn-mse=0.0624(0.0370) gcn-final-mse=0.0364(0.0519)
2020-08-04 21:05:15 10000-10553 loss=0.2945(0.1751+0.1194)-0.1511(0.0920+0.0591) sod-mse=0.0803(0.0317) gcn-mse=0.0911(0.0370) gcn-final-mse=0.0365(0.0519)

2020-08-04 21:07:31    0-5019 loss=1.0134(0.5738+0.4396)-1.0134(0.5738+0.4396) sod-mse=0.1249(0.1249) gcn-mse=0.1250(0.1250) gcn-final-mse=0.1176(0.1292)
2020-08-04 21:09:23 1000-5019 loss=0.0526(0.0361+0.0165)-0.3547(0.1877+0.1670) sod-mse=0.0150(0.0755) gcn-mse=0.0167(0.0731) gcn-final-mse=0.0732(0.0861)
2020-08-04 21:11:14 2000-5019 loss=0.6686(0.3443+0.3242)-0.3565(0.1885+0.1680) sod-mse=0.1056(0.0761) gcn-mse=0.0983(0.0738) gcn-final-mse=0.0739(0.0867)
2020-08-04 21:13:05 3000-5019 loss=0.0494(0.0358+0.0136)-0.3579(0.1893+0.1686) sod-mse=0.0075(0.0767) gcn-mse=0.0085(0.0745) gcn-final-mse=0.0746(0.0874)
2020-08-04 21:14:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 21:14:56 4000-5019 loss=0.1955(0.1171+0.0785)-0.3537(0.1876+0.1661) sod-mse=0.0488(0.0763) gcn-mse=0.0534(0.0742) gcn-final-mse=0.0743(0.0871)
2020-08-04 21:15:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 21:16:46 5000-5019 loss=0.5019(0.2381+0.2638)-0.3529(0.1874+0.1655) sod-mse=0.0982(0.0763) gcn-mse=0.0860(0.0742) gcn-final-mse=0.0742(0.0870)
2020-08-04 21:16:48 E:12, Train sod-mae-score=0.0321-0.9568 gcn-mae-score=0.0374-0.9262 gcn-final-mse-score=0.0369-0.9289(0.0524/0.9289) loss=0.1525(0.0927+0.0599)
2020-08-04 21:16:48 E:12, Test  sod-mae-score=0.0763-0.8128 gcn-mae-score=0.0742-0.7558 gcn-final-mse-score=0.0742-0.7620(0.0870/0.7620) loss=0.3525(0.1872+0.1653)

2020-08-04 21:16:48 Start Epoch 13
2020-08-04 21:16:48 Epoch:13,lr=0.0001
2020-08-04 21:16:50    0-10553 loss=0.5795(0.3280+0.2515)-0.5795(0.3280+0.2515) sod-mse=0.1341(0.1341) gcn-mse=0.1565(0.1565) gcn-final-mse=0.1608(0.1725)
2020-08-04 21:20:56 1000-10553 loss=0.4027(0.2223+0.1804)-0.1475(0.0901+0.0574) sod-mse=0.0679(0.0305) gcn-mse=0.0761(0.0355) gcn-final-mse=0.0350(0.0508)
2020-08-04 21:22:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 21:25:01 2000-10553 loss=0.0540(0.0402+0.0138)-0.1403(0.0867+0.0536) sod-mse=0.0079(0.0286) gcn-mse=0.0131(0.0339) gcn-final-mse=0.0334(0.0491)
2020-08-04 21:29:05 3000-10553 loss=0.0618(0.0481+0.0137)-0.1404(0.0867+0.0537) sod-mse=0.0082(0.0286) gcn-mse=0.0119(0.0340) gcn-final-mse=0.0335(0.0491)
2020-08-04 21:33:07 4000-10553 loss=0.1332(0.0944+0.0388)-0.1430(0.0879+0.0551) sod-mse=0.0269(0.0293) gcn-mse=0.0497(0.0346) gcn-final-mse=0.0340(0.0496)
2020-08-04 21:35:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 21:37:12 5000-10553 loss=0.0842(0.0532+0.0310)-0.1437(0.0882+0.0555) sod-mse=0.0173(0.0295) gcn-mse=0.0150(0.0348) gcn-final-mse=0.0343(0.0498)
2020-08-04 21:41:16 6000-10553 loss=0.1569(0.1125+0.0445)-0.1448(0.0889+0.0559) sod-mse=0.0235(0.0297) gcn-mse=0.0484(0.0350) gcn-final-mse=0.0345(0.0501)
2020-08-04 21:45:19 7000-10553 loss=0.0409(0.0320+0.0090)-0.1469(0.0898+0.0571) sod-mse=0.0047(0.0304) gcn-mse=0.0072(0.0356) gcn-final-mse=0.0350(0.0506)
2020-08-04 21:49:21 8000-10553 loss=0.0323(0.0218+0.0105)-0.1481(0.0903+0.0578) sod-mse=0.0043(0.0308) gcn-mse=0.0090(0.0360) gcn-final-mse=0.0355(0.0510)
2020-08-04 21:53:24 9000-10553 loss=0.1325(0.0902+0.0423)-0.1490(0.0908+0.0582) sod-mse=0.0302(0.0311) gcn-mse=0.0424(0.0363) gcn-final-mse=0.0358(0.0513)
2020-08-04 21:54:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 21:57:27 10000-10553 loss=0.0728(0.0522+0.0206)-0.1492(0.0909+0.0582) sod-mse=0.0109(0.0311) gcn-mse=0.0165(0.0363) gcn-final-mse=0.0358(0.0513)

2020-08-04 21:59:43    0-5019 loss=0.9574(0.5410+0.4164)-0.9574(0.5410+0.4164) sod-mse=0.0887(0.0887) gcn-mse=0.0954(0.0954) gcn-final-mse=0.0880(0.1010)
2020-08-04 22:01:35 1000-5019 loss=0.0714(0.0498+0.0216)-0.3583(0.1814+0.1769) sod-mse=0.0179(0.0660) gcn-mse=0.0258(0.0686) gcn-final-mse=0.0686(0.0821)
2020-08-04 22:03:26 2000-5019 loss=0.7376(0.3783+0.3593)-0.3634(0.1838+0.1796) sod-mse=0.0919(0.0667) gcn-mse=0.0889(0.0694) gcn-final-mse=0.0694(0.0828)
2020-08-04 22:05:17 3000-5019 loss=0.0471(0.0351+0.0121)-0.3690(0.1860+0.1830) sod-mse=0.0057(0.0677) gcn-mse=0.0085(0.0702) gcn-final-mse=0.0703(0.0836)
2020-08-04 22:06:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 22:07:08 4000-5019 loss=0.1118(0.0733+0.0385)-0.3652(0.1847+0.1806) sod-mse=0.0187(0.0670) gcn-mse=0.0174(0.0699) gcn-final-mse=0.0699(0.0832)
2020-08-04 22:07:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 22:08:59 5000-5019 loss=0.9937(0.4413+0.5524)-0.3635(0.1841+0.1794) sod-mse=0.1389(0.0670) gcn-mse=0.1313(0.0700) gcn-final-mse=0.0699(0.0833)
2020-08-04 22:09:01 E:13, Train sod-mae-score=0.0311-0.9582 gcn-mae-score=0.0363-0.9271 gcn-final-mse-score=0.0358-0.9298(0.0514/0.9298) loss=0.1489(0.0908+0.0581)
2020-08-04 22:09:01 E:13, Test  sod-mae-score=0.0671-0.8197 gcn-mae-score=0.0700-0.7670 gcn-final-mse-score=0.0699-0.7726(0.0833/0.7726) loss=0.3634(0.1841+0.1793)

2020-08-04 22:09:01 Start Epoch 14
2020-08-04 22:09:01 Epoch:14,lr=0.0001
2020-08-04 22:09:02    0-10553 loss=0.0492(0.0382+0.0110)-0.0492(0.0382+0.0110) sod-mse=0.0056(0.0056) gcn-mse=0.0088(0.0088) gcn-final-mse=0.0080(0.0209)
2020-08-04 22:13:06 1000-10553 loss=0.0411(0.0310+0.0102)-0.1259(0.0795+0.0464) sod-mse=0.0063(0.0244) gcn-mse=0.0087(0.0297) gcn-final-mse=0.0292(0.0449)
2020-08-04 22:17:08 2000-10553 loss=0.0637(0.0448+0.0189)-0.1316(0.0823+0.0493) sod-mse=0.0084(0.0262) gcn-mse=0.0101(0.0315) gcn-final-mse=0.0310(0.0467)
2020-08-04 22:17:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 22:21:11 3000-10553 loss=0.1707(0.1076+0.0631)-0.1355(0.0845+0.0510) sod-mse=0.0433(0.0272) gcn-mse=0.0553(0.0327) gcn-final-mse=0.0321(0.0477)
2020-08-04 22:21:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 22:25:15 4000-10553 loss=0.1624(0.1054+0.0570)-0.1375(0.0853+0.0522) sod-mse=0.0335(0.0277) gcn-mse=0.0303(0.0329) gcn-final-mse=0.0323(0.0480)
2020-08-04 22:29:18 5000-10553 loss=0.0453(0.0312+0.0141)-0.1362(0.0847+0.0515) sod-mse=0.0099(0.0273) gcn-mse=0.0102(0.0327) gcn-final-mse=0.0321(0.0477)
2020-08-04 22:33:23 6000-10553 loss=0.1829(0.1231+0.0598)-0.1371(0.0851+0.0520) sod-mse=0.0370(0.0276) gcn-mse=0.0512(0.0330) gcn-final-mse=0.0324(0.0479)
2020-08-04 22:37:28 7000-10553 loss=0.1891(0.1197+0.0694)-0.1382(0.0856+0.0527) sod-mse=0.0417(0.0279) gcn-mse=0.0633(0.0332) gcn-final-mse=0.0326(0.0481)
2020-08-04 22:41:31 8000-10553 loss=0.2855(0.1423+0.1431)-0.1380(0.0855+0.0525) sod-mse=0.0534(0.0279) gcn-mse=0.0617(0.0332) gcn-final-mse=0.0326(0.0481)
2020-08-04 22:45:37 9000-10553 loss=0.1597(0.1026+0.0571)-0.1384(0.0856+0.0527) sod-mse=0.0381(0.0280) gcn-mse=0.0346(0.0333) gcn-final-mse=0.0327(0.0483)
2020-08-04 22:49:41 10000-10553 loss=0.1558(0.0839+0.0720)-0.1391(0.0861+0.0531) sod-mse=0.0477(0.0282) gcn-mse=0.0463(0.0335) gcn-final-mse=0.0330(0.0485)
2020-08-04 22:50:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg

2020-08-04 22:51:57    0-5019 loss=0.8858(0.5040+0.3818)-0.8858(0.5040+0.3818) sod-mse=0.0872(0.0872) gcn-mse=0.0992(0.0992) gcn-final-mse=0.0917(0.1049)
2020-08-04 22:53:49 1000-5019 loss=0.0395(0.0293+0.0101)-0.3431(0.1765+0.1666) sod-mse=0.0087(0.0702) gcn-mse=0.0112(0.0695) gcn-final-mse=0.0694(0.0831)
2020-08-04 22:55:40 2000-5019 loss=0.6678(0.3516+0.3162)-0.3534(0.1811+0.1723) sod-mse=0.0851(0.0719) gcn-mse=0.0868(0.0710) gcn-final-mse=0.0709(0.0844)
2020-08-04 22:57:30 3000-5019 loss=0.0500(0.0367+0.0133)-0.3566(0.1824+0.1742) sod-mse=0.0064(0.0727) gcn-mse=0.0092(0.0717) gcn-final-mse=0.0716(0.0851)
2020-08-04 22:58:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 22:59:21 4000-5019 loss=0.1151(0.0771+0.0380)-0.3522(0.1806+0.1716) sod-mse=0.0187(0.0720) gcn-mse=0.0179(0.0713) gcn-final-mse=0.0712(0.0847)
2020-08-04 22:59:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 23:01:11 5000-5019 loss=0.4067(0.2115+0.1952)-0.3524(0.1810+0.1714) sod-mse=0.1012(0.0722) gcn-mse=0.0953(0.0716) gcn-final-mse=0.0714(0.0849)
2020-08-04 23:01:13 E:14, Train sod-mae-score=0.0284-0.9615 gcn-mae-score=0.0337-0.9308 gcn-final-mse-score=0.0331-0.9336(0.0487/0.9336) loss=0.1398(0.0864+0.0534)
2020-08-04 23:01:13 E:14, Test  sod-mae-score=0.0722-0.8112 gcn-mae-score=0.0716-0.7584 gcn-final-mse-score=0.0714-0.7641(0.0849/0.7641) loss=0.3522(0.1809+0.1713)

2020-08-04 23:01:13 Start Epoch 15
2020-08-04 23:01:13 Epoch:15,lr=0.0001
2020-08-04 23:01:14    0-10553 loss=0.0279(0.0169+0.0110)-0.0279(0.0169+0.0110) sod-mse=0.0090(0.0090) gcn-mse=0.0070(0.0070) gcn-final-mse=0.0076(0.0149)
2020-08-04 23:05:18 1000-10553 loss=0.1300(0.0858+0.0442)-0.1245(0.0787+0.0459) sod-mse=0.0288(0.0241) gcn-mse=0.0315(0.0289) gcn-final-mse=0.0283(0.0442)
2020-08-04 23:09:22 2000-10553 loss=0.0929(0.0573+0.0356)-0.1265(0.0795+0.0470) sod-mse=0.0199(0.0244) gcn-mse=0.0245(0.0295) gcn-final-mse=0.0288(0.0445)
2020-08-04 23:12:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-04 23:13:25 3000-10553 loss=0.0722(0.0521+0.0201)-0.1332(0.0829+0.0503) sod-mse=0.0107(0.0265) gcn-mse=0.0152(0.0315) gcn-final-mse=0.0309(0.0466)
2020-08-04 23:17:30 4000-10553 loss=0.2796(0.1523+0.1273)-0.1330(0.0827+0.0502) sod-mse=0.0542(0.0264) gcn-mse=0.0649(0.0313) gcn-final-mse=0.0307(0.0464)
2020-08-04 23:18:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-04 23:21:35 5000-10553 loss=0.1126(0.0731+0.0395)-0.1329(0.0827+0.0502) sod-mse=0.0266(0.0265) gcn-mse=0.0268(0.0315) gcn-final-mse=0.0308(0.0465)
2020-08-04 23:25:39 6000-10553 loss=0.0923(0.0662+0.0260)-0.1330(0.0828+0.0502) sod-mse=0.0138(0.0265) gcn-mse=0.0262(0.0315) gcn-final-mse=0.0309(0.0466)
2020-08-04 23:29:43 7000-10553 loss=0.1042(0.0642+0.0399)-0.1349(0.0838+0.0512) sod-mse=0.0233(0.0271) gcn-mse=0.0236(0.0320) gcn-final-mse=0.0315(0.0471)
2020-08-04 23:33:46 8000-10553 loss=0.4765(0.2391+0.2374)-0.1353(0.0840+0.0514) sod-mse=0.0329(0.0273) gcn-mse=0.0354(0.0322) gcn-final-mse=0.0316(0.0473)
2020-08-04 23:37:50 9000-10553 loss=0.1075(0.0498+0.0577)-0.1366(0.0846+0.0520) sod-mse=0.0179(0.0276) gcn-mse=0.0164(0.0325) gcn-final-mse=0.0319(0.0476)
2020-08-04 23:40:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-04 23:41:55 10000-10553 loss=0.2070(0.1178+0.0892)-0.1360(0.0843+0.0516) sod-mse=0.0375(0.0274) gcn-mse=0.0366(0.0323) gcn-final-mse=0.0318(0.0475)

2020-08-04 23:44:12    0-5019 loss=0.9516(0.5462+0.4054)-0.9516(0.5462+0.4054) sod-mse=0.0886(0.0886) gcn-mse=0.0995(0.0995) gcn-final-mse=0.0924(0.1061)
2020-08-04 23:46:04 1000-5019 loss=0.0417(0.0323+0.0094)-0.3316(0.1730+0.1587) sod-mse=0.0083(0.0661) gcn-mse=0.0137(0.0691) gcn-final-mse=0.0691(0.0825)
2020-08-04 23:47:55 2000-5019 loss=0.7017(0.3734+0.3283)-0.3406(0.1774+0.1632) sod-mse=0.0909(0.0676) gcn-mse=0.0923(0.0709) gcn-final-mse=0.0708(0.0841)
2020-08-04 23:49:46 3000-5019 loss=0.0518(0.0374+0.0143)-0.3451(0.1796+0.1655) sod-mse=0.0066(0.0682) gcn-mse=0.0087(0.0718) gcn-final-mse=0.0718(0.0850)
2020-08-04 23:50:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-04 23:51:37 4000-5019 loss=0.1226(0.0834+0.0393)-0.3449(0.1799+0.1650) sod-mse=0.0215(0.0679) gcn-mse=0.0272(0.0717) gcn-final-mse=0.0716(0.0849)
2020-08-04 23:52:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-04 23:53:27 5000-5019 loss=0.5476(0.2710+0.2765)-0.3461(0.1805+0.1656) sod-mse=0.0830(0.0678) gcn-mse=0.0780(0.0718) gcn-final-mse=0.0716(0.0848)
2020-08-04 23:53:29 E:15, Train sod-mae-score=0.0274-0.9621 gcn-mae-score=0.0324-0.9317 gcn-final-mse-score=0.0318-0.9344(0.0475/0.9344) loss=0.1361(0.0844+0.0517)
2020-08-04 23:53:29 E:15, Test  sod-mae-score=0.0679-0.8255 gcn-mae-score=0.0718-0.7681 gcn-final-mse-score=0.0716-0.7740(0.0848/0.7740) loss=0.3460(0.1805+0.1655)

2020-08-04 23:53:29 Start Epoch 16
2020-08-04 23:53:29 Epoch:16,lr=0.0001
2020-08-04 23:53:30    0-10553 loss=0.2487(0.1522+0.0966)-0.2487(0.1522+0.0966) sod-mse=0.0629(0.0629) gcn-mse=0.0822(0.0822) gcn-final-mse=0.0730(0.0944)
2020-08-04 23:57:36 1000-10553 loss=0.1603(0.0988+0.0615)-0.1369(0.0851+0.0518) sod-mse=0.0310(0.0276) gcn-mse=0.0374(0.0326) gcn-final-mse=0.0320(0.0478)
2020-08-05 00:01:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 00:01:39 2000-10553 loss=0.0538(0.0378+0.0160)-0.1351(0.0841+0.0510) sod-mse=0.0119(0.0272) gcn-mse=0.0154(0.0324) gcn-final-mse=0.0319(0.0475)
2020-08-05 00:05:44 3000-10553 loss=0.1494(0.0862+0.0632)-0.1315(0.0822+0.0493) sod-mse=0.0468(0.0262) gcn-mse=0.0489(0.0312) gcn-final-mse=0.0306(0.0463)
2020-08-05 00:06:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 00:09:48 4000-10553 loss=0.1319(0.0807+0.0513)-0.1311(0.0821+0.0490) sod-mse=0.0360(0.0261) gcn-mse=0.0349(0.0313) gcn-final-mse=0.0306(0.0463)
2020-08-05 00:13:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 00:13:51 5000-10553 loss=0.1173(0.0741+0.0432)-0.1291(0.0812+0.0480) sod-mse=0.0243(0.0255) gcn-mse=0.0218(0.0307) gcn-final-mse=0.0301(0.0457)
2020-08-05 00:17:53 6000-10553 loss=0.0857(0.0681+0.0176)-0.1307(0.0819+0.0488) sod-mse=0.0118(0.0259) gcn-mse=0.0188(0.0310) gcn-final-mse=0.0304(0.0461)
2020-08-05 00:21:56 7000-10553 loss=1.5467(0.8058+0.7409)-0.1308(0.0821+0.0488) sod-mse=0.2136(0.0259) gcn-mse=0.2212(0.0311) gcn-final-mse=0.0305(0.0462)
2020-08-05 00:26:02 8000-10553 loss=0.0755(0.0543+0.0213)-0.1312(0.0823+0.0489) sod-mse=0.0097(0.0260) gcn-mse=0.0152(0.0313) gcn-final-mse=0.0307(0.0464)
2020-08-05 00:30:06 9000-10553 loss=0.3551(0.1481+0.2070)-0.1306(0.0820+0.0486) sod-mse=0.0820(0.0258) gcn-mse=0.0633(0.0311) gcn-final-mse=0.0305(0.0463)
2020-08-05 00:34:11 10000-10553 loss=0.3905(0.1974+0.1931)-0.1321(0.0828+0.0493) sod-mse=0.0776(0.0262) gcn-mse=0.0747(0.0315) gcn-final-mse=0.0309(0.0466)

2020-08-05 00:36:26    0-5019 loss=0.9333(0.5065+0.4268)-0.9333(0.5065+0.4268) sod-mse=0.1504(0.1504) gcn-mse=0.1336(0.1336) gcn-final-mse=0.1267(0.1388)
2020-08-05 00:38:18 1000-5019 loss=0.0745(0.0441+0.0304)-0.3618(0.1944+0.1674) sod-mse=0.0268(0.0769) gcn-mse=0.0247(0.0704) gcn-final-mse=0.0702(0.0839)
2020-08-05 00:40:08 2000-5019 loss=0.7133(0.3878+0.3255)-0.3635(0.1950+0.1686) sod-mse=0.1069(0.0777) gcn-mse=0.0890(0.0712) gcn-final-mse=0.0711(0.0847)
2020-08-05 00:41:59 3000-5019 loss=0.0508(0.0353+0.0155)-0.3645(0.1951+0.1694) sod-mse=0.0098(0.0781) gcn-mse=0.0097(0.0716) gcn-final-mse=0.0715(0.0852)
2020-08-05 00:43:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 00:43:49 4000-5019 loss=0.1340(0.0847+0.0492)-0.3645(0.1953+0.1692) sod-mse=0.0322(0.0781) gcn-mse=0.0247(0.0717) gcn-final-mse=0.0716(0.0852)
2020-08-05 00:44:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 00:45:39 5000-5019 loss=0.7076(0.3562+0.3514)-0.3661(0.1962+0.1700) sod-mse=0.1287(0.0783) gcn-mse=0.1205(0.0719) gcn-final-mse=0.0718(0.0854)
2020-08-05 00:45:41 E:16, Train sod-mae-score=0.0263-0.9634 gcn-mae-score=0.0315-0.9331 gcn-final-mse-score=0.0309-0.9358(0.0467/0.9358) loss=0.1325(0.0829+0.0496)
2020-08-05 00:45:41 E:16, Test  sod-mae-score=0.0783-0.8265 gcn-mae-score=0.0719-0.7706 gcn-final-mse-score=0.0718-0.7770(0.0853/0.7770) loss=0.3658(0.1960+0.1698)

2020-08-05 00:45:41 Start Epoch 17
2020-08-05 00:45:41 Epoch:17,lr=0.0001
2020-08-05 00:45:43    0-10553 loss=0.2392(0.1434+0.0958)-0.2392(0.1434+0.0958) sod-mse=0.0661(0.0661) gcn-mse=0.0639(0.0639) gcn-final-mse=0.0678(0.0842)
2020-08-05 00:49:47 1000-10553 loss=0.1101(0.0738+0.0363)-0.1339(0.0841+0.0498) sod-mse=0.0153(0.0270) gcn-mse=0.0200(0.0325) gcn-final-mse=0.0318(0.0479)
2020-08-05 00:53:51 2000-10553 loss=0.1031(0.0737+0.0294)-0.1293(0.0815+0.0478) sod-mse=0.0131(0.0254) gcn-mse=0.0176(0.0307) gcn-final-mse=0.0300(0.0459)
2020-08-05 00:55:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 00:57:54 3000-10553 loss=0.0453(0.0370+0.0083)-0.1285(0.0812+0.0473) sod-mse=0.0035(0.0251) gcn-mse=0.0057(0.0303) gcn-final-mse=0.0296(0.0457)
2020-08-05 01:01:58 4000-10553 loss=0.0761(0.0554+0.0207)-0.1346(0.0839+0.0507) sod-mse=0.0112(0.0270) gcn-mse=0.0190(0.0319) gcn-final-mse=0.0313(0.0472)
2020-08-05 01:06:01 5000-10553 loss=0.1585(0.1022+0.0564)-0.1325(0.0829+0.0496) sod-mse=0.0346(0.0264) gcn-mse=0.0337(0.0314) gcn-final-mse=0.0308(0.0467)
2020-08-05 01:10:04 6000-10553 loss=0.0885(0.0639+0.0247)-0.1304(0.0818+0.0486) sod-mse=0.0144(0.0258) gcn-mse=0.0217(0.0308) gcn-final-mse=0.0302(0.0461)
2020-08-05 01:11:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 01:14:09 7000-10553 loss=0.1305(0.0829+0.0476)-0.1302(0.0817+0.0486) sod-mse=0.0276(0.0258) gcn-mse=0.0336(0.0307) gcn-final-mse=0.0301(0.0460)
2020-08-05 01:18:12 8000-10553 loss=0.0789(0.0575+0.0213)-0.1287(0.0809+0.0478) sod-mse=0.0123(0.0254) gcn-mse=0.0129(0.0303) gcn-final-mse=0.0297(0.0455)
2020-08-05 01:22:15 9000-10553 loss=0.0174(0.0103+0.0071)-0.1291(0.0812+0.0480) sod-mse=0.0057(0.0255) gcn-mse=0.0044(0.0305) gcn-final-mse=0.0299(0.0457)
2020-08-05 01:26:19 10000-10553 loss=0.0909(0.0616+0.0293)-0.1290(0.0810+0.0479) sod-mse=0.0202(0.0255) gcn-mse=0.0244(0.0305) gcn-final-mse=0.0298(0.0456)
2020-08-05 01:26:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg

2020-08-05 01:28:35    0-5019 loss=0.9351(0.4593+0.4758)-0.9351(0.4593+0.4758) sod-mse=0.0993(0.0993) gcn-mse=0.1041(0.1041) gcn-final-mse=0.0965(0.1088)
2020-08-05 01:30:28 1000-5019 loss=0.0448(0.0363+0.0085)-0.3751(0.1804+0.1948) sod-mse=0.0073(0.0581) gcn-mse=0.0169(0.0607) gcn-final-mse=0.0606(0.0732)
2020-08-05 01:32:19 2000-5019 loss=1.0247(0.4623+0.5624)-0.3769(0.1813+0.1956) sod-mse=0.1004(0.0584) gcn-mse=0.0961(0.0610) gcn-final-mse=0.0609(0.0735)
2020-08-05 01:34:11 3000-5019 loss=0.0504(0.0363+0.0141)-0.3808(0.1822+0.1986) sod-mse=0.0073(0.0587) gcn-mse=0.0094(0.0614) gcn-final-mse=0.0613(0.0739)
2020-08-05 01:35:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 01:36:02 4000-5019 loss=0.1517(0.0919+0.0599)-0.3772(0.1813+0.1959) sod-mse=0.0291(0.0583) gcn-mse=0.0291(0.0612) gcn-final-mse=0.0611(0.0736)
2020-08-05 01:36:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 01:37:53 5000-5019 loss=1.1859(0.4762+0.7097)-0.3791(0.1821+0.1970) sod-mse=0.0905(0.0587) gcn-mse=0.0931(0.0616) gcn-final-mse=0.0614(0.0739)
2020-08-05 01:37:55 E:17, Train sod-mae-score=0.0254-0.9653 gcn-mae-score=0.0303-0.9351 gcn-final-mse-score=0.0297-0.9377(0.0455/0.9377) loss=0.1287(0.0809+0.0478)
2020-08-05 01:37:55 E:17, Test  sod-mae-score=0.0587-0.8388 gcn-mae-score=0.0616-0.7823 gcn-final-mse-score=0.0614-0.7884(0.0739/0.7884) loss=0.3787(0.1820+0.1968)

2020-08-05 01:37:55 Start Epoch 18
2020-08-05 01:37:55 Epoch:18,lr=0.0001
2020-08-05 01:37:56    0-10553 loss=0.0939(0.0636+0.0303)-0.0939(0.0636+0.0303) sod-mse=0.0182(0.0182) gcn-mse=0.0195(0.0195) gcn-final-mse=0.0204(0.0408)
2020-08-05 01:42:01 1000-10553 loss=0.0580(0.0398+0.0182)-0.1195(0.0766+0.0428) sod-mse=0.0099(0.0227) gcn-mse=0.0071(0.0278) gcn-final-mse=0.0271(0.0431)
2020-08-05 01:46:04 2000-10553 loss=0.0389(0.0283+0.0106)-0.1143(0.0740+0.0404) sod-mse=0.0062(0.0213) gcn-mse=0.0051(0.0263) gcn-final-mse=0.0257(0.0416)
2020-08-05 01:49:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 01:50:08 3000-10553 loss=0.1808(0.1176+0.0632)-0.1170(0.0751+0.0419) sod-mse=0.0387(0.0220) gcn-mse=0.0622(0.0269) gcn-final-mse=0.0263(0.0423)
2020-08-05 01:52:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 01:54:12 4000-10553 loss=0.0888(0.0665+0.0222)-0.1172(0.0751+0.0420) sod-mse=0.0115(0.0221) gcn-mse=0.0124(0.0270) gcn-final-mse=0.0264(0.0422)
2020-08-05 01:58:16 5000-10553 loss=0.0783(0.0540+0.0243)-0.1176(0.0754+0.0422) sod-mse=0.0138(0.0223) gcn-mse=0.0175(0.0272) gcn-final-mse=0.0266(0.0424)
2020-08-05 02:02:22 6000-10553 loss=0.1885(0.1167+0.0718)-0.1184(0.0757+0.0426) sod-mse=0.0457(0.0225) gcn-mse=0.0461(0.0274) gcn-final-mse=0.0268(0.0426)
2020-08-05 02:06:28 7000-10553 loss=0.0715(0.0504+0.0211)-0.1205(0.0767+0.0438) sod-mse=0.0114(0.0231) gcn-mse=0.0181(0.0279) gcn-final-mse=0.0273(0.0431)
2020-08-05 02:10:34 8000-10553 loss=0.1071(0.0745+0.0326)-0.1231(0.0780+0.0451) sod-mse=0.0191(0.0238) gcn-mse=0.0360(0.0286) gcn-final-mse=0.0280(0.0438)
2020-08-05 02:13:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 02:14:38 9000-10553 loss=0.2175(0.1599+0.0577)-0.1233(0.0782+0.0452) sod-mse=0.0392(0.0238) gcn-mse=0.0756(0.0287) gcn-final-mse=0.0280(0.0439)
2020-08-05 02:18:42 10000-10553 loss=0.1827(0.0977+0.0850)-0.1231(0.0780+0.0451) sod-mse=0.0357(0.0238) gcn-mse=0.0318(0.0286) gcn-final-mse=0.0280(0.0438)

2020-08-05 02:20:58    0-5019 loss=0.8427(0.4366+0.4061)-0.8427(0.4366+0.4061) sod-mse=0.0932(0.0932) gcn-mse=0.1014(0.1014) gcn-final-mse=0.0944(0.1078)
2020-08-05 02:22:49 1000-5019 loss=0.0402(0.0323+0.0080)-0.3376(0.1747+0.1629) sod-mse=0.0070(0.0591) gcn-mse=0.0138(0.0621) gcn-final-mse=0.0620(0.0755)
2020-08-05 02:24:40 2000-5019 loss=0.5893(0.2971+0.2922)-0.3377(0.1746+0.1630) sod-mse=0.0859(0.0597) gcn-mse=0.0868(0.0628) gcn-final-mse=0.0626(0.0761)
2020-08-05 02:26:31 3000-5019 loss=0.0504(0.0378+0.0126)-0.3421(0.1762+0.1659) sod-mse=0.0067(0.0605) gcn-mse=0.0108(0.0636) gcn-final-mse=0.0635(0.0769)
2020-08-05 02:27:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 02:28:21 4000-5019 loss=0.1313(0.0829+0.0484)-0.3401(0.1755+0.1646) sod-mse=0.0227(0.0602) gcn-mse=0.0236(0.0634) gcn-final-mse=0.0633(0.0767)
2020-08-05 02:28:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 02:30:12 5000-5019 loss=1.0743(0.4606+0.6136)-0.3393(0.1751+0.1643) sod-mse=0.0901(0.0603) gcn-mse=0.0918(0.0635) gcn-final-mse=0.0633(0.0767)
2020-08-05 02:30:13 E:18, Train sod-mae-score=0.0239-0.9667 gcn-mae-score=0.0287-0.9367 gcn-final-mse-score=0.0281-0.9394(0.0439/0.9394) loss=0.1233(0.0781+0.0451)
2020-08-05 02:30:13 E:18, Test  sod-mae-score=0.0603-0.8320 gcn-mae-score=0.0635-0.7750 gcn-final-mse-score=0.0633-0.7811(0.0767/0.7811) loss=0.3392(0.1750+0.1642)

2020-08-05 02:30:13 Start Epoch 19
2020-08-05 02:30:13 Epoch:19,lr=0.0001
2020-08-05 02:30:15    0-10553 loss=0.0531(0.0374+0.0157)-0.0531(0.0374+0.0157) sod-mse=0.0066(0.0066) gcn-mse=0.0093(0.0093) gcn-final-mse=0.0086(0.0212)
2020-08-05 02:34:20 1000-10553 loss=1.3800(0.5679+0.8121)-0.1104(0.0714+0.0390) sod-mse=0.2552(0.0201) gcn-mse=0.2149(0.0246) gcn-final-mse=0.0240(0.0401)
2020-08-05 02:38:24 2000-10553 loss=0.0415(0.0271+0.0145)-0.1188(0.0758+0.0430) sod-mse=0.0092(0.0225) gcn-mse=0.0057(0.0273) gcn-final-mse=0.0268(0.0426)
2020-08-05 02:42:29 3000-10553 loss=0.0617(0.0485+0.0132)-0.1194(0.0762+0.0432) sod-mse=0.0077(0.0227) gcn-mse=0.0115(0.0276) gcn-final-mse=0.0270(0.0429)
2020-08-05 02:46:31 4000-10553 loss=0.0514(0.0361+0.0153)-0.1183(0.0757+0.0427) sod-mse=0.0071(0.0224) gcn-mse=0.0098(0.0273) gcn-final-mse=0.0267(0.0426)
2020-08-05 02:48:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 02:50:35 5000-10553 loss=0.1223(0.0828+0.0394)-0.1175(0.0752+0.0423) sod-mse=0.0202(0.0222) gcn-mse=0.0303(0.0270) gcn-final-mse=0.0264(0.0423)
2020-08-05 02:54:38 6000-10553 loss=0.0612(0.0437+0.0174)-0.1172(0.0751+0.0420) sod-mse=0.0088(0.0221) gcn-mse=0.0078(0.0269) gcn-final-mse=0.0263(0.0422)
2020-08-05 02:58:40 7000-10553 loss=0.0531(0.0408+0.0124)-0.1180(0.0755+0.0425) sod-mse=0.0065(0.0222) gcn-mse=0.0139(0.0270) gcn-final-mse=0.0264(0.0424)
2020-08-05 03:02:45 8000-10553 loss=0.0632(0.0482+0.0150)-0.1191(0.0761+0.0430) sod-mse=0.0099(0.0225) gcn-mse=0.0099(0.0274) gcn-final-mse=0.0267(0.0427)
2020-08-05 03:05:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 03:06:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 03:06:49 9000-10553 loss=0.0595(0.0412+0.0184)-0.1207(0.0768+0.0439) sod-mse=0.0112(0.0230) gcn-mse=0.0137(0.0277) gcn-final-mse=0.0271(0.0431)
2020-08-05 03:10:53 10000-10553 loss=0.1826(0.1103+0.0723)-0.1204(0.0767+0.0438) sod-mse=0.0377(0.0230) gcn-mse=0.0357(0.0277) gcn-final-mse=0.0270(0.0430)

2020-08-05 03:13:10    0-5019 loss=0.8756(0.4443+0.4313)-0.8756(0.4443+0.4313) sod-mse=0.0970(0.0970) gcn-mse=0.1011(0.1011) gcn-final-mse=0.0939(0.1065)
2020-08-05 03:15:02 1000-5019 loss=0.0443(0.0323+0.0119)-0.3465(0.1820+0.1645) sod-mse=0.0103(0.0634) gcn-mse=0.0134(0.0658) gcn-final-mse=0.0658(0.0797)
2020-08-05 03:16:53 2000-5019 loss=0.8045(0.3891+0.4154)-0.3497(0.1833+0.1664) sod-mse=0.0876(0.0641) gcn-mse=0.0831(0.0665) gcn-final-mse=0.0665(0.0803)
2020-08-05 03:18:45 3000-5019 loss=0.0470(0.0351+0.0119)-0.3554(0.1855+0.1699) sod-mse=0.0061(0.0648) gcn-mse=0.0085(0.0673) gcn-final-mse=0.0674(0.0811)
2020-08-05 03:19:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 03:20:37 4000-5019 loss=0.1535(0.0975+0.0560)-0.3514(0.1840+0.1674) sod-mse=0.0280(0.0644) gcn-mse=0.0295(0.0671) gcn-final-mse=0.0671(0.0808)
2020-08-05 03:21:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 03:22:30 5000-5019 loss=1.3067(0.5466+0.7601)-0.3524(0.1845+0.1680) sod-mse=0.0967(0.0646) gcn-mse=0.0983(0.0673) gcn-final-mse=0.0673(0.0810)
2020-08-05 03:22:32 E:19, Train sod-mae-score=0.0231-0.9675 gcn-mae-score=0.0277-0.9375 gcn-final-mse-score=0.0271-0.9402(0.0431/0.9402) loss=0.1208(0.0768+0.0440)
2020-08-05 03:22:32 E:19, Test  sod-mae-score=0.0646-0.8249 gcn-mae-score=0.0673-0.7685 gcn-final-mse-score=0.0673-0.7744(0.0810/0.7744) loss=0.3523(0.1844+0.1678)

2020-08-05 03:22:32 Start Epoch 20
2020-08-05 03:22:32 Epoch:20,lr=0.0000
2020-08-05 03:22:33    0-10553 loss=0.0687(0.0472+0.0215)-0.0687(0.0472+0.0215) sod-mse=0.0121(0.0121) gcn-mse=0.0123(0.0123) gcn-final-mse=0.0118(0.0321)
2020-08-05 03:24:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 03:26:39 1000-10553 loss=0.0345(0.0247+0.0098)-0.1071(0.0704+0.0367) sod-mse=0.0054(0.0201) gcn-mse=0.0059(0.0249) gcn-final-mse=0.0242(0.0407)
2020-08-05 03:26:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 03:30:45 2000-10553 loss=0.1055(0.0726+0.0328)-0.1036(0.0684+0.0352) sod-mse=0.0194(0.0188) gcn-mse=0.0255(0.0235) gcn-final-mse=0.0228(0.0392)
2020-08-05 03:34:50 3000-10553 loss=0.0240(0.0189+0.0050)-0.1021(0.0677+0.0345) sod-mse=0.0023(0.0182) gcn-mse=0.0035(0.0228) gcn-final-mse=0.0221(0.0385)
2020-08-05 03:36:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 03:38:54 4000-10553 loss=0.1362(0.0898+0.0464)-0.1004(0.0667+0.0337) sod-mse=0.0259(0.0178) gcn-mse=0.0293(0.0223) gcn-final-mse=0.0216(0.0379)
2020-08-05 03:42:57 5000-10553 loss=0.2622(0.1402+0.1220)-0.0996(0.0662+0.0334) sod-mse=0.0559(0.0175) gcn-mse=0.0606(0.0220) gcn-final-mse=0.0213(0.0375)
2020-08-05 03:47:01 6000-10553 loss=0.0583(0.0471+0.0112)-0.0985(0.0657+0.0329) sod-mse=0.0046(0.0173) gcn-mse=0.0075(0.0216) gcn-final-mse=0.0209(0.0372)
2020-08-05 03:51:05 7000-10553 loss=0.1218(0.0926+0.0292)-0.0980(0.0653+0.0327) sod-mse=0.0135(0.0171) gcn-mse=0.0168(0.0213) gcn-final-mse=0.0207(0.0370)
2020-08-05 03:55:08 8000-10553 loss=0.1124(0.0675+0.0450)-0.0979(0.0653+0.0326) sod-mse=0.0211(0.0170) gcn-mse=0.0240(0.0212) gcn-final-mse=0.0205(0.0369)
2020-08-05 03:59:14 9000-10553 loss=0.0681(0.0487+0.0193)-0.0981(0.0654+0.0327) sod-mse=0.0085(0.0170) gcn-mse=0.0115(0.0211) gcn-final-mse=0.0204(0.0368)
2020-08-05 04:03:18 10000-10553 loss=0.1405(0.0862+0.0543)-0.0978(0.0652+0.0326) sod-mse=0.0231(0.0169) gcn-mse=0.0275(0.0210) gcn-final-mse=0.0203(0.0367)

2020-08-05 04:05:34    0-5019 loss=0.9605(0.5480+0.4125)-0.9605(0.5480+0.4125) sod-mse=0.0914(0.0914) gcn-mse=0.1005(0.1005) gcn-final-mse=0.0931(0.1048)
2020-08-05 04:07:26 1000-5019 loss=0.0339(0.0272+0.0067)-0.3255(0.1694+0.1561) sod-mse=0.0057(0.0548) gcn-mse=0.0083(0.0573) gcn-final-mse=0.0572(0.0710)
2020-08-05 04:09:17 2000-5019 loss=0.7903(0.4128+0.3775)-0.3336(0.1730+0.1607) sod-mse=0.0878(0.0561) gcn-mse=0.0862(0.0585) gcn-final-mse=0.0583(0.0721)
2020-08-05 04:11:08 3000-5019 loss=0.0483(0.0361+0.0122)-0.3373(0.1742+0.1632) sod-mse=0.0061(0.0569) gcn-mse=0.0095(0.0593) gcn-final-mse=0.0591(0.0729)
2020-08-05 04:12:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 04:12:58 4000-5019 loss=0.1035(0.0716+0.0319)-0.3351(0.1733+0.1618) sod-mse=0.0161(0.0563) gcn-mse=0.0164(0.0589) gcn-final-mse=0.0587(0.0725)
2020-08-05 04:13:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 04:14:49 5000-5019 loss=1.1414(0.5090+0.6324)-0.3332(0.1725+0.1607) sod-mse=0.1085(0.0563) gcn-mse=0.1058(0.0589) gcn-final-mse=0.0586(0.0724)
2020-08-05 04:14:51 E:20, Train sod-mae-score=0.0168-0.9753 gcn-mae-score=0.0209-0.9466 gcn-final-mse-score=0.0202-0.9492(0.0366/0.9492) loss=0.0975(0.0650+0.0325)
2020-08-05 04:14:51 E:20, Test  sod-mae-score=0.0563-0.8439 gcn-mae-score=0.0589-0.7865 gcn-final-mse-score=0.0587-0.7922(0.0724/0.7922) loss=0.3331(0.1724+0.1607)

2020-08-05 04:14:51 Start Epoch 21
2020-08-05 04:14:51 Epoch:21,lr=0.0000
2020-08-05 04:14:52    0-10553 loss=0.0486(0.0341+0.0145)-0.0486(0.0341+0.0145) sod-mse=0.0093(0.0093) gcn-mse=0.0106(0.0106) gcn-final-mse=0.0101(0.0215)
2020-08-05 04:18:58 1000-10553 loss=0.0254(0.0211+0.0043)-0.0899(0.0613+0.0286) sod-mse=0.0019(0.0147) gcn-mse=0.0026(0.0186) gcn-final-mse=0.0178(0.0346)
2020-08-05 04:23:02 2000-10553 loss=0.2682(0.1497+0.1185)-0.0899(0.0613+0.0285) sod-mse=0.0408(0.0146) gcn-mse=0.0393(0.0184) gcn-final-mse=0.0176(0.0344)
2020-08-05 04:27:07 3000-10553 loss=0.1720(0.1012+0.0708)-0.0889(0.0607+0.0282) sod-mse=0.0357(0.0144) gcn-mse=0.0360(0.0182) gcn-final-mse=0.0175(0.0341)
2020-08-05 04:28:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 04:31:11 4000-10553 loss=0.1183(0.0811+0.0372)-0.0900(0.0611+0.0289) sod-mse=0.0220(0.0147) gcn-mse=0.0301(0.0184) gcn-final-mse=0.0177(0.0343)
2020-08-05 04:35:15 5000-10553 loss=0.0480(0.0358+0.0122)-0.0901(0.0612+0.0290) sod-mse=0.0080(0.0147) gcn-mse=0.0105(0.0183) gcn-final-mse=0.0176(0.0342)
2020-08-05 04:39:18 6000-10553 loss=0.0617(0.0462+0.0155)-0.0905(0.0613+0.0292) sod-mse=0.0077(0.0148) gcn-mse=0.0116(0.0184) gcn-final-mse=0.0177(0.0343)
2020-08-05 04:39:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 04:43:21 7000-10553 loss=0.0386(0.0316+0.0070)-0.0903(0.0613+0.0290) sod-mse=0.0034(0.0148) gcn-mse=0.0090(0.0184) gcn-final-mse=0.0177(0.0343)
2020-08-05 04:47:24 8000-10553 loss=0.0849(0.0629+0.0220)-0.0902(0.0612+0.0290) sod-mse=0.0124(0.0148) gcn-mse=0.0162(0.0184) gcn-final-mse=0.0176(0.0342)
2020-08-05 04:51:29 9000-10553 loss=0.0216(0.0139+0.0077)-0.0903(0.0612+0.0290) sod-mse=0.0055(0.0148) gcn-mse=0.0055(0.0184) gcn-final-mse=0.0177(0.0342)
2020-08-05 04:51:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 04:55:33 10000-10553 loss=0.1216(0.0835+0.0381)-0.0902(0.0612+0.0290) sod-mse=0.0175(0.0148) gcn-mse=0.0196(0.0184) gcn-final-mse=0.0177(0.0343)

2020-08-05 04:57:48    0-5019 loss=1.0146(0.5758+0.4388)-1.0146(0.5758+0.4388) sod-mse=0.0923(0.0923) gcn-mse=0.1012(0.1012) gcn-final-mse=0.0937(0.1054)
2020-08-05 04:59:40 1000-5019 loss=0.0328(0.0266+0.0062)-0.3374(0.1727+0.1647) sod-mse=0.0051(0.0535) gcn-mse=0.0077(0.0562) gcn-final-mse=0.0560(0.0697)
2020-08-05 05:01:30 2000-5019 loss=0.8365(0.4234+0.4131)-0.3442(0.1754+0.1688) sod-mse=0.0896(0.0548) gcn-mse=0.0868(0.0573) gcn-final-mse=0.0570(0.0707)
2020-08-05 05:03:21 3000-5019 loss=0.0476(0.0355+0.0121)-0.3465(0.1759+0.1707) sod-mse=0.0058(0.0553) gcn-mse=0.0085(0.0579) gcn-final-mse=0.0577(0.0713)
2020-08-05 05:04:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 05:05:11 4000-5019 loss=0.1007(0.0695+0.0313)-0.3444(0.1752+0.1692) sod-mse=0.0154(0.0548) gcn-mse=0.0146(0.0575) gcn-final-mse=0.0573(0.0709)
2020-08-05 05:05:43 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 05:07:01 5000-5019 loss=1.2017(0.5152+0.6865)-0.3428(0.1745+0.1683) sod-mse=0.1129(0.0547) gcn-mse=0.1084(0.0575) gcn-final-mse=0.0572(0.0708)
2020-08-05 05:07:03 E:21, Train sod-mae-score=0.0148-0.9779 gcn-mae-score=0.0184-0.9488 gcn-final-mse-score=0.0176-0.9514(0.0342/0.9514) loss=0.0902(0.0612+0.0290)
2020-08-05 05:07:03 E:21, Test  sod-mae-score=0.0547-0.8452 gcn-mae-score=0.0575-0.7907 gcn-final-mse-score=0.0572-0.7965(0.0708/0.7965) loss=0.3426(0.1744+0.1682)

2020-08-05 05:07:03 Start Epoch 22
2020-08-05 05:07:03 Epoch:22,lr=0.0000
2020-08-05 05:07:05    0-10553 loss=0.0352(0.0224+0.0128)-0.0352(0.0224+0.0128) sod-mse=0.0078(0.0078) gcn-mse=0.0075(0.0075) gcn-final-mse=0.0090(0.0165)
2020-08-05 05:11:09 1000-10553 loss=0.0630(0.0455+0.0175)-0.0837(0.0574+0.0263) sod-mse=0.0080(0.0131) gcn-mse=0.0080(0.0165) gcn-final-mse=0.0157(0.0317)
2020-08-05 05:15:13 2000-10553 loss=0.0705(0.0533+0.0173)-0.0868(0.0592+0.0276) sod-mse=0.0073(0.0138) gcn-mse=0.0085(0.0173) gcn-final-mse=0.0166(0.0329)
2020-08-05 05:19:17 3000-10553 loss=0.0256(0.0205+0.0051)-0.0871(0.0593+0.0278) sod-mse=0.0021(0.0139) gcn-mse=0.0024(0.0174) gcn-final-mse=0.0167(0.0331)
2020-08-05 05:23:19 4000-10553 loss=0.0598(0.0458+0.0140)-0.0867(0.0591+0.0276) sod-mse=0.0058(0.0139) gcn-mse=0.0079(0.0173) gcn-final-mse=0.0166(0.0330)
2020-08-05 05:26:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 05:26:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 05:27:22 5000-10553 loss=0.1016(0.0633+0.0383)-0.0866(0.0592+0.0274) sod-mse=0.0114(0.0138) gcn-mse=0.0133(0.0173) gcn-final-mse=0.0166(0.0329)
2020-08-05 05:31:27 6000-10553 loss=0.1118(0.0723+0.0394)-0.0865(0.0592+0.0273) sod-mse=0.0206(0.0138) gcn-mse=0.0248(0.0173) gcn-final-mse=0.0165(0.0330)
2020-08-05 05:35:30 7000-10553 loss=0.0658(0.0464+0.0194)-0.0868(0.0594+0.0274) sod-mse=0.0105(0.0138) gcn-mse=0.0135(0.0173) gcn-final-mse=0.0165(0.0331)
2020-08-05 05:38:44 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 05:39:34 8000-10553 loss=0.1230(0.0812+0.0418)-0.0867(0.0594+0.0273) sod-mse=0.0195(0.0139) gcn-mse=0.0208(0.0173) gcn-final-mse=0.0166(0.0331)
2020-08-05 05:43:37 9000-10553 loss=0.0823(0.0627+0.0196)-0.0869(0.0595+0.0274) sod-mse=0.0114(0.0139) gcn-mse=0.0137(0.0173) gcn-final-mse=0.0166(0.0332)
2020-08-05 05:47:40 10000-10553 loss=0.1121(0.0766+0.0354)-0.0871(0.0596+0.0275) sod-mse=0.0239(0.0139) gcn-mse=0.0383(0.0174) gcn-final-mse=0.0166(0.0333)

2020-08-05 05:49:56    0-5019 loss=1.0416(0.5936+0.4479)-1.0416(0.5936+0.4479) sod-mse=0.0957(0.0957) gcn-mse=0.1022(0.1022) gcn-final-mse=0.0947(0.1081)
2020-08-05 05:51:50 1000-5019 loss=0.0314(0.0253+0.0061)-0.3344(0.1704+0.1641) sod-mse=0.0050(0.0522) gcn-mse=0.0069(0.0549) gcn-final-mse=0.0547(0.0684)
2020-08-05 05:53:42 2000-5019 loss=0.8860(0.4449+0.4411)-0.3410(0.1731+0.1679) sod-mse=0.0909(0.0533) gcn-mse=0.0897(0.0559) gcn-final-mse=0.0557(0.0693)
2020-08-05 05:55:35 3000-5019 loss=0.0486(0.0364+0.0121)-0.3438(0.1736+0.1702) sod-mse=0.0059(0.0537) gcn-mse=0.0100(0.0564) gcn-final-mse=0.0562(0.0698)
2020-08-05 05:56:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 05:57:28 4000-5019 loss=0.1041(0.0720+0.0321)-0.3416(0.1730+0.1686) sod-mse=0.0155(0.0532) gcn-mse=0.0161(0.0560) gcn-final-mse=0.0558(0.0694)
2020-08-05 05:58:01 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 05:59:21 5000-5019 loss=1.1115(0.4668+0.6447)-0.3406(0.1725+0.1681) sod-mse=0.0955(0.0532) gcn-mse=0.0961(0.0561) gcn-final-mse=0.0558(0.0694)
2020-08-05 05:59:23 E:22, Train sod-mae-score=0.0139-0.9789 gcn-mae-score=0.0174-0.9502 gcn-final-mse-score=0.0166-0.9527(0.0333/0.9527) loss=0.0870(0.0596+0.0274)
2020-08-05 05:59:23 E:22, Test  sod-mae-score=0.0532-0.8471 gcn-mae-score=0.0561-0.7911 gcn-final-mse-score=0.0558-0.7965(0.0694/0.7965) loss=0.3404(0.1724+0.1680)

2020-08-05 05:59:23 Start Epoch 23
2020-08-05 05:59:23 Epoch:23,lr=0.0000
2020-08-05 05:59:24    0-10553 loss=0.0296(0.0241+0.0055)-0.0296(0.0241+0.0055) sod-mse=0.0024(0.0024) gcn-mse=0.0028(0.0028) gcn-final-mse=0.0030(0.0155)
2020-08-05 06:03:30 1000-10553 loss=0.0514(0.0350+0.0164)-0.0850(0.0589+0.0262) sod-mse=0.0104(0.0132) gcn-mse=0.0089(0.0166) gcn-final-mse=0.0157(0.0326)
2020-08-05 06:07:32 2000-10553 loss=0.0800(0.0597+0.0203)-0.0849(0.0584+0.0265) sod-mse=0.0121(0.0132) gcn-mse=0.0172(0.0164) gcn-final-mse=0.0157(0.0323)
2020-08-05 06:09:58 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 06:11:37 3000-10553 loss=0.3508(0.1801+0.1707)-0.0863(0.0590+0.0273) sod-mse=0.0571(0.0134) gcn-mse=0.0533(0.0166) gcn-final-mse=0.0159(0.0325)
2020-08-05 06:15:41 4000-10553 loss=0.0539(0.0417+0.0122)-0.0855(0.0586+0.0269) sod-mse=0.0053(0.0133) gcn-mse=0.0109(0.0166) gcn-final-mse=0.0158(0.0324)
2020-08-05 06:18:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 06:19:46 5000-10553 loss=0.0334(0.0216+0.0118)-0.0851(0.0583+0.0268) sod-mse=0.0100(0.0134) gcn-mse=0.0129(0.0166) gcn-final-mse=0.0158(0.0324)
2020-08-05 06:23:51 6000-10553 loss=0.0765(0.0483+0.0283)-0.0852(0.0584+0.0268) sod-mse=0.0146(0.0134) gcn-mse=0.0140(0.0167) gcn-final-mse=0.0159(0.0325)
2020-08-05 06:27:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 06:27:55 7000-10553 loss=0.0986(0.0737+0.0249)-0.0850(0.0584+0.0266) sod-mse=0.0147(0.0134) gcn-mse=0.0187(0.0166) gcn-final-mse=0.0158(0.0325)
2020-08-05 06:32:00 8000-10553 loss=0.0768(0.0476+0.0291)-0.0853(0.0586+0.0267) sod-mse=0.0142(0.0134) gcn-mse=0.0158(0.0167) gcn-final-mse=0.0159(0.0326)
2020-08-05 06:36:03 9000-10553 loss=0.0749(0.0581+0.0169)-0.0850(0.0584+0.0266) sod-mse=0.0097(0.0134) gcn-mse=0.0169(0.0166) gcn-final-mse=0.0159(0.0326)
2020-08-05 06:40:08 10000-10553 loss=0.0658(0.0501+0.0156)-0.0853(0.0587+0.0267) sod-mse=0.0093(0.0134) gcn-mse=0.0145(0.0167) gcn-final-mse=0.0159(0.0327)

2020-08-05 06:42:23    0-5019 loss=1.0513(0.6050+0.4464)-1.0513(0.6050+0.4464) sod-mse=0.0947(0.0947) gcn-mse=0.1008(0.1008) gcn-final-mse=0.0933(0.1071)
2020-08-05 06:44:15 1000-5019 loss=0.0312(0.0252+0.0060)-0.3508(0.1784+0.1724) sod-mse=0.0050(0.0540) gcn-mse=0.0068(0.0567) gcn-final-mse=0.0564(0.0701)
2020-08-05 06:46:05 2000-5019 loss=0.8979(0.4552+0.4426)-0.3601(0.1823+0.1778) sod-mse=0.0893(0.0554) gcn-mse=0.0881(0.0579) gcn-final-mse=0.0576(0.0712)
2020-08-05 06:47:56 3000-5019 loss=0.0478(0.0357+0.0121)-0.3622(0.1824+0.1798) sod-mse=0.0058(0.0559) gcn-mse=0.0090(0.0584) gcn-final-mse=0.0582(0.0717)
2020-08-05 06:48:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 06:49:46 4000-5019 loss=0.1012(0.0705+0.0306)-0.3607(0.1818+0.1788) sod-mse=0.0149(0.0554) gcn-mse=0.0145(0.0580) gcn-final-mse=0.0578(0.0714)
2020-08-05 06:50:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 06:51:37 5000-5019 loss=1.0761(0.4362+0.6399)-0.3588(0.1810+0.1778) sod-mse=0.1171(0.0554) gcn-mse=0.1109(0.0580) gcn-final-mse=0.0577(0.0712)
2020-08-05 06:51:39 E:23, Train sod-mae-score=0.0134-0.9799 gcn-mae-score=0.0166-0.9510 gcn-final-mse-score=0.0159-0.9536(0.0326/0.9536) loss=0.0851(0.0585+0.0266)
2020-08-05 06:51:39 E:23, Test  sod-mae-score=0.0553-0.8414 gcn-mae-score=0.0580-0.7872 gcn-final-mse-score=0.0577-0.7930(0.0712/0.7930) loss=0.3586(0.1809+0.1777)

2020-08-05 06:51:39 Start Epoch 24
2020-08-05 06:51:39 Epoch:24,lr=0.0000
2020-08-05 06:51:41    0-10553 loss=0.0349(0.0266+0.0083)-0.0349(0.0266+0.0083) sod-mse=0.0042(0.0042) gcn-mse=0.0071(0.0071) gcn-final-mse=0.0058(0.0129)
2020-08-05 06:55:47 1000-10553 loss=0.0826(0.0626+0.0200)-0.0839(0.0583+0.0256) sod-mse=0.0093(0.0129) gcn-mse=0.0114(0.0162) gcn-final-mse=0.0154(0.0325)
2020-08-05 06:58:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 06:59:50 2000-10553 loss=0.0735(0.0550+0.0185)-0.0828(0.0578+0.0250) sod-mse=0.0099(0.0127) gcn-mse=0.0124(0.0160) gcn-final-mse=0.0152(0.0324)
2020-08-05 07:03:54 3000-10553 loss=0.0776(0.0569+0.0207)-0.0822(0.0573+0.0249) sod-mse=0.0128(0.0126) gcn-mse=0.0216(0.0160) gcn-final-mse=0.0152(0.0321)
2020-08-05 07:07:58 4000-10553 loss=0.0984(0.0710+0.0274)-0.0842(0.0582+0.0261) sod-mse=0.0135(0.0130) gcn-mse=0.0174(0.0163) gcn-final-mse=0.0155(0.0324)
2020-08-05 07:12:01 5000-10553 loss=0.0255(0.0186+0.0069)-0.0839(0.0581+0.0258) sod-mse=0.0032(0.0129) gcn-mse=0.0033(0.0162) gcn-final-mse=0.0154(0.0324)
2020-08-05 07:16:05 6000-10553 loss=0.0744(0.0584+0.0160)-0.0838(0.0580+0.0258) sod-mse=0.0074(0.0129) gcn-mse=0.0108(0.0162) gcn-final-mse=0.0154(0.0323)
2020-08-05 07:20:09 7000-10553 loss=0.1150(0.0787+0.0363)-0.0836(0.0579+0.0257) sod-mse=0.0230(0.0129) gcn-mse=0.0290(0.0161) gcn-final-mse=0.0154(0.0323)
2020-08-05 07:24:12 8000-10553 loss=0.1091(0.0826+0.0265)-0.0836(0.0579+0.0257) sod-mse=0.0106(0.0129) gcn-mse=0.0174(0.0161) gcn-final-mse=0.0154(0.0323)
2020-08-05 07:25:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 07:28:15 9000-10553 loss=0.0359(0.0294+0.0064)-0.0832(0.0577+0.0255) sod-mse=0.0032(0.0128) gcn-mse=0.0075(0.0161) gcn-final-mse=0.0153(0.0322)
2020-08-05 07:31:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 07:32:19 10000-10553 loss=0.0420(0.0342+0.0078)-0.0831(0.0576+0.0255) sod-mse=0.0028(0.0128) gcn-mse=0.0032(0.0160) gcn-final-mse=0.0153(0.0321)

2020-08-05 07:34:34    0-5019 loss=1.0492(0.6116+0.4376)-1.0492(0.6116+0.4376) sod-mse=0.0950(0.0950) gcn-mse=0.1002(0.1002) gcn-final-mse=0.0927(0.1060)
2020-08-05 07:36:26 1000-5019 loss=0.0318(0.0252+0.0065)-0.3442(0.1764+0.1678) sod-mse=0.0054(0.0531) gcn-mse=0.0070(0.0555) gcn-final-mse=0.0553(0.0690)
2020-08-05 07:38:17 2000-5019 loss=1.0060(0.5109+0.4951)-0.3515(0.1795+0.1720) sod-mse=0.0933(0.0545) gcn-mse=0.0930(0.0567) gcn-final-mse=0.0564(0.0700)
2020-08-05 07:40:08 3000-5019 loss=0.0487(0.0366+0.0121)-0.3527(0.1794+0.1733) sod-mse=0.0058(0.0549) gcn-mse=0.0101(0.0572) gcn-final-mse=0.0570(0.0705)
2020-08-05 07:41:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 07:41:58 4000-5019 loss=0.1020(0.0715+0.0305)-0.3508(0.1789+0.1719) sod-mse=0.0149(0.0544) gcn-mse=0.0148(0.0568) gcn-final-mse=0.0566(0.0701)
2020-08-05 07:42:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 07:43:48 5000-5019 loss=1.0665(0.4549+0.6116)-0.3483(0.1778+0.1706) sod-mse=0.1130(0.0542) gcn-mse=0.1087(0.0567) gcn-final-mse=0.0564(0.0699)
2020-08-05 07:43:50 E:24, Train sod-mae-score=0.0129-0.9807 gcn-mae-score=0.0161-0.9515 gcn-final-mse-score=0.0153-0.9540(0.0322/0.9540) loss=0.0834(0.0577+0.0257)
2020-08-05 07:43:50 E:24, Test  sod-mae-score=0.0542-0.8445 gcn-mae-score=0.0567-0.7899 gcn-final-mse-score=0.0564-0.7956(0.0699/0.7956) loss=0.3482(0.1777+0.1705)

2020-08-05 07:43:50 Start Epoch 25
2020-08-05 07:43:50 Epoch:25,lr=0.0000
2020-08-05 07:43:52    0-10553 loss=0.0836(0.0608+0.0228)-0.0836(0.0608+0.0228) sod-mse=0.0113(0.0113) gcn-mse=0.0154(0.0154) gcn-final-mse=0.0128(0.0294)
2020-08-05 07:47:57 1000-10553 loss=0.0969(0.0618+0.0351)-0.0795(0.0557+0.0239) sod-mse=0.0214(0.0122) gcn-mse=0.0341(0.0154) gcn-final-mse=0.0146(0.0311)
2020-08-05 07:51:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 07:52:02 2000-10553 loss=0.0483(0.0372+0.0110)-0.0796(0.0558+0.0238) sod-mse=0.0062(0.0122) gcn-mse=0.0107(0.0154) gcn-final-mse=0.0146(0.0312)
2020-08-05 07:56:04 3000-10553 loss=0.1025(0.0681+0.0345)-0.0800(0.0561+0.0239) sod-mse=0.0147(0.0121) gcn-mse=0.0188(0.0154) gcn-final-mse=0.0146(0.0313)
2020-08-05 08:00:09 4000-10553 loss=0.0251(0.0209+0.0042)-0.0806(0.0564+0.0242) sod-mse=0.0019(0.0122) gcn-mse=0.0023(0.0155) gcn-final-mse=0.0147(0.0315)
2020-08-05 08:04:13 5000-10553 loss=0.2478(0.1670+0.0808)-0.0818(0.0570+0.0248) sod-mse=0.0462(0.0124) gcn-mse=0.0503(0.0156) gcn-final-mse=0.0148(0.0317)
2020-08-05 08:08:16 6000-10553 loss=0.0396(0.0259+0.0137)-0.0820(0.0571+0.0249) sod-mse=0.0065(0.0125) gcn-mse=0.0058(0.0157) gcn-final-mse=0.0149(0.0318)
2020-08-05 08:11:34 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 08:12:19 7000-10553 loss=0.1390(0.0942+0.0448)-0.0815(0.0568+0.0247) sod-mse=0.0230(0.0124) gcn-mse=0.0279(0.0155) gcn-final-mse=0.0148(0.0316)
2020-08-05 08:16:23 8000-10553 loss=0.0575(0.0327+0.0247)-0.0814(0.0567+0.0246) sod-mse=0.0172(0.0124) gcn-mse=0.0196(0.0155) gcn-final-mse=0.0148(0.0316)
2020-08-05 08:20:27 9000-10553 loss=0.0647(0.0518+0.0129)-0.0816(0.0569+0.0247) sod-mse=0.0055(0.0124) gcn-mse=0.0079(0.0156) gcn-final-mse=0.0148(0.0317)
2020-08-05 08:23:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 08:24:31 10000-10553 loss=0.0318(0.0244+0.0073)-0.0816(0.0568+0.0248) sod-mse=0.0036(0.0124) gcn-mse=0.0058(0.0156) gcn-final-mse=0.0148(0.0317)

2020-08-05 08:26:47    0-5019 loss=1.0807(0.6310+0.4496)-1.0807(0.6310+0.4496) sod-mse=0.0968(0.0968) gcn-mse=0.1019(0.1019) gcn-final-mse=0.0941(0.1077)
2020-08-05 08:28:40 1000-5019 loss=0.0306(0.0248+0.0058)-0.3486(0.1770+0.1716) sod-mse=0.0047(0.0525) gcn-mse=0.0065(0.0548) gcn-final-mse=0.0546(0.0684)
2020-08-05 08:30:33 2000-5019 loss=1.0600(0.5264+0.5336)-0.3558(0.1802+0.1756) sod-mse=0.0930(0.0537) gcn-mse=0.0923(0.0559) gcn-final-mse=0.0556(0.0693)
2020-08-05 08:32:26 3000-5019 loss=0.0478(0.0358+0.0120)-0.3582(0.1806+0.1776) sod-mse=0.0057(0.0541) gcn-mse=0.0091(0.0564) gcn-final-mse=0.0562(0.0698)
2020-08-05 08:33:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 08:34:18 4000-5019 loss=0.1001(0.0705+0.0296)-0.3568(0.1802+0.1766) sod-mse=0.0145(0.0536) gcn-mse=0.0141(0.0560) gcn-final-mse=0.0558(0.0694)
2020-08-05 08:34:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 08:36:11 5000-5019 loss=1.0878(0.4585+0.6293)-0.3557(0.1797+0.1760) sod-mse=0.0966(0.0535) gcn-mse=0.0977(0.0560) gcn-final-mse=0.0557(0.0693)
2020-08-05 08:36:12 E:25, Train sod-mae-score=0.0124-0.9812 gcn-mae-score=0.0156-0.9522 gcn-final-mse-score=0.0148-0.9548(0.0317/0.9548) loss=0.0816(0.0569+0.0248)
2020-08-05 08:36:12 E:25, Test  sod-mae-score=0.0535-0.8436 gcn-mae-score=0.0560-0.7889 gcn-final-mse-score=0.0557-0.7945(0.0693/0.7945) loss=0.3556(0.1796+0.1759)

2020-08-05 08:36:12 Start Epoch 26
2020-08-05 08:36:12 Epoch:26,lr=0.0000
2020-08-05 08:36:14    0-10553 loss=0.0666(0.0503+0.0164)-0.0666(0.0503+0.0164) sod-mse=0.0074(0.0074) gcn-mse=0.0097(0.0097) gcn-final-mse=0.0087(0.0274)
2020-08-05 08:37:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 08:40:18 1000-10553 loss=0.0618(0.0461+0.0156)-0.0839(0.0579+0.0260) sod-mse=0.0100(0.0127) gcn-mse=0.0124(0.0156) gcn-final-mse=0.0148(0.0320)
2020-08-05 08:40:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 08:44:22 2000-10553 loss=0.1512(0.1130+0.0382)-0.0841(0.0583+0.0258) sod-mse=0.0207(0.0128) gcn-mse=0.0240(0.0158) gcn-final-mse=0.0150(0.0324)
2020-08-05 08:48:26 3000-10553 loss=0.2300(0.1369+0.0930)-0.0830(0.0577+0.0252) sod-mse=0.0388(0.0125) gcn-mse=0.0400(0.0156) gcn-final-mse=0.0148(0.0321)
2020-08-05 08:52:30 4000-10553 loss=0.0316(0.0246+0.0070)-0.0814(0.0568+0.0246) sod-mse=0.0032(0.0123) gcn-mse=0.0065(0.0153) gcn-final-mse=0.0145(0.0317)
2020-08-05 08:56:34 5000-10553 loss=0.0874(0.0693+0.0180)-0.0808(0.0565+0.0244) sod-mse=0.0077(0.0122) gcn-mse=0.0172(0.0152) gcn-final-mse=0.0144(0.0315)
2020-08-05 09:00:38 6000-10553 loss=0.0298(0.0211+0.0087)-0.0808(0.0564+0.0244) sod-mse=0.0045(0.0122) gcn-mse=0.0056(0.0152) gcn-final-mse=0.0144(0.0314)
2020-08-05 09:01:05 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 09:04:43 7000-10553 loss=0.0975(0.0780+0.0196)-0.0809(0.0564+0.0245) sod-mse=0.0094(0.0122) gcn-mse=0.0151(0.0152) gcn-final-mse=0.0145(0.0314)
2020-08-05 09:08:48 8000-10553 loss=0.0768(0.0548+0.0220)-0.0809(0.0564+0.0245) sod-mse=0.0162(0.0122) gcn-mse=0.0134(0.0152) gcn-final-mse=0.0145(0.0314)
2020-08-05 09:12:51 9000-10553 loss=0.0627(0.0367+0.0261)-0.0807(0.0563+0.0244) sod-mse=0.0154(0.0122) gcn-mse=0.0170(0.0152) gcn-final-mse=0.0145(0.0314)
2020-08-05 09:16:56 10000-10553 loss=0.1115(0.0807+0.0307)-0.0806(0.0563+0.0243) sod-mse=0.0186(0.0122) gcn-mse=0.0269(0.0152) gcn-final-mse=0.0144(0.0314)

2020-08-05 09:19:12    0-5019 loss=1.1229(0.6558+0.4670)-1.1229(0.6558+0.4670) sod-mse=0.0980(0.0980) gcn-mse=0.1034(0.1034) gcn-final-mse=0.0958(0.1095)
2020-08-05 09:21:05 1000-5019 loss=0.0303(0.0244+0.0059)-0.3630(0.1815+0.1815) sod-mse=0.0048(0.0532) gcn-mse=0.0065(0.0558) gcn-final-mse=0.0555(0.0694)
2020-08-05 09:22:57 2000-5019 loss=1.0249(0.5025+0.5223)-0.3708(0.1848+0.1860) sod-mse=0.0910(0.0544) gcn-mse=0.0901(0.0569) gcn-final-mse=0.0566(0.0703)
2020-08-05 09:24:49 3000-5019 loss=0.0481(0.0357+0.0123)-0.3721(0.1847+0.1874) sod-mse=0.0058(0.0550) gcn-mse=0.0087(0.0575) gcn-final-mse=0.0572(0.0709)
2020-08-05 09:25:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 09:26:42 4000-5019 loss=0.0980(0.0689+0.0292)-0.3706(0.1842+0.1864) sod-mse=0.0143(0.0545) gcn-mse=0.0129(0.0572) gcn-final-mse=0.0569(0.0705)
2020-08-05 09:27:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 09:28:35 5000-5019 loss=0.9584(0.3713+0.5871)-0.3692(0.1836+0.1856) sod-mse=0.1159(0.0545) gcn-mse=0.1093(0.0571) gcn-final-mse=0.0568(0.0704)
2020-08-05 09:28:37 E:26, Train sod-mae-score=0.0121-0.9815 gcn-mae-score=0.0152-0.9525 gcn-final-mse-score=0.0145-0.9549(0.0314/0.9549) loss=0.0806(0.0563+0.0243)
2020-08-05 09:28:37 E:26, Test  sod-mae-score=0.0544-0.8420 gcn-mae-score=0.0571-0.7877 gcn-final-mse-score=0.0568-0.7934(0.0704/0.7934) loss=0.3690(0.1835+0.1855)

2020-08-05 09:28:37 Start Epoch 27
2020-08-05 09:28:37 Epoch:27,lr=0.0000
2020-08-05 09:28:38    0-10553 loss=0.0402(0.0268+0.0134)-0.0402(0.0268+0.0134) sod-mse=0.0068(0.0068) gcn-mse=0.0068(0.0068) gcn-final-mse=0.0068(0.0153)
2020-08-05 09:32:45 1000-10553 loss=0.0441(0.0375+0.0066)-0.0795(0.0555+0.0240) sod-mse=0.0023(0.0121) gcn-mse=0.0047(0.0149) gcn-final-mse=0.0142(0.0311)
2020-08-05 09:36:49 2000-10553 loss=0.0574(0.0404+0.0170)-0.0792(0.0555+0.0237) sod-mse=0.0090(0.0119) gcn-mse=0.0134(0.0148) gcn-final-mse=0.0141(0.0310)
2020-08-05 09:40:54 3000-10553 loss=0.0844(0.0588+0.0256)-0.0787(0.0553+0.0234) sod-mse=0.0141(0.0118) gcn-mse=0.0146(0.0147) gcn-final-mse=0.0140(0.0310)
2020-08-05 09:43:21 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 09:44:59 4000-10553 loss=0.0603(0.0427+0.0176)-0.0783(0.0551+0.0232) sod-mse=0.0093(0.0116) gcn-mse=0.0083(0.0146) gcn-final-mse=0.0138(0.0308)
2020-08-05 09:49:01 5000-10553 loss=0.0388(0.0303+0.0085)-0.0780(0.0550+0.0230) sod-mse=0.0058(0.0116) gcn-mse=0.0053(0.0146) gcn-final-mse=0.0138(0.0308)
2020-08-05 09:52:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 09:53:05 6000-10553 loss=0.0559(0.0377+0.0183)-0.0785(0.0553+0.0232) sod-mse=0.0094(0.0116) gcn-mse=0.0112(0.0147) gcn-final-mse=0.0139(0.0309)
2020-08-05 09:56:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 09:57:08 7000-10553 loss=0.0361(0.0265+0.0096)-0.0787(0.0554+0.0233) sod-mse=0.0068(0.0117) gcn-mse=0.0094(0.0147) gcn-final-mse=0.0139(0.0309)
2020-08-05 10:01:11 8000-10553 loss=0.0914(0.0690+0.0224)-0.0787(0.0554+0.0233) sod-mse=0.0117(0.0117) gcn-mse=0.0204(0.0147) gcn-final-mse=0.0139(0.0309)
2020-08-05 10:05:15 9000-10553 loss=0.1567(0.1022+0.0545)-0.0786(0.0553+0.0232) sod-mse=0.0327(0.0116) gcn-mse=0.0335(0.0146) gcn-final-mse=0.0139(0.0309)
2020-08-05 10:09:17 10000-10553 loss=0.0619(0.0491+0.0129)-0.0788(0.0554+0.0233) sod-mse=0.0093(0.0117) gcn-mse=0.0171(0.0147) gcn-final-mse=0.0139(0.0310)

2020-08-05 10:11:33    0-5019 loss=1.1432(0.6698+0.4735)-1.1432(0.6698+0.4735) sod-mse=0.1022(0.1022) gcn-mse=0.1073(0.1073) gcn-final-mse=0.0994(0.1128)
2020-08-05 10:13:25 1000-5019 loss=0.0299(0.0242+0.0057)-0.3604(0.1810+0.1794) sod-mse=0.0047(0.0517) gcn-mse=0.0060(0.0542) gcn-final-mse=0.0539(0.0676)
2020-08-05 10:15:16 2000-5019 loss=1.1272(0.5481+0.5791)-0.3659(0.1833+0.1827) sod-mse=0.0941(0.0527) gcn-mse=0.0941(0.0551) gcn-final-mse=0.0548(0.0684)
2020-08-05 10:17:07 3000-5019 loss=0.0485(0.0364+0.0121)-0.3678(0.1834+0.1844) sod-mse=0.0058(0.0532) gcn-mse=0.0093(0.0556) gcn-final-mse=0.0554(0.0689)
2020-08-05 10:18:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 10:18:58 4000-5019 loss=0.0996(0.0704+0.0292)-0.3656(0.1828+0.1828) sod-mse=0.0142(0.0527) gcn-mse=0.0139(0.0552) gcn-final-mse=0.0549(0.0685)
2020-08-05 10:19:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 10:20:48 5000-5019 loss=1.1490(0.4673+0.6818)-0.3647(0.1824+0.1823) sod-mse=0.0951(0.0526) gcn-mse=0.0969(0.0552) gcn-final-mse=0.0549(0.0684)
2020-08-05 10:20:50 E:27, Train sod-mae-score=0.0117-0.9823 gcn-mae-score=0.0147-0.9537 gcn-final-mse-score=0.0139-0.9562(0.0309/0.9562) loss=0.0787(0.0554+0.0233)
2020-08-05 10:20:50 E:27, Test  sod-mae-score=0.0526-0.8469 gcn-mae-score=0.0552-0.7912 gcn-final-mse-score=0.0549-0.7971(0.0684/0.7971) loss=0.3644(0.1823+0.1822)

2020-08-05 10:20:50 Start Epoch 28
2020-08-05 10:20:50 Epoch:28,lr=0.0000
2020-08-05 10:20:51    0-10553 loss=0.0931(0.0570+0.0361)-0.0931(0.0570+0.0361) sod-mse=0.0140(0.0140) gcn-mse=0.0164(0.0164) gcn-final-mse=0.0134(0.0242)
2020-08-05 10:24:56 1000-10553 loss=0.0735(0.0542+0.0193)-0.0791(0.0558+0.0233) sod-mse=0.0119(0.0116) gcn-mse=0.0179(0.0146) gcn-final-mse=0.0139(0.0312)
2020-08-05 10:29:01 2000-10553 loss=0.1104(0.0702+0.0402)-0.0783(0.0551+0.0231) sod-mse=0.0191(0.0115) gcn-mse=0.0157(0.0144) gcn-final-mse=0.0137(0.0308)
2020-08-05 10:30:43 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 10:33:05 3000-10553 loss=0.1074(0.0805+0.0269)-0.0777(0.0548+0.0229) sod-mse=0.0139(0.0115) gcn-mse=0.0189(0.0144) gcn-final-mse=0.0136(0.0306)
2020-08-05 10:37:09 4000-10553 loss=0.0723(0.0521+0.0202)-0.0782(0.0551+0.0232) sod-mse=0.0126(0.0115) gcn-mse=0.0128(0.0144) gcn-final-mse=0.0137(0.0307)
2020-08-05 10:40:01 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 10:41:12 5000-10553 loss=0.0985(0.0588+0.0397)-0.0784(0.0552+0.0232) sod-mse=0.0140(0.0116) gcn-mse=0.0202(0.0145) gcn-final-mse=0.0137(0.0308)
2020-08-05 10:45:15 6000-10553 loss=0.0564(0.0410+0.0154)-0.0780(0.0550+0.0230) sod-mse=0.0082(0.0115) gcn-mse=0.0142(0.0144) gcn-final-mse=0.0136(0.0307)
2020-08-05 10:49:17 7000-10553 loss=0.0681(0.0501+0.0179)-0.0777(0.0549+0.0229) sod-mse=0.0120(0.0114) gcn-mse=0.0144(0.0144) gcn-final-mse=0.0136(0.0307)
2020-08-05 10:53:21 8000-10553 loss=0.0788(0.0567+0.0222)-0.0779(0.0550+0.0229) sod-mse=0.0095(0.0114) gcn-mse=0.0104(0.0144) gcn-final-mse=0.0136(0.0307)
2020-08-05 10:57:27 9000-10553 loss=0.0728(0.0543+0.0185)-0.0777(0.0548+0.0229) sod-mse=0.0091(0.0114) gcn-mse=0.0104(0.0143) gcn-final-mse=0.0136(0.0306)
2020-08-05 11:00:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 11:01:30 10000-10553 loss=0.0717(0.0548+0.0169)-0.0779(0.0549+0.0229) sod-mse=0.0090(0.0114) gcn-mse=0.0123(0.0144) gcn-final-mse=0.0136(0.0307)

2020-08-05 11:03:45    0-5019 loss=1.1434(0.6602+0.4832)-1.1434(0.6602+0.4832) sod-mse=0.1019(0.1019) gcn-mse=0.1058(0.1058) gcn-final-mse=0.0978(0.1121)
2020-08-05 11:05:36 1000-5019 loss=0.0292(0.0235+0.0057)-0.3679(0.1831+0.1847) sod-mse=0.0046(0.0517) gcn-mse=0.0054(0.0540) gcn-final-mse=0.0538(0.0674)
2020-08-05 11:07:27 2000-5019 loss=1.2607(0.6010+0.6598)-0.3740(0.1858+0.1882) sod-mse=0.0931(0.0526) gcn-mse=0.0941(0.0549) gcn-final-mse=0.0547(0.0682)
2020-08-05 11:09:18 3000-5019 loss=0.0482(0.0361+0.0121)-0.3769(0.1861+0.1907) sod-mse=0.0057(0.0530) gcn-mse=0.0095(0.0555) gcn-final-mse=0.0552(0.0687)
2020-08-05 11:10:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 11:11:09 4000-5019 loss=0.0993(0.0702+0.0291)-0.3751(0.1858+0.1893) sod-mse=0.0140(0.0525) gcn-mse=0.0134(0.0551) gcn-final-mse=0.0548(0.0683)
2020-08-05 11:11:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 11:13:00 5000-5019 loss=1.2313(0.4858+0.7455)-0.3743(0.1854+0.1889) sod-mse=0.1123(0.0525) gcn-mse=0.1111(0.0551) gcn-final-mse=0.0547(0.0682)
2020-08-05 11:13:02 E:28, Train sod-mae-score=0.0115-0.9827 gcn-mae-score=0.0144-0.9538 gcn-final-mse-score=0.0136-0.9563(0.0307/0.9563) loss=0.0781(0.0550+0.0230)
2020-08-05 11:13:02 E:28, Test  sod-mae-score=0.0524-0.8447 gcn-mae-score=0.0551-0.7894 gcn-final-mse-score=0.0547-0.7951(0.0682/0.7951) loss=0.3741(0.1853+0.1887)

2020-08-05 11:13:02 Start Epoch 29
2020-08-05 11:13:02 Epoch:29,lr=0.0000
2020-08-05 11:13:04    0-10553 loss=0.1005(0.0716+0.0288)-0.1005(0.0716+0.0288) sod-mse=0.0152(0.0152) gcn-mse=0.0291(0.0291) gcn-final-mse=0.0257(0.0378)
2020-08-05 11:17:09 1000-10553 loss=0.0894(0.0538+0.0356)-0.0756(0.0535+0.0221) sod-mse=0.0150(0.0112) gcn-mse=0.0177(0.0140) gcn-final-mse=0.0132(0.0301)
2020-08-05 11:21:13 2000-10553 loss=0.0603(0.0472+0.0131)-0.0757(0.0538+0.0219) sod-mse=0.0086(0.0110) gcn-mse=0.0093(0.0139) gcn-final-mse=0.0131(0.0302)
2020-08-05 11:21:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 11:22:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 11:25:15 3000-10553 loss=0.1063(0.0813+0.0250)-0.0763(0.0542+0.0222) sod-mse=0.0128(0.0111) gcn-mse=0.0247(0.0140) gcn-final-mse=0.0132(0.0303)
2020-08-05 11:29:21 4000-10553 loss=0.0729(0.0576+0.0153)-0.0761(0.0540+0.0221) sod-mse=0.0079(0.0111) gcn-mse=0.0112(0.0140) gcn-final-mse=0.0132(0.0303)
2020-08-05 11:33:27 5000-10553 loss=0.0972(0.0677+0.0295)-0.0759(0.0538+0.0220) sod-mse=0.0118(0.0110) gcn-mse=0.0134(0.0139) gcn-final-mse=0.0132(0.0302)
2020-08-05 11:37:31 6000-10553 loss=0.0796(0.0614+0.0182)-0.0758(0.0538+0.0219) sod-mse=0.0089(0.0110) gcn-mse=0.0122(0.0139) gcn-final-mse=0.0131(0.0302)
2020-08-05 11:41:35 7000-10553 loss=0.0777(0.0555+0.0223)-0.0762(0.0541+0.0221) sod-mse=0.0144(0.0111) gcn-mse=0.0149(0.0140) gcn-final-mse=0.0132(0.0303)
2020-08-05 11:45:39 8000-10553 loss=0.0906(0.0724+0.0182)-0.0766(0.0543+0.0223) sod-mse=0.0077(0.0111) gcn-mse=0.0094(0.0140) gcn-final-mse=0.0133(0.0303)
2020-08-05 11:49:44 9000-10553 loss=0.0703(0.0556+0.0147)-0.0765(0.0542+0.0222) sod-mse=0.0111(0.0111) gcn-mse=0.0181(0.0140) gcn-final-mse=0.0133(0.0303)
2020-08-05 11:51:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 11:53:49 10000-10553 loss=0.0793(0.0626+0.0167)-0.0765(0.0543+0.0222) sod-mse=0.0063(0.0111) gcn-mse=0.0103(0.0141) gcn-final-mse=0.0133(0.0304)

2020-08-05 11:56:05    0-5019 loss=1.1154(0.6734+0.4419)-1.1154(0.6734+0.4419) sod-mse=0.0998(0.0998) gcn-mse=0.1047(0.1047) gcn-final-mse=0.0966(0.1103)
2020-08-05 11:57:57 1000-5019 loss=0.0295(0.0235+0.0060)-0.3554(0.1805+0.1748) sod-mse=0.0049(0.0523) gcn-mse=0.0054(0.0537) gcn-final-mse=0.0535(0.0669)
2020-08-05 11:59:48 2000-5019 loss=1.1726(0.5860+0.5866)-0.3612(0.1829+0.1783) sod-mse=0.0947(0.0533) gcn-mse=0.0948(0.0547) gcn-final-mse=0.0545(0.0678)
2020-08-05 12:01:39 3000-5019 loss=0.0479(0.0360+0.0118)-0.3632(0.1831+0.1801) sod-mse=0.0057(0.0538) gcn-mse=0.0090(0.0552) gcn-final-mse=0.0550(0.0683)
2020-08-05 12:02:41 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 12:03:31 4000-5019 loss=0.0983(0.0699+0.0284)-0.3623(0.1831+0.1791) sod-mse=0.0136(0.0533) gcn-mse=0.0134(0.0549) gcn-final-mse=0.0547(0.0680)
2020-08-05 12:04:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 12:05:22 5000-5019 loss=1.3082(0.5620+0.7462)-0.3610(0.1826+0.1783) sod-mse=0.0896(0.0532) gcn-mse=0.0916(0.0549) gcn-final-mse=0.0546(0.0679)
2020-08-05 12:05:24 E:29, Train sod-mae-score=0.0111-0.9829 gcn-mae-score=0.0140-0.9540 gcn-final-mse-score=0.0133-0.9565(0.0304/0.9565) loss=0.0766(0.0544+0.0223)
2020-08-05 12:05:24 E:29, Test  sod-mae-score=0.0532-0.8430 gcn-mae-score=0.0549-0.7899 gcn-final-mse-score=0.0546-0.7958(0.0679/0.7958) loss=0.3608(0.1826+0.1782)

2020-08-05 12:05:24 Start Epoch 30
2020-08-05 12:05:24 Epoch:30,lr=0.0000
2020-08-05 12:05:25    0-10553 loss=0.0380(0.0301+0.0079)-0.0380(0.0301+0.0079) sod-mse=0.0054(0.0054) gcn-mse=0.0056(0.0056) gcn-final-mse=0.0054(0.0210)
2020-08-05 12:09:30 1000-10553 loss=0.0499(0.0373+0.0126)-0.0766(0.0545+0.0222) sod-mse=0.0067(0.0113) gcn-mse=0.0102(0.0141) gcn-final-mse=0.0133(0.0306)
2020-08-05 12:13:36 2000-10553 loss=0.1366(0.0817+0.0549)-0.0771(0.0547+0.0224) sod-mse=0.0223(0.0112) gcn-mse=0.0208(0.0140) gcn-final-mse=0.0132(0.0304)
2020-08-05 12:17:40 3000-10553 loss=0.0602(0.0429+0.0173)-0.0763(0.0542+0.0221) sod-mse=0.0099(0.0110) gcn-mse=0.0082(0.0138) gcn-final-mse=0.0130(0.0302)
2020-08-05 12:20:26 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 12:21:46 4000-10553 loss=0.0582(0.0422+0.0160)-0.0760(0.0540+0.0221) sod-mse=0.0082(0.0110) gcn-mse=0.0099(0.0138) gcn-final-mse=0.0130(0.0301)
2020-08-05 12:25:50 5000-10553 loss=0.0656(0.0467+0.0189)-0.0763(0.0541+0.0222) sod-mse=0.0094(0.0111) gcn-mse=0.0130(0.0139) gcn-final-mse=0.0131(0.0302)
2020-08-05 12:29:54 6000-10553 loss=0.0830(0.0556+0.0274)-0.0760(0.0539+0.0221) sod-mse=0.0132(0.0110) gcn-mse=0.0162(0.0138) gcn-final-mse=0.0130(0.0301)
2020-08-05 12:33:58 7000-10553 loss=0.0924(0.0691+0.0233)-0.0756(0.0537+0.0219) sod-mse=0.0106(0.0109) gcn-mse=0.0126(0.0137) gcn-final-mse=0.0130(0.0300)
2020-08-05 12:38:00 8000-10553 loss=0.0605(0.0467+0.0137)-0.0757(0.0538+0.0219) sod-mse=0.0099(0.0109) gcn-mse=0.0125(0.0137) gcn-final-mse=0.0130(0.0301)
2020-08-05 12:39:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 12:42:04 9000-10553 loss=0.1128(0.0844+0.0284)-0.0757(0.0538+0.0219) sod-mse=0.0137(0.0109) gcn-mse=0.0186(0.0137) gcn-final-mse=0.0129(0.0301)
2020-08-05 12:46:09 10000-10553 loss=0.0676(0.0487+0.0189)-0.0757(0.0539+0.0219) sod-mse=0.0107(0.0109) gcn-mse=0.0140(0.0137) gcn-final-mse=0.0129(0.0301)
2020-08-05 12:46:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg

2020-08-05 12:48:24    0-5019 loss=1.1904(0.6744+0.5161)-1.1904(0.6744+0.5161) sod-mse=0.1021(0.1021) gcn-mse=0.1074(0.1074) gcn-final-mse=0.0996(0.1132)
2020-08-05 12:50:19 1000-5019 loss=0.0293(0.0235+0.0058)-0.3810(0.1850+0.1960) sod-mse=0.0047(0.0521) gcn-mse=0.0057(0.0542) gcn-final-mse=0.0540(0.0677)
2020-08-05 12:52:13 2000-5019 loss=1.3107(0.5968+0.7139)-0.3875(0.1877+0.1998) sod-mse=0.0939(0.0529) gcn-mse=0.0946(0.0551) gcn-final-mse=0.0548(0.0683)
2020-08-05 12:54:07 3000-5019 loss=0.0482(0.0360+0.0121)-0.3899(0.1880+0.2019) sod-mse=0.0057(0.0534) gcn-mse=0.0090(0.0556) gcn-final-mse=0.0553(0.0688)
2020-08-05 12:55:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 12:56:02 4000-5019 loss=0.0966(0.0685+0.0281)-0.3871(0.1872+0.1998) sod-mse=0.0135(0.0528) gcn-mse=0.0128(0.0552) gcn-final-mse=0.0549(0.0684)
2020-08-05 12:56:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 12:57:55 5000-5019 loss=1.3444(0.5109+0.8335)-0.3871(0.1872+0.1999) sod-mse=0.1103(0.0528) gcn-mse=0.1072(0.0552) gcn-final-mse=0.0549(0.0684)
2020-08-05 12:57:57 E:30, Train sod-mae-score=0.0109-0.9833 gcn-mae-score=0.0137-0.9545 gcn-final-mse-score=0.0130-0.9570(0.0301/0.9570) loss=0.0758(0.0539+0.0219)
2020-08-05 12:57:57 E:30, Test  sod-mae-score=0.0528-0.8410 gcn-mae-score=0.0552-0.7879 gcn-final-mse-score=0.0549-0.7936(0.0684/0.7936) loss=0.3869(0.1871+0.1998)

2020-08-05 12:57:57 Start Epoch 31
2020-08-05 12:57:57 Epoch:31,lr=0.0000
2020-08-05 12:57:58    0-10553 loss=0.0716(0.0442+0.0274)-0.0716(0.0442+0.0274) sod-mse=0.0148(0.0148) gcn-mse=0.0102(0.0102) gcn-final-mse=0.0103(0.0267)
2020-08-05 13:02:07 1000-10553 loss=0.0683(0.0522+0.0162)-0.0733(0.0527+0.0207) sod-mse=0.0108(0.0104) gcn-mse=0.0137(0.0132) gcn-final-mse=0.0125(0.0297)
2020-08-05 13:06:13 2000-10553 loss=0.1276(0.0826+0.0450)-0.0747(0.0536+0.0211) sod-mse=0.0254(0.0106) gcn-mse=0.0205(0.0134) gcn-final-mse=0.0127(0.0301)
2020-08-05 13:10:14 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 13:10:21 3000-10553 loss=0.1205(0.0781+0.0424)-0.0746(0.0535+0.0211) sod-mse=0.0244(0.0106) gcn-mse=0.0239(0.0134) gcn-final-mse=0.0126(0.0301)
2020-08-05 13:14:28 4000-10553 loss=0.0930(0.0675+0.0256)-0.0747(0.0535+0.0212) sod-mse=0.0130(0.0106) gcn-mse=0.0156(0.0134) gcn-final-mse=0.0126(0.0301)
2020-08-05 13:17:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 13:18:38 5000-10553 loss=0.0819(0.0584+0.0234)-0.0750(0.0536+0.0213) sod-mse=0.0142(0.0106) gcn-mse=0.0183(0.0135) gcn-final-mse=0.0127(0.0301)
2020-08-05 13:22:48 6000-10553 loss=0.0706(0.0526+0.0179)-0.0749(0.0536+0.0213) sod-mse=0.0080(0.0106) gcn-mse=0.0101(0.0135) gcn-final-mse=0.0127(0.0301)
2020-08-05 13:26:57 7000-10553 loss=0.0497(0.0408+0.0089)-0.0748(0.0535+0.0214) sod-mse=0.0034(0.0107) gcn-mse=0.0068(0.0135) gcn-final-mse=0.0127(0.0300)
2020-08-05 13:31:06 8000-10553 loss=0.0854(0.0578+0.0276)-0.0752(0.0537+0.0216) sod-mse=0.0094(0.0107) gcn-mse=0.0069(0.0135) gcn-final-mse=0.0127(0.0300)
2020-08-05 13:35:13 9000-10553 loss=0.0750(0.0574+0.0176)-0.0748(0.0534+0.0214) sod-mse=0.0104(0.0106) gcn-mse=0.0187(0.0134) gcn-final-mse=0.0126(0.0299)
2020-08-05 13:37:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 13:39:19 10000-10553 loss=0.0413(0.0356+0.0057)-0.0747(0.0534+0.0213) sod-mse=0.0045(0.0106) gcn-mse=0.0049(0.0134) gcn-final-mse=0.0126(0.0299)

2020-08-05 13:41:37    0-5019 loss=1.1173(0.6452+0.4721)-1.1173(0.6452+0.4721) sod-mse=0.0986(0.0986) gcn-mse=0.1030(0.1030) gcn-final-mse=0.0953(0.1095)
2020-08-05 13:43:31 1000-5019 loss=0.0291(0.0232+0.0058)-0.3836(0.1887+0.1949) sod-mse=0.0047(0.0527) gcn-mse=0.0054(0.0547) gcn-final-mse=0.0545(0.0681)
2020-08-05 13:45:22 2000-5019 loss=1.2396(0.5684+0.6712)-0.3903(0.1915+0.1988) sod-mse=0.0954(0.0537) gcn-mse=0.0947(0.0558) gcn-final-mse=0.0555(0.0690)
2020-08-05 13:47:14 3000-5019 loss=0.0483(0.0361+0.0122)-0.3906(0.1909+0.1997) sod-mse=0.0057(0.0540) gcn-mse=0.0089(0.0562) gcn-final-mse=0.0559(0.0694)
2020-08-05 13:48:16 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 13:49:05 4000-5019 loss=0.0983(0.0697+0.0286)-0.3879(0.1901+0.1978) sod-mse=0.0135(0.0535) gcn-mse=0.0127(0.0558) gcn-final-mse=0.0555(0.0690)
2020-08-05 13:49:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 13:50:56 5000-5019 loss=1.3535(0.5191+0.8344)-0.3865(0.1896+0.1970) sod-mse=0.1162(0.0535) gcn-mse=0.1135(0.0558) gcn-final-mse=0.0555(0.0690)
2020-08-05 13:50:58 E:31, Train sod-mae-score=0.0106-0.9837 gcn-mae-score=0.0134-0.9550 gcn-final-mse-score=0.0126-0.9575(0.0299/0.9575) loss=0.0747(0.0534+0.0214)
2020-08-05 13:50:58 E:31, Test  sod-mae-score=0.0535-0.8393 gcn-mae-score=0.0558-0.7854 gcn-final-mse-score=0.0555-0.7910(0.0690/0.7910) loss=0.3864(0.1895+0.1969)

2020-08-05 13:50:58 Start Epoch 32
2020-08-05 13:50:58 Epoch:32,lr=0.0000
2020-08-05 13:51:00    0-10553 loss=0.1011(0.0659+0.0351)-0.1011(0.0659+0.0351) sod-mse=0.0189(0.0189) gcn-mse=0.0188(0.0188) gcn-final-mse=0.0178(0.0366)
2020-08-05 13:55:04 1000-10553 loss=0.0627(0.0381+0.0247)-0.0734(0.0522+0.0212) sod-mse=0.0139(0.0103) gcn-mse=0.0135(0.0129) gcn-final-mse=0.0121(0.0289)
2020-08-05 13:56:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 13:59:08 2000-10553 loss=0.0524(0.0409+0.0115)-0.0737(0.0526+0.0211) sod-mse=0.0068(0.0104) gcn-mse=0.0084(0.0129) gcn-final-mse=0.0122(0.0292)
2020-08-05 13:59:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 14:03:10 3000-10553 loss=0.0822(0.0537+0.0285)-0.0736(0.0526+0.0210) sod-mse=0.0135(0.0104) gcn-mse=0.0169(0.0129) gcn-final-mse=0.0122(0.0294)
2020-08-05 14:07:13 4000-10553 loss=0.0697(0.0456+0.0242)-0.0735(0.0526+0.0208) sod-mse=0.0115(0.0103) gcn-mse=0.0166(0.0130) gcn-final-mse=0.0122(0.0294)
2020-08-05 14:11:16 5000-10553 loss=0.0817(0.0601+0.0217)-0.0734(0.0526+0.0208) sod-mse=0.0112(0.0103) gcn-mse=0.0148(0.0130) gcn-final-mse=0.0122(0.0294)
2020-08-05 14:15:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 14:15:19 6000-10553 loss=0.0931(0.0535+0.0396)-0.0734(0.0526+0.0208) sod-mse=0.0158(0.0103) gcn-mse=0.0111(0.0130) gcn-final-mse=0.0122(0.0294)
2020-08-05 14:19:23 7000-10553 loss=0.0664(0.0493+0.0172)-0.0743(0.0530+0.0213) sod-mse=0.0086(0.0105) gcn-mse=0.0078(0.0131) gcn-final-mse=0.0124(0.0295)
2020-08-05 14:23:26 8000-10553 loss=0.0612(0.0483+0.0129)-0.0745(0.0531+0.0213) sod-mse=0.0068(0.0105) gcn-mse=0.0116(0.0132) gcn-final-mse=0.0124(0.0296)
2020-08-05 14:27:32 9000-10553 loss=0.0585(0.0426+0.0159)-0.0748(0.0533+0.0215) sod-mse=0.0074(0.0106) gcn-mse=0.0079(0.0132) gcn-final-mse=0.0125(0.0297)
2020-08-05 14:31:34 10000-10553 loss=0.0389(0.0285+0.0104)-0.0747(0.0533+0.0214) sod-mse=0.0066(0.0106) gcn-mse=0.0098(0.0132) gcn-final-mse=0.0125(0.0298)

2020-08-05 14:33:50    0-5019 loss=1.2352(0.6905+0.5447)-1.2352(0.6905+0.5447) sod-mse=0.1039(0.1039) gcn-mse=0.1082(0.1082) gcn-final-mse=0.1002(0.1141)
2020-08-05 14:35:43 1000-5019 loss=0.0284(0.0231+0.0052)-0.3875(0.1876+0.1999) sod-mse=0.0042(0.0514) gcn-mse=0.0053(0.0535) gcn-final-mse=0.0532(0.0668)
2020-08-05 14:37:35 2000-5019 loss=1.2347(0.5630+0.6718)-0.3935(0.1899+0.2036) sod-mse=0.0928(0.0520) gcn-mse=0.0931(0.0542) gcn-final-mse=0.0539(0.0673)
2020-08-05 14:39:28 3000-5019 loss=0.0485(0.0365+0.0120)-0.3960(0.1901+0.2059) sod-mse=0.0056(0.0523) gcn-mse=0.0086(0.0545) gcn-final-mse=0.0543(0.0677)
2020-08-05 14:40:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 14:41:21 4000-5019 loss=0.0983(0.0696+0.0287)-0.3928(0.1893+0.2035) sod-mse=0.0132(0.0517) gcn-mse=0.0130(0.0541) gcn-final-mse=0.0538(0.0672)
2020-08-05 14:41:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 14:43:15 5000-5019 loss=1.3765(0.5236+0.8529)-0.3923(0.1892+0.2030) sod-mse=0.1139(0.0518) gcn-mse=0.1113(0.0542) gcn-final-mse=0.0539(0.0672)
2020-08-05 14:43:16 E:32, Train sod-mae-score=0.0105-0.9841 gcn-mae-score=0.0132-0.9550 gcn-final-mse-score=0.0124-0.9575(0.0297/0.9575) loss=0.0746(0.0532+0.0213)
2020-08-05 14:43:16 E:32, Test  sod-mae-score=0.0518-0.8448 gcn-mae-score=0.0542-0.7888 gcn-final-mse-score=0.0539-0.7946(0.0672/0.7946) loss=0.3920(0.1891+0.2029)

2020-08-05 14:43:16 Start Epoch 33
2020-08-05 14:43:16 Epoch:33,lr=0.0000
2020-08-05 14:43:18    0-10553 loss=0.0431(0.0328+0.0102)-0.0431(0.0328+0.0102) sod-mse=0.0050(0.0050) gcn-mse=0.0128(0.0128) gcn-final-mse=0.0107(0.0176)
2020-08-05 14:47:24 1000-10553 loss=0.0723(0.0522+0.0202)-0.0741(0.0531+0.0210) sod-mse=0.0107(0.0106) gcn-mse=0.0190(0.0132) gcn-final-mse=0.0124(0.0296)
2020-08-05 14:51:29 2000-10553 loss=0.0623(0.0478+0.0145)-0.0734(0.0526+0.0208) sod-mse=0.0072(0.0104) gcn-mse=0.0101(0.0130) gcn-final-mse=0.0122(0.0294)
2020-08-05 14:52:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 14:55:34 3000-10553 loss=0.0858(0.0641+0.0217)-0.0733(0.0526+0.0207) sod-mse=0.0140(0.0103) gcn-mse=0.0154(0.0129) gcn-final-mse=0.0121(0.0293)
2020-08-05 14:59:38 4000-10553 loss=0.0548(0.0420+0.0128)-0.0729(0.0524+0.0205) sod-mse=0.0061(0.0102) gcn-mse=0.0065(0.0129) gcn-final-mse=0.0121(0.0293)
2020-08-05 15:01:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 15:03:42 5000-10553 loss=0.0400(0.0297+0.0103)-0.0726(0.0522+0.0204) sod-mse=0.0054(0.0101) gcn-mse=0.0086(0.0128) gcn-final-mse=0.0120(0.0292)
2020-08-05 15:07:47 6000-10553 loss=0.0365(0.0266+0.0098)-0.0729(0.0523+0.0206) sod-mse=0.0070(0.0102) gcn-mse=0.0108(0.0128) gcn-final-mse=0.0121(0.0293)
2020-08-05 15:11:49 7000-10553 loss=0.0482(0.0350+0.0132)-0.0728(0.0523+0.0205) sod-mse=0.0089(0.0102) gcn-mse=0.0091(0.0128) gcn-final-mse=0.0120(0.0293)
2020-08-05 15:15:05 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 15:15:53 8000-10553 loss=0.1124(0.0712+0.0412)-0.0730(0.0525+0.0205) sod-mse=0.0147(0.0102) gcn-mse=0.0195(0.0128) gcn-final-mse=0.0121(0.0294)
2020-08-05 15:19:58 9000-10553 loss=0.0879(0.0671+0.0208)-0.0730(0.0525+0.0205) sod-mse=0.0112(0.0102) gcn-mse=0.0144(0.0128) gcn-final-mse=0.0120(0.0293)
2020-08-05 15:24:01 10000-10553 loss=0.0773(0.0586+0.0187)-0.0731(0.0525+0.0206) sod-mse=0.0086(0.0102) gcn-mse=0.0158(0.0128) gcn-final-mse=0.0121(0.0294)

2020-08-05 15:26:17    0-5019 loss=1.1809(0.6761+0.5048)-1.1809(0.6761+0.5048) sod-mse=0.0999(0.0999) gcn-mse=0.1077(0.1077) gcn-final-mse=0.0994(0.1130)
2020-08-05 15:28:10 1000-5019 loss=0.0286(0.0232+0.0054)-0.3886(0.1883+0.2004) sod-mse=0.0043(0.0517) gcn-mse=0.0054(0.0538) gcn-final-mse=0.0536(0.0672)
2020-08-05 15:30:01 2000-5019 loss=1.2538(0.5774+0.6765)-0.3965(0.1914+0.2051) sod-mse=0.0931(0.0525) gcn-mse=0.0929(0.0546) gcn-final-mse=0.0543(0.0678)
2020-08-05 15:31:52 3000-5019 loss=0.0476(0.0357+0.0119)-0.3978(0.1913+0.2065) sod-mse=0.0054(0.0528) gcn-mse=0.0083(0.0550) gcn-final-mse=0.0548(0.0682)
2020-08-05 15:32:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 15:33:43 4000-5019 loss=0.0972(0.0684+0.0288)-0.3956(0.1908+0.2048) sod-mse=0.0131(0.0523) gcn-mse=0.0124(0.0547) gcn-final-mse=0.0544(0.0678)
2020-08-05 15:34:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 15:35:34 5000-5019 loss=1.1776(0.4388+0.7388)-0.3942(0.1904+0.2039) sod-mse=0.1014(0.0523) gcn-mse=0.1011(0.0547) gcn-final-mse=0.0544(0.0678)
2020-08-05 15:35:36 E:33, Train sod-mae-score=0.0102-0.9844 gcn-mae-score=0.0129-0.9554 gcn-final-mse-score=0.0121-0.9579(0.0294/0.9579) loss=0.0732(0.0526+0.0206)
2020-08-05 15:35:36 E:33, Test  sod-mae-score=0.0523-0.8413 gcn-mae-score=0.0547-0.7872 gcn-final-mse-score=0.0544-0.7929(0.0678/0.7929) loss=0.3941(0.1903+0.2038)

2020-08-05 15:35:36 Start Epoch 34
2020-08-05 15:35:36 Epoch:34,lr=0.0000
2020-08-05 15:35:37    0-10553 loss=0.0699(0.0508+0.0191)-0.0699(0.0508+0.0191) sod-mse=0.0081(0.0081) gcn-mse=0.0082(0.0082) gcn-final-mse=0.0095(0.0394)
2020-08-05 15:39:24 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 15:39:42 1000-10553 loss=0.1417(0.0940+0.0477)-0.0726(0.0522+0.0204) sod-mse=0.0265(0.0102) gcn-mse=0.0236(0.0128) gcn-final-mse=0.0120(0.0293)
2020-08-05 15:43:46 2000-10553 loss=0.0457(0.0354+0.0103)-0.0720(0.0518+0.0202) sod-mse=0.0046(0.0100) gcn-mse=0.0063(0.0127) gcn-final-mse=0.0118(0.0289)
2020-08-05 15:47:50 3000-10553 loss=0.0481(0.0405+0.0076)-0.0724(0.0521+0.0203) sod-mse=0.0041(0.0101) gcn-mse=0.0082(0.0127) gcn-final-mse=0.0119(0.0291)
2020-08-05 15:51:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 15:51:53 4000-10553 loss=0.1116(0.0802+0.0314)-0.0727(0.0523+0.0205) sod-mse=0.0152(0.0101) gcn-mse=0.0162(0.0127) gcn-final-mse=0.0120(0.0292)
2020-08-05 15:55:57 5000-10553 loss=0.1316(0.1028+0.0288)-0.0726(0.0523+0.0204) sod-mse=0.0198(0.0101) gcn-mse=0.0359(0.0127) gcn-final-mse=0.0119(0.0292)
2020-08-05 16:00:03 6000-10553 loss=0.0773(0.0544+0.0229)-0.0731(0.0524+0.0207) sod-mse=0.0137(0.0101) gcn-mse=0.0192(0.0127) gcn-final-mse=0.0119(0.0292)
2020-08-05 16:04:08 7000-10553 loss=0.0471(0.0355+0.0116)-0.0735(0.0525+0.0209) sod-mse=0.0058(0.0102) gcn-mse=0.0041(0.0127) gcn-final-mse=0.0120(0.0292)
2020-08-05 16:04:52 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 16:08:11 8000-10553 loss=0.1653(0.1076+0.0578)-0.0733(0.0525+0.0208) sod-mse=0.0263(0.0102) gcn-mse=0.0265(0.0127) gcn-final-mse=0.0119(0.0292)
2020-08-05 16:12:14 9000-10553 loss=0.0349(0.0295+0.0054)-0.0737(0.0527+0.0210) sod-mse=0.0039(0.0102) gcn-mse=0.0070(0.0128) gcn-final-mse=0.0120(0.0293)
2020-08-05 16:16:17 10000-10553 loss=0.0652(0.0437+0.0215)-0.0735(0.0526+0.0209) sod-mse=0.0110(0.0102) gcn-mse=0.0076(0.0127) gcn-final-mse=0.0120(0.0293)

2020-08-05 16:18:33    0-5019 loss=1.1895(0.6780+0.5114)-1.1895(0.6780+0.5114) sod-mse=0.0991(0.0991) gcn-mse=0.1042(0.1042) gcn-final-mse=0.0962(0.1103)
2020-08-05 16:20:25 1000-5019 loss=0.0283(0.0230+0.0053)-0.3815(0.1873+0.1942) sod-mse=0.0042(0.0523) gcn-mse=0.0052(0.0540) gcn-final-mse=0.0537(0.0674)
2020-08-05 16:22:16 2000-5019 loss=1.3144(0.6196+0.6948)-0.3873(0.1899+0.1974) sod-mse=0.0937(0.0530) gcn-mse=0.0946(0.0548) gcn-final-mse=0.0545(0.0680)
2020-08-05 16:24:06 3000-5019 loss=0.0475(0.0357+0.0118)-0.3883(0.1894+0.1989) sod-mse=0.0055(0.0533) gcn-mse=0.0089(0.0551) gcn-final-mse=0.0549(0.0684)
2020-08-05 16:25:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 16:25:57 4000-5019 loss=0.0988(0.0701+0.0287)-0.3863(0.1890+0.1973) sod-mse=0.0131(0.0529) gcn-mse=0.0130(0.0548) gcn-final-mse=0.0545(0.0680)
2020-08-05 16:26:29 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 16:27:47 5000-5019 loss=1.3318(0.5330+0.7988)-0.3853(0.1888+0.1965) sod-mse=0.1263(0.0529) gcn-mse=0.1183(0.0548) gcn-final-mse=0.0545(0.0680)
2020-08-05 16:27:49 E:34, Train sod-mae-score=0.0102-0.9846 gcn-mae-score=0.0127-0.9557 gcn-final-mse-score=0.0120-0.9582(0.0293/0.9582) loss=0.0734(0.0526+0.0208)
2020-08-05 16:27:49 E:34, Test  sod-mae-score=0.0529-0.8412 gcn-mae-score=0.0548-0.7877 gcn-final-mse-score=0.0545-0.7933(0.0680/0.7933) loss=0.3851(0.1887+0.1964)

2020-08-05 16:27:49 Start Epoch 35
2020-08-05 16:27:49 Epoch:35,lr=0.0000
2020-08-05 16:27:51    0-10553 loss=0.0394(0.0295+0.0098)-0.0394(0.0295+0.0098) sod-mse=0.0068(0.0068) gcn-mse=0.0064(0.0064) gcn-final-mse=0.0062(0.0171)
2020-08-05 16:31:55 1000-10553 loss=0.1371(0.0943+0.0428)-0.0713(0.0518+0.0195) sod-mse=0.0151(0.0101) gcn-mse=0.0192(0.0126) gcn-final-mse=0.0119(0.0296)
2020-08-05 16:35:58 2000-10553 loss=0.0674(0.0498+0.0175)-0.0709(0.0516+0.0194) sod-mse=0.0096(0.0099) gcn-mse=0.0117(0.0124) gcn-final-mse=0.0117(0.0293)
2020-08-05 16:40:00 3000-10553 loss=0.0661(0.0501+0.0160)-0.0722(0.0522+0.0200) sod-mse=0.0116(0.0100) gcn-mse=0.0117(0.0125) gcn-final-mse=0.0117(0.0294)
2020-08-05 16:44:04 4000-10553 loss=0.0595(0.0407+0.0187)-0.0719(0.0520+0.0199) sod-mse=0.0094(0.0099) gcn-mse=0.0117(0.0124) gcn-final-mse=0.0117(0.0292)
2020-08-05 16:45:10 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 16:48:09 5000-10553 loss=0.0679(0.0519+0.0160)-0.0720(0.0520+0.0200) sod-mse=0.0067(0.0100) gcn-mse=0.0113(0.0124) gcn-final-mse=0.0117(0.0293)
2020-08-05 16:50:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 16:52:14 6000-10553 loss=0.0470(0.0392+0.0077)-0.0715(0.0517+0.0198) sod-mse=0.0038(0.0099) gcn-mse=0.0053(0.0124) gcn-final-mse=0.0116(0.0291)
2020-08-05 16:56:19 7000-10553 loss=0.1015(0.0828+0.0187)-0.0714(0.0517+0.0197) sod-mse=0.0135(0.0098) gcn-mse=0.0146(0.0123) gcn-final-mse=0.0116(0.0291)
2020-08-05 17:00:23 8000-10553 loss=0.0381(0.0332+0.0050)-0.0714(0.0516+0.0197) sod-mse=0.0021(0.0098) gcn-mse=0.0046(0.0123) gcn-final-mse=0.0115(0.0290)
2020-08-05 17:02:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 17:04:27 9000-10553 loss=0.0474(0.0367+0.0107)-0.0716(0.0517+0.0199) sod-mse=0.0055(0.0098) gcn-mse=0.0074(0.0123) gcn-final-mse=0.0116(0.0291)
2020-08-05 17:08:29 10000-10553 loss=0.0964(0.0735+0.0229)-0.0717(0.0518+0.0199) sod-mse=0.0127(0.0099) gcn-mse=0.0154(0.0123) gcn-final-mse=0.0116(0.0290)

2020-08-05 17:10:46    0-5019 loss=1.1936(0.6819+0.5117)-1.1936(0.6819+0.5117) sod-mse=0.0989(0.0989) gcn-mse=0.1044(0.1044) gcn-final-mse=0.0963(0.1102)
2020-08-05 17:12:37 1000-5019 loss=0.0282(0.0229+0.0053)-0.3887(0.1899+0.1987) sod-mse=0.0043(0.0520) gcn-mse=0.0051(0.0538) gcn-final-mse=0.0536(0.0672)
2020-08-05 17:14:27 2000-5019 loss=1.3162(0.6156+0.7007)-0.3947(0.1925+0.2022) sod-mse=0.0944(0.0527) gcn-mse=0.0942(0.0546) gcn-final-mse=0.0543(0.0677)
2020-08-05 17:16:17 3000-5019 loss=0.0473(0.0354+0.0118)-0.3958(0.1921+0.2037) sod-mse=0.0054(0.0531) gcn-mse=0.0083(0.0550) gcn-final-mse=0.0547(0.0681)
2020-08-05 17:17:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 17:18:08 4000-5019 loss=0.0990(0.0704+0.0286)-0.3936(0.1916+0.2020) sod-mse=0.0131(0.0526) gcn-mse=0.0131(0.0547) gcn-final-mse=0.0544(0.0677)
2020-08-05 17:18:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 17:19:58 5000-5019 loss=1.3586(0.5412+0.8174)-0.3926(0.1914+0.2012) sod-mse=0.1252(0.0526) gcn-mse=0.1178(0.0547) gcn-final-mse=0.0544(0.0677)
2020-08-05 17:20:00 E:35, Train sod-mae-score=0.0099-0.9850 gcn-mae-score=0.0123-0.9561 gcn-final-mse-score=0.0116-0.9585(0.0290/0.9585) loss=0.0717(0.0518+0.0199)
2020-08-05 17:20:00 E:35, Test  sod-mae-score=0.0526-0.8403 gcn-mae-score=0.0547-0.7879 gcn-final-mse-score=0.0543-0.7937(0.0677/0.7937) loss=0.3924(0.1913+0.2011)

2020-08-05 17:20:00 Start Epoch 36
2020-08-05 17:20:00 Epoch:36,lr=0.0000
2020-08-05 17:20:01    0-10553 loss=0.1245(0.0712+0.0532)-0.1245(0.0712+0.0532) sod-mse=0.0250(0.0250) gcn-mse=0.0264(0.0264) gcn-final-mse=0.0241(0.0350)
2020-08-05 17:24:08 1000-10553 loss=0.0504(0.0374+0.0130)-0.0724(0.0522+0.0202) sod-mse=0.0063(0.0098) gcn-mse=0.0083(0.0121) gcn-final-mse=0.0113(0.0291)
2020-08-05 17:28:12 2000-10553 loss=0.0847(0.0538+0.0309)-0.0724(0.0522+0.0202) sod-mse=0.0159(0.0098) gcn-mse=0.0128(0.0122) gcn-final-mse=0.0114(0.0291)
2020-08-05 17:29:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 17:32:17 3000-10553 loss=0.0827(0.0633+0.0194)-0.0721(0.0520+0.0201) sod-mse=0.0099(0.0099) gcn-mse=0.0180(0.0122) gcn-final-mse=0.0115(0.0291)
2020-08-05 17:36:22 4000-10553 loss=0.0626(0.0449+0.0177)-0.0720(0.0519+0.0201) sod-mse=0.0083(0.0099) gcn-mse=0.0121(0.0122) gcn-final-mse=0.0115(0.0289)
2020-08-05 17:37:03 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 17:40:26 5000-10553 loss=0.0472(0.0343+0.0129)-0.0720(0.0519+0.0200) sod-mse=0.0068(0.0099) gcn-mse=0.0093(0.0123) gcn-final-mse=0.0115(0.0290)
2020-08-05 17:44:29 6000-10553 loss=0.0499(0.0384+0.0115)-0.0716(0.0517+0.0199) sod-mse=0.0061(0.0098) gcn-mse=0.0089(0.0122) gcn-final-mse=0.0115(0.0289)
2020-08-05 17:44:38 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 17:48:33 7000-10553 loss=0.0189(0.0146+0.0043)-0.0716(0.0517+0.0199) sod-mse=0.0021(0.0098) gcn-mse=0.0034(0.0122) gcn-final-mse=0.0115(0.0290)
2020-08-05 17:52:36 8000-10553 loss=0.0827(0.0584+0.0243)-0.0717(0.0518+0.0199) sod-mse=0.0153(0.0098) gcn-mse=0.0131(0.0122) gcn-final-mse=0.0115(0.0290)
2020-08-05 17:56:39 9000-10553 loss=0.0904(0.0714+0.0190)-0.0714(0.0517+0.0198) sod-mse=0.0112(0.0098) gcn-mse=0.0225(0.0122) gcn-final-mse=0.0114(0.0289)
2020-08-05 18:00:43 10000-10553 loss=0.0593(0.0476+0.0117)-0.0713(0.0516+0.0197) sod-mse=0.0050(0.0098) gcn-mse=0.0100(0.0122) gcn-final-mse=0.0114(0.0289)

2020-08-05 18:02:59    0-5019 loss=1.2179(0.6927+0.5252)-1.2179(0.6927+0.5252) sod-mse=0.0998(0.0998) gcn-mse=0.1053(0.1053) gcn-final-mse=0.0972(0.1111)
2020-08-05 18:04:50 1000-5019 loss=0.0280(0.0228+0.0051)-0.3942(0.1901+0.2042) sod-mse=0.0041(0.0514) gcn-mse=0.0050(0.0534) gcn-final-mse=0.0531(0.0666)
2020-08-05 18:06:40 2000-5019 loss=1.3597(0.6239+0.7358)-0.3994(0.1922+0.2071) sod-mse=0.0945(0.0520) gcn-mse=0.0943(0.0541) gcn-final-mse=0.0538(0.0672)
2020-08-05 18:08:30 3000-5019 loss=0.0473(0.0354+0.0119)-0.4008(0.1920+0.2088) sod-mse=0.0054(0.0524) gcn-mse=0.0083(0.0545) gcn-final-mse=0.0543(0.0676)
2020-08-05 18:09:31 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 18:10:20 4000-5019 loss=0.0991(0.0703+0.0288)-0.3985(0.1915+0.2070) sod-mse=0.0131(0.0520) gcn-mse=0.0129(0.0542) gcn-final-mse=0.0539(0.0672)
2020-08-05 18:10:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 18:12:11 5000-5019 loss=1.4084(0.5552+0.8532)-0.3977(0.1913+0.2064) sod-mse=0.1137(0.0520) gcn-mse=0.1111(0.0542) gcn-final-mse=0.0539(0.0672)
2020-08-05 18:12:13 E:36, Train sod-mae-score=0.0098-0.9851 gcn-mae-score=0.0122-0.9562 gcn-final-mse-score=0.0115-0.9587(0.0289/0.9587) loss=0.0713(0.0516+0.0197)
2020-08-05 18:12:13 E:36, Test  sod-mae-score=0.0520-0.8415 gcn-mae-score=0.0542-0.7891 gcn-final-mse-score=0.0539-0.7949(0.0672/0.7949) loss=0.3974(0.1912+0.2063)

2020-08-05 18:12:13 Start Epoch 37
2020-08-05 18:12:13 Epoch:37,lr=0.0000
2020-08-05 18:12:14    0-10553 loss=0.0742(0.0576+0.0166)-0.0742(0.0576+0.0166) sod-mse=0.0080(0.0080) gcn-mse=0.0107(0.0107) gcn-final-mse=0.0096(0.0307)
2020-08-05 18:16:20 1000-10553 loss=0.0263(0.0222+0.0041)-0.0713(0.0516+0.0197) sod-mse=0.0031(0.0098) gcn-mse=0.0037(0.0121) gcn-final-mse=0.0114(0.0290)
2020-08-05 18:20:24 2000-10553 loss=0.0794(0.0577+0.0217)-0.0712(0.0516+0.0196) sod-mse=0.0102(0.0097) gcn-mse=0.0148(0.0120) gcn-final-mse=0.0113(0.0288)
2020-08-05 18:24:29 3000-10553 loss=0.0450(0.0354+0.0096)-0.0714(0.0517+0.0197) sod-mse=0.0045(0.0098) gcn-mse=0.0093(0.0122) gcn-final-mse=0.0114(0.0290)
2020-08-05 18:28:36 4000-10553 loss=0.0463(0.0369+0.0094)-0.0710(0.0514+0.0195) sod-mse=0.0068(0.0097) gcn-mse=0.0052(0.0121) gcn-final-mse=0.0114(0.0288)
2020-08-05 18:32:38 5000-10553 loss=0.1057(0.0637+0.0419)-0.0705(0.0512+0.0194) sod-mse=0.0214(0.0096) gcn-mse=0.0201(0.0121) gcn-final-mse=0.0113(0.0288)
2020-08-05 18:36:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 18:36:42 6000-10553 loss=0.0792(0.0556+0.0236)-0.0706(0.0512+0.0194) sod-mse=0.0139(0.0096) gcn-mse=0.0175(0.0121) gcn-final-mse=0.0113(0.0287)
2020-08-05 18:40:46 7000-10553 loss=0.0671(0.0415+0.0256)-0.0708(0.0513+0.0195) sod-mse=0.0109(0.0096) gcn-mse=0.0098(0.0121) gcn-final-mse=0.0113(0.0288)
2020-08-05 18:42:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 18:44:50 8000-10553 loss=0.0842(0.0595+0.0248)-0.0708(0.0513+0.0195) sod-mse=0.0130(0.0096) gcn-mse=0.0123(0.0121) gcn-final-mse=0.0113(0.0288)
2020-08-05 18:48:54 9000-10553 loss=0.0639(0.0486+0.0153)-0.0710(0.0514+0.0196) sod-mse=0.0075(0.0097) gcn-mse=0.0107(0.0121) gcn-final-mse=0.0113(0.0288)
2020-08-05 18:52:18 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 18:52:57 10000-10553 loss=0.1248(0.1038+0.0210)-0.0711(0.0515+0.0196) sod-mse=0.0101(0.0097) gcn-mse=0.0175(0.0121) gcn-final-mse=0.0114(0.0289)

2020-08-05 18:55:12    0-5019 loss=1.2035(0.6878+0.5158)-1.2035(0.6878+0.5158) sod-mse=0.0993(0.0993) gcn-mse=0.1049(0.1049) gcn-final-mse=0.0970(0.1109)
2020-08-05 18:57:05 1000-5019 loss=0.0280(0.0228+0.0053)-0.3960(0.1911+0.2049) sod-mse=0.0042(0.0515) gcn-mse=0.0050(0.0534) gcn-final-mse=0.0532(0.0667)
2020-08-05 18:58:55 2000-5019 loss=1.3588(0.6268+0.7319)-0.4014(0.1933+0.2081) sod-mse=0.0945(0.0522) gcn-mse=0.0942(0.0542) gcn-final-mse=0.0539(0.0672)
2020-08-05 19:00:46 3000-5019 loss=0.0474(0.0355+0.0119)-0.4025(0.1929+0.2096) sod-mse=0.0054(0.0526) gcn-mse=0.0084(0.0546) gcn-final-mse=0.0543(0.0677)
2020-08-05 19:01:48 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 19:02:37 4000-5019 loss=0.0988(0.0702+0.0286)-0.4001(0.1924+0.2077) sod-mse=0.0130(0.0521) gcn-mse=0.0128(0.0542) gcn-final-mse=0.0540(0.0673)
2020-08-05 19:03:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 19:04:27 5000-5019 loss=1.3804(0.5437+0.8367)-0.3993(0.1921+0.2071) sod-mse=0.1120(0.0521) gcn-mse=0.1096(0.0543) gcn-final-mse=0.0540(0.0673)
2020-08-05 19:04:29 E:37, Train sod-mae-score=0.0097-0.9852 gcn-mae-score=0.0121-0.9562 gcn-final-mse-score=0.0114-0.9586(0.0289/0.9586) loss=0.0710(0.0515+0.0196)
2020-08-05 19:04:29 E:37, Test  sod-mae-score=0.0521-0.8412 gcn-mae-score=0.0543-0.7887 gcn-final-mse-score=0.0540-0.7945(0.0673/0.7945) loss=0.3991(0.1921+0.2070)

2020-08-05 19:04:29 Start Epoch 38
2020-08-05 19:04:29 Epoch:38,lr=0.0000
2020-08-05 19:04:30    0-10553 loss=0.1071(0.0798+0.0273)-0.1071(0.0798+0.0273) sod-mse=0.0099(0.0099) gcn-mse=0.0130(0.0130) gcn-final-mse=0.0120(0.0453)
2020-08-05 19:08:35 1000-10553 loss=0.0433(0.0269+0.0164)-0.0703(0.0511+0.0192) sod-mse=0.0073(0.0093) gcn-mse=0.0060(0.0119) gcn-final-mse=0.0111(0.0286)
2020-08-05 19:12:39 2000-10553 loss=0.0805(0.0619+0.0186)-0.0709(0.0514+0.0195) sod-mse=0.0093(0.0095) gcn-mse=0.0104(0.0120) gcn-final-mse=0.0112(0.0286)
2020-08-05 19:16:46 3000-10553 loss=0.0583(0.0439+0.0144)-0.0702(0.0509+0.0193) sod-mse=0.0078(0.0095) gcn-mse=0.0069(0.0119) gcn-final-mse=0.0111(0.0285)
2020-08-05 19:20:51 4000-10553 loss=0.0444(0.0317+0.0128)-0.0703(0.0510+0.0193) sod-mse=0.0072(0.0095) gcn-mse=0.0091(0.0119) gcn-final-mse=0.0112(0.0286)
2020-08-05 19:24:54 5000-10553 loss=0.0739(0.0578+0.0161)-0.0702(0.0511+0.0192) sod-mse=0.0111(0.0095) gcn-mse=0.0097(0.0119) gcn-final-mse=0.0112(0.0287)
2020-08-05 19:28:59 6000-10553 loss=0.0821(0.0580+0.0241)-0.0702(0.0510+0.0192) sod-mse=0.0138(0.0095) gcn-mse=0.0109(0.0119) gcn-final-mse=0.0111(0.0287)
2020-08-05 19:33:02 7000-10553 loss=0.0289(0.0208+0.0080)-0.0705(0.0512+0.0193) sod-mse=0.0042(0.0095) gcn-mse=0.0039(0.0120) gcn-final-mse=0.0112(0.0287)
2020-08-05 19:34:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 19:37:05 8000-10553 loss=0.0882(0.0697+0.0184)-0.0709(0.0515+0.0194) sod-mse=0.0088(0.0096) gcn-mse=0.0139(0.0120) gcn-final-mse=0.0113(0.0289)
2020-08-05 19:41:11 9000-10553 loss=0.1105(0.0829+0.0276)-0.0709(0.0515+0.0195) sod-mse=0.0196(0.0096) gcn-mse=0.0181(0.0121) gcn-final-mse=0.0113(0.0289)
2020-08-05 19:44:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 19:45:14 10000-10553 loss=0.1006(0.0763+0.0243)-0.0709(0.0514+0.0195) sod-mse=0.0109(0.0096) gcn-mse=0.0145(0.0121) gcn-final-mse=0.0113(0.0288)
2020-08-05 19:45:44 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg

2020-08-05 19:47:29    0-5019 loss=1.2120(0.6937+0.5183)-1.2120(0.6937+0.5183) sod-mse=0.1001(0.1001) gcn-mse=0.1057(0.1057) gcn-final-mse=0.0977(0.1115)
2020-08-05 19:49:22 1000-5019 loss=0.0280(0.0228+0.0052)-0.3978(0.1911+0.2067) sod-mse=0.0042(0.0513) gcn-mse=0.0051(0.0533) gcn-final-mse=0.0530(0.0666)
2020-08-05 19:51:14 2000-5019 loss=1.3595(0.6271+0.7324)-0.4034(0.1934+0.2100) sod-mse=0.0946(0.0520) gcn-mse=0.0944(0.0540) gcn-final-mse=0.0537(0.0671)
2020-08-05 19:53:06 3000-5019 loss=0.0475(0.0356+0.0119)-0.4046(0.1931+0.2115) sod-mse=0.0054(0.0524) gcn-mse=0.0085(0.0545) gcn-final-mse=0.0542(0.0675)
2020-08-05 19:54:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 19:54:57 4000-5019 loss=0.0987(0.0701+0.0286)-0.4023(0.1926+0.2097) sod-mse=0.0131(0.0519) gcn-mse=0.0129(0.0541) gcn-final-mse=0.0538(0.0672)
2020-08-05 19:55:30 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 19:56:48 5000-5019 loss=1.3573(0.5303+0.8270)-0.4016(0.1924+0.2092) sod-mse=0.1103(0.0519) gcn-mse=0.1081(0.0542) gcn-final-mse=0.0539(0.0672)
2020-08-05 19:56:50 E:38, Train sod-mae-score=0.0096-0.9854 gcn-mae-score=0.0121-0.9565 gcn-final-mse-score=0.0113-0.9590(0.0288/0.9590) loss=0.0708(0.0514+0.0194)
2020-08-05 19:56:50 E:38, Test  sod-mae-score=0.0519-0.8412 gcn-mae-score=0.0542-0.7886 gcn-final-mse-score=0.0538-0.7946(0.0672/0.7946) loss=0.4014(0.1923+0.2091)

2020-08-05 19:56:50 Start Epoch 39
2020-08-05 19:56:50 Epoch:39,lr=0.0000
2020-08-05 19:56:52    0-10553 loss=0.0451(0.0352+0.0099)-0.0451(0.0352+0.0099) sod-mse=0.0067(0.0067) gcn-mse=0.0068(0.0068) gcn-final-mse=0.0065(0.0200)
2020-08-05 20:00:56 1000-10553 loss=0.0669(0.0513+0.0156)-0.0682(0.0498+0.0183) sod-mse=0.0071(0.0091) gcn-mse=0.0103(0.0117) gcn-final-mse=0.0110(0.0281)
2020-08-05 20:05:00 2000-10553 loss=0.0825(0.0625+0.0200)-0.0693(0.0505+0.0187) sod-mse=0.0089(0.0093) gcn-mse=0.0108(0.0118) gcn-final-mse=0.0111(0.0284)
2020-08-05 20:09:04 3000-10553 loss=0.1013(0.0766+0.0247)-0.0704(0.0512+0.0191) sod-mse=0.0113(0.0094) gcn-mse=0.0194(0.0119) gcn-final-mse=0.0112(0.0287)
2020-08-05 20:13:08 4000-10553 loss=0.0401(0.0308+0.0093)-0.0703(0.0512+0.0191) sod-mse=0.0069(0.0094) gcn-mse=0.0071(0.0119) gcn-final-mse=0.0112(0.0286)
2020-08-05 20:13:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 20:15:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 20:16:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 20:17:10 5000-10553 loss=0.0769(0.0567+0.0202)-0.0705(0.0513+0.0192) sod-mse=0.0088(0.0095) gcn-mse=0.0135(0.0120) gcn-final-mse=0.0112(0.0287)
2020-08-05 20:21:13 6000-10553 loss=0.0588(0.0410+0.0178)-0.0704(0.0512+0.0192) sod-mse=0.0077(0.0095) gcn-mse=0.0068(0.0120) gcn-final-mse=0.0112(0.0287)
2020-08-05 20:25:16 7000-10553 loss=0.0547(0.0416+0.0131)-0.0704(0.0513+0.0192) sod-mse=0.0061(0.0095) gcn-mse=0.0093(0.0120) gcn-final-mse=0.0112(0.0288)
2020-08-05 20:29:20 8000-10553 loss=0.0647(0.0479+0.0169)-0.0704(0.0512+0.0192) sod-mse=0.0099(0.0095) gcn-mse=0.0115(0.0120) gcn-final-mse=0.0112(0.0288)
2020-08-05 20:33:24 9000-10553 loss=0.0243(0.0197+0.0045)-0.0704(0.0512+0.0192) sod-mse=0.0037(0.0095) gcn-mse=0.0035(0.0120) gcn-final-mse=0.0112(0.0288)
2020-08-05 20:37:26 10000-10553 loss=0.0398(0.0262+0.0136)-0.0705(0.0513+0.0193) sod-mse=0.0068(0.0095) gcn-mse=0.0054(0.0120) gcn-final-mse=0.0113(0.0288)

2020-08-05 20:39:42    0-5019 loss=1.2002(0.6916+0.5086)-1.2002(0.6916+0.5086) sod-mse=0.1001(0.1001) gcn-mse=0.1056(0.1056) gcn-final-mse=0.0976(0.1114)
2020-08-05 20:41:35 1000-5019 loss=0.0281(0.0228+0.0053)-0.3951(0.1905+0.2046) sod-mse=0.0043(0.0514) gcn-mse=0.0050(0.0532) gcn-final-mse=0.0530(0.0665)
2020-08-05 20:43:28 2000-5019 loss=1.3579(0.6303+0.7275)-0.4007(0.1928+0.2079) sod-mse=0.0946(0.0520) gcn-mse=0.0943(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-05 20:45:22 3000-5019 loss=0.0474(0.0355+0.0119)-0.4020(0.1925+0.2095) sod-mse=0.0055(0.0524) gcn-mse=0.0084(0.0544) gcn-final-mse=0.0541(0.0675)
2020-08-05 20:46:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 20:47:15 4000-5019 loss=0.0987(0.0702+0.0284)-0.3995(0.1919+0.2076) sod-mse=0.0131(0.0519) gcn-mse=0.0130(0.0540) gcn-final-mse=0.0538(0.0671)
2020-08-05 20:47:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 20:49:06 5000-5019 loss=1.3716(0.5391+0.8325)-0.3989(0.1918+0.2071) sod-mse=0.1053(0.0520) gcn-mse=0.1048(0.0541) gcn-final-mse=0.0538(0.0671)
2020-08-05 20:49:08 E:39, Train sod-mae-score=0.0095-0.9854 gcn-mae-score=0.0120-0.9566 gcn-final-mse-score=0.0113-0.9590(0.0288/0.9590) loss=0.0706(0.0513+0.0193)
2020-08-05 20:49:08 E:39, Test  sod-mae-score=0.0519-0.8415 gcn-mae-score=0.0541-0.7891 gcn-final-mse-score=0.0538-0.7950(0.0671/0.7950) loss=0.3987(0.1917+0.2070)

2020-08-05 20:49:08 Start Epoch 40
2020-08-05 20:49:08 Epoch:40,lr=0.0000
2020-08-05 20:49:09    0-10553 loss=0.0660(0.0501+0.0159)-0.0660(0.0501+0.0159) sod-mse=0.0109(0.0109) gcn-mse=0.0133(0.0133) gcn-final-mse=0.0126(0.0352)
2020-08-05 20:51:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 20:53:13 1000-10553 loss=0.0624(0.0360+0.0265)-0.0722(0.0523+0.0199) sod-mse=0.0139(0.0099) gcn-mse=0.0158(0.0122) gcn-final-mse=0.0115(0.0292)
2020-08-05 20:57:20 2000-10553 loss=0.0765(0.0587+0.0178)-0.0708(0.0511+0.0197) sod-mse=0.0093(0.0098) gcn-mse=0.0128(0.0121) gcn-final-mse=0.0114(0.0286)
2020-08-05 21:00:22 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 21:01:22 3000-10553 loss=0.1286(0.0913+0.0374)-0.0705(0.0510+0.0195) sod-mse=0.0222(0.0097) gcn-mse=0.0221(0.0120) gcn-final-mse=0.0113(0.0287)
2020-08-05 21:05:25 4000-10553 loss=0.0494(0.0365+0.0129)-0.0705(0.0511+0.0194) sod-mse=0.0060(0.0096) gcn-mse=0.0085(0.0120) gcn-final-mse=0.0112(0.0287)
2020-08-05 21:09:28 5000-10553 loss=0.0360(0.0271+0.0089)-0.0704(0.0511+0.0194) sod-mse=0.0044(0.0096) gcn-mse=0.0049(0.0119) gcn-final-mse=0.0112(0.0287)
2020-08-05 21:13:31 6000-10553 loss=0.0898(0.0652+0.0246)-0.0705(0.0511+0.0194) sod-mse=0.0159(0.0096) gcn-mse=0.0186(0.0119) gcn-final-mse=0.0112(0.0286)
2020-08-05 21:17:33 7000-10553 loss=0.0802(0.0634+0.0167)-0.0702(0.0510+0.0192) sod-mse=0.0087(0.0095) gcn-mse=0.0196(0.0119) gcn-final-mse=0.0111(0.0286)
2020-08-05 21:21:38 8000-10553 loss=0.0877(0.0689+0.0189)-0.0704(0.0511+0.0193) sod-mse=0.0091(0.0095) gcn-mse=0.0127(0.0119) gcn-final-mse=0.0112(0.0287)
2020-08-05 21:23:56 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 21:25:41 9000-10553 loss=0.0876(0.0620+0.0255)-0.0705(0.0512+0.0193) sod-mse=0.0130(0.0095) gcn-mse=0.0178(0.0119) gcn-final-mse=0.0112(0.0287)
2020-08-05 21:29:45 10000-10553 loss=0.0419(0.0339+0.0081)-0.0705(0.0512+0.0193) sod-mse=0.0045(0.0096) gcn-mse=0.0146(0.0120) gcn-final-mse=0.0112(0.0287)

2020-08-05 21:32:00    0-5019 loss=1.2087(0.6953+0.5134)-1.2087(0.6953+0.5134) sod-mse=0.0992(0.0992) gcn-mse=0.1050(0.1050) gcn-final-mse=0.0970(0.1109)
2020-08-05 21:33:52 1000-5019 loss=0.0280(0.0228+0.0053)-0.4032(0.1926+0.2106) sod-mse=0.0042(0.0515) gcn-mse=0.0050(0.0535) gcn-final-mse=0.0532(0.0667)
2020-08-05 21:35:43 2000-5019 loss=1.3616(0.6252+0.7364)-0.4090(0.1949+0.2141) sod-mse=0.0944(0.0521) gcn-mse=0.0941(0.0542) gcn-final-mse=0.0539(0.0673)
2020-08-05 21:37:34 3000-5019 loss=0.0476(0.0357+0.0120)-0.4102(0.1946+0.2156) sod-mse=0.0055(0.0525) gcn-mse=0.0085(0.0546) gcn-final-mse=0.0544(0.0678)
2020-08-05 21:38:35 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 21:39:25 4000-5019 loss=0.0985(0.0701+0.0284)-0.4078(0.1941+0.2137) sod-mse=0.0131(0.0521) gcn-mse=0.0129(0.0543) gcn-final-mse=0.0540(0.0674)
2020-08-05 21:39:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 21:41:15 5000-5019 loss=1.4229(0.5526+0.8703)-0.4069(0.1938+0.2131) sod-mse=0.1081(0.0521) gcn-mse=0.1067(0.0543) gcn-final-mse=0.0540(0.0674)
2020-08-05 21:41:17 E:40, Train sod-mae-score=0.0096-0.9854 gcn-mae-score=0.0120-0.9566 gcn-final-mse-score=0.0112-0.9591(0.0287/0.9591) loss=0.0706(0.0512+0.0193)
2020-08-05 21:41:17 E:40, Test  sod-mae-score=0.0521-0.8402 gcn-mae-score=0.0543-0.7874 gcn-final-mse-score=0.0540-0.7932(0.0674/0.7932) loss=0.4067(0.1937+0.2130)

2020-08-05 21:41:17 Start Epoch 41
2020-08-05 21:41:17 Epoch:41,lr=0.0000
2020-08-05 21:41:18    0-10553 loss=0.0915(0.0584+0.0331)-0.0915(0.0584+0.0331) sod-mse=0.0168(0.0168) gcn-mse=0.0227(0.0227) gcn-final-mse=0.0205(0.0323)
2020-08-05 21:45:22 1000-10553 loss=0.0280(0.0230+0.0050)-0.0703(0.0512+0.0192) sod-mse=0.0019(0.0096) gcn-mse=0.0023(0.0120) gcn-final-mse=0.0113(0.0290)
2020-08-05 21:48:54 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 21:49:27 2000-10553 loss=0.0860(0.0656+0.0205)-0.0716(0.0520+0.0196) sod-mse=0.0084(0.0097) gcn-mse=0.0111(0.0122) gcn-final-mse=0.0114(0.0293)
2020-08-05 21:53:31 3000-10553 loss=0.0802(0.0603+0.0199)-0.0713(0.0517+0.0195) sod-mse=0.0097(0.0096) gcn-mse=0.0143(0.0121) gcn-final-mse=0.0114(0.0291)
2020-08-05 21:57:35 4000-10553 loss=0.0744(0.0483+0.0261)-0.0711(0.0516+0.0196) sod-mse=0.0117(0.0096) gcn-mse=0.0132(0.0121) gcn-final-mse=0.0113(0.0290)
2020-08-05 22:01:39 5000-10553 loss=0.0606(0.0467+0.0139)-0.0707(0.0514+0.0193) sod-mse=0.0073(0.0095) gcn-mse=0.0108(0.0120) gcn-final-mse=0.0113(0.0289)
2020-08-05 22:05:42 6000-10553 loss=0.0208(0.0168+0.0040)-0.0707(0.0514+0.0193) sod-mse=0.0019(0.0095) gcn-mse=0.0015(0.0120) gcn-final-mse=0.0112(0.0288)
2020-08-05 22:09:43 7000-10553 loss=0.0523(0.0406+0.0116)-0.0707(0.0514+0.0193) sod-mse=0.0080(0.0095) gcn-mse=0.0070(0.0120) gcn-final-mse=0.0112(0.0288)
2020-08-05 22:10:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 22:13:48 8000-10553 loss=0.0791(0.0592+0.0198)-0.0708(0.0514+0.0194) sod-mse=0.0106(0.0096) gcn-mse=0.0146(0.0120) gcn-final-mse=0.0113(0.0288)
2020-08-05 22:17:51 9000-10553 loss=0.0725(0.0525+0.0200)-0.0705(0.0512+0.0193) sod-mse=0.0126(0.0095) gcn-mse=0.0157(0.0120) gcn-final-mse=0.0112(0.0287)
2020-08-05 22:21:44 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 22:21:55 10000-10553 loss=0.0120(0.0093+0.0026)-0.0705(0.0512+0.0193) sod-mse=0.0019(0.0095) gcn-mse=0.0020(0.0119) gcn-final-mse=0.0112(0.0287)

2020-08-05 22:24:10    0-5019 loss=1.2239(0.6994+0.5246)-1.2239(0.6994+0.5246) sod-mse=0.1000(0.1000) gcn-mse=0.1058(0.1058) gcn-final-mse=0.0978(0.1116)
2020-08-05 22:26:02 1000-5019 loss=0.0280(0.0227+0.0052)-0.4072(0.1930+0.2143) sod-mse=0.0042(0.0511) gcn-mse=0.0050(0.0532) gcn-final-mse=0.0530(0.0664)
2020-08-05 22:27:53 2000-5019 loss=1.4008(0.6336+0.7672)-0.4127(0.1952+0.2175) sod-mse=0.0948(0.0518) gcn-mse=0.0943(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-05 22:29:44 3000-5019 loss=0.0477(0.0357+0.0120)-0.4140(0.1949+0.2191) sod-mse=0.0055(0.0522) gcn-mse=0.0084(0.0544) gcn-final-mse=0.0542(0.0674)
2020-08-05 22:30:46 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 22:31:35 4000-5019 loss=0.0987(0.0704+0.0283)-0.4113(0.1943+0.2171) sod-mse=0.0130(0.0517) gcn-mse=0.0130(0.0541) gcn-final-mse=0.0538(0.0671)
2020-08-05 22:32:08 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 22:33:27 5000-5019 loss=1.4439(0.5527+0.8911)-0.4107(0.1941+0.2166) sod-mse=0.1030(0.0518) gcn-mse=0.1033(0.0541) gcn-final-mse=0.0538(0.0671)
2020-08-05 22:33:29 E:41, Train sod-mae-score=0.0095-0.9855 gcn-mae-score=0.0119-0.9563 gcn-final-mse-score=0.0112-0.9588(0.0287/0.9588) loss=0.0705(0.0512+0.0193)
2020-08-05 22:33:29 E:41, Test  sod-mae-score=0.0517-0.8409 gcn-mae-score=0.0541-0.7881 gcn-final-mse-score=0.0538-0.7940(0.0671/0.7940) loss=0.4105(0.1940+0.2165)

2020-08-05 22:33:29 Start Epoch 42
2020-08-05 22:33:29 Epoch:42,lr=0.0000
2020-08-05 22:33:30    0-10553 loss=0.1000(0.0676+0.0325)-0.1000(0.0676+0.0325) sod-mse=0.0161(0.0161) gcn-mse=0.0196(0.0196) gcn-final-mse=0.0182(0.0394)
2020-08-05 22:37:34 1000-10553 loss=0.1026(0.0761+0.0265)-0.0697(0.0510+0.0187) sod-mse=0.0125(0.0093) gcn-mse=0.0190(0.0118) gcn-final-mse=0.0111(0.0287)
2020-08-05 22:41:38 2000-10553 loss=0.0883(0.0611+0.0271)-0.0696(0.0506+0.0190) sod-mse=0.0119(0.0094) gcn-mse=0.0119(0.0117) gcn-final-mse=0.0110(0.0284)
2020-08-05 22:45:43 3000-10553 loss=0.0918(0.0646+0.0272)-0.0705(0.0512+0.0192) sod-mse=0.0127(0.0094) gcn-mse=0.0161(0.0119) gcn-final-mse=0.0111(0.0287)
2020-08-05 22:49:48 4000-10553 loss=0.0362(0.0267+0.0094)-0.0698(0.0508+0.0190) sod-mse=0.0049(0.0094) gcn-mse=0.0065(0.0118) gcn-final-mse=0.0110(0.0285)
2020-08-05 22:53:51 5000-10553 loss=0.0338(0.0244+0.0094)-0.0701(0.0510+0.0191) sod-mse=0.0049(0.0094) gcn-mse=0.0049(0.0118) gcn-final-mse=0.0111(0.0286)
2020-08-05 22:54:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 22:57:55 6000-10553 loss=0.1044(0.0658+0.0386)-0.0701(0.0510+0.0191) sod-mse=0.0155(0.0094) gcn-mse=0.0238(0.0118) gcn-final-mse=0.0110(0.0286)
2020-08-05 23:01:57 7000-10553 loss=0.0763(0.0600+0.0163)-0.0704(0.0512+0.0192) sod-mse=0.0074(0.0094) gcn-mse=0.0163(0.0119) gcn-final-mse=0.0111(0.0287)
2020-08-05 23:02:09 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 23:06:00 8000-10553 loss=0.0245(0.0183+0.0061)-0.0705(0.0513+0.0192) sod-mse=0.0035(0.0095) gcn-mse=0.0050(0.0119) gcn-final-mse=0.0111(0.0287)
2020-08-05 23:10:03 9000-10553 loss=0.0994(0.0775+0.0219)-0.0707(0.0514+0.0194) sod-mse=0.0127(0.0095) gcn-mse=0.0256(0.0120) gcn-final-mse=0.0112(0.0287)
2020-08-05 23:13:07 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-05 23:14:08 10000-10553 loss=0.0367(0.0289+0.0078)-0.0705(0.0512+0.0193) sod-mse=0.0033(0.0095) gcn-mse=0.0035(0.0119) gcn-final-mse=0.0112(0.0287)

2020-08-05 23:16:24    0-5019 loss=1.2239(0.6987+0.5252)-1.2239(0.6987+0.5252) sod-mse=0.1004(0.1004) gcn-mse=0.1063(0.1063) gcn-final-mse=0.0983(0.1119)
2020-08-05 23:18:17 1000-5019 loss=0.0279(0.0227+0.0052)-0.4078(0.1933+0.2146) sod-mse=0.0041(0.0511) gcn-mse=0.0049(0.0532) gcn-final-mse=0.0530(0.0664)
2020-08-05 23:20:09 2000-5019 loss=1.3989(0.6357+0.7633)-0.4127(0.1953+0.2175) sod-mse=0.0948(0.0518) gcn-mse=0.0944(0.0539) gcn-final-mse=0.0537(0.0669)
2020-08-05 23:22:00 3000-5019 loss=0.0476(0.0357+0.0119)-0.4140(0.1950+0.2190) sod-mse=0.0054(0.0522) gcn-mse=0.0084(0.0544) gcn-final-mse=0.0541(0.0674)
2020-08-05 23:23:02 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-05 23:23:51 4000-5019 loss=0.0988(0.0703+0.0284)-0.4113(0.1944+0.2169) sod-mse=0.0130(0.0517) gcn-mse=0.0131(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-05 23:24:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-05 23:25:42 5000-5019 loss=1.4506(0.5575+0.8931)-0.4107(0.1942+0.2165) sod-mse=0.1024(0.0517) gcn-mse=0.1034(0.0541) gcn-final-mse=0.0537(0.0670)
2020-08-05 23:25:43 E:42, Train sod-mae-score=0.0095-0.9856 gcn-mae-score=0.0119-0.9564 gcn-final-mse-score=0.0111-0.9589(0.0287/0.9589) loss=0.0704(0.0511+0.0192)
2020-08-05 23:25:43 E:42, Test  sod-mae-score=0.0517-0.8407 gcn-mae-score=0.0541-0.7880 gcn-final-mse-score=0.0537-0.7938(0.0670/0.7938) loss=0.4105(0.1941+0.2163)

2020-08-05 23:25:43 Start Epoch 43
2020-08-05 23:25:43 Epoch:43,lr=0.0000
2020-08-05 23:25:45    0-10553 loss=0.0280(0.0203+0.0077)-0.0280(0.0203+0.0077) sod-mse=0.0038(0.0038) gcn-mse=0.0028(0.0028) gcn-final-mse=0.0028(0.0138)
2020-08-05 23:29:48 1000-10553 loss=0.0542(0.0416+0.0126)-0.0702(0.0510+0.0191) sod-mse=0.0085(0.0095) gcn-mse=0.0068(0.0118) gcn-final-mse=0.0111(0.0286)
2020-08-05 23:31:45 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-05 23:33:51 2000-10553 loss=0.0382(0.0290+0.0092)-0.0691(0.0505+0.0187) sod-mse=0.0043(0.0093) gcn-mse=0.0044(0.0117) gcn-final-mse=0.0109(0.0284)
2020-08-05 23:37:57 3000-10553 loss=0.0666(0.0521+0.0145)-0.0703(0.0510+0.0193) sod-mse=0.0062(0.0095) gcn-mse=0.0072(0.0118) gcn-final-mse=0.0111(0.0285)
2020-08-05 23:41:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-05 23:42:00 4000-10553 loss=0.0773(0.0527+0.0246)-0.0699(0.0509+0.0190) sod-mse=0.0113(0.0094) gcn-mse=0.0128(0.0118) gcn-final-mse=0.0111(0.0285)
2020-08-05 23:46:03 5000-10553 loss=0.0612(0.0467+0.0144)-0.0701(0.0510+0.0191) sod-mse=0.0071(0.0094) gcn-mse=0.0141(0.0118) gcn-final-mse=0.0111(0.0286)
2020-08-05 23:50:07 6000-10553 loss=0.0994(0.0718+0.0276)-0.0701(0.0510+0.0191) sod-mse=0.0115(0.0094) gcn-mse=0.0166(0.0118) gcn-final-mse=0.0111(0.0285)
2020-08-05 23:54:12 7000-10553 loss=0.0774(0.0528+0.0246)-0.0699(0.0509+0.0190) sod-mse=0.0102(0.0094) gcn-mse=0.0117(0.0118) gcn-final-mse=0.0110(0.0285)
2020-08-05 23:58:15 8000-10553 loss=0.1196(0.0901+0.0295)-0.0700(0.0510+0.0190) sod-mse=0.0143(0.0094) gcn-mse=0.0200(0.0118) gcn-final-mse=0.0111(0.0286)
2020-08-06 00:02:18 9000-10553 loss=0.1128(0.0880+0.0248)-0.0701(0.0510+0.0191) sod-mse=0.0105(0.0094) gcn-mse=0.0111(0.0118) gcn-final-mse=0.0111(0.0286)
2020-08-06 00:06:22 10000-10553 loss=0.0952(0.0649+0.0303)-0.0702(0.0510+0.0191) sod-mse=0.0145(0.0094) gcn-mse=0.0142(0.0118) gcn-final-mse=0.0111(0.0286)
2020-08-06 00:07:57 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg

2020-08-06 00:08:37    0-5019 loss=1.2198(0.6991+0.5208)-1.2198(0.6991+0.5208) sod-mse=0.1000(0.1000) gcn-mse=0.1061(0.1061) gcn-final-mse=0.0981(0.1119)
2020-08-06 00:10:31 1000-5019 loss=0.0279(0.0227+0.0052)-0.4069(0.1930+0.2139) sod-mse=0.0042(0.0511) gcn-mse=0.0049(0.0532) gcn-final-mse=0.0529(0.0664)
2020-08-06 00:12:23 2000-5019 loss=1.4007(0.6384+0.7622)-0.4119(0.1950+0.2169) sod-mse=0.0948(0.0518) gcn-mse=0.0944(0.0539) gcn-final-mse=0.0536(0.0669)
2020-08-06 00:14:16 3000-5019 loss=0.0476(0.0356+0.0120)-0.4131(0.1947+0.2184) sod-mse=0.0055(0.0522) gcn-mse=0.0085(0.0543) gcn-final-mse=0.0540(0.0674)
2020-08-06 00:15:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 00:16:09 4000-5019 loss=0.0985(0.0703+0.0282)-0.4107(0.1942+0.2165) sod-mse=0.0129(0.0517) gcn-mse=0.0130(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-06 00:16:42 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 00:18:02 5000-5019 loss=1.4370(0.5546+0.8824)-0.4101(0.1941+0.2160) sod-mse=0.1004(0.0517) gcn-mse=0.1012(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-06 00:18:04 E:43, Train sod-mae-score=0.0094-0.9856 gcn-mae-score=0.0118-0.9568 gcn-final-mse-score=0.0111-0.9591(0.0286/0.9591) loss=0.0702(0.0510+0.0191)
2020-08-06 00:18:04 E:43, Test  sod-mae-score=0.0517-0.8406 gcn-mae-score=0.0540-0.7880 gcn-final-mse-score=0.0537-0.7939(0.0670/0.7939) loss=0.4099(0.1940+0.2159)

2020-08-06 00:18:04 Start Epoch 44
2020-08-06 00:18:04 Epoch:44,lr=0.0000
2020-08-06 00:18:05    0-10553 loss=0.0789(0.0577+0.0212)-0.0789(0.0577+0.0212) sod-mse=0.0104(0.0104) gcn-mse=0.0183(0.0183) gcn-final-mse=0.0177(0.0329)
2020-08-06 00:22:09 1000-10553 loss=0.0359(0.0255+0.0104)-0.0705(0.0516+0.0189) sod-mse=0.0055(0.0093) gcn-mse=0.0089(0.0118) gcn-final-mse=0.0110(0.0288)
2020-08-06 00:26:14 2000-10553 loss=0.0120(0.0092+0.0029)-0.0694(0.0507+0.0187) sod-mse=0.0019(0.0093) gcn-mse=0.0021(0.0117) gcn-final-mse=0.0109(0.0284)
2020-08-06 00:30:18 3000-10553 loss=0.0636(0.0421+0.0214)-0.0694(0.0508+0.0186) sod-mse=0.0132(0.0092) gcn-mse=0.0122(0.0116) gcn-final-mse=0.0109(0.0284)
2020-08-06 00:34:22 4000-10553 loss=0.0686(0.0476+0.0210)-0.0696(0.0509+0.0186) sod-mse=0.0098(0.0093) gcn-mse=0.0094(0.0117) gcn-final-mse=0.0109(0.0286)
2020-08-06 00:38:25 5000-10553 loss=0.0909(0.0686+0.0223)-0.0701(0.0513+0.0189) sod-mse=0.0115(0.0093) gcn-mse=0.0137(0.0118) gcn-final-mse=0.0110(0.0288)
2020-08-06 00:41:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 00:42:29 6000-10553 loss=0.0331(0.0196+0.0135)-0.0699(0.0511+0.0188) sod-mse=0.0060(0.0093) gcn-mse=0.0056(0.0118) gcn-final-mse=0.0110(0.0287)
2020-08-06 00:46:25 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 00:46:32 7000-10553 loss=0.0833(0.0682+0.0150)-0.0700(0.0511+0.0189) sod-mse=0.0079(0.0093) gcn-mse=0.0157(0.0118) gcn-final-mse=0.0110(0.0287)
2020-08-06 00:50:35 8000-10553 loss=0.0636(0.0478+0.0158)-0.0702(0.0512+0.0190) sod-mse=0.0075(0.0094) gcn-mse=0.0113(0.0118) gcn-final-mse=0.0111(0.0287)
2020-08-06 00:54:38 9000-10553 loss=0.1713(0.1203+0.0510)-0.0701(0.0511+0.0190) sod-mse=0.0227(0.0094) gcn-mse=0.0276(0.0118) gcn-final-mse=0.0111(0.0287)
2020-08-06 00:56:00 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 00:58:41 10000-10553 loss=0.1031(0.0684+0.0347)-0.0700(0.0510+0.0190) sod-mse=0.0153(0.0094) gcn-mse=0.0190(0.0118) gcn-final-mse=0.0110(0.0286)

2020-08-06 01:00:58    0-5019 loss=1.2202(0.7005+0.5197)-1.2202(0.7005+0.5197) sod-mse=0.1003(0.1003) gcn-mse=0.1065(0.1065) gcn-final-mse=0.0985(0.1121)
2020-08-06 01:02:49 1000-5019 loss=0.0279(0.0227+0.0052)-0.4090(0.1938+0.2152) sod-mse=0.0042(0.0512) gcn-mse=0.0049(0.0532) gcn-final-mse=0.0529(0.0664)
2020-08-06 01:04:40 2000-5019 loss=1.4020(0.6426+0.7594)-0.4139(0.1958+0.2181) sod-mse=0.0947(0.0519) gcn-mse=0.0944(0.0539) gcn-final-mse=0.0536(0.0670)
2020-08-06 01:06:31 3000-5019 loss=0.0475(0.0356+0.0119)-0.4150(0.1954+0.2196) sod-mse=0.0054(0.0522) gcn-mse=0.0084(0.0543) gcn-final-mse=0.0541(0.0674)
2020-08-06 01:07:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 01:08:21 4000-5019 loss=0.0987(0.0703+0.0283)-0.4123(0.1948+0.2175) sod-mse=0.0131(0.0517) gcn-mse=0.0131(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-06 01:08:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 01:10:11 5000-5019 loss=1.4339(0.5515+0.8824)-0.4119(0.1947+0.2172) sod-mse=0.1000(0.0518) gcn-mse=0.1006(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-06 01:10:13 E:44, Train sod-mae-score=0.0094-0.9857 gcn-mae-score=0.0118-0.9567 gcn-final-mse-score=0.0110-0.9591(0.0286/0.9591) loss=0.0700(0.0510+0.0190)
2020-08-06 01:10:13 E:44, Test  sod-mae-score=0.0518-0.8404 gcn-mae-score=0.0540-0.7880 gcn-final-mse-score=0.0537-0.7938(0.0670/0.7938) loss=0.4116(0.1946+0.2170)

2020-08-06 01:10:13 Start Epoch 45
2020-08-06 01:10:13 Epoch:45,lr=0.0000
2020-08-06 01:10:14    0-10553 loss=0.0529(0.0375+0.0155)-0.0529(0.0375+0.0155) sod-mse=0.0072(0.0072) gcn-mse=0.0102(0.0102) gcn-final-mse=0.0089(0.0205)
2020-08-06 01:14:18 1000-10553 loss=0.0440(0.0339+0.0101)-0.0707(0.0514+0.0193) sod-mse=0.0048(0.0095) gcn-mse=0.0071(0.0120) gcn-final-mse=0.0112(0.0288)
2020-08-06 01:18:21 2000-10553 loss=0.1294(0.0949+0.0345)-0.0718(0.0519+0.0199) sod-mse=0.0203(0.0097) gcn-mse=0.0255(0.0121) gcn-final-mse=0.0113(0.0289)
2020-08-06 01:22:24 3000-10553 loss=0.0399(0.0282+0.0117)-0.0717(0.0519+0.0199) sod-mse=0.0060(0.0098) gcn-mse=0.0056(0.0121) gcn-final-mse=0.0113(0.0290)
2020-08-06 01:26:28 4000-10553 loss=0.0892(0.0638+0.0254)-0.0712(0.0517+0.0195) sod-mse=0.0122(0.0096) gcn-mse=0.0143(0.0120) gcn-final-mse=0.0113(0.0290)
2020-08-06 01:30:34 5000-10553 loss=0.0844(0.0554+0.0289)-0.0710(0.0515+0.0195) sod-mse=0.0136(0.0096) gcn-mse=0.0153(0.0120) gcn-final-mse=0.0112(0.0289)
2020-08-06 01:33:47 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 01:34:12 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 01:34:38 6000-10553 loss=0.0519(0.0407+0.0113)-0.0712(0.0517+0.0195) sod-mse=0.0049(0.0096) gcn-mse=0.0079(0.0120) gcn-final-mse=0.0112(0.0290)
2020-08-06 01:38:43 7000-10553 loss=0.0633(0.0393+0.0241)-0.0707(0.0514+0.0193) sod-mse=0.0119(0.0095) gcn-mse=0.0112(0.0119) gcn-final-mse=0.0112(0.0288)
2020-08-06 01:42:48 8000-10553 loss=0.0594(0.0443+0.0151)-0.0706(0.0513+0.0193) sod-mse=0.0068(0.0095) gcn-mse=0.0105(0.0119) gcn-final-mse=0.0111(0.0288)
2020-08-06 01:46:54 9000-10553 loss=0.0508(0.0435+0.0073)-0.0702(0.0511+0.0191) sod-mse=0.0054(0.0094) gcn-mse=0.0096(0.0118) gcn-final-mse=0.0111(0.0287)
2020-08-06 01:47:40 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 01:50:58 10000-10553 loss=0.0648(0.0476+0.0172)-0.0701(0.0511+0.0191) sod-mse=0.0088(0.0094) gcn-mse=0.0080(0.0118) gcn-final-mse=0.0111(0.0287)

2020-08-06 01:53:15    0-5019 loss=1.2367(0.7069+0.5299)-1.2367(0.7069+0.5299) sod-mse=0.1006(0.1006) gcn-mse=0.1067(0.1067) gcn-final-mse=0.0986(0.1125)
2020-08-06 01:55:07 1000-5019 loss=0.0279(0.0227+0.0052)-0.4112(0.1937+0.2175) sod-mse=0.0041(0.0510) gcn-mse=0.0050(0.0531) gcn-final-mse=0.0528(0.0663)
2020-08-06 01:56:59 2000-5019 loss=1.4317(0.6470+0.7847)-0.4161(0.1957+0.2204) sod-mse=0.0948(0.0517) gcn-mse=0.0947(0.0538) gcn-final-mse=0.0535(0.0668)
2020-08-06 01:58:53 3000-5019 loss=0.0477(0.0357+0.0120)-0.4171(0.1953+0.2218) sod-mse=0.0054(0.0520) gcn-mse=0.0086(0.0542) gcn-final-mse=0.0539(0.0672)
2020-08-06 01:59:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 02:00:45 4000-5019 loss=0.0987(0.0704+0.0283)-0.4146(0.1948+0.2198) sod-mse=0.0129(0.0516) gcn-mse=0.0131(0.0538) gcn-final-mse=0.0535(0.0669)
2020-08-06 02:01:17 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 02:02:36 5000-5019 loss=1.4745(0.5603+0.9142)-0.4141(0.1947+0.2194) sod-mse=0.0988(0.0516) gcn-mse=0.1002(0.0539) gcn-final-mse=0.0536(0.0669)
2020-08-06 02:02:38 E:45, Train sod-mae-score=0.0094-0.9856 gcn-mae-score=0.0118-0.9566 gcn-final-mse-score=0.0110-0.9590(0.0286/0.9590) loss=0.0700(0.0510+0.0190)
2020-08-06 02:02:38 E:45, Test  sod-mae-score=0.0516-0.8406 gcn-mae-score=0.0539-0.7881 gcn-final-mse-score=0.0536-0.7939(0.0669/0.7939) loss=0.4139(0.1946+0.2193)

2020-08-06 02:02:38 Start Epoch 46
2020-08-06 02:02:38 Epoch:46,lr=0.0000
2020-08-06 02:02:39    0-10553 loss=0.0511(0.0358+0.0153)-0.0511(0.0358+0.0153) sod-mse=0.0081(0.0081) gcn-mse=0.0083(0.0083) gcn-final-mse=0.0062(0.0168)
2020-08-06 02:06:44 1000-10553 loss=0.0716(0.0539+0.0177)-0.0700(0.0510+0.0191) sod-mse=0.0090(0.0095) gcn-mse=0.0122(0.0118) gcn-final-mse=0.0111(0.0289)
2020-08-06 02:10:48 2000-10553 loss=0.0342(0.0240+0.0102)-0.0703(0.0512+0.0190) sod-mse=0.0055(0.0095) gcn-mse=0.0089(0.0118) gcn-final-mse=0.0111(0.0289)
2020-08-06 02:13:19 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 02:14:51 3000-10553 loss=0.0629(0.0437+0.0192)-0.0707(0.0515+0.0192) sod-mse=0.0096(0.0095) gcn-mse=0.0117(0.0118) gcn-final-mse=0.0111(0.0289)
2020-08-06 02:18:55 4000-10553 loss=0.0592(0.0430+0.0162)-0.0705(0.0513+0.0192) sod-mse=0.0071(0.0094) gcn-mse=0.0078(0.0118) gcn-final-mse=0.0110(0.0288)
2020-08-06 02:22:04 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 02:22:59 5000-10553 loss=0.0714(0.0584+0.0129)-0.0702(0.0511+0.0191) sod-mse=0.0053(0.0094) gcn-mse=0.0086(0.0118) gcn-final-mse=0.0111(0.0288)
2020-08-06 02:27:03 6000-10553 loss=0.0600(0.0460+0.0140)-0.0705(0.0511+0.0193) sod-mse=0.0090(0.0095) gcn-mse=0.0110(0.0118) gcn-final-mse=0.0111(0.0287)
2020-08-06 02:31:06 7000-10553 loss=0.0157(0.0110+0.0047)-0.0703(0.0510+0.0193) sod-mse=0.0037(0.0095) gcn-mse=0.0049(0.0118) gcn-final-mse=0.0110(0.0286)
2020-08-06 02:33:28 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 02:35:11 8000-10553 loss=0.0944(0.0626+0.0318)-0.0701(0.0509+0.0192) sod-mse=0.0165(0.0094) gcn-mse=0.0176(0.0117) gcn-final-mse=0.0110(0.0286)
2020-08-06 02:39:16 9000-10553 loss=0.0417(0.0327+0.0090)-0.0700(0.0509+0.0191) sod-mse=0.0061(0.0094) gcn-mse=0.0071(0.0117) gcn-final-mse=0.0110(0.0285)
2020-08-06 02:43:21 10000-10553 loss=0.0304(0.0223+0.0081)-0.0699(0.0508+0.0190) sod-mse=0.0033(0.0094) gcn-mse=0.0050(0.0117) gcn-final-mse=0.0110(0.0285)

2020-08-06 02:45:38    0-5019 loss=1.2352(0.7054+0.5298)-1.2352(0.7054+0.5298) sod-mse=0.1003(0.1003) gcn-mse=0.1066(0.1066) gcn-final-mse=0.0985(0.1122)
2020-08-06 02:47:31 1000-5019 loss=0.0278(0.0227+0.0051)-0.4156(0.1948+0.2208) sod-mse=0.0041(0.0511) gcn-mse=0.0049(0.0532) gcn-final-mse=0.0530(0.0664)
2020-08-06 02:49:23 2000-5019 loss=1.4387(0.6447+0.7941)-0.4207(0.1969+0.2238) sod-mse=0.0948(0.0518) gcn-mse=0.0946(0.0539) gcn-final-mse=0.0536(0.0669)
2020-08-06 02:51:14 3000-5019 loss=0.0477(0.0358+0.0119)-0.4217(0.1964+0.2252) sod-mse=0.0054(0.0521) gcn-mse=0.0086(0.0543) gcn-final-mse=0.0541(0.0673)
2020-08-06 02:52:15 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 02:53:04 4000-5019 loss=0.0984(0.0701+0.0283)-0.4191(0.1959+0.2232) sod-mse=0.0129(0.0516) gcn-mse=0.0131(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-06 02:53:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 02:54:55 5000-5019 loss=1.4888(0.5604+0.9284)-0.4184(0.1958+0.2226) sod-mse=0.0991(0.0517) gcn-mse=0.1003(0.0541) gcn-final-mse=0.0537(0.0670)
2020-08-06 02:54:57 E:46, Train sod-mae-score=0.0094-0.9857 gcn-mae-score=0.0117-0.9570 gcn-final-mse-score=0.0110-0.9593(0.0285/0.9593) loss=0.0698(0.0508+0.0190)
2020-08-06 02:54:57 E:46, Test  sod-mae-score=0.0517-0.8398 gcn-mae-score=0.0540-0.7878 gcn-final-mse-score=0.0537-0.7935(0.0670/0.7935) loss=0.4182(0.1957+0.2225)

2020-08-06 02:54:57 Start Epoch 47
2020-08-06 02:54:57 Epoch:47,lr=0.0000
2020-08-06 02:54:58    0-10553 loss=0.0973(0.0588+0.0385)-0.0973(0.0588+0.0385) sod-mse=0.0201(0.0201) gcn-mse=0.0189(0.0189) gcn-final-mse=0.0180(0.0328)
2020-08-06 02:59:04 1000-10553 loss=0.0755(0.0537+0.0218)-0.0684(0.0502+0.0183) sod-mse=0.0087(0.0091) gcn-mse=0.0107(0.0115) gcn-final-mse=0.0108(0.0281)
2020-08-06 03:03:08 2000-10553 loss=0.0831(0.0614+0.0217)-0.0687(0.0503+0.0184) sod-mse=0.0097(0.0091) gcn-mse=0.0121(0.0115) gcn-final-mse=0.0108(0.0283)
2020-08-06 03:07:13 3000-10553 loss=0.0381(0.0305+0.0075)-0.0688(0.0502+0.0185) sod-mse=0.0033(0.0092) gcn-mse=0.0045(0.0116) gcn-final-mse=0.0108(0.0282)
2020-08-06 03:11:17 4000-10553 loss=0.0631(0.0513+0.0118)-0.0697(0.0508+0.0189) sod-mse=0.0077(0.0093) gcn-mse=0.0115(0.0117) gcn-final-mse=0.0110(0.0284)
2020-08-06 03:15:21 5000-10553 loss=0.0452(0.0338+0.0115)-0.0696(0.0507+0.0189) sod-mse=0.0056(0.0093) gcn-mse=0.0090(0.0117) gcn-final-mse=0.0109(0.0285)
2020-08-06 03:19:26 6000-10553 loss=0.1305(0.0974+0.0332)-0.0696(0.0508+0.0189) sod-mse=0.0157(0.0093) gcn-mse=0.0219(0.0117) gcn-final-mse=0.0109(0.0285)
2020-08-06 03:20:23 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 03:23:30 7000-10553 loss=0.0619(0.0438+0.0181)-0.0696(0.0508+0.0188) sod-mse=0.0120(0.0093) gcn-mse=0.0116(0.0116) gcn-final-mse=0.0109(0.0285)
2020-08-06 03:23:36 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 03:27:33 8000-10553 loss=0.0721(0.0611+0.0110)-0.0697(0.0508+0.0189) sod-mse=0.0049(0.0093) gcn-mse=0.0201(0.0117) gcn-final-mse=0.0109(0.0285)
2020-08-06 03:28:20 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 03:31:37 9000-10553 loss=0.0482(0.0376+0.0106)-0.0700(0.0510+0.0190) sod-mse=0.0051(0.0093) gcn-mse=0.0093(0.0117) gcn-final-mse=0.0110(0.0286)
2020-08-06 03:35:42 10000-10553 loss=0.1185(0.0673+0.0511)-0.0699(0.0509+0.0190) sod-mse=0.0204(0.0093) gcn-mse=0.0189(0.0117) gcn-final-mse=0.0110(0.0286)

2020-08-06 03:37:58    0-5019 loss=1.2110(0.6937+0.5173)-1.2110(0.6937+0.5173) sod-mse=0.0994(0.0994) gcn-mse=0.1051(0.1051) gcn-final-mse=0.0971(0.1111)
2020-08-06 03:39:50 1000-5019 loss=0.0280(0.0227+0.0053)-0.4129(0.1943+0.2185) sod-mse=0.0042(0.0509) gcn-mse=0.0050(0.0530) gcn-final-mse=0.0527(0.0662)
2020-08-06 03:41:40 2000-5019 loss=1.4326(0.6477+0.7849)-0.4187(0.1966+0.2221) sod-mse=0.0950(0.0516) gcn-mse=0.0947(0.0537) gcn-final-mse=0.0534(0.0667)
2020-08-06 03:43:30 3000-5019 loss=0.0477(0.0358+0.0120)-0.4198(0.1963+0.2235) sod-mse=0.0055(0.0520) gcn-mse=0.0085(0.0541) gcn-final-mse=0.0539(0.0671)
2020-08-06 03:44:32 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 03:45:21 4000-5019 loss=0.0976(0.0695+0.0281)-0.4172(0.1957+0.2215) sod-mse=0.0130(0.0515) gcn-mse=0.0129(0.0538) gcn-final-mse=0.0535(0.0668)
2020-08-06 03:45:53 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 03:47:11 5000-5019 loss=1.5052(0.5747+0.9305)-0.4168(0.1957+0.2211) sod-mse=0.0999(0.0516) gcn-mse=0.1008(0.0538) gcn-final-mse=0.0535(0.0668)
2020-08-06 03:47:13 E:47, Train sod-mae-score=0.0093-0.9858 gcn-mae-score=0.0117-0.9568 gcn-final-mse-score=0.0109-0.9592(0.0285/0.9592) loss=0.0697(0.0508+0.0189)
2020-08-06 03:47:13 E:47, Test  sod-mae-score=0.0515-0.8406 gcn-mae-score=0.0538-0.7881 gcn-final-mse-score=0.0535-0.7940(0.0668/0.7940) loss=0.4166(0.1956+0.2210)

2020-08-06 03:47:13 Start Epoch 48
2020-08-06 03:47:13 Epoch:48,lr=0.0000
2020-08-06 03:47:15    0-10553 loss=0.0421(0.0321+0.0099)-0.0421(0.0321+0.0099) sod-mse=0.0046(0.0046) gcn-mse=0.0057(0.0057) gcn-final-mse=0.0055(0.0183)
2020-08-06 03:51:20 1000-10553 loss=0.0995(0.0635+0.0360)-0.0687(0.0502+0.0185) sod-mse=0.0149(0.0093) gcn-mse=0.0156(0.0117) gcn-final-mse=0.0108(0.0281)
2020-08-06 03:52:39 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 03:55:25 2000-10553 loss=0.0874(0.0591+0.0284)-0.0696(0.0508+0.0188) sod-mse=0.0141(0.0093) gcn-mse=0.0112(0.0118) gcn-final-mse=0.0110(0.0284)
2020-08-06 03:59:29 3000-10553 loss=0.0605(0.0430+0.0175)-0.0697(0.0509+0.0188) sod-mse=0.0117(0.0093) gcn-mse=0.0082(0.0117) gcn-final-mse=0.0110(0.0285)
2020-08-06 04:03:33 4000-10553 loss=0.0256(0.0183+0.0073)-0.0700(0.0510+0.0190) sod-mse=0.0036(0.0094) gcn-mse=0.0049(0.0117) gcn-final-mse=0.0110(0.0285)
2020-08-06 04:07:36 5000-10553 loss=0.0866(0.0561+0.0305)-0.0699(0.0510+0.0190) sod-mse=0.0143(0.0094) gcn-mse=0.0119(0.0117) gcn-final-mse=0.0110(0.0286)
2020-08-06 04:11:39 6000-10553 loss=0.0240(0.0180+0.0060)-0.0701(0.0510+0.0191) sod-mse=0.0029(0.0094) gcn-mse=0.0056(0.0117) gcn-final-mse=0.0110(0.0286)
2020-08-06 04:15:43 7000-10553 loss=0.0729(0.0520+0.0210)-0.0696(0.0507+0.0189) sod-mse=0.0096(0.0093) gcn-mse=0.0104(0.0116) gcn-final-mse=0.0109(0.0284)
2020-08-06 04:19:48 8000-10553 loss=0.0384(0.0312+0.0072)-0.0694(0.0506+0.0188) sod-mse=0.0053(0.0093) gcn-mse=0.0066(0.0116) gcn-final-mse=0.0109(0.0284)
2020-08-06 04:19:51 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 04:23:52 9000-10553 loss=0.0559(0.0416+0.0143)-0.0696(0.0507+0.0189) sod-mse=0.0063(0.0093) gcn-mse=0.0074(0.0117) gcn-final-mse=0.0109(0.0285)
2020-08-06 04:27:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 04:27:54 10000-10553 loss=0.0448(0.0329+0.0119)-0.0697(0.0508+0.0189) sod-mse=0.0085(0.0093) gcn-mse=0.0081(0.0117) gcn-final-mse=0.0109(0.0285)

2020-08-06 04:30:10    0-5019 loss=1.2331(0.7069+0.5262)-1.2331(0.7069+0.5262) sod-mse=0.1004(0.1004) gcn-mse=0.1067(0.1067) gcn-final-mse=0.0987(0.1125)
2020-08-06 04:32:03 1000-5019 loss=0.0278(0.0227+0.0051)-0.4167(0.1959+0.2209) sod-mse=0.0041(0.0511) gcn-mse=0.0049(0.0532) gcn-final-mse=0.0529(0.0664)
2020-08-06 04:33:55 2000-5019 loss=1.4406(0.6521+0.7884)-0.4219(0.1979+0.2240) sod-mse=0.0947(0.0518) gcn-mse=0.0946(0.0539) gcn-final-mse=0.0536(0.0669)
2020-08-06 04:35:47 3000-5019 loss=0.0477(0.0357+0.0120)-0.4230(0.1975+0.2255) sod-mse=0.0054(0.0522) gcn-mse=0.0085(0.0543) gcn-final-mse=0.0541(0.0673)
2020-08-06 04:36:50 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 04:37:40 4000-5019 loss=0.0984(0.0700+0.0284)-0.4205(0.1970+0.2235) sod-mse=0.0130(0.0517) gcn-mse=0.0129(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-06 04:38:13 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 04:39:33 5000-5019 loss=1.4946(0.5687+0.9259)-0.4200(0.1969+0.2231) sod-mse=0.0999(0.0518) gcn-mse=0.1008(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-06 04:39:35 E:48, Train sod-mae-score=0.0093-0.9858 gcn-mae-score=0.0117-0.9568 gcn-final-mse-score=0.0109-0.9593(0.0285/0.9593) loss=0.0697(0.0508+0.0189)
2020-08-06 04:39:35 E:48, Test  sod-mae-score=0.0517-0.8393 gcn-mae-score=0.0540-0.7874 gcn-final-mse-score=0.0537-0.7933(0.0670/0.7933) loss=0.4198(0.1968+0.2229)

2020-08-06 04:39:35 Start Epoch 49
2020-08-06 04:39:35 Epoch:49,lr=0.0000
2020-08-06 04:39:36    0-10553 loss=0.0417(0.0297+0.0121)-0.0417(0.0297+0.0121) sod-mse=0.0068(0.0068) gcn-mse=0.0106(0.0106) gcn-final-mse=0.0091(0.0168)
2020-08-06 04:43:44 1000-10553 loss=0.0433(0.0335+0.0098)-0.0707(0.0513+0.0194) sod-mse=0.0057(0.0095) gcn-mse=0.0085(0.0119) gcn-final-mse=0.0111(0.0287)
2020-08-06 04:45:49 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n01532829_13482.jpg
2020-08-06 04:47:53 2000-10553 loss=0.0988(0.0687+0.0301)-0.0704(0.0511+0.0193) sod-mse=0.0151(0.0094) gcn-mse=0.0169(0.0118) gcn-final-mse=0.0110(0.0286)
2020-08-06 04:52:03 3000-10553 loss=0.0509(0.0384+0.0126)-0.0697(0.0507+0.0190) sod-mse=0.0050(0.0093) gcn-mse=0.0082(0.0117) gcn-final-mse=0.0109(0.0285)
2020-08-06 04:56:10 4000-10553 loss=0.1325(0.0726+0.0599)-0.0696(0.0507+0.0188) sod-mse=0.0142(0.0093) gcn-mse=0.0173(0.0117) gcn-final-mse=0.0109(0.0284)
2020-08-06 05:00:17 5000-10553 loss=0.0783(0.0538+0.0244)-0.0700(0.0510+0.0190) sod-mse=0.0113(0.0093) gcn-mse=0.0144(0.0117) gcn-final-mse=0.0109(0.0285)
2020-08-06 05:02:37 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/ILSVRC2014_train_00023530.jpg
2020-08-06 05:04:23 6000-10553 loss=0.0282(0.0221+0.0061)-0.0700(0.0510+0.0190) sod-mse=0.0039(0.0093) gcn-mse=0.0026(0.0117) gcn-final-mse=0.0109(0.0285)
2020-08-06 05:08:29 7000-10553 loss=0.0483(0.0327+0.0156)-0.0698(0.0509+0.0189) sod-mse=0.0066(0.0093) gcn-mse=0.0077(0.0116) gcn-final-mse=0.0109(0.0285)
2020-08-06 05:12:11 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TR/DUTS-TR-Image/n04442312_17818.jpg
2020-08-06 05:12:31 8000-10553 loss=0.0294(0.0231+0.0062)-0.0694(0.0507+0.0187) sod-mse=0.0031(0.0092) gcn-mse=0.0026(0.0116) gcn-final-mse=0.0108(0.0284)
2020-08-06 05:16:35 9000-10553 loss=0.0708(0.0525+0.0183)-0.0693(0.0506+0.0187) sod-mse=0.0096(0.0092) gcn-mse=0.0107(0.0116) gcn-final-mse=0.0108(0.0284)
2020-08-06 05:20:38 10000-10553 loss=0.0368(0.0300+0.0069)-0.0694(0.0507+0.0187) sod-mse=0.0049(0.0092) gcn-mse=0.0054(0.0116) gcn-final-mse=0.0109(0.0285)

2020-08-06 05:22:54    0-5019 loss=1.2348(0.7046+0.5301)-1.2348(0.7046+0.5301) sod-mse=0.1003(0.1003) gcn-mse=0.1065(0.1065) gcn-final-mse=0.0984(0.1123)
2020-08-06 05:24:47 1000-5019 loss=0.0279(0.0227+0.0052)-0.4201(0.1962+0.2240) sod-mse=0.0041(0.0512) gcn-mse=0.0050(0.0533) gcn-final-mse=0.0530(0.0665)
2020-08-06 05:26:38 2000-5019 loss=1.4661(0.6560+0.8102)-0.4255(0.1982+0.2272) sod-mse=0.0948(0.0518) gcn-mse=0.0948(0.0540) gcn-final-mse=0.0537(0.0670)
2020-08-06 05:28:30 3000-5019 loss=0.0477(0.0357+0.0120)-0.4263(0.1977+0.2286) sod-mse=0.0055(0.0522) gcn-mse=0.0085(0.0544) gcn-final-mse=0.0542(0.0674)
2020-08-06 05:29:33 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/ILSVRC2012_test_00036002.jpg
2020-08-06 05:30:23 4000-5019 loss=0.0983(0.0700+0.0283)-0.4237(0.1972+0.2265) sod-mse=0.0130(0.0517) gcn-mse=0.0128(0.0541) gcn-final-mse=0.0538(0.0671)
2020-08-06 05:30:55 IMAGE ERROR, PASSING /mnt/4T/Data/SOD/DUTS/DUTS-TE/DUTS-TE-Image/sun_bcogaqperiljqupq.jpg
2020-08-06 05:32:15 5000-5019 loss=1.5159(0.5693+0.9465)-0.4232(0.1971+0.2261) sod-mse=0.0994(0.0518) gcn-mse=0.1002(0.0541) gcn-final-mse=0.0538(0.0671)
2020-08-06 05:32:17 E:49, Train sod-mae-score=0.0092-0.9859 gcn-mae-score=0.0116-0.9570 gcn-final-mse-score=0.0108-0.9594(0.0284/0.9594) loss=0.0694(0.0507+0.0187)
2020-08-06 05:32:17 E:49, Test  sod-mae-score=0.0518-0.8393 gcn-mae-score=0.0541-0.7871 gcn-final-mse-score=0.0538-0.7929(0.0671/0.7929) loss=0.4229(0.1970+0.2259)

Process finished with exit code 0
