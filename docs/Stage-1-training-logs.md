╔══════════════════════════════════════════════════════════╗
║  HFAQE Embedding Training                                ║
╚══════════════════════════════════════════════════════════╝
[config] V=256  d=256  r=64  K=128  B=64
[config] lr=3.00e-04  epochs=5  batch=64  seq=256
[config] data=Data  ckpt=checkpoints
[data] Loading Data/train.txt ...
[data] Loading Data/validation.txt ...
[data] train=23767 lines  val=2461 lines
[model] Computing token frequencies ...
[model] Frequency computation complete.
[model] Building frequency tiers (K=128) ...
[model] Initializing weights (SVD) ...
[model] Pinning hot tier (mlock) ...
[model] Setting up unified training state ...
[model] Hot=128 int8[128×256]  Cold=128 bf16[128×64]  Basis bf16[256×64]
[model] Param RAM: 0.08 MB  (baseline BF16 would be 0.12 MB)
[train] 372 steps/epoch × 5 epochs = 1860 total steps
[train] ep=1  step=    20/1860  loss=5.5404  ppl= 254.79  gnorm=6.416  lr=1.50e-05  tok/s=   9918  ratio_hc=1.00  ETA=33m46s   
[train] ep=1  step=    40/1860  loss=5.5398  ppl= 254.62  gnorm=6.515  lr=3.00e-05  tok/s=   9691  ratio_hc=1.00  ETA=33m42s   
[train] ep=1  step=    60/1860  loss=5.5383  ppl= 254.25  gnorm=6.849  lr=4.50e-05  tok/s=   9507  ratio_hc=1.00  ETA=33m56s   
[train] ep=1  step=    80/1860  loss=5.5373  ppl= 254.00  gnorm=5.400  lr=6.00e-05  tok/s=   9500  ratio_hc=1.00  ETA=33m56s   
[train] ep=1  step=   100/1860  loss=5.5357  ppl= 253.57  gnorm=7.262  lr=7.50e-05  tok/s=   9880  ratio_hc=1.00  ETA=33m09s   
[train] ep=1  step=   120/1860  loss=5.5308  ppl= 252.35  gnorm=5.573  lr=9.00e-05  tok/s=   9485  ratio_hc=1.00  ETA=32m45s   
[train] ep=1  step=   140/1860  loss=5.5259  ppl= 251.11  gnorm=5.402  lr=1.05e-04  tok/s=   9496  ratio_hc=1.00  ETA=32m24s   
[train] ep=1  step=   160/1860  loss=5.5205  ppl= 249.75  gnorm=6.260  lr=1.20e-04  tok/s=   9863  ratio_hc=1.00  ETA=31m55s   
[train] ep=1  step=   180/1860  loss=5.5100  ppl= 247.15  gnorm=5.940  lr=1.35e-04  tok/s=   9440  ratio_hc=1.00  ETA=31m43s   
[train] ep=1  step=   200/1860  loss=5.4924  ppl= 242.83  gnorm=6.642  lr=1.50e-04  tok/s=   9448  ratio_hc=1.00  ETA=31m25s   
[train] ep=1  step=   220/1860  loss=5.4728  ppl= 238.13  gnorm=6.522  lr=1.65e-04  tok/s=   9654  ratio_hc=1.00  ETA=30m57s   
[train] ep=1  step=   240/1860  loss=5.4491  ppl= 232.56  gnorm=7.302  lr=1.80e-04  tok/s=   9656  ratio_hc=1.00  ETA=30m39s   
[train] ep=1  step=   260/1860  loss=5.4157  ppl= 224.91  gnorm=7.792  lr=1.95e-04  tok/s=   9413  ratio_hc=1.00  ETA=30m18s   
[train] ep=1  step=   280/1860  loss=5.3696  ppl= 214.78  gnorm=7.975  lr=2.10e-04  tok/s=   9333  ratio_hc=1.00  ETA=29m59s   
[train] ep=1  step=   300/1860  loss=5.3054  ppl= 201.42  gnorm=9.287  lr=2.25e-04  tok/s=   9821  ratio_hc=1.00  ETA=29m35s   
[val]   step=   300  val_loss=5.2630  val_ppl= 193.06  (35.3s)
[ckpt] saved checkpoints/hfaqe_best.nex  (0.00 MB, 9.5 ms)
[train] ep=1  step=   320/1860  loss=5.2196  ppl= 184.86  gnorm=9.900  lr=2.40e-04  tok/s=   3703  ratio_hc=1.00  ETA=32m02s   
[train] ep=1  step=   340/1860  loss=5.1085  ppl= 165.42  gnorm=10.599  lr=2.55e-04  tok/s=   9722  ratio_hc=1.00  ETA=31m24s   
[train] ep=1  step=   360/1860  loss=4.9751  ppl= 144.76  gnorm=11.571  lr=2.70e-04  tok/s=   9543  ratio_hc=1.00  ETA=30m51s   
[epoch] 1/5 complete.
[val]   step=   372  val_loss=4.7963  val_ppl= 121.06  (0.0s)
[ckpt] saved checkpoints/hfaqe_epoch_01.nex  (0.00 MB, 8.6 ms)
[train] ep=2  step=   380/1860  loss=4.8167  ppl= 123.56  gnorm=12.774  lr=2.85e-04  tok/s=   3694  ratio_hc=1.00  ETA=32m32s   
[train] ep=2  step=   400/1860  loss=4.6405  ppl= 103.59  gnorm=11.227  lr=3.00e-04  tok/s=   9609  ratio_hc=1.00  ETA=31m50s   
[train] ep=2  step=   420/1860  loss=4.4501  ppl=  85.64  gnorm=13.021  lr=3.00e-04  tok/s=   9426  ratio_hc=1.00  ETA=31m14s   
[train] ep=2  step=   440/1860  loss=4.2848  ppl=  72.58  gnorm=11.080  lr=2.99e-04  tok/s=   9561  ratio_hc=1.00  ETA=30m36s   
[train] ep=2  step=   460/1860  loss=4.1604  ppl=  64.09  gnorm=10.462  lr=2.99e-04  tok/s=   9715  ratio_hc=1.00  ETA=30m01s   
[train] ep=2  step=   480/1860  loss=4.0720  ppl=  58.68  gnorm=10.085  lr=2.98e-04  tok/s=   9432  ratio_hc=1.00  ETA=29m29s   
[train] ep=2  step=   500/1860  loss=4.0005  ppl=  54.63  gnorm=9.703  lr=2.97e-04  tok/s=   9429  ratio_hc=1.00  ETA=28m56s   
[train] ep=2  step=   520/1860  loss=3.9553  ppl=  52.21  gnorm=9.367  lr=2.95e-04  tok/s=   9825  ratio_hc=1.00  ETA=28m20s   
[train] ep=2  step=   540/1860  loss=3.9216  ppl=  50.48  gnorm=9.130  lr=2.94e-04  tok/s=   9458  ratio_hc=1.00  ETA=27m49s   
[train] ep=2  step=   560/1860  loss=3.8996  ppl=  49.38  gnorm=8.422  lr=2.92e-04  tok/s=   9419  ratio_hc=1.00  ETA=27m19s   
[train] ep=2  step=   580/1860  loss=3.8846  ppl=  48.65  gnorm=7.735  lr=2.89e-04  tok/s=   9638  ratio_hc=1.00  ETA=26m48s   
[train] ep=2  step=   600/1860  loss=3.8677  ppl=  47.83  gnorm=7.408  lr=2.87e-04  tok/s=   9714  ratio_hc=1.00  ETA=26m17s   
[val]   step=   600  val_loss=3.8678  val_ppl=  47.84  (35.2s)
[ckpt] saved checkpoints/hfaqe_best.nex  (0.00 MB, 8.5 ms)
[train] ep=2  step=   620/1860  loss=3.9093  ppl=  49.87  gnorm=7.077  lr=2.84e-04  tok/s=   3781  ratio_hc=1.00  ETA=26m59s   
[train] ep=2  step=   640/1860  loss=3.9038  ppl=  49.59  gnorm=7.703  lr=2.81e-04  tok/s=   9860  ratio_hc=1.00  ETA=26m25s   
[train] ep=2  step=   660/1860  loss=3.8966  ppl=  49.23  gnorm=7.226  lr=2.78e-04  tok/s=   9500  ratio_hc=1.00  ETA=25m53s   
[train] ep=2  step=   680/1860  loss=3.8970  ppl=  49.25  gnorm=7.706  lr=2.75e-04  tok/s=   9501  ratio_hc=1.00  ETA=25m22s   
[train] ep=2  step=   700/1860  loss=3.8960  ppl=  49.21  gnorm=7.167  lr=2.71e-04  tok/s=   9654  ratio_hc=1.00  ETA=24m52s   
[train] ep=2  step=   720/1860  loss=3.8955  ppl=  49.18  gnorm=7.064  lr=2.67e-04  tok/s=   9618  ratio_hc=1.00  ETA=24m21s   
[train] ep=2  step=   740/1860  loss=3.8938  ppl=  49.10  gnorm=8.012  lr=2.63e-04  tok/s=   9496  ratio_hc=1.00  ETA=23m51s   
[epoch] 2/5 complete.
[val]   step=   744  val_loss=3.8971  val_ppl=  49.26  (0.0s)
[ckpt] saved checkpoints/hfaqe_epoch_02.nex  (0.00 MB, 8.6 ms)
[train] ep=3  step=   760/1860  loss=3.8978  ppl=  49.29  gnorm=7.209  lr=2.59e-04  tok/s=   3744  ratio_hc=1.00  ETA=24m10s   
[train] ep=3  step=   780/1860  loss=3.8999  ppl=  49.40  gnorm=6.336  lr=2.54e-04  tok/s=   9464  ratio_hc=1.00  ETA=23m39s   
[train] ep=3  step=   800/1860  loss=3.9043  ppl=  49.61  gnorm=6.711  lr=2.50e-04  tok/s=   9447  ratio_hc=1.00  ETA=23m09s   
[train] ep=3  step=   820/1860  loss=3.9035  ppl=  49.58  gnorm=7.123  lr=2.45e-04  tok/s=   9856  ratio_hc=1.00  ETA=22m37s   
[train] ep=3  step=   840/1860  loss=3.9036  ppl=  49.58  gnorm=7.315  lr=2.40e-04  tok/s=   9454  ratio_hc=1.00  ETA=22m07s   
[train] ep=3  step=   860/1860  loss=3.9095  ppl=  49.87  gnorm=6.964  lr=2.35e-04  tok/s=   9463  ratio_hc=1.00  ETA=21m38s   
[train] ep=3  step=   880/1860  loss=3.9080  ppl=  49.80  gnorm=6.798  lr=2.30e-04  tok/s=   9587  ratio_hc=1.00  ETA=21m09s   
[train] ep=3  step=   900/1860  loss=3.9110  ppl=  49.95  gnorm=6.735  lr=2.24e-04  tok/s=   9742  ratio_hc=1.00  ETA=20m40s   
[val]   step=   900  val_loss=3.9116  val_ppl=  49.98  (35.1s)
[train] ep=3  step=   920/1860  loss=3.8654  ppl=  47.72  gnorm=8.306  lr=2.19e-04  tok/s=   3726  ratio_hc=1.00  ETA=20m47s   
[train] ep=3  step=   940/1860  loss=3.8627  ppl=  47.60  gnorm=7.819  lr=2.13e-04  tok/s=   9861  ratio_hc=1.00  ETA=20m16s   
[train] ep=3  step=   960/1860  loss=3.8628  ppl=  47.60  gnorm=7.177  lr=2.07e-04  tok/s=   9449  ratio_hc=1.00  ETA=19m45s   
[train] ep=3  step=   980/1860  loss=3.8684  ppl=  47.87  gnorm=6.772  lr=2.01e-04  tok/s=   9502  ratio_hc=1.00  ETA=19m16s   
[train] ep=3  step=  1000/1860  loss=3.8673  ppl=  47.81  gnorm=6.698  lr=1.95e-04  tok/s=   9784  ratio_hc=1.00  ETA=18m47s   
[ckpt] saved checkpoints/hfaqe_step_0001000.nex  (0.00 MB, 8.3 ms)
[train] ep=3  step=  1020/1860  loss=3.8663  ppl=  47.77  gnorm=6.711  lr=1.89e-04  tok/s=   9642  ratio_hc=1.00  ETA=18m17s   
[train] ep=3  step=  1040/1860  loss=3.8681  ppl=  47.85  gnorm=7.676  lr=1.83e-04  tok/s=   9515  ratio_hc=1.00  ETA=17m48s   
[train] ep=3  step=  1060/1860  loss=3.8634  ppl=  47.63  gnorm=6.758  lr=1.77e-04  tok/s=   9676  ratio_hc=1.00  ETA=17m20s   
[train] ep=3  step=  1080/1860  loss=3.8654  ppl=  47.72  gnorm=7.363  lr=1.71e-04  tok/s=   9823  ratio_hc=1.00  ETA=16m51s   
[train] ep=3  step=  1100/1860  loss=3.8713  ppl=  48.01  gnorm=6.738  lr=1.65e-04  tok/s=   9558  ratio_hc=1.00  ETA=16m23s   
[epoch] 3/5 complete.
[val]   step=  1116  val_loss=3.8663  val_ppl=  47.76  (0.0s)
[ckpt] saved checkpoints/hfaqe_epoch_03.nex  (0.00 MB, 9.5 ms)
[train] ep=4  step=  1120/1860  loss=3.8667  ppl=  47.78  gnorm=7.161  lr=1.58e-04  tok/s=   3814  ratio_hc=1.00  ETA=16m17s   
[train] ep=4  step=  1140/1860  loss=3.8665  ppl=  47.77  gnorm=7.623  lr=1.52e-04  tok/s=   9421  ratio_hc=1.00  ETA=15m48s   
[train] ep=4  step=  1160/1860  loss=3.8673  ppl=  47.81  gnorm=7.462  lr=1.46e-04  tok/s=   9481  ratio_hc=1.00  ETA=15m20s   
[train] ep=4  step=  1180/1860  loss=3.8684  ppl=  47.86  gnorm=6.810  lr=1.40e-04  tok/s=   9802  ratio_hc=1.00  ETA=14m51s   
[train] ep=4  step=  1200/1860  loss=3.8629  ppl=  47.60  gnorm=7.198  lr=1.34e-04  tok/s=   9549  ratio_hc=1.00  ETA=14m23s   
[val]   step=  1200  val_loss=3.8672  val_ppl=  47.81  (35.0s)
[train] ep=4  step=  1220/1860  loss=4.0863  ppl=  59.52  gnorm=5.972  lr=1.27e-04  tok/s=   3816  ratio_hc=1.00  ETA=14m14s   
[train] ep=4  step=  1240/1860  loss=4.0392  ppl=  56.78  gnorm=6.715  lr=1.21e-04  tok/s=   9497  ratio_hc=1.00  ETA=13m45s   
[train] ep=4  step=  1260/1860  loss=4.0346  ppl=  56.52  gnorm=5.375  lr=1.15e-04  tok/s=   9503  ratio_hc=1.00  ETA=13m17s   
[train] ep=4  step=  1280/1860  loss=4.0336  ppl=  56.46  gnorm=6.911  lr=1.09e-04  tok/s=   9636  ratio_hc=1.00  ETA=12m48s   
[train] ep=4  step=  1300/1860  loss=4.0345  ppl=  56.51  gnorm=5.824  lr=1.03e-04  tok/s=   9757  ratio_hc=1.00  ETA=12m20s   
[train] ep=4  step=  1320/1860  loss=4.0302  ppl=  56.27  gnorm=5.509  lr=9.76e-05  tok/s=   9511  ratio_hc=1.00  ETA=11m52s   
[train] ep=4  step=  1340/1860  loss=4.0344  ppl=  56.51  gnorm=8.082  lr=9.20e-05  tok/s=   9592  ratio_hc=1.00  ETA=11m24s   
[train] ep=4  step=  1360/1860  loss=4.0356  ppl=  56.57  gnorm=6.109  lr=8.64e-05  tok/s=   9829  ratio_hc=1.00  ETA=10m56s   
[train] ep=4  step=  1380/1860  loss=4.0369  ppl=  56.65  gnorm=7.091  lr=8.10e-05  tok/s=   9512  ratio_hc=1.00  ETA=10m29s   
[train] ep=4  step=  1400/1860  loss=4.0352  ppl=  56.55  gnorm=7.117  lr=7.57e-05  tok/s=   9503  ratio_hc=1.00  ETA=10m01s   
[train] ep=4  step=  1420/1860  loss=4.0398  ppl=  56.81  gnorm=6.224  lr=7.05e-05  tok/s=   9898  ratio_hc=1.00  ETA=9m34s   
[train] ep=4  step=  1440/1860  loss=4.0369  ppl=  56.65  gnorm=5.971  lr=6.55e-05  tok/s=   9535  ratio_hc=1.00  ETA=9m07s   
[train] ep=4  step=  1460/1860  loss=4.0361  ppl=  56.61  gnorm=6.105  lr=6.07e-05  tok/s=   9491  ratio_hc=1.00  ETA=8m40s   
[train] ep=4  step=  1480/1860  loss=4.0343  ppl=  56.50  gnorm=6.970  lr=5.61e-05  tok/s=   9547  ratio_hc=1.00  ETA=8m13s   
[epoch] 4/5 complete.
[val]   step=  1488  val_loss=4.0363  val_ppl=  56.62  (0.0s)
[ckpt] saved checkpoints/hfaqe_epoch_04.nex  (0.00 MB, 9.3 ms)
[train] ep=5  step=  1500/1860  loss=4.0379  ppl=  56.71  gnorm=6.956  lr=5.16e-05  tok/s=   3691  ratio_hc=1.00  ETA=7m55s   
[val]   step=  1500  val_loss=4.0364  val_ppl=  56.62  (35.0s)
[train] ep=5  step=  1520/1860  loss=3.8831  ppl=  48.57  gnorm=7.380  lr=4.73e-05  tok/s=   3786  ratio_hc=1.00  ETA=7m35s   
[train] ep=5  step=  1540/1860  loss=3.8730  ppl=  48.09  gnorm=7.048  lr=4.32e-05  tok/s=   9529  ratio_hc=1.00  ETA=7m08s   
[train] ep=5  step=  1560/1860  loss=3.8771  ppl=  48.29  gnorm=8.043  lr=3.94e-05  tok/s=   9534  ratio_hc=1.00  ETA=6m40s   
[train] ep=5  step=  1580/1860  loss=3.8753  ppl=  48.19  gnorm=7.039  lr=3.57e-05  tok/s=   9934  ratio_hc=1.00  ETA=6m13s   
[train] ep=5  step=  1600/1860  loss=3.8749  ppl=  48.18  gnorm=7.036  lr=3.23e-05  tok/s=   9505  ratio_hc=1.00  ETA=5m45s   
[train] ep=5  step=  1620/1860  loss=3.8790  ppl=  48.37  gnorm=7.002  lr=2.91e-05  tok/s=   9495  ratio_hc=1.00  ETA=5m18s   
[train] ep=5  step=  1640/1860  loss=3.8711  ppl=  48.00  gnorm=8.188  lr=2.61e-05  tok/s=   9925  ratio_hc=1.00  ETA=4m51s   
[train] ep=5  step=  1660/1860  loss=3.8750  ppl=  48.18  gnorm=7.211  lr=2.34e-05  tok/s=   9492  ratio_hc=1.00  ETA=4m24s   
[train] ep=5  step=  1680/1860  loss=3.8762  ppl=  48.24  gnorm=7.070  lr=2.09e-05  tok/s=   9466  ratio_hc=1.00  ETA=3m57s   
[train] ep=5  step=  1700/1860  loss=3.8729  ppl=  48.08  gnorm=7.641  lr=1.86e-05  tok/s=   9751  ratio_hc=1.00  ETA=3m30s   
[train] ep=5  step=  1720/1860  loss=3.8746  ppl=  48.16  gnorm=7.773  lr=1.66e-05  tok/s=   9671  ratio_hc=1.00  ETA=3m04s   
[train] ep=5  step=  1740/1860  loss=3.8727  ppl=  48.07  gnorm=7.105  lr=1.49e-05  tok/s=   9485  ratio_hc=1.00  ETA=2m37s   
[train] ep=5  step=  1760/1860  loss=3.8746  ppl=  48.16  gnorm=6.951  lr=1.34e-05  tok/s=   9568  ratio_hc=1.00  ETA=2m11s   
[train] ep=5  step=  1780/1860  loss=3.8770  ppl=  48.28  gnorm=7.658  lr=1.22e-05  tok/s=   9929  ratio_hc=1.00  ETA=1m44s   
[train] ep=5  step=  1800/1860  loss=3.8740  ppl=  48.13  gnorm=6.880  lr=1.12e-05  tok/s=   9566  ratio_hc=1.00  ETA=1m18s   
[val]   step=  1800  val_loss=3.8748  val_ppl=  48.17  (35.0s)
[train] ep=5  step=  1820/1860  loss=4.0310  ppl=  56.32  gnorm=6.442  lr=1.06e-05  tok/s=   3803  ratio_hc=1.00  ETA=0m52s   
[train] ep=5  step=  1840/1860  loss=4.0401  ppl=  56.83  gnorm=5.435  lr=1.01e-05  tok/s=   9512  ratio_hc=1.00  ETA=0m26s   
[train] ep=5  step=  1860/1860  loss=4.0392  ppl=  56.78  gnorm=2.278  lr=1.00e-05  tok/s=   9451  ratio_hc=1.00  ETA=0m00s   
[epoch] 5/5 complete.
[val]   step=  1860  val_loss=4.0385  val_ppl=  56.74  (0.0s)
[ckpt] saved checkpoints/hfaqe_epoch_05.nex  (0.00 MB, 9.7 ms)
[ckpt] saved checkpoints/hfaqe_final.nex  (0.00 MB, 9.5 ms)

╔══════════════════════════════════════╗
║  Training complete                   ║
║  Steps      : 1860                   ║
║  Best val loss: 3.8663               ║
║  Best val PPL : 47.76                ║
║  Wall time  : 2491.2 s               ║
╚══════════════════════════════════════╝
