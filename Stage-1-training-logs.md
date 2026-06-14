Training logs



╔══════════════════════════════════════════════════════════╗ ║  HFAQE Embedding Training                                ║ ╚══════════════════════════════════════════════════════════╝ [config] V=256  d=256  r=64  K=128  B=64 [config] lr=3.00e-04  epochs=5  batch=64  seq=256 [config] data=Data  ckpt=checkpoints [data] Loading Data/train.txt ... [data] Loading Data/validation.txt ... [data] train=23767 lines  val=2461 lines [model] Computing token frequencies ... [model] Hot=128 int8[128×256]  Cold=128 bf16[128×64]  Basis bf16[256×64] [model] Param RAM: 0.08 MB  (baseline BF16 would be 0.12 MB) [train] 372 steps/epoch × 5 epochs = 1860 total steps [train] ep=1  step=    20/1860  loss=5.5404  ppl= 254.79  gnorm=0.000  lr=1.50e-05  tok/s=  16957  ETA=19m45s    [train] ep=1  step=    40/1860  loss=5.5398  ppl= 254.63  gnorm=0.000  lr=3.00e-05  tok/s=  16868  ETA=19m32s    [train] ep=1  step=    60/1860  loss=5.5393  ppl= 254.50  gnorm=0.000  lr=4.50e-05  tok/s=  16398  ETA=19m40s    [train] ep=1  step=    80/1860  loss=5.5395  ppl= 254.54  gnorm=0.000  lr=6.00e-05  tok/s=  17166  ETA=19m26s    [train] ep=1  step=   100/1860  loss=5.5403  ppl= 254.75  gnorm=0.000  lr=7.50e-05  tok/s=  17116  ETA=19m01s    [train] ep=1  step=   120/1860  loss=5.5399  ppl= 254.65  gnorm=0.000  lr=9.00e-05  tok/s=  17084  ETA=18m41s    [train] ep=1  step=   140/1860  loss=5.5398  ppl= 254.63  gnorm=0.000  lr=1.05e-04  tok/s=  17206  ETA=18m24s    [train] ep=1  step=   160/1860  loss=5.5402  ppl= 254.73  gnorm=0.000  lr=1.20e-04  tok/s=  17136  ETA=18m09s    [train] ep=1  step=   180/1860  loss=5.5405  ppl= 254.79  gnorm=0.000  lr=1.35e-04  tok/s=  17309  ETA=17m57s    [train] ep=1  step=   200/1860  loss=5.5397  ppl= 254.59  gnorm=0.000  lr=1.50e-04  tok/s=  17225  ETA=17m44s    [train] ep=1  step=   220/1860  loss=5.5397  ppl= 254.61  gnorm=0.000  lr=1.65e-04  tok/s=  16625  ETA=17m31s    [train] ep=1  step=   240/1860  loss=5.5396  ppl= 254.58  gnorm=0.000  lr=1.80e-04  tok/s=  17049  ETA=17m20s    [train] ep=1  step=   260/1860  loss=5.5400  ppl= 254.69  gnorm=0.000  lr=1.95e-04  tok/s=  17184  ETA=17m06s    [train] ep=1  step=   280/1860  loss=5.5397  ppl= 254.60  gnorm=0.000  lr=2.10e-04  tok/s=  17324  ETA=16m52s    [train] ep=1  step=   300/1860  loss=5.5398  ppl= 254.63  gnorm=0.000  lr=2.25e-04  tok/s=  17276  ETA=16m39s    [val]   step=   300  val_loss=5.5388  val_ppl= 254.37  (22.0s) [ckpt] saved checkpoints/hfaqe_best.nex  (0.00 MB, 5.2 ms) [train] ep=1  step=   320/1860  loss=5.5401  ppl= 254.71  gnorm=0.000  lr=2.40e-04  tok/s=   6240  ETA=18m10s    [train] ep=1  step=   340/1860  loss=5.5391  ppl= 254.45  gnorm=0.000  lr=2.55e-04  tok/s=  17152  ETA=17m49s    [train] ep=1  step=   360/1860  loss=5.5399  ppl= 254.66  gnorm=0.000  lr=2.70e-04  tok/s=  17275  ETA=17m29s    [epoch] 1/5 complete. [val]   step=   372  val_loss=5.5389  val_ppl= 254.39  (0.0s) [ckpt] saved checkpoints/hfaqe_epoch_01.nex  (0.00 MB, 6.8 ms) [train] ep=2  step=   380/1860  loss=5.5399  ppl= 254.65  gnorm=0.000  lr=2.85e-04  tok/s=   6073  ETA=18m35s    [train] ep=2  step=   400/1860  loss=5.5401  ppl= 254.70  gnorm=0.000  lr=3.00e-04  tok/s=  17152  ETA=18m10s    [train] ep=2  step=   420/1860  loss=5.5407  ppl= 254.85  gnorm=0.000  lr=3.00e-04  tok/s=  17085  ETA=17m48s    [train] ep=2  step=   440/1860  loss=5.5405  ppl= 254.80  gnorm=0.000  lr=2.99e-04  tok/s=  16386  ETA=17m27s    [train] ep=2  step=   460/1860  loss=5.5403  ppl= 254.76  gnorm=0.000  lr=2.99e-04  tok/s=  16959  ETA=17m08s    [train] ep=2  step=   480/1860  loss=5.5403  ppl= 254.76  gnorm=0.000  lr=2.98e-04  tok/s=  16875  ETA=16m48s    [train] ep=2  step=   500/1860  loss=5.5406  ppl= 254.83  gnorm=0.000  lr=2.97e-04  tok/s=  17058  ETA=16m29s    [train] ep=2  step=   520/1860  loss=5.5407  ppl= 254.86  gnorm=0.000  lr=2.95e-04  tok/s=  16876  ETA=16m09s    [train] ep=2  step=   540/1860  loss=5.5403  ppl= 254.76  gnorm=0.000  lr=2.94e-04  tok/s=  16969  ETA=15m50s    [train] ep=2  step=   560/1860  loss=5.5412  ppl= 254.99  gnorm=0.000  lr=2.92e-04  tok/s=  17049  ETA=15m32s    [train] ep=2  step=   580/1860  loss=5.5412  ppl= 254.97  gnorm=0.000  lr=2.89e-04  tok/s=  16804  ETA=15m15s    [train] ep=2  step=   600/1860  loss=5.5411  ppl= 254.96  gnorm=0.000  lr=2.87e-04  tok/s=  16649  ETA=14m58s    [val]   step=   600  val_loss=5.5396  val_ppl= 254.57  (21.7s) [train] ep=2  step=   620/1860  loss=5.5404  ppl= 254.79  gnorm=0.000  lr=2.84e-04  tok/s=   6289  ETA=15m25s    [train] ep=2  step=   640/1860  loss=5.5407  ppl= 254.85  gnorm=0.000  lr=2.81e-04  tok/s=  17252  ETA=15m06s    [train] ep=2  step=   660/1860  loss=5.5413  ppl= 255.00  gnorm=0.000  lr=2.78e-04  tok/s=  17212  ETA=14m47s    [train] ep=2  step=   680/1860  loss=5.5411  ppl= 254.95  gnorm=0.000  lr=2.75e-04  tok/s=  17174  ETA=14m29s    [train] ep=2  step=   700/1860  loss=5.5408  ppl= 254.87  gnorm=0.000  lr=2.71e-04  tok/s=  17247  ETA=14m11s    [train] ep=2  step=   720/1860  loss=5.5413  ppl= 255.00  gnorm=0.000  lr=2.67e-04  tok/s=  17183  ETA=13m53s    [train] ep=2  step=   740/1860  loss=5.5417  ppl= 255.12  gnorm=0.000  lr=2.63e-04  tok/s=  17095  ETA=13m36s    [epoch] 2/5 complete. [val]   step=   744  val_loss=5.5404  val_ppl= 254.78  (0.0s) [ckpt] saved checkpoints/hfaqe_epoch_02.nex  (0.00 MB, 6.1 ms) [train] ep=3  step=   760/1860  loss=5.5417  ppl= 255.12  gnorm=0.000  lr=2.59e-04  tok/s=   6121  ETA=13m50s    [train] ep=3  step=   780/1860  loss=5.5422  ppl= 255.23  gnorm=0.000  lr=2.54e-04  tok/s=  17042  ETA=13m31s    [train] ep=3  step=   800/1860  loss=5.5419  ppl= 255.17  gnorm=0.000  lr=2.50e-04  tok/s=  16846  ETA=13m14s    [train] ep=3  step=   820/1860  loss=5.5424  ppl= 255.29  gnorm=0.000  lr=2.45e-04  tok/s=  16480  ETA=12m57s    [train] ep=3  step=   840/1860  loss=5.5423  ppl= 255.26  gnorm=0.000  lr=2.40e-04  tok/s=  16964  ETA=12m39s    [train] ep=3  step=   860/1860  loss=5.5422  ppl= 255.24  gnorm=0.000  lr=2.35e-04  tok/s=  16998  ETA=12m22s    [train] ep=3  step=   880/1860  loss=5.5421  ppl= 255.22  gnorm=0.000  lr=2.30e-04  tok/s=  17011  ETA=12m05s    [train] ep=3  step=   900/1860  loss=5.5422  ppl= 255.24  gnorm=0.000  lr=2.24e-04  tok/s=  16981  ETA=11m48s    [val]   step=   900  val_loss=5.5412  val_ppl= 254.99  (22.1s) [train] ep=3  step=   920/1860  loss=5.5425  ppl= 255.31  gnorm=0.000  lr=2.19e-04  tok/s=   6238  ETA=11m54s    [train] ep=3  step=   940/1860  loss=5.5426  ppl= 255.35  gnorm=0.000  lr=2.13e-04  tok/s=  17140  ETA=11m36s    [train] ep=3  step=   960/1860  loss=5.5429  ppl= 255.43  gnorm=0.000  lr=2.07e-04  tok/s=  17175  ETA=11m18s    [train] ep=3  step=   980/1860  loss=5.5423  ppl= 255.27  gnorm=0.000  lr=2.01e-04  tok/s=  16964  ETA=11m01s    [train] ep=3  step=  1000/1860  loss=5.5423  ppl= 255.27  gnorm=0.000  lr=1.95e-04  tok/s=  16538  ETA=10m45s    [ckpt] saved checkpoints/hfaqe_step_0001000.nex  (0.00 MB, 5.7 ms) [train] ep=3  step=  1020/1860  loss=5.5431  ppl= 255.46  gnorm=0.000  lr=1.89e-04  tok/s=  17172  ETA=10m28s    [train] ep=3  step=  1040/1860  loss=5.5430  ppl= 255.45  gnorm=0.000  lr=1.83e-04  tok/s=  17198  ETA=10m11s    [train] ep=3  step=  1060/1860  loss=5.5429  ppl= 255.41  gnorm=0.000  lr=1.77e-04  tok/s=  17282  ETA=9m54s    [train] ep=3  step=  1080/1860  loss=5.5427  ppl= 255.37  gnorm=0.000  lr=1.71e-04  tok/s=  17085  ETA=9m38s    [train] ep=3  step=  1100/1860  loss=5.5430  ppl= 255.43  gnorm=0.000  lr=1.65e-04  tok/s=  17037  ETA=9m21s    [epoch] 3/5 complete. [val]   step=  1116  val_loss=5.5418  val_ppl= 255.12  (0.0s) [ckpt] saved checkpoints/hfaqe_epoch_03.nex  (0.00 MB, 6.6 ms) [train] ep=4  step=  1120/1860  loss=5.5427  ppl= 255.37  gnorm=0.000  lr=1.58e-04  tok/s=   6224  ETA=9m20s    [train] ep=4  step=  1140/1860  loss=5.5434  ppl= 255.56  gnorm=0.000  lr=1.52e-04  tok/s=  17051  ETA=9m03s    [train] ep=4  step=  1160/1860  loss=5.5433  ppl= 255.52  gnorm=0.000  lr=1.46e-04  tok/s=  17163  ETA=8m46s    [train] ep=4  step=  1180/1860  loss=5.5436  ppl= 255.59  gnorm=0.000  lr=1.40e-04  tok/s=  17069  ETA=8m30s    [train] ep=4  step=  1200/1860  loss=5.5427  ppl= 255.36  gnorm=0.000  lr=1.34e-04  tok/s=  16619  ETA=8m14s    [val]   step=  1200  val_loss=5.5421  val_ppl= 255.21  (22.0s) [train] ep=4  step=  1220/1860  loss=5.5436  ppl= 255.61  gnorm=0.000  lr=1.27e-04  tok/s=   6295  ETA=8m10s    [train] ep=4  step=  1240/1860  loss=5.5434  ppl= 255.54  gnorm=0.000  lr=1.21e-04  tok/s=  17307  ETA=7m53s    [train] ep=4  step=  1260/1860  loss=5.5432  ppl= 255.51  gnorm=0.000  lr=1.15e-04  tok/s=  17180  ETA=7m36s    [train] ep=4  step=  1280/1860  loss=5.5438  ppl= 255.65  gnorm=0.000  lr=1.09e-04  tok/s=  17271  ETA=7m20s    [train] ep=4  step=  1300/1860  loss=5.5437  ppl= 255.62  gnorm=0.000  lr=1.03e-04  tok/s=  17220  ETA=7m04s    [train] ep=4  step=  1320/1860  loss=5.5426  ppl= 255.33  gnorm=0.000  lr=9.76e-05  tok/s=  17310  ETA=6m47s    [train] ep=4  step=  1340/1860  loss=5.5432  ppl= 255.49  gnorm=0.000  lr=9.20e-05  tok/s=  17244  ETA=6m31s    [train] ep=4  step=  1360/1860  loss=5.5431  ppl= 255.47  gnorm=0.000  lr=8.64e-05  tok/s=  17329  ETA=6m15s    [train] ep=4  step=  1380/1860  loss=5.5436  ppl= 255.59  gnorm=0.000  lr=8.10e-05  tok/s=  16616  ETA=6m00s    [train] ep=4  step=  1400/1860  loss=5.5437  ppl= 255.62  gnorm=0.000  lr=7.57e-05  tok/s=  17087  ETA=5m44s    [train] ep=4  step=  1420/1860  loss=5.5436  ppl= 255.60  gnorm=0.000  lr=7.05e-05  tok/s=  17188  ETA=5m28s    [train] ep=4  step=  1440/1860  loss=5.5433  ppl= 255.51  gnorm=0.000  lr=6.55e-05  tok/s=  17242  ETA=5m13s    [train] ep=4  step=  1460/1860  loss=5.5434  ppl= 255.55  gnorm=0.001  lr=6.07e-05  tok/s=  17292  ETA=4m57s    [train] ep=4  step=  1480/1860  loss=5.5437  ppl= 255.63  gnorm=0.000  lr=5.61e-05  tok/s=  17103  ETA=4m42s    [epoch] 4/5 complete. [val]   step=  1488  val_loss=5.5422  val_ppl= 255.24  (0.0s) [ckpt] saved checkpoints/hfaqe_epoch_04.nex  (0.00 MB, 8.2 ms) [train] ep=5  step=  1500/1860  loss=5.5437  ppl= 255.61  gnorm=0.000  lr=5.16e-05  tok/s=   6109  ETA=4m32s    [val]   step=  1500  val_loss=5.5422  val_ppl= 255.24  (22.0s) [train] ep=5  step=  1520/1860  loss=5.5435  ppl= 255.58  gnorm=0.001  lr=4.73e-05  tok/s=   6243  ETA=4m21s    [train] ep=5  step=  1540/1860  loss=5.5436  ppl= 255.61  gnorm=0.000  lr=4.32e-05  tok/s=  16936  ETA=4m05s    [train] ep=5  step=  1560/1860  loss=5.5434  ppl= 255.53  gnorm=0.000  lr=3.94e-05  tok/s=  16441  ETA=3m49s    [train] ep=5  step=  1580/1860  loss=5.5439  ppl= 255.67  gnorm=0.000  lr=3.57e-05  tok/s=  16727  ETA=3m33s    [train] ep=5  step=  1600/1860  loss=5.5440  ppl= 255.70  gnorm=0.000  lr=3.23e-05  tok/s=  17135  ETA=3m18s    [train] ep=5  step=  1620/1860  loss=5.5435  ppl= 255.56  gnorm=0.000  lr=2.91e-05  tok/s=  17106  ETA=3m02s    [train] ep=5  step=  1640/1860  loss=5.5436  ppl= 255.59  gnorm=0.000  lr=2.61e-05  tok/s=  17234  ETA=2m47s    [train] ep=5  step=  1660/1860  loss=5.5443  ppl= 255.79  gnorm=0.000  lr=2.34e-05  tok/s=  17104  ETA=2m31s    [train] ep=5  step=  1680/1860  loss=5.5436  ppl= 255.60  gnorm=0.000  lr=2.09e-05  tok/s=  17043  ETA=2m16s    [train] ep=5  step=  1700/1860  loss=5.5434  ppl= 255.54  gnorm=0.000  lr=1.86e-05  tok/s=  16404  ETA=2m00s    [train] ep=5  step=  1720/1860  loss=5.5433  ppl= 255.52  gnorm=0.000  lr=1.66e-05  tok/s=  15001  ETA=1m45s    [train] ep=5  step=  1740/1860  loss=5.5432  ppl= 255.49  gnorm=0.000  lr=1.49e-05  tok/s=  16501  ETA=1m30s    [train] ep=5  step=  1760/1860  loss=5.5433  ppl= 255.52  gnorm=0.000  lr=1.34e-05  tok/s=  17115  ETA=1m15s    [train] ep=5  step=  1780/1860  loss=5.5428  ppl= 255.40  gnorm=0.000  lr=1.22e-05  tok/s=  17023  ETA=1m00s    [train] ep=5  step=  1800/1860  loss=5.5429  ppl= 255.41  gnorm=0.000  lr=1.12e-05  tok/s=  17031  ETA=0m45s    [val]   step=  1800  val_loss=5.5422  val_ppl= 255.25  (22.2s) [train] ep=5  step=  1820/1860  loss=5.5434  ppl= 255.54  gnorm=0.000  lr=1.06e-05  tok/s=   6173  ETA=0m30s    [train] ep=5  step=  1840/1860  loss=5.5434  ppl= 255.55  gnorm=0.000  lr=1.01e-05  tok/s=  16787  ETA=0m15s    [train] ep=5  step=  1860/1860  loss=5.5435  ppl= 255.58  gnorm=0.000  lr=1.00e-05  tok/s=  16967  ETA=0m00s    [epoch] 5/5 complete. [val]   step=  1860  val_loss=5.5422  val_ppl= 255.25  (0.0s) [ckpt] saved checkpoints/hfaqe_epoch_05.nex  (0.00 MB, 7.0 ms) [ckpt] saved checkpoints/hfaqe_final.nex  (0.00 MB, 6.2 ms)  ╔══════════════════════════════════════╗ ║  Training complete                   ║ ║  Steps      : 1860                   ║ ║  Best val loss: 5.5388               ║ ║  Best val PPL : 254.37               ║ ║  Wall time  : 1433.3 s               ║ ╚══════════════════════════════════════╝ 

Evaluation logs

╔══════════════════════════════════════════════════════════════╗
║  HFAQE Evaluation                                            ║
╚══════════════════════════════════════════════════════════════╝
  Checkpoint : checkpoints/hfaqe_best.nex
  Data dir   : Data

[load] Opening checkpoint...
[load] Model loaded: V=256 d=256 r=64 K=128  hot=128 cold=128
[data] Loading test corpus...
[data] test=2891 lines  train=23767 lines

[eval] Task 1: Perplexity...
  train PPL=254.6855  test PPL=254.7130
[eval] Task 2: Nearest neighbour retrieval...
[eval] Task 3: Embedding space geometry...
[eval] Task 4: Tier fidelity...
[eval] Task 5: Throughput...
[eval] Task 6: Anisotropy...

╔══════════════════════════════════════════════════════════════
╗
║  HFAQE EMBEDDING EVALUATION REPORT                           ║
╠══════════════════════════════════════════════════════════════
╣
║  Model : checkpoints/hfaqe_best.nex                            ║
║  Step  : 300        Epoch : 0      Best val PPL : 254.37   ║
║  V=256     d=256    r=64    K=128    B=64              ║
╠══════════════════════════════════════════════════════════════
╣
║  TASK 1 — PERPLEXITY  (lower = better; random baseline = 256)  ║
╠══════════════════════════════════════════════════════════════
╣
║  Train set PPL                   254.6855  (4047526 tokens, 298.6 s)     ║
║  Test  set PPL                   254.7130  (487076 tokens, 36.0 s)     ║
║  Train PPL improvement             +0.51%  vs uniform random           ║
║  Test  PPL improvement             +0.50%  vs uniform random           ║
╠══════════════════════════════════════════════════════════════
╣
║  TASK 4 — TIER FIDELITY                                      ║
╠══════════════════════════════════════════════════════════════
╣
║  Hot tier (int8 quantisation):                               ║
║    Mean relative error  : 0.000000   (SPEC bound: < 1/254)   ║
║    Max  relative error  : 0.000000                            ║
║    RMSE (element-wise)  : 0.000000                            ║
║  Cold tier (low-rank B·α):                                   ║
║    Mean relative error  : 0.000000   (SPEC bound: < 0.02)    ║
║    Max  relative error  : 0.000000                            ║
║    Hot  fidelity check  : ✓ PASS                            ║
║    Cold fidelity check  : ✓ PASS                            ║
╠══════════════════════════════════════════════════════════════
╣
║  TASK 5 — THROUGHPUT & MEMORY                                ║
╠══════════════════════════════════════════════════════════════
╣
║  Hot  gather  :  14220721 tok/s                               ║
║  Cold reconstruct :    519591 tok/s                           ║
║  LM-head      :    0.0689 ms/token                            ║
║  HFAQE RAM    :     0.080 MB                                  ║
║  Baseline BF16:     0.125 MB                                  ║
║  RAM reduction:     35.9%                                   ║
╠══════════════════════════════════════════════════════════════
╣
║  TASK 3 — EMBEDDING SPACE GEOMETRY                           ║
╠══════════════════════════════════════════════════════════════
╣
║  Mean embedding L2 norm :   0.9393  (±0.0790)               ║
║  Anisotropy             :   0.0560  (random=0.0, bad=1.0)  ║
║  Intra-digit similarity :   0.0052  (avg cos('0'..'9'))     ║
║  Intra-alpha similarity :  -0.0037  (avg cos('a'..'z'))     ║
║  Inter-class similarity :   0.0074  (digit vs alpha)        ║
║  Cluster ratio          :   0.1015  (>1.0 = classes cluster)║
║    → – no strong clustering                                 ║
╠══════════════════════════════════════════════════════════════
╣
║  TASK 2 — NEAREST NEIGHBOUR RETRIEVAL  (cosine similarity)   ║
╠══════════════════════════════════════════════════════════════
╣
║  Query '0' (0x30)  →  top-5 neighbours:                     ║
║    'V'(0.160)  '?'(0.135)  '?'(0.135)  '?'(0.128)  'O'(0.126)  ║
║  Query 'a' (0x61)  →  top-5 neighbours:                     ║
║    '#'(0.178)  '?'(0.158)  '?'(0.153)  '?'(0.148)  'g'(0.138)  ║
║  Query 'A' (0x41)  →  top-5 neighbours:                     ║
║    '\'(0.147)  'i'(0.124)  ','(0.123)  '?'(0.112)  '?'(0.105)  ║
║  Query ' ' (0x20)  →  top-5 neighbours:                     ║
║    '?'(0.198)  '?'(0.186)  '*'(0.177)  '?'(0.149)  '@'(0.137)  ║
║  Query '.' (0x2E)  →  top-5 neighbours:                     ║
║    '?'(0.212)  '?'(0.142)  '?'(0.138)  ';'(0.123)  '?'(0.121)  ║
║  Query '?' (0x0A)  →  top-5 neighbours:                     ║
║    '?'(0.305)  '?'(0.233)  '?'(0.201)  '?'(0.201)  '?'(0.199)  ║
║  Query 'z' (0x7A)  →  top-5 neighbours:                     ║
║    'Y'(0.184)  '`'(0.182)  '?'(0.137)  '?'(0.136)  '?'(0.133)  ║
║  Query '9' (0x39)  →  top-5 neighbours:                     ║
║    'N'(0.154)  '?'(0.149)  'd'(0.143)  '?'(0.141)  '?'(0.140)  ║
║  Query '!' (0x21)  →  top-5 neighbours:                     ║
║    '?'(0.155)  '{'(0.129)  '?'(0.110)  '?'(0.106)  '?'(0.101)  ║
╠══════════════════════════════════════════════════════════════
╣
║  SUMMARY                                                     ║
╠══════════════════════════════════════════════════════════════
╣
║  Perplexity improvement  :   0.3 / 40.0 pts               ║
║  Tier fidelity           :  20.0 / 20.0 pts               ║
║  Embedding clustering    :   1.0 / 20.0 pts               ║
║  Low anisotropy          :   8.9 / 10.0 pts               ║
║  Memory compression      :   3.6 / 10.0 pts               ║
╠══════════════════════════════════════════════════════════════
╣
║  TOTAL SCORE             :  33.8 / 100.0                  ║
║  Grade: WEAK      — training needs fixing                   ║
╚══════════════════════════════════════════════════════════════
╝