# HNU-Net
# Hybrid Nonlinear Unmixing for Hyperspectral image via Evidential Deep Learning with Orthogoal Sparse Regularziation
**Note**: HMU-Net codes are intended for academic communication only and may not be used commercially.
## Architectures and Systems
+ HNU-Net can run under X86 architectures.
+ environment: `python 3.11.5`, `torch 2.1.1` and `CUDA 10.1`.
```
├── utils: All tools used
|  ├──Dir_CVAE.py: Dirichelet VAE
|  ├──Wavelet_Transform_Separation.py: Separation of original HSI using threshold wavelet transform
|  ├──attention.py: Multi-head band attention mechanism
|  └──tools.py: other Other functions used
├── main-jasper.py: code for Jasper Riden dataset
├── main-samson.py: code for Samson dataset
└── main-urban.py: code for Urban dataset
```
