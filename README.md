# TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis with Missing and Noisy Modalities
The code for TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis with Missing and Noisy Modalities, which is accepted by AAAI 2026 [paper](https://arxiv.org/abs/2511.10325).

### The Framework of TMDC:
![image](https://github.com/YetZzzzzz/TMDC/blob/main/framework.png)
Figure 1: Illustration of the proposed TMDC. TMDC includes two training stages. In the first stage, TMDC learns from complete modality information using two denoising modules. The modality-specific denoising module applies separate networks to each modality to remove noise while preserving unique modality information. Simultaneously, the modality-common denoising module employs a shared network to filter noise across multiple modalities and extract common information. In the second stage, the learned shared information is used to supplement missing modalities.

### Datasets:
**Please move the CMU-MOSI, CMU-MOSEI and IEMOCAP datasets into directory ./gcnet_datasets/.**

These dataset can be downloaded according to [GCNet](https://github.com/zeroQiaoba/GCNet) or [MoMKE](https://github.com/wxxv/MoMKE/tree/main).

### Prerequisites:
```
* Python 3.8.10
* CUDA 11.5
* pytorch 1.12.1+cu113
* sentence-transformers 3.1.1
* transformers 4.30.2
```
**Note that the torch version can be changed to your cuda version, but please keep the transformers==4.30.2 as some functions will change in later versions**

### Run TMDC
For CMU-MOSI, please run the following code:
```
bash run_TMDC_cmumosi.sh
```
For CMU-MOSEI, please run the following code:
```
bash run_TMDC_cmumosei.sh
```
For IEMOCAP, please run the following code:
```
bash run_TMDC_iemocap4.sh
```


### Citation:
Please cite our paper if you find our work useful for your research:
```
@article{zhuang2025tmdc,
  title={TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis with Missing and Noisy Modalities},
  author={Zhuang, Yan and Liu, Minhao and Zhang, Yanru and Deng, Jiawen and Ren, Fuji},
  journal={arXiv preprint arXiv:2511.10325},
  year={2025}
}
```
### Acknowledgement
Thanks to  [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck),  [MoMKE](https://github.com/wxxv/MoMKE/tree/main), and [GCNet](https://github.com/zeroQiaoba/GCNet) for their great help to our codes and research. 
