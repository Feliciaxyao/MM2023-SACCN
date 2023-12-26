# MM2023 - SAMCN
## Introduction

![image](img/SAMCN.png)

Video Entailment via Reaching a Structure-Aware Cross-modal Consensus

Xuan Yao, Junyu Gao, Mengyuan Chen, **Changsheng Xu***

*Correspondece should be addressed to C.X.

State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences.

[Paper Link on ACM MM 2023](https://dl.acm.org/doi/10.1145/3581783.3612345) 


## Prerequisites
### Environment

```
conda create -n samcn python=3.9
pip install -r requirements.txt
```
* `requirements.txt` contains the core requirements for running the code in the `SAMCN` packages.
NOTE: pytorch > = 1.2

### Data Preparation
#### VIOLIN dataset:

We use the visual features, statements and subtitles provided by CVPR 2020 paper: [VIOLIN: A Large-Scale Dataset for Video-and-Language Inference](https://arxiv.org/pdf/2003.11618.pdf). 

1. Download the visual features([C3D features](https://drive.google.com/file/d/10MQ_ceFdhtJYP3CYmm1JoBAQSmnvzv-w/view?pli=1)), [statements and subtitles](https://drive.google.com/file/d/15XS7F_En90CHnSLrRmQ0M1bqEObuqt1-/view) and unzip it under the `./dataset/violin` folder.

We represent the statement and subtitles using the pretrained RoBERTa encoder provided by arXiv 2019 paper: [Roberta: A robustly optimized bert pretraining approach](https://arxiv.org/pdf/1907.11692.pdf). 


2. Download the [pre-trained Roberta model]( https://huggingface.co/roberta-base/tree/main) and put it into the `./roberta.base` folder.

#### VLEP dataset (TODO)



## Training

```
python violin_main.py --results_dir_base 'YOUR OUTPUT PATH' \
                      --feat_dir ./dataset/violin \
                      --bert_dir ./roberta.base \
                      --model VlepModel \
                      --data ViolinDataset \
                      --lr1 5e-6 \
                      --beta1 0.9 \
                      --first_n_epoch 60 \
                      --batch_size 8 \
                      --test_batch_size 8 \
                      --feat c3d \
                      --input_streams vid sub \
                      --dropout 0.3 \
                      --cmcm \
                      --cmcm_loss \
```




## Testing
Download the pretrained model: [Baidu Wangpan](https://pan.baidu.com/s/1QJPWia5226Qxx5_u34434g) (pwd: 200d)

Put our provided pretrained model VIOLIN_SACCN_checkpoint.pth under the project folder, and run:
```
python violin_main.py --ccl --ccl_loss --test 
```


