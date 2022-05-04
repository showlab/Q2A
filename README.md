# Question-to-Actions for AssistQ

This repo provides a baseline model for our proposed task: AssistQ: Affordance-centric Question-driven Task Completion for Egocentric Assistant. 

[[Page]](https://showlab.github.io/assistq/)  [[Paper]](https://arxiv.org/abs/2203.04203)   [[LOVEU@CVPR'22 Challenge]](https://sites.google.com/view/loveucvpr22/track-3?authuser=0) [[CodaLab Leaderboard]](https://codalab.lisn.upsaclay.fr/competitions/4642#results)

Click to know the task:

[![Click to see the demo](https://img.youtube.com/vi/3v8ceel9Mos/0.jpg)](https://www.youtube.com/watch?v=3v8ceel9Mos)

Model Architecture (see [[Paper]](https://arxiv.org/abs/2203.04203) for details):

![arch](https://user-images.githubusercontent.com/20626415/162197041-f1b06325-098c-448a-9b65-d746d8bfe08d.png)


## Install
(1) PyTorch. See https://pytorch.org/ for instruction. For example,
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
(2) PyTorch Lightning. See https://www.pytorchlightning.ai/ for instruction. For example,
```
pip install pytorch-lightning
```

## Data

Download training set and testing set (without ground-truth labels) by filling in the [[AssistQ Downloading Agreement]](https://forms.gle/h9A8GxHksWJfPByf7).

Then carefully set your data path in the config file ;)

## Encoding

Before starting, you should encode the instructional videos, scripts, QAs. See [encoder.md](https://github.com/showlab/Q2A/blob/master/encoder/README.md).

## Training & Evaluation

Select the config file and simply train, e.g.,

```
CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/q2a_gru+fps1+maskx-1_vit_b16+bert_b.yaml
```

To inference a model, e.g.,

```
CUDA_VISIBLE_DEVICES=0 python inference.py --cfg configs/q2a_gru+fps1+maskx-1_vit_b16+bert_b.yaml CKPT "outputs/q2a_gru+fps1+maskx-1_vit_b16+bert_b/lightning_logs/version_0/checkpoints/epoch=5-step=155.ckpt"
```


The evaluation will be performed after each epoch. You can use Tensorboard, or just terminal outputs to record evaluation results.

## Baseline Performance for LOVEU@CVPR2022 Challenge: 80 videos' QA samples for training, 20 videos' QA samples for testing

|  Model   | Recall@1 ↑ | Recall@3 ↑ | MR (Mean Rank) ↓ | MRR (Mean Reciprocal Rank) ↑ |
|  ----  |  ----  |  ----  |  ----  |  ----  |
| Q2A ([configs/q2a_gru+fps1+maskx-1_vit_b16+bert_b.yaml](configs/q2a_gru+fps1+maskx-1_vit_b16+bert_b.yaml)) | 30.2 | 62.3 | 3.2 | 3.2 |

## Contact

Feel free to contact us if you have any problems: joyachen97@gmail.com, or leave an issue in this repo.
