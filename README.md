# GRHP: Graph-Fused Hierarchical Planning forEmbodied Long-Horizon Robotic Task

## Overview


**This repository serves as the official implementation of the paper "GRHP: Graph-Fused Hierarchical Planning for Embodied Long-Horizon Robotic Task".**




## 1. INSTALLATION

First, install the dependent python environments using the following commands

```
conda env create -f environment.yml
```

You need to use [**ALFRED's official GitHub page**](https://github.com/askforalfred/alfred) to download the dataset(Trajectory JSONs and Resnet feats (~17GB)).

## 2.USAGE

After following the official tutorial to configure the environment and download the dataset, then train the model using the following commands.

```
python models/train/train_seq2seq.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
```



