
This repository implements the SaaS algorithm proposed in [[1]](https://arxiv.org/abs/1805.00980) using PyTorch.

### SaaS

First phase of the SaaS: To get the estimates on unlabeled data, run either of the following:

```python saas_nozca.py --dataset 'svhn' --net_name resnet ```

```python saas_nozca.py --dataset 'cifar10' --net_name resnet ```

Second phase of the SaaS: train on labeled data augmented with the estimates of unlabeled data from the first phase. This phase is simply a supervised learning. 

```python saas_sup_nozca.py --dataset 'svhn' --net_name resnet```

```python saas_sup_nozca.py --dataset 'cifar10' --net_name resnet ```

### Fig-1

The minimal code to reproduce the figure-1 in the paper SaaS is inside the folder fig1.

## Reference
[1] Cicek, S., A. Fawzi, Soatto, S.: *SaaS: Speed as a Supervisor for Semi-supervised Learning*.  In Proceedings of the European Conference on Computer Vision (ECCV ’18).

