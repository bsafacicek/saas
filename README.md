### SaaS

First phase of the SaaS: To get the estimates on unlabeled data, run either of the following:

```python saas_nozca.py --dataset 'svhn' --net_name resnet ```

```python saas_nozca.py --dataset 'svhn' --net_name conv_large ```

```python saas_nozca.py --dataset 'cifar10' --net_name resnet ```

When we use conv-large instead of ResNet18 for CIFAR-10 dataset, we have employed ZCA preprocessing. 

```python saas_zca.py --dataset 'cifar10' --net_name conv_large ```

Second phase of the SaaS: train on labeled data augmented with the estimates of unlabeled data from the first phase. This phase is simply a supervised learning. 

```python saas_sup_nozca.py --dataset 'svhn' --net_name resnet```

```python saas_sup_nozca.py --dataset 'svhn' --net_name conv_large ```

```python saas_sup_nozca.py --dataset 'cifar10' --net_name resnet ```

When we use conv-large instead of ResNet18 for CIFAR-10 dataset, we have employed ZCA preprocessing. 

```python saas_sup_zca.py  --dataset 'cifar10' --net_name conv_large```

### Fig-1

The minimal code to reproduce the figure-1 in the paper SaaS is inside the folder fig1.
