

### This folder contains the minimal code to reproduce the figure-1 in the paper SaaS. 

Run

```python train_on_random_labels.py```

to train network on random labels several times and to save the losses to .npy file. Then, run 

``` python fig1.py ``` 

to get the plots in the paper. These results in the figure-1 are obtained by fixed learning rate of 0.1, with
no data augmentation or weight decay.
