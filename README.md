# Contrastive Learning Loss Comparision

This repository is to compare the performance of different contrastive learning loss functions on the CIFAR-100 dataset. The following loss functions are compared:
* [InfoNCE Loss](https://arxiv.org/pdf/1807.03748v2 "Representation Learning with Contrastive Predictive Coding")
* [NT-Xent Loss (SimCLR)](https://arxiv.org/abs/2002.05709 "A Simple Framework for Contrastive Learning of Visual Representations")
* [DCL Loss (Decoupled Contrastive Learning)](https://arxiv.org/abs/2110.06848 "Decoupled Contrastive Learning")
* [DHEL Loss (Decoupled Hypershperical Energy Loss)](https://arxiv.org/abs/2405.18045 "Bridging Mini-Batch and Asymptotic Analysis in Contrastive Learning: From InfoNCE to Kernel-Based Losses")

InfoNCE loss variants are implemented in the [lib/losses.py](lib/losses.py) file.

## Methodology
* The CIFAR-100 dataset is used to train a ResNet-18 model. 
* The model is pre-trained on 75% of the data for 200 epochs using the contrastive learning loss functions mentioned above.
* The model is then fine-tuned on the remaining 20% of the data for 10 epochs using the cross-entropy loss function and 5% of the data is used for validation.
* The accuracy of the model is evaluated on the test set.

## Usage
To run the code, follow the steps below:
```bash
$ python train.py --loss_func <loss_function>
# loss_function: info_nce, dcl, dcl_symmetric, nt_xent, dhel
# Check python train.py --help for more options
```

## Results
The following table shows the performance of different contrastive learning loss functions on the CIFAR-100 dataset. The results are reported in terms of the top-1 accuracy.

| Loss Function | Top-1 Accuracy | Top-5 Accuracy |
| ------------- | -------------- | -------------- |
| InfoNCE Loss  |                |                |
| NT-Xent Loss  |                |                |
| DCL Loss      |                |                |
| DHEL Loss     |                |                |