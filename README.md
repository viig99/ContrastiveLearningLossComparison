# Contrastive Learning Loss Comparision

This repository is re-create the results from the paper [Bridging Mini-Batch and Asymptotic Analysis in Contrastive Learning: From InfoNCE to Kernel-Based Losses](https://arxiv.org/abs/2405.18045).
It compare's the performance of different contrastive learning loss functions on the CIFAR-100 dataset. 
The following loss functions are compared:
  * [InfoNCE Loss](https://arxiv.org/pdf/1807.03748v2 "Representation Learning with Contrastive Predictive Coding")
  * [NT-Xent Loss (SimCLR)](https://arxiv.org/abs/2002.05709 "A Simple Framework for Contrastive Learning of Visual Representations")
  * [DCL Loss (Decoupled Contrastive Learning)](https://arxiv.org/abs/2110.06848 "Decoupled Contrastive Learning")
  * [DHEL Loss (Decoupled Hypershperical Energy Loss)](https://arxiv.org/abs/2405.18045 "Bridging Mini-Batch and Asymptotic Analysis in Contrastive Learning: From InfoNCE to Kernel-Based Losses")
  * [VICReg Loss (Variance-Invariance-Covariance Regularization)](https://arxiv.org/pdf/2105.04906 "Vicreg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning")

InfoNCE loss variants are implemented in the [lib/losses.py](lib/losses.py) file.

## Methodology
* The CIFAR-100 dataset is used to train a ResNet-18 model. 
* The model is pre-trained on all the training data for 200 epochs using the contrastive learning loss functions mentioned above.
* The model is then fine-tuned for classification on 90% of the data for another 200 epochs and the rest 10% of the data is used for validation.
* The accuracy of the model is then evaluated on the test set.

## Usage
To run the code, follow the steps below:
```bash
$ python train.py --loss_func <loss_function> --continue_pretrain --continue_finetune
# loss_function: info_nce, dcl, dcl_symmetric, nt_xent, dhel, vicreg
# Check python train.py --help for more options
```

## Results
The following table shows the performance of different contrastive learning loss functions on the CIFAR-100 dataset.

| Loss Function | Top-1 Accuracy | Top-5 Accuracy | Additional Notes                                           |
| ------------- | -------------- | -------------- | ---------------------------------------------------------- |
| InfoNCE Loss  |                |                |                                                            |
| NT-Xent Loss  | 0.5364         | 0.8115         | -                                                          |
| DCL Loss      | 0.5629         | 0.8322         | faster than NT-Xent but slower than DHEL.                  |
| DHEL Loss     | 0.5614         | 0.8256         | Classification accuracy converges fast  (high uniformity?) |
| VICReg Loss   |                |                |                                                            |