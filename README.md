# Contrastive Learning Loss Comparision

This repository is to compare the performance of different contrastive learning loss functions on the CIFAR-100 dataset. The following loss functions are compared:
* InfoNCE Loss
* NT-Xent Loss (SimCLR)
* DCL Loss (Decoupled Contrastive Learning)
* DHEL Loss (Decoupled Hypershperical Energy Loss)

## Methodology
* The CIFAR-100 dataset is used to train a ResNet-18 model. 
* The model is pre-trained on 75% of the data for 200 epochs using the contrastive learning loss functions mentioned above.
* The model is then fine-tuned on the remaining 20% of the data for 10 epochs using the cross-entropy loss function and 5% of the data is used for validation.
* The accuracy of the model is evaluated on the test set.

## Results
The following table shows the performance of different contrastive learning loss functions on the CIFAR-100 dataset. The results are reported in terms of the top-1 accuracy.

| Loss Function | Top-1 Accuracy |
| ------------- | -------------- |
| InfoNCE Loss  | 0.425          |
| NT-Xent Loss  |                |
| DCL Loss      |                |
| DHEL Loss     |                |