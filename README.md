# Sleep Deprivation in the Forward Forward Algorithm

This repository is the implementation of our submission to the [Tiny papers section of ICLR 2023](https://iclr.cc/Conferences/2023/CallForTinyPapers).

The project contains 3 files and one notebook:

- data.py - contains helpers and dataloaders for MNIST, CIFAR10 and FashionMNIST.
- FFLayer.py - a pytorch implementation of a Forward Forward layer.
- FF.py - a pytorch module that combines multiple FFLayers.

- Notebook.ipynb - the main and only notebook that allows training an FF model for all of the implemented datasets and allows tweaking negative data generation and all other parameters.


In this study we explore the behaviour of the Forward Forward Algorithm by Hinton with separated positive and negative phases. We find a relation between the type of negative data and the behaviour at large separations.

Even though the CIFAR-10 dataset is included in this repo, it was not included in the study as it experienced close to no learning when the two phases were separated.

## Results
### Table 1: Accuracy of models with balanced awake and sleep phases.

There's a clear performance advantage when using masks as negative data for larger phases.

| Dataset | Negative data | Unseparated | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | Wrong label | 97.7% | 89% | 88% | 81% | 63% | 35% | 11% | 10% | 9% |
| MNIST | Masks | 96% | 88% | 84% | 85% | 78% | 74% | 49% | 23% | 11% |
| Fashion-MNIST | Wrong label | 88.7% | 73% | 63% | 59% | 54% | 20% | 14% | 10% | 10% |
| Fashion-MNIST | Masks | 84% | 56% | 58% | 53% | 50% | 45% | 36% | 30% | 22% |

### Table 2: Accuracy of models using unequal phases.

The phase size of the positive data ranges from 1 to 16, and the negative phase is fixed at 1. The awake phase's learning rate is scaled down to have the same effect as the negative phase in the training procedure. The empty lines in the table represent models that experienced no learning.

| Dataset | Negative data | Unseparated | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|--|
| MNIST | Wrong label | 97.7% | 89% | 10% | 9% | - | - |
| MNIST | Masks | 96% | 88% | 78% | 75% | 73% | 10% |
| Fashion-MNIST | Wrong label | 88.7% | 73% | 56% | 47% | 14% | 10% |
| Fashion-MNIST | Masks | 84% | 56% | 55% | 54% | 52% | 49% |