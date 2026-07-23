# ML Learning Journey

Hands-on deep learning fundamentals, documented notebook by notebook. Each folder is a self-contained experiment covering one core concept, with working code rather than just notes.

## Topics Covered

| Topic | What it covers |
|---|---|
| [`data_scaling`](./data_scaling) | Feature scaling and normalization techniques for neural network inputs |
| [`weight_initialization`](./weight_initialization) | How initial weight distributions affect training dynamics |
| [`batch_Normalization`](./batch_Normalization) | Batch norm implementation and its effect on convergence |
| [`Regularization`](./Regularization) | L1/L2 regularization and dropout to reduce overfitting |
| [`optimizers`](./optimizers) | Comparing optimizers (SGD, Adam, RMSprop, etc.) |
| [`hyperparameter-tuning`](./hyperparameter-tuning) | Systematic approaches to tuning learning rate, batch size, etc. |
| [`Data-Augmentation`](./Data-Augmentation) | Image augmentation techniques for improving generalization |
| [`CNN`](./CNN) | Convolutional neural network architectures from scratch |
| [`LSTM`](./LSTM) | Sequence modeling with LSTMs |
| [`Image-classification`](./Image-classification) | End-to-end image classification pipelines |
| [`cats vs dogs`](./cats%20vs%20dogs) | Classic binary image classification project |
| [`using-pretrained-model`](./using-pretrained-model) | Transfer learning with pretrained architectures |

## Why this repo exists

Most tutorials show the finished model. This repo is the opposite — it's the scratch work: testing what happens when you change one variable at a time (initialization, batch size, regularization strength) so the *why* behind deep learning best practices actually sticks.

## Tech Stack

Python, PyTorch/TensorFlow, Jupyter Notebook

## Notes

Each folder's notebook is self-contained and can be run independently. See individual notebooks for setup/dependency details.

<!--
FILL IN BEFORE PUBLISHING (delete this block):
- Confirm whether you used PyTorch or TensorFlow (or both) — adjust "Tech Stack" line.
- If any folder has a requirements.txt, link it here.
- Consider picking 1-2 of your strongest notebooks and adding a results screenshot/plot
  to the top of the README — recruiters respond well to a visual proof-of-work.
-->
