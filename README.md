# Delayed Mini-Batch Sampling Method for Accelerated Training

Official implementation of the paper **"Delayed Mini-Batch Sampling Method for Accelerated Training"** by **Tuan Nguyen** and **Stephen Becker** (University of Colorado Boulder).

---

## Overview

In standard Stochastic Gradient Descent (SGD), a new minibatch of samples is drawn at every single gradient step. This repository introduces a approach of re-using the same minibatch for several consecutive gradient steps.

The method is inspired by time-varying optimization. By utilizing "stale" information strategically, we can achieve faster convergence. Our research provides the mathematical analysis to validate this viewpoint and demonstrates that this method consistently outperforms tuned SGD and Adam in training loss.

---

## Repo Structure

The code is organized to mirror the sections of the paper:

| Directory | Description |
| :--- | :--- |
| `olnm.py` | The main optimizer in PyTorch. |
| `numerical/` | Contains scripts demonstrating the core conceptual mechanics of the algorithm. |
| `logistic-reg-mnist/` | Experiments using a simple model on the MNIST dataset comparing our solution against SGD and Adam. |
| `cnn-fmnist/` | Deep learning implementation using a CNN on Fashion-MNIST to validate performance in deep architectures. |

---

## Experiments

We evaluated the Delayed Mini-Batch Sampling method on multiple datasets:
1.  MNIST: Using both squared and cross-entropy loss.
2.  Fashion-MNIST: Validating the deep learning (CNN) approach.

In all cases, the proposed method showed superior performance compared to tuned baseline optimizers.

![OLNM Performance using CNN on Fashion MNIST](cnn-fmnist/imgs/svg/figure-5a-fmnist.svg)


---

## Authors
Tuan Nguyen & Prof. Steven Becker - [Applied Mathematics, CU Boulder](https://www.colorado.edu/amath/)