# Pytorch implementaion of CapsNet supporting multiMNIST
This is an implementation of CapsNet from Dynamic Routing Between Capsules by Sabour and Hinton.
This implementation can generate MultiMNIST data and the CapsNet implmentation is modified to support MultiMNIST.

## Requirements

- Python 3.x
- Pytorch 0.3.0 or above
- Numpy
- tqdm (to make display better, of course you can replace it with 'print')

## Run

To train and test CapsNet, run `python test_capsnet.py` in at the root of the repo.
To create visualization of routing data, run `python routing_visualization.py`

## Acknowledgements
This implementaion was modified from [Pytorch-CapsuleNet by jindongwang](https://github.com/jindongwang/Pytorch-CapsuleNet)
