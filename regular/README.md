## Directional kernel on regular graph
this code is an example of using directional kernel for regular graph (section 6.3). You could easily modify it for other kernels. Reinforcement learning takes a while for training.

## How to run

make sure the dataset is located under ```../data/```, make directory called model under this directory for saving trained model file.
```
# Runs 16x16 grid world with default parameter using imitation learning
python train.py

# Runs 16x16 grid world with default parameter using reinforcement learning
python vin_rl_train.py

```
