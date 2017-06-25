## Embedding kernel on irregular graph
This is implementation of embedding kernel described in paper 6.3. Note that this implementation is not optimized. You should modify the code by using sparse tensor to address large graph data set.


## Training
Download the datasets (irregular100.mat) from [here](https://drive.google.com/file/d/0B4eFbZCPIAvMSjRyZmNVR3dNbEU/view?usp=sharing), put it in the directionary ```../data/```

```
# Runs 100 nodes graph with default parameter using imitation learning
python train.py
```

