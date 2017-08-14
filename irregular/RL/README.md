## Embedding kernel on irregular graph
This is implementation of embedding kernel described in paper 6.3. Note that this implementation is not optimized. You should modify the code by using sparse tensor to address large graph data set.


## Training
It is trained via episodic-Q learning, the data set is 10 nodes.

```
# Runs 10 nodes graph with default parameter using reinforcement learning
python rl_nstep.py 
```

