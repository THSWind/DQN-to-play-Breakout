# DQN-to-play-Breakout
A tensorflow implementation of Deep Q Network (DQN), Double Deep Q Network (DDQN), Dueling Deep Q Network (DuelingDQN)

<div align=center><img src="https://github.com/Checkmate986212/DQN-to-play-Breakout/blob/master/image_result/result.gif" /></div>

## Requirments

* python3
* tensorflow
* gym
* opencv-python

## Train
### DQN

`python main.py --train_dqn`
    
### DDQN

`python main.py --train_dqn --double_dqn=1`

### DuelingDQN

`python main.py --train_dqn --dueling_dqn=1`


## Test
### DQN

`python main.py --test_dqn`
    
### DDQN

`python main.py --test_dqn --double_dqn=1`

### DuelingDQN

`python main.py --test_dqn --dueling_dqn=1`

## Additional notes
You can change the model in `agent_dqn.py`

```python
    self.q_eval = self.build_net(self.s, 'eval_net') # online Q
    self.q_target = self.build_net(self.s_, 'target_net') # target Q

    self.q_eval = self.build_net_resnet(self.s, 'eval_net') # online Q
    self.q_target = self.build_net_resnet(self.s_, 'target_net') # target Q

    self.q_eval = self.build_net_alex(self.s, 'eval_net') # online Q
    self.q_target = self.build_net_alex(self.s_, 'target_net') # target Q
```

## Results

### Different model
![](https://github.com/Checkmate986212/DQN-to-play-Breakout/blob/master/image_result/models.png)

### Different method
![](https://github.com/Checkmate986212/DQN-to-play-Breakout/blob/master/image_result/methods.png)

### Different learning rate
![](https://github.com/Checkmate986212/DQN-to-play-Breakout/blob/master/image_result/learningrate.png)

### Test of DQN
![](https://github.com/Checkmate986212/DQN-to-play-Breakout/blob/master/image_result/test.png)
