# DQN-to-play-Breakout
A tensorflow implementation of Deep Q Network (DQN), Double Deep Q Network (DDQN), Dueling Deep Q Network (DuelingDQN)

<div align=center><img src="https://github.com/Checkmate986212/DQN-to-play-Breakout/blob/master/image_result/result.gif" /></div>

## Requirment

* python3
* tensorflow
* gym
* opencv-python

## Train
DQN
`python main.py --train_dqn`
    
DDQN
    python main.py --train_dqn --double_dqn=1
DuelingDQN
    python main.py --train_dqn --dueling_dqn=1

