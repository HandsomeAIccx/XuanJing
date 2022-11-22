import numpy as np
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch as T



import time
env = gym.make('CartPole-v0')

class NN(T.nn.Module):
    def __init__(self,inputDims=1,output=1):
        super().__init__()
        self.layers=T.nn.ModuleList()
        self.layers.append(T.nn.Linear(inputDims,256))
        self.layers.append(T.nn.Linear(256,output))
    def state2Value(self,X):
        o=T.tensor(X,dtype=T.float)
        for i in self.layers:
            o=i(o)
        return np.argmax(o.detach().numpy())
    def setWeights(self,weights):
      with T.no_grad():
        self.layers[0].weight=T.nn.parameter.Parameter(weights[:(self.layers[0].weight).view(-1).shape[0]].reshape(self.layers[0].weight.shape))
        self.layers[1].weight=T.nn.parameter.Parameter(weights[(self.layers[0].weight).view(-1).shape[0]:].reshape(self.layers[1].weight.shape))
    def getWeights(self):
      return T.cat((self.layers[0].weight.reshape(-1),self.layers[1].weight.reshape(-1)),0)

nn=NN(env.observation_space.shape[0],env.action_space.n)

def runEvolution(rewardFunction,populationSize,generationSize,sigma,lr,NN):
  parameters=NN.getWeights()
  rewards=[]
  for g in range(generationSize):
    noise=T.randn(populationSize,parameters.shape[0])
    springs=noise*sigma
    reward=T.zeros(populationSize)
    for i in range(populationSize):
      spring=parameters+springs[i]
      NN.setWeights(spring)
      reward[i]=rewardFunction(NN)
    rewards.append(reward.mean())
    advantage=(reward-reward.mean())/reward.std()
    # print(T.matmul(noise.T,advantage))
    parameters=parameters+lr/(populationSize*sigma)*T.matmul(noise.T,advantage)
    print(f"generation: {g} Average Reward: {reward.mean()} Best reward:{max(reward)}")
    lr *= 0.992354
  return parameters,rewards
def rewardFunction(NN):
  totalRewards=[]
  for i in range(10):
    done=False
    state=env.reset()
    rewards=0
    while not done:
      action=NN.state2Value(state)
      nextState,reward,done,info=env.step(action)
      rewards+=reward
      state=nextState
    totalRewards.append(rewards)
  return sum(totalRewards)/len(totalRewards)
parameters,rewards=runEvolution(rewardFunction,populationSize=50,generationSize=50,sigma=0.1,lr=0.01,NN=nn)
plt.title(f'Average Rewards')
plt.plot(rewards)
plt.savefig("Average Rewards",dpi=200)