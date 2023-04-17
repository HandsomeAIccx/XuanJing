from XuanJing.env.sample.patch import Patch
import collections
import random
import copy


class ReplayBuffer(object):
    def __init__(self, capacity=None):
        assert capacity != None, "Parameter capacity Must Given!"
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, data):
        for idx in range(len(data)):
            self.buffer.append(data[idx])

    def random_pop(self, batch_size):
        transition_patch = Patch()
        transitions = random.sample(self.buffer, batch_size)
        for transition in transitions:
            transition_patch.add(copy.deepcopy(transition))
        return transition_patch

    def __len__(self):
        return len(self.buffer)
