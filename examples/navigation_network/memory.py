import random
import numpy as np

class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    Instead of using tuples (as other implementations do), the information is stored in lists 
    that get returned as another list of dictionaries with each key corresponding to either 
    "state", "action", "reward", "nextState" or "isFinal".
    """
    def __init__(self, size):
        self.size = size
        self.currentPosition = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.finals = []

    def getMiniBatch(self, size):
        # Efficiently sample indices without creating a large list
        indices = np.random.choice(len(self.states), size=min(size, len(self.states)), replace=False)
        
        # Generate the mini-batch from the sampled indices
        miniBatch = [{
            'state': self.states[index],
            'action': self.actions[index],
            'reward': self.rewards[index],
            'newState': self.newStates[index],
            'isFinal': self.finals[index]
        } for index in indices]
        
        return miniBatch



    def getCurrentSize(self) :
        return len(self.states)

    def getMemory(self, index): 
        return {'state': self.states[index],'action': self.actions[index], 'reward': self.rewards[index], 'newState': self.newStates[index], 'isFinal': self.finals[index]}

    def addMemory(self, state, action, reward, newState, isFinal):
        # Ensure state and newState are not None or malformed
        if state is None or newState is None:
            print("Error: Attempting to add None state or newState to memory!")
            return
        
        # Overwrite memory if capacity is reached
        if len(self.states) < self.size:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.finals.append(isFinal)
        else:
            self.states[self.currentPosition] = state
            self.actions[self.currentPosition] = action
            self.rewards[self.currentPosition] = reward
            self.newStates[self.currentPosition] = newState
            self.finals[self.currentPosition] = isFinal
        self.currentPosition = (self.currentPosition + 1) % self.size
        