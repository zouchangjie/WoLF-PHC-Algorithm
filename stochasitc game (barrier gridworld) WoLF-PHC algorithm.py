import numpy as np
import matplotlib.pyplot as plt
import datetime
from itertools import permutations
import random
import multiprocessing


def generateRandomFromDistribution (distribution):
    randomIndex = 0
    randomSum = distribution[randomIndex]
    randomFlag = np.random.random_sample()
    while randomFlag > randomSum:
        randomIndex += 1
        randomSum += distribution[randomIndex]
    return randomIndex

# world height
WORLD_HEIGHT = 3

# world width
WORLD_WIDTH = 3

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

gridIndexList = []
for i in range(0, WORLD_HEIGHT):
    for j in range(0, WORLD_WIDTH):
        gridIndexList.append(WORLD_WIDTH * i + j)

actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
statesAllOne = []
locationValidActions = {}

for i in permutations(gridIndexList, 2):
    statesAllOne.append(i)

# In gridworld with barrier, the agent's target in 7
statesAllOne.append((7, 7))

for i in  gridIndexList:
    locationValidActions[i] = []

for i in range(0, WORLD_HEIGHT):
    for j in range(0, WORLD_WIDTH):
        gridIndexNumber = WORLD_WIDTH * i + j
        if i != WORLD_HEIGHT - 1:
            locationValidActions[gridIndexNumber].append(ACTION_UP)
        if i != 0:
            locationValidActions[gridIndexNumber].append(ACTION_DOWN)
        if j != 0:
            locationValidActions[gridIndexNumber].append(ACTION_LEFT)
        if j != WORLD_WIDTH - 1:
            locationValidActions[gridIndexNumber].append(ACTION_RIGHT)

class agent:
    def __init__(self, agentIndex = 0, startLocationIndex = 0, gammma = 0.9, delta = 0.0001):
        self.timeStep = 0
        self.alpha = 1 / (10 + 0.002 * self.timeStep)
        self.gamma = gammma
        self.currentState = ()
        self.nextState = ()
        self.strategy = {}
        self.agentIndex = agentIndex
        self.startLocationIndex = startLocationIndex
        self.locationIndex = startLocationIndex
        self.currentAction = 0
        self.stateActionValues = {}
        self.maxStateAction = 0
        self.deltaStateAction = {}
        self.deltaStateActionTop = {}
        self.delta = delta
        self.EPSILON = 0.5 / (1 + 0.0001 * self.timeStep)
        self.stateCount = {}
        self.deltaWin = 0.0
        self.deltaLose = 0.0
        self.averageStrategy = {}
        for i in statesAllOne:
            self.averageStrategy[i] = {}
            lengthOfAction = len(locationValidActions[i[self.agentIndex]])
            for j in locationValidActions[i[self.agentIndex]]:
                self.averageStrategy[i][j] = 1 / lengthOfAction
        self.sumActionValue = {}
        self.sumAverageActionValue = {}
        # self.actionRewards = np.zeros((2))
        # self.currentActionIndex = 0
        # self.nextActionIndex = 0
        # self.currentAction = np.random.choice(self.actions)
        # self.currentReward = 0
        # self.maxAction = np.random.choice(self.actions)
        # self.deltaAction = np.zeros((self.lengthOfAction))
        # self.deltaActionTop = np.zeros((self.lengthOfAction))


    def initialSelfStrategy (self):
        for i in statesAllOne:
            self.strategy[i] = {}
            lengthOfAction = len(locationValidActions[i[self.agentIndex]])
            for j in locationValidActions[i[self.agentIndex]]:
                self.strategy[i][j] = 1 / lengthOfAction

    def initialActionValues (self):
        for i in statesAllOne:
            self.stateActionValues[i] = {}
            for j in locationValidActions[i[self.agentIndex]]:
                self.stateActionValues[i][j] = 0

    def initialStateCount(self):
        for i in statesAllOne:
            self.stateCount[i] = 0

    def initialDeltaStateAction (self):
        for i in statesAllOne:
            self.deltaStateAction[i] = {}
            for j in locationValidActions[i[self.agentIndex]]:
                self.deltaStateAction[i][j] = 0

    def initialDeltaStateActionTop (self):
        for i in statesAllOne:
            self.deltaStateActionTop[i] = {}
            for j in locationValidActions[i[self.agentIndex]]:
                self.deltaStateActionTop[i][j] = 0

    # when agent choose action, there are two options:
    # 1: chooseActionEpsilon = self.EPSILON, which means exploration-exploition method
    # 2: chooseActionEpsilon = 0, which means np.random.binomial(1, chooseActionEpsilon) == 0, will only choose the most probability action.
    def chooseAction (self, currentState, chooseActionEpsilon):
        if np.random.binomial(1, chooseActionEpsilon) == 1:
            self.currentAction = random.choice(locationValidActions[currentState[self.agentIndex]])
        else:
            actionProbabilityList = list(self.strategy[currentState].values())
            actionIndex = generateRandomFromDistribution(actionProbabilityList)
            actionProbability = actionProbabilityList[actionIndex]
            potentialactions = []
            for actionkeys in self.strategy[currentState].keys():
                if self.strategy[currentState][actionkeys] == actionProbability:
                    potentialactions.append(actionkeys)
            self.currentAction = random.choice(potentialactions)
        return self.currentAction

    def chooseActionWithFxiedStrategy (self, currentState):
        self.currentAction = locationValidActions[currentState[self.agentIndex]][generateRandomFromDistribution(self.strategy[currentState])]

    def updateActionValues (self,currentState, nextState, agentReward):
        self.stateActionValues[currentState][self.currentAction] = (1 - self.alpha) * self.stateActionValues[currentState][self.currentAction] \
                                                 + self.alpha * (agentReward + self.gamma * max(self.stateActionValues[nextState].values()))
    def updateStrategy (self, currentState):
        self.stateCount[currentState] += 1.0
        self.deltaWin = 1.0 / (1000 + self.timeStep / 10)  #notice the parameter settings
        self.deltaLose = 4.0 * self.deltaWin # reference to Multiagent Learning Using a Variable Learning Rate
        lengthOfAction = len(locationValidActions[currentState[self.agentIndex]])
        for j in locationValidActions[currentState[self.agentIndex]]:
            self.averageStrategy[currentState][j] += (1.0 / self.stateCount[currentState]) * (self.strategy[currentState][j] - self.averageStrategy[currentState][j])
        self.sumActionValue[currentState] = 0.0
        self.sumAverageActionValue[currentState] = 0.0
        for j in locationValidActions[currentState[self.agentIndex]]:
            self.sumActionValue[currentState] += self.strategy[currentState][j] * self.stateActionValues[currentState][j]
            self.sumAverageActionValue[currentState] += self.averageStrategy[currentState][j] * self.stateActionValues[currentState][j]
        if self.sumActionValue[currentState] > self.sumAverageActionValue[currentState]:
            self.delta = self.deltaWin
        else:
            self.delta = self.deltaLose

        maxAction = max(self.stateActionValues[currentState], key=lambda x:self.stateActionValues[currentState][x])
        for j in locationValidActions[currentState[self.agentIndex]]:
            self.deltaStateAction[currentState][j] = min([self.strategy[currentState][j], self.delta / (lengthOfAction - 1)])
        sumDeltaStateAction = 0
        for action_i in [action_j for action_j in locationValidActions[currentState[self.agentIndex]] if action_j != maxAction]:
            self.deltaStateActionTop[currentState][action_i] = -self.deltaStateAction[currentState][action_i]
            sumDeltaStateAction += self.deltaStateAction[currentState][action_i]
        self.deltaStateActionTop[currentState][maxAction] = sumDeltaStateAction
        for j in locationValidActions[currentState[self.agentIndex]]:
            self.strategy[currentState][j] += self.deltaStateActionTop[currentState][j]

        # if self.currentAction != self.maxAction:
        #     self.deltaActionTop[self.currentAction] = -self.deltaAction[self.currentAction]
        # else:
        #     self.sumDeltaAction = 0
        #     for action_i in [action_j for action_j in self.actions if action_j != self.currentAction]:
        #         self.sumDeltaAction += self.deltaAction[action_i]
        #     self.deltaActionTop[self.currentAction] = self.sumDeltaAction
        # self.strategy[self.currentAction] += self.deltaActionTop[self.currentAction]

    def updateTimeStep (self):
        self.timeStep += 1

    def updateEpsilon (self):
        self.EPSILON = 0.5 / (1 + 0.0001 * self.timeStep)

    def updateAlpha (self): #notice the parameter alpha
        self.alpha = 1 / (10 + 0.002 * self.timeStep)

def nextGridIndex (action, gridIndex):
    action = action
    index_i = int(gridIndex / 3)
    index_j = gridIndex - index_i * 3
    if (action == 0):
        index_i += 1
    elif (action == 1):
        index_i -= 1
    elif (action == 2):
        index_j -= 1
    elif (action == 3):
        index_j += 1
    nextIndex = index_i * 3 + index_j
    return nextIndex

def gridGameOne(action_0, action_1, currentState):
    action_0 = action_0
    action_1 = action_1
    currentState = currentState
    reward_0 = 0
    reward_1 = 0
    endGameFlag = 0

    currentIndex_0 = currentState[0]
    currentIndex_1 = currentState[1]
    nextIndex_0 = nextGridIndex(action_0, currentIndex_0)
    nextIndex_1 = nextGridIndex(action_1, currentIndex_1)

    if currentIndex_0 == 0 or currentIndex_0 == 2:
        if action_0 == 0:
            if random.uniform(0, 1) < 0.5:
                nextIndex_0 = currentIndex_0

    if currentIndex_1 == 0 or currentIndex_1 == 2:
        if action_1 == 0:
            if random.uniform(0, 1) < 0.5:
                nextIndex_1 = currentIndex_1

    if (nextIndex_0 == 7 and nextIndex_1 == 7):
        reward_0 = 100
        reward_1 = 100
        nextState = (nextIndex_0, nextIndex_1)
        endGameFlag = 1
    elif (nextIndex_0 == 7 and nextIndex_1 != 7):
        reward_0 = 100# maybe there is a difference between 100 and 50, after I adopt the parameter, the resutl will always converge to (8, 6)
                # and do not appear (8, 5) and (3, 6)
        reward_1 = -1
        nextState = (nextIndex_0, nextIndex_1)
        endGameFlag = 1
        # if (nextIndex_1 == 8):
        #     reward_1 = 0
        #     nextState = (nextIndex_0, currentIndex_1)
        # else:
        #     reward_1 = 0
        #     nextState = (nextIndex_0, nextIndex_1)
    elif (nextIndex_0 != 7 and nextIndex_1 == 7):
        reward_0 = -1
        reward_1 = 100
        nextState = (nextIndex_0, nextIndex_1)
        endGameFlag = 1
        # if (nextIndex_0 == 6):
        #     reward_0 = 0
        #     nextState = (currentIndex_0, nextIndex_1)
        # else:
        #     reward_0 = 0
        #     nextState = (nextIndex_0, nextIndex_1)
    elif (nextIndex_0 != 7 and nextIndex_1 != 7 and nextIndex_0 == nextIndex_1):
        reward_0 = -10
        reward_1 = -10
        nextState = (currentIndex_0, currentIndex_1)
        endGameFlag = 0
    else:
        reward_0 = -1
        reward_1 = -1
        nextState = (nextIndex_0, nextIndex_1)
        endGameFlag = 0
    return reward_0, reward_1, nextState, endGameFlag

def resetStartState():
    agent_0LocationIndex = np.random.choice([x for x in gridIndexList if x not in [7]])
    while True:
        agent_1LocationIndex = np.random.choice([x for x in gridIndexList if x not in [7]])
        if agent_1LocationIndex != agent_0LocationIndex:
            break
    return (agent_0LocationIndex, agent_1LocationIndex)

def playGameOne(agent_0 = agent, agent_1 = agent):
    agent_0 = agent_0
    agent_1 = agent_1
    currentState = (agent_0.startLocationIndex, agent_1.startLocationIndex)
    episodes = 0
    endGameFlag = 0
    agent_0.initialSelfStrategy()
    agent_1.initialSelfStrategy()
    agent_0.initialActionValues()
    agent_1.initialActionValues()
    agent_0.initialStateCount()
    agent_1.initialStateCount()
    agent_0.initialDeltaStateAction()
    agent_1.initialDeltaStateAction()
    agent_0.initialDeltaStateActionTop()
    agent_1.initialDeltaStateActionTop()
    while episodes < 1000000: #notice the episodes time, it need enough long time to train
        print (episodes)
        while True:
            agent0Action = agent_0.chooseAction(currentState, agent_0.EPSILON)
            agent1Action = agent_1.chooseAction(currentState, agent_1.EPSILON)
            reward_0, reward_1, nextState, endGameFlag = gridGameOne(agent0Action, agent1Action, currentState)
            agent_0.updateActionValues(currentState, nextState, reward_0)
            agent_1.updateActionValues(currentState, nextState, reward_1)
            agent_0.updateStrategy(currentState)
            agent_1.updateStrategy(currentState)
            agent_0.updateTimeStep()
            agent_1.updateTimeStep()
            agent_0.updateEpsilon()
            agent_1.updateEpsilon()
            agent_0.updateAlpha()
            agent_1.updateAlpha()
            if (endGameFlag == 1): # one episode of the game ends
                episodes += 1
                currentState = resetStartState()
                break
            currentState = nextState

def test (agent_0 = agent, agent_1 = agent):
    agent_0 = agent_0
    agent_1 = agent_1
    startState = (0, 2)
    endGameFlag = 0
    runs = 0
    agentActionList = []
    currentState = startState
    endGameFlag = 0
    while endGameFlag != 1:
        agent0Action = agent_0.chooseAction(currentState, 0)
        agent1Action = agent_1.chooseAction(currentState, 0)
        agentActionList.append([agent0Action, agent1Action])
        reward_0, reward_1, nextState, endGameFlag = gridGameOne(agent0Action, agent1Action, currentState)
        currentState = nextState
    agentActionList.append(currentState)
    return agentActionList

agent_0 = agent(agentIndex=0, startLocationIndex=0)
agent_1 = agent(agentIndex=1, startLocationIndex=2)

# starttime = datetime.datetime.now()
# playGameOne(agent_0, agent_1)
# runGameResult = test(agent_0, agent_1)
# endtime = datetime.datetime.now()
# intervaltime = (endtime - starttime).seconds
# print (runGameResult)
# print (intervaltime)

def rungame (agent_0 = agent, agent_1 = agent):
    agent_0 = agent_0
    agent_1 = agent_1
    playGameOne(agent_0, agent_1)
    runGameResult = test(agent_0, agent_1)
    return runGameResult

if __name__ == "__main__":
    # playGameOne(agent_0, agent_1)
    pool = multiprocessing.Pool(processes=20)
    agentActionList = []
    for i in range(20):
        agentActionList.append(pool.apply_async(rungame, (agent_0, agent_1)))
    pool.close()
    pool.join()

    # print (agent_0.qTable[0][3, 6])
    # print (agent_0.qTable[1][3, 6])
    # print (agent_1.qTable[0][8, 5])
    # print (agent_1.qTable[1][8, 5])
    for res in agentActionList:
        print (res.get())