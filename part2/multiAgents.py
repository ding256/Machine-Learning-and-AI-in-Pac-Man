# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if currentGameState.getPacmanPosition() == successorGameState.getPacmanPosition():
            return -999
        for a in currentGameState.getGhostPositions():
            if manhattanDistance(a, newPos) <= 2:
                return -999

        if oldFood[newPos[0]][newPos[1]]:
            return 100

        closest_food = [100, 100]
        row = 0
        col = 0
        for a in newFood:
            for b in a:
                if b and (manhattanDistance(newPos, (col, row)) < manhattanDistance(newPos, closest_food)):
                    closest_food = [col, row]
                row += 1
            col += 1
            row = 0

        return 50 - manhattanDistance(newPos, closest_food)

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    '''def recurse(self, gameState: GameState, agent, depth):
        if depth > self.depth:
            return (gameState, Directions.STOP)

        list_of_actions = gameState.getLegalActions(agent)
        if len(list_of_actions) == 0:
            return (gameState, Directions.STOP)

        new_states = [gameState.generateSuccessor(agent, action) for action in list_of_actions]
        list_of_states = []

        if agent < gameState.getNumAgents() - 1:
            for states in new_states:
                temp = self.recurse(states, agent + 1, depth)
                if temp:
                    list_of_states.append(temp[0])
        else:
            for states in new_states:
                temp = self.recurse(states, 0, depth + 1)
                if temp:
                    list_of_states.append(temp[0])

        max_index = 0
        min_index = 0
        if agent == 0:
            for a in range(len(list_of_states)):
                if self.evaluationFunction(list_of_states[a]) > self.evaluationFunction(list_of_states[max_index]):
                    max_index = a
            return (list_of_states[max_index], list_of_actions[max_index])
        else:
            for a in range(len(list_of_states)):
                if self.evaluationFunction(list_of_states[a]) < self.evaluationFunction(list_of_states[min_index]):
                    min_index = a
            return (list_of_states[min_index], list_of_actions[min_index])


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.recurse(gameState, 0, 1)[1]'''

    def recurse(self, gameState: GameState, agent, depth):
        if gameState.isWin():
            return gameState
        if gameState.isLose():
            return gameState
        if depth > self.depth:
            return gameState

        list_of_actions = gameState.getLegalActions(agent)
        if len(list_of_actions) == 0:
            return gameState

        maxi = None
        mini = None
        for action in list_of_actions:
            if agent < gameState.getNumAgents() - 1:
                temp = self.recurse(gameState.generateSuccessor(agent, action), agent + 1, depth)
            else:
                temp = self.recurse(gameState.generateSuccessor(agent, action), 0, depth + 1)
            if agent == 0:
                if maxi is None:
                    maxi = temp
                if self.evaluationFunction(temp) > self.evaluationFunction(maxi):
                    maxi = temp
            else:
                if mini is None:
                    mini = temp
                if self.evaluationFunction(temp) < self.evaluationFunction(mini):
                    mini = temp
        if agent == 0:
            return maxi
        else:
            return mini

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        list_of_actions = gameState.getLegalActions()
        list_of_states = [gameState.generateSuccessor(0, action) for action in list_of_actions]
        maxi = 0
        temp = []
        nextAgent = 1
        if gameState.getNumAgents() == 1:
            nextAgent = 0
        for i in range(len(list_of_states)):
            temp.append(self.recurse(list_of_states[i], nextAgent, 1))
            if self.evaluationFunction(temp[i]) > self.evaluationFunction(temp[maxi]):
                maxi = i
        return list_of_actions[maxi]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def recurse(self, gameState: GameState, agent, depth, alpha, beta):
        alphaa = alpha
        betaa = beta
        if gameState.isWin():
            return gameState
        if gameState.isLose():
            return gameState
        if depth > self.depth:
            return gameState

        list_of_actions = gameState.getLegalActions(agent)
        if len(list_of_actions) == 0:
            return gameState

        maxi = None
        mini = None
        for action in list_of_actions:
            if agent < gameState.getNumAgents() - 1:
                temp = self.recurse(gameState.generateSuccessor(agent, action), agent + 1, depth, alphaa, betaa)
            else:
                temp = self.recurse(gameState.generateSuccessor(agent, action), 0, depth + 1, alphaa, betaa)
            if agent == 0:
                if maxi is None:
                    maxi = temp
                if self.evaluationFunction(temp) > self.evaluationFunction(maxi):
                    maxi = temp
                if self.evaluationFunction(maxi) > betaa:
                    return maxi
                alphaa = max(alphaa, self.evaluationFunction(maxi))
            else:
                if mini is None:
                    mini = temp
                if self.evaluationFunction(temp) < self.evaluationFunction(mini):
                    mini = temp
                if self.evaluationFunction(mini) < alphaa:
                    return mini
                betaa = min(betaa, self.evaluationFunction(mini))
        if agent == 0:
            return maxi
        else:
            return mini

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        list_of_actions = gameState.getLegalActions()
        list_of_states = [gameState.generateSuccessor(0, action) for action in list_of_actions]
        maxi = 0
        alpha = -10000
        beta = 10000
        temp = []
        nextAgent = 1
        if gameState.getNumAgents() == 1:
            nextAgent = 0
        for i in range(len(list_of_states)):
            temp.append(self.recurse(list_of_states[i], nextAgent, 1, alpha, beta))
            if self.evaluationFunction(temp[i]) > self.evaluationFunction(temp[maxi]):
                maxi = i
            alpha = max(alpha, self.evaluationFunction(temp[maxi]))
        return list_of_actions[maxi]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def recurse(self, gameState: GameState, agent, depth):
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        if depth > self.depth:
            return self.evaluationFunction(gameState)

        list_of_actions = gameState.getLegalActions(agent)
        if len(list_of_actions) == 0:
            return self.evaluationFunction(gameState)

        available_choices = []
        for action in list_of_actions:
            if agent < gameState.getNumAgents() - 1:
                temp = self.recurse(gameState.generateSuccessor(agent, action), agent + 1, depth)
                available_choices.append(temp)
            else:
                temp = self.recurse(gameState.generateSuccessor(agent, action), 0, depth + 1)
                available_choices.append(temp)
        if agent == 0:
            return max(available_choices)
        else:
            final_score = 0
            for a in available_choices:
                final_score += a * (1 / len(available_choices))
            return final_score

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        list_of_actions = gameState.getLegalActions()
        list_of_states = [gameState.generateSuccessor(0, action) for action in list_of_actions]
        maxi = 0
        temp = []
        nextAgent = 1
        if gameState.getNumAgents() == 1:
            nextAgent = 0
        for i in range(len(list_of_states)):
            temp.append(self.recurse(list_of_states[i], nextAgent, 1))
            if temp[i] > temp[maxi]:
                maxi = i
        return list_of_actions[maxi]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    '''if currentGameState.isWin():
        return 10000000
    if currentGameState.isLose():
        return -99999'''
    actions = currentGameState.getLegalActions()
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]

    for i in range(1, currentGameState.getNumAgents()):
        if manhattanDistance(currentGameState.getGhostPosition(i), position) <= 2 and scared_times[i - 1] == 0:
            return -3
        elif currentGameState.getGhostPosition(i) == position:
            return -3

    if food[position[0] + 1][position[1]] or food[position[0] - 1][position[1]] or food[position[0]][position[1] + 1] or food[position[0]][position[1] - 1]:
        return currentGameState.getScore()

    closest_food = [100, 100]
    row = 0
    col = 0
    for a in food:
        for b in a:
            if b and (manhattanDistance(position, (col, row)) < manhattanDistance(position, closest_food)):
                closest_food = [col, row]
            row += 1
        col += 1
        row = 0

    return currentGameState.getScore() - manhattanDistance(position, closest_food)

# Abbreviation
better = betterEvaluationFunction

'''
if currentGameState.isWin():
    return 10000000
if currentGameState.isLose():
    return -99999
actions = currentGameState.getLegalActions()
position = currentGameState.getPacmanPosition()
food = currentGameState.getFood()
ghost_states = currentGameState.getGhostStates()
scared_times = [ghostState.scaredTimer for ghostState in ghost_states]

for i in range(1, currentGameState.getNumAgents()):
    if manhattanDistance(currentGameState.getGhostPosition(i), position) <= 1 and scared_times[i - 1] == 0:
        return -1
    elif currentGameState.getGhostPosition(i) == position:
        return -1

if food[position[0] + 1][position[1]] or food[position[0] - 1][position[1]] or food[position[0]][position[1] + 1] or food[position[0]][position[1] - 1]:
    return currentGameState.getScore()

closest_food = [100, 100]
row = 0
col = 0
for a in food:
    for b in a:
        if b and (manhattanDistance(position, (col, row)) < manhattanDistance(position, closest_food)):
            closest_food = [col, row]
        row += 1
    col += 1
    row = 0

return currentGameState.getScore() - manhattanDistance(position, closest_food)
'''