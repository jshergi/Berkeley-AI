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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        inf = float('inf')
        ghostPositions = successorGameState.getGhostPositions()
        
        for ghost in ghostPositions:
            # if ghost is in position or next to position 
            if util.manhattanDistance(newPos, ghost) < 2:
                return -inf
            # if food is in new position
            elif currentGameState.getFood()[newPos[0]][newPos[1]]:
                return inf
            
        minDistance = inf
        # check for food when not in danger
        for food in newFood.asList():
            distance = manhattanDistance(newPos, food)
            minDistance = min(minDistance, distance)
            
        # smaller distance better the reward
        return 1.0 / minDistance


def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        # Following pseudo code from wikipedia and slide set for minimax:

        # function minimax(node, depth, maximizingPlayer) is
        #     if depth = 0 or node is a terminal node then
        #         return the heuristic value of node
        #     if maximizingPlayer then
        #         value := −∞
        #         for each child of node do
        #             value := max(value, minimax(child, depth − 1, FALSE))
        #         return value
        #     else (* minimizing player *)
        #         value := +∞
        #         for each child of node do
        #             value := min(value, minimax(child, depth − 1, TRUE))
        #         return value

        # max plays first
        actions = gameState.getLegalActions(0)
        resultAction = max(actions, key=lambda action: self.minimax(gameState.generateSuccessor(0, action), 0, 1))
        return resultAction 

    def minimax(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
             return self.evaluationFunction(gameState)
        if agent == 0:
            v = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor = gameState.generateSuccessor(0, action) 
                v = max(v, self.minimax(successor, depth, 1))
            return v
        else:
            v = float('inf')
            actions = gameState.getLegalActions(agent)
            numGhosts = gameState.getNumAgents() -1 
            for action in actions:
                successor =  gameState.generateSuccessor(agent, action) 
                if agent < numGhosts:
                    v = min(v, self.minimax(successor, depth, agent + 1))
                else:
                    # increase depth here because a single search ply is considered to be one Pacman move and all the ghosts'
                    v = min(v, self.minimax(successor, depth + 1, 0))
            return v


       # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Following pseudo code from wikipedia and slide set for alpha beta prunning:
        # function alphabeta(node, depth, α, β, maximizingPlayer) is
        #     if depth == 0 or node is terminal then
        #         return the heuristic value of node
        #     if maximizingPlayer then
        #         value := −∞
        #         for each child of node do
        #             value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
        #             if value > β then
        #                 break (* β cutoff *)
        #             α := max(α, value)
        #         return value
        #     else
        #         value := +∞
        #         for each child of node do
        #             value := min(value, alphabeta(child, depth − 1, α, β, TRUE))
        #             if value < α then
        #                 break (* α cutoff *)
        #             β := min(β, value)
        #         return value

        # max will play first
        alpha = float('-inf')
        beta = float('inf')
        actions = gameState.getLegalActions(0)
        maxV = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0,action)
            v = self.alphabeta(successor, 0, 1, alpha, beta)
            if v > maxV:
                maxV, resultAction = v, action
                alpha = max(alpha,v)
        return resultAction 

    def alphabeta(self, gameState, depth, agent, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
             return self.evaluationFunction(gameState)
        if agent == 0:
            v = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor = gameState.generateSuccessor(0, action) 
                v = max(v, self.alphabeta(successor, depth, 1, alpha, beta))
                if v > beta: 
                    return v
                alpha = max(alpha,v)
            return v
        else:
            v = float('inf')
            actions = gameState.getLegalActions(agent)
            numGhosts = gameState.getNumAgents() -1 
            for action in actions:
                successor =  gameState.generateSuccessor(agent, action) 
                if agent < numGhosts:
                    v = min(v, self.alphabeta(successor, depth, agent + 1, alpha, beta))
                else:
                    # increase depth here because a single search ply is considered to be one Pacman move and all the ghosts'
                    v = min(v, self.alphabeta(successor, depth + 1, 0, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta,v)
            return v
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Following code from slide set. 
        # max will play first
        actions = gameState.getLegalActions(0)
        resultAction = max(actions, key=lambda action: self.expectimax(gameState.generateSuccessor(0, action), 0, 1))
        return resultAction 

    def expectimax(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
             return self.evaluationFunction(gameState)
        if agent == 0:
            v = float('-inf')
            actions = gameState.getLegalActions(0)
            for action in actions:
                successor = gameState.generateSuccessor(0, action) 
                v = max(v, self.expectimax(successor, depth, 1))
            return v
        else:
            v = 0
            actions = gameState.getLegalActions(agent)
            numGhosts = gameState.getNumAgents() -1 
            for action in actions:
                successor =  gameState.generateSuccessor(agent, action)
                p = 1 / len(actions)
                if agent < numGhosts:
                    v += p *(self.expectimax(successor, depth, agent + 1))
                else:
                    # increase depth here because a single search ply is considered to be one Pacman move and all the ghosts'
                    v += p *(self.expectimax(successor, depth + 1, 0))
            return v
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function guides Pac-Man's decisions by balancing safety and efficiency. 
                 It encourages Pac-Man to eat scared ghosts, prioritize nearby food pellets, and avoid non-scared ghosts. 
                 It also considers capsules but doesn't prioritize them heavily.

    The key components of the evaluation function include:
    - Rewarding Pac-Man for getting closer to scared ghosts.
    - Heavily penalizing Pac-Man for being near non-scared ghosts.
    - Rewarding Pac-Man for approaching food pellets.
    - Considering capsules but not prioritizing them heavily.
    
    """

    "*** YOUR CODE HERE ***"
    #Extract useful information from game state
    currFood = currentGameState.getFood().asList() 
    pacmanPos = currentGameState.getPacmanPosition() 
    ghostPos = currentGameState.getGhostPositions()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    from util import manhattanDistance

    # intialize evaluation to pacman's current score x 10
    evaluation = currentGameState.getScore()*10
    # get scared time for each ghost
    scaredTime = [ghost.scaredTimer for ghost in ghostStates]
    # Evaluate each componet of game state

    for ghost in range(len(ghostPos)):
        if scaredTime[ghost] > 0:
            distance = manhattanDistance(pacmanPos, ghostPos[ghost])
            # pacman is rewared for getting closer to scared ghosts
            evaluation = evaluation + (-1.0) + -0.5*distance
        elif manhattanDistance(ghostPos[ghost], pacmanPos) < 2:
            # pacman is heavily penalized for being next to a non-scared ghost
            evaluation += -10000
    for food in currFood:
        distance = manhattanDistance(pacmanPos, food)
        # pacman is rewarded for getting closer to food pellets (more than getting closer to scared ghost -> prioritize food)
        evaluation = evaluation + (-0.5) + (-0.5)*distance
    
    for capsule in capsules:
        distance = manhattanDistance(pacmanPos, capsule)
        # pacman is rewared / penalized for getting closer to capsules
        # heavier weight (-1.5) to not prioritize capsules
        evaluation = evaluation + (-1.5) + -0.5*distance

    # return final evaluation
    return evaluation 

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
