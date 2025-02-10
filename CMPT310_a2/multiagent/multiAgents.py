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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Calculate the distance of the closest food (similar to the food heuristic from assignment 1)

        foodDistances = []
        foodList = newFood.asList()

        # If the next move leads to eating the last piece of food, always take it
        if len(foodList) == 0:
            return float('inf') # Ensures no other move can have a higher score
        
        for foodPos in foodList:
            foodDistances.append(util.manhattanDistance(newPos, foodPos))

        closestFoodDistance = min(foodDistances)

        # Calculate the distance of the closest ghost

        ghostDistances = []
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDistances.append(util.manhattanDistance(newPos, ghostPos))

        closestGhostDistance = min(ghostDistances)

        # Added a term that prefers staying away from ghosts and staying close to food
        return successorGameState.getScore() + closestGhostDistance / (2.0 * closestFoodDistance + 1.0)

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

        # Start with the maximizing agent (pacman)
        return self.minimax(gameState)[0]

    # Minimax algorithm which returns the optimal action and its associated value
    def minimax(self, gameState: GameState, agentIndex: int=0, depth: int=0):
        # Stop condition (Max depth reached, win/lose game, or terminal state)
        if (depth == self.depth or gameState.isWin() or gameState.isLose() or
            len(gameState.getLegalActions(agentIndex)) == 0):
            return (Directions.STOP, self.evaluationFunction(gameState))

        # Maximizing agent (pacman)
        if agentIndex == 0:
            maxAction = Directions.STOP
            maxValue = -float('inf')

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                
                # Loop through the agents in order of increasing index
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth

                value = self.minimax(successor, nextAgentIndex, nextDepth)[1]

                if value > maxValue:
                    maxAction = action
                    maxValue = value

            return (maxAction, maxValue)

        # Minimizing agent(s) (ghost)
        else:
            minAction = Directions.STOP
            minValue = float('inf')

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                
                # Loop through the agents in order of increasing index
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                # If the index has looped back around to 0, each agent has moved once in turn,
                # so this is one search ply
                nextDepth = depth + (1 if nextAgentIndex == 0 else 0)

                value = self.minimax(successor, nextAgentIndex, nextDepth)[1]

                if value < minValue:
                    minAction = action
                    minValue = value

            return (minAction, minValue)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        return self.minimaxAlphaBeta(gameState)[0]

    # The minimax algorithm above, but with alpha-beta pruning
    def minimaxAlphaBeta(self, gameState: GameState, agentIndex: int=0, depth: int=0, alpha: int=-float('inf'), beta: int=float('inf')):
        # Stop condition (Max depth reached, win/lose game, or terminal state)
        if (depth == self.depth or gameState.isWin() or gameState.isLose() or
            len(gameState.getLegalActions(agentIndex)) == 0):
            return (Directions.STOP, self.evaluationFunction(gameState))

        # Maximizing agent (pacman)
        if agentIndex == 0:
            maxAction = Directions.STOP
            maxValue = -float('inf')

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                
                # Loop through the agents in order of increasing index
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth

                value = self.minimaxAlphaBeta(successor, nextAgentIndex, nextDepth, alpha, beta)[1]

                if value > maxValue:
                    maxAction = action
                    maxValue = value

                if value > beta:
                    return (maxAction, maxValue)
                
                alpha = max(alpha, maxValue)

            return (maxAction, maxValue)

        # Minimizing agent(s) (ghost)
        else:
            minAction = Directions.STOP
            minValue = float('inf')

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                
                # Loop through the agents in order of increasing index
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                # If the index has looped back around to 0, each agent has moved once in turn,
                # so this is one search ply
                nextDepth = depth + (1 if nextAgentIndex == 0 else 0)

                value = self.minimaxAlphaBeta(successor, nextAgentIndex, nextDepth, alpha, beta)[1]

                if value < minValue:
                    minAction = action
                    minValue = value

                if value < alpha:
                    return (minAction, minValue)
                
                beta = min(beta, minValue)

            return (minAction, minValue)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        return self.expectimax(gameState)[0]

    def expectimax(self, gameState: GameState, agentIndex: int=0, depth: int=0):
        # Stop condition (Max depth reached, win/lose game, or terminal state)
        if (depth == self.depth or gameState.isWin() or gameState.isLose() or
            len(gameState.getLegalActions(agentIndex)) == 0):
            return (Directions.STOP, self.evaluationFunction(gameState))

        # Maximizing agent (pacman)
        if agentIndex == 0:
            maxAction = Directions.STOP
            maxValue = -float('inf')

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                
                # Loop through the agents in order of increasing index
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth

                value = self.expectimax(successor, nextAgentIndex, nextDepth)[1]

                if value > maxValue:
                    maxAction = action
                    maxValue = value

            return (maxAction, maxValue)

        # Sub-optimal agent(s) (ghost)
        else:
            expAction = Directions.STOP
            expValue = 0

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                
                # Loop through the agents in order of increasing index
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
                # If the index has looped back around to 0, each agent has moved once in turn,
                # so this is one search ply
                nextDepth = depth + (1 if nextAgentIndex == 0 else 0)

                # Each successor state is equally weighted
                probability = 1.0 / len(gameState.getLegalActions(agentIndex))
                expValue += probability * self.expectimax(successor, nextAgentIndex, nextDepth)[1]

                expAction = action

            return (expAction, expValue)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
