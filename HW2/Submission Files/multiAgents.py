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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        # Return - infinity to ban "Stop" action
        if action == "Stop":
            return -1 * float("inf")
        # Get the list of remaining capsule positions
        newCapsules = successorGameState.getCapsules()
        # Initialize the score with current state's game score value
        score = successorGameState.getScore()

        if len(newCapsules) > 0:
            # Penalize 100 points for each remaining capsule
            score -= len(newCapsules) * 100
            # Find the distance to the closest capsule
            capsuleDist = [manhattanDistance(newPos, capsule) for capsule in newCapsules]
            minCapsuleDist = min(capsuleDist)

        # Iterate over the ghost states
        for i in range(len(newGhostStates)):
            # Get the scared time of the ith ghost
            scaredTime = newScaredTimes[i]
            # Calculate the manhattan distance between pacman and the ith ghost
            ghostDist = manhattanDistance(newPos, newGhostStates[i].getPosition())
            # To avoid division by 0, set ghost distance to 0.1 if it is 0
            if ghostDist == 0:
                ghostDist = 0.1
            # Check if the ghost is scared and scared time is larger than the distance between ghost&pacman
            if scaredTime > 0 and (scaredTime - ghostDist) > 0:
                # Increase the score proportionally with the distance
                score += 200 - ghostDist * 20
            else:
                # If ghost is not scared, penalize the closeness of pacman to the ghost
                score -= 100 / ghostDist
                if len(newCapsules) > 0:
                    # Reward the closeness of pacman to the closest capsule
                    score += 50 / (minCapsuleDist + ghostDist)

        # Get the list of remaining food
        foodList = newFood.asList()
        if len(foodList) > 0:
            # Penalize 100 points for each remaining food
            score -= len(foodList) * 100
            # Calculate the manhattan distance between pacman and the closest food
            foodDist = [manhattanDistance(newPos, foodPos) for foodPos in foodList]
            minFoodDist = min(foodDist)
            # Penalize 10 points for each 1 unit of distance to the closest food
            score -= 10 * minFoodDist

        return score


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
        """
        "*** YOUR CODE HERE ***"
        # Get possible actions
        actions = gameState.getLegalActions(0)
        values = []
        currentDepth = 0
        # For each action, call the value() function passing the successor state to it
        for action in actions:
            val = self.value(gameState.generateSuccessor(0, action), currentDepth, 1)
            values.append(val)

        # Return the action that results in the max value
        return actions[values.index(max(values))]

    def value(self, gameState, currentDepth, agentIndex):
        # If game is over or depth limit is reached, call the evaluation function
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        # If the current agent is pacman, call maxValue()
        elif agentIndex == 0:
            return self.maxValue(gameState, currentDepth)
        # If the current agent is a ghost, call minValue()
        else:
            return self.minValue(gameState, currentDepth, agentIndex)

    def maxValue(self, gameState, currentDepth):
        # Initialize maxVal to - infinity
        maxVal = -1 * float("inf")
        # For each possible action, call the value() function passing the successor state to it
        for action in gameState.getLegalActions(0):
            val = self.value(gameState.generateSuccessor(0, action), currentDepth, 1)
            # Find the maximum value
            if val > maxVal:
                maxVal = val
        # Return the maximum value
        return maxVal

    def minValue(self, gameState, currentDepth, agentIndex):
        # Initialize minVal to infinity
        minVal = float("inf")
        # If the last agent is not reached yet, increase agentIndex by 1
        if agentIndex < gameState.getNumAgents() - 1:
            nextAgentIndex = agentIndex + 1
        # If the last agent is reached, set the nextAgentIndex to 0 (pacman),
        # and increase the depth by 1
        else:
            currentDepth += 1
            nextAgentIndex = 0
        # For each possible action, call the value() function passing the successor state to it
        for action in gameState.getLegalActions(agentIndex):
            val = self.value(gameState.generateSuccessor(agentIndex, action), currentDepth, nextAgentIndex)
            # Find the minimum value
            if val < minVal:
                minVal = val
        # Return the minimum value
        return minVal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Initialize the needed variables
        actions = gameState.getLegalActions(0)
        maxVal = -1 * float("inf")
        alpha = -1 * float("inf")
        beta = float("inf")
        actionToTake = "Stop"
        currentDepth = 0
        # For each action, call the value() function passing the successor state to it
        for action in actions:
            val = self.value(gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta)
            if val > maxVal:
                actionToTake = action
                maxVal = val
                if val > beta:
                    return action
                # Update alpha
                alpha = max(alpha, maxVal)

        # Return the action that results in the max value
        return actionToTake

    def value(self, gameState, currentDepth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState, currentDepth, alpha, beta)
        else:
            return self.minValue(gameState, currentDepth, agentIndex, alpha, beta)

    def maxValue(self, gameState, currentDepth, alpha, beta):
        maxVal = -1 * float("inf")
        for action in gameState.getLegalActions(0):
            val = self.value(gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta)
            if val > maxVal:
                maxVal = val
                # If value is greater than beta, prune by returning the value
                if val > beta:
                    return val
                # Update alpha
                alpha = max(alpha, maxVal)
        return maxVal

    def minValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        minVal = float("inf")

        if agentIndex < gameState.getNumAgents() - 1:
            nextAgentIndex = agentIndex + 1
        else:
            currentDepth += 1
            nextAgentIndex = 0

        for action in gameState.getLegalActions(agentIndex):
            val = self.value(gameState.generateSuccessor(agentIndex, action), currentDepth, nextAgentIndex, alpha, beta)
            if val < minVal:
                minVal = val
                # If value is less than alpha, prune by returning the value
                if val < alpha:
                    return val
                # Update beta
                beta = min(beta, minVal)
        return minVal

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
        actions = gameState.getLegalActions(0)
        values = []
        currentDepth = 0
        for action in actions:
            val = self.value(gameState.generateSuccessor(0, action), currentDepth, 1)
            values.append(val)

        # Return the action that results in the max value
        return actions[values.index(max(values))]

    def value(self, gameState, currentDepth, agentIndex):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState, currentDepth)
        else:
            return self.expValue(gameState, currentDepth, agentIndex)

    def maxValue(self, gameState, currentDepth):
        maxVal = -1 * float("inf")
        for action in gameState.getLegalActions(0):
            val = self.value(gameState.generateSuccessor(0, action), currentDepth, 1)
            if val > maxVal:
                maxVal = val
        return maxVal

    def expValue(self, gameState, currentDepth, agentIndex):
        if agentIndex < gameState.getNumAgents() - 1:
            nextAgentIndex = agentIndex + 1
        else:
            currentDepth += 1
            nextAgentIndex = 0

        # Put the values of each possible successor state into a list
        values = []
        for action in gameState.getLegalActions(agentIndex):
            val = self.value(gameState.generateSuccessor(agentIndex, action), currentDepth, nextAgentIndex)
            values.append(val)
        # Return the average of the values (expected value)
        return (sum(values) / len(values)) if len(values) != 0 else float("inf")

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: The score is initialized to the score of the game in the
                current state.

                The number of remaining capsules is found.
                There is a 149 point penalty for each remaining capsule.

                For each ghost state, it is checked if the ghost is scared.
                Also, the manhattan distance between pacman and ghost is computed.
                If ghost is scared, the closeness to the ghost is rewarded, else,
                it is penalized.

                There is a 85 point penalty for each remaining food. Also, the
                closeness to the nearest food is rewarded.
    """
    "*** YOUR CODE HERE ***"
    # Initialize the score with current state's game score value
    score = currentGameState.getScore()
    # Get the properties to be used as features
    capsules = currentGameState.getCapsules()
    foods = currentGameState.getFood().asList()
    pos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    # Check if any capsule left
    if len(capsules) > 0:
        # Penalize 149 points for each remaining capsule
        score -= 149 * len(capsules)
        # Calculate the manhattan distance to the closest capsule
        capsuleDist = [manhattanDistance(pos, capsule) for capsule in capsules]
        # Increase the score proportionally with the closeness to the closest capsule
        score += max(0, 59 - min(capsuleDist)*7)
        # Get the closest capsule to use later
        closestCapsule = capsules[capsuleDist.index(min(capsuleDist))]

    # Iterate over the ghost states
    for ghostState in ghostStates:
        # Get the scared time of the current ghost
        scaredTime = ghostState.scaredTimer
        # Calculate the manhattan distance between pacman and the current ghost
        ghostDist = manhattanDistance(pos, ghostState.getPosition())
        # To avoid division by 0, set ghost distance to 0.1 if it is 0
        if ghostDist == 0:
            ghostDist = 0.1
        # Check if the ghost is scared and scared time is larger than the distance between ghost&pacman
        if scaredTime > 0 and (scaredTime - ghostDist) > 0:
            # Increase the score proportionally with the distance to the ghost
            score += max(33, 199 - (ghostDist * 9))
        else:
            # If ghost is not scared, penalize the distance of ghost to the capsule
            if len(capsules) > 0:
                score -= max(29, 5 * manhattanDistance(ghostState.getPosition(), closestCapsule))
            # If ghost is not scared, penalize the closeness of pacman to the ghost
            score -= 159 / ghostDist

    # Get the list of remaining food
    if len(foods) > 0:
        # Penalize 85 points for each remaining food
        score -= len(foods) * 85
        # Calculate the manhattan distance between pacman and the closest food
        foodDist = [manhattanDistance(pos, foodPos) for foodPos in foods]
        minFoodDist = min(foodDist)
        # Increase the score proportionally with the closeness to the closest food
        score += max(0, 80 - 9 * minFoodDist)

    return score

# Abbreviation
better = betterEvaluationFunction

