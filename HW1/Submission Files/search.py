# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def genericSearch(problem, frontier):
    """
    This is a generic search function which can be called with different
    types of data structures (as the frontier) from util.py file.
    The frontier doesn't store the list of states alone, rather it stores
    paths to relevant states (to be able to access all actions until a goal state)
    :param problem: the problem to solve
    :param frontier: the container to keep states(paths)
    :return: list of actions from starting state to the goal
    """
    startState = problem.getStartState()
    visitedStates = []
    frontier.push([(startState, "Stop", 0)])

    # loop until no states left in the frontier
    while not frontier.isEmpty():
        currentPath = frontier.pop()
        currentState = currentPath[-1][0]

        # If current state is already visited, go to next iteration
        if currentState in visitedStates:
            continue

        # If current state is the goal state, return a list of actions in the path
        if problem.isGoalState(currentState):
            currentActions = [x[1] for x in currentPath][1:]
            return currentActions

        # Mark the state as visited by placing it in the visited list
        visitedStates.append(currentState)

        # Find all possible successor states and put their path into the frontier
        successors = problem.getSuccessors(currentState)
        for successor in successors:
            successorState = successor[0]
            if successorState not in visitedStates:
                successorPath = currentPath[:]
                successorPath.append(successor)
                frontier.push(successorPath)
    return []

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    # Calls the generic search function with Stack data structure
    return genericSearch(problem, util.Stack())


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Calls the generic search function with Queue data structure
    return genericSearch(problem, util.Queue())

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # Calls the generic search function with Priority Queue
    # Priority function takes a path and returns the total cost of actions
    pri_func = lambda path: problem.getCostOfActions([x[1] for x in path][1:])
    return genericSearch(problem, util.PriorityQueueWithFunction(pri_func))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Calls the generic search function with Priority Queue
    # Priority function takes a path and computes the sum of costs and the heuristic of the last state
    pri_func = lambda path: problem.getCostOfActions([x[1] for x in path][1:]) + heuristic(path[-1][0], problem)
    return genericSearch(problem, util.PriorityQueueWithFunction(pri_func))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
