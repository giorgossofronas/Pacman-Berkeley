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

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    from util import Stack

    # keep track of visited states in a closed set
    closed = set()

    # frontier consists of tuples of (state, path_to_this_state)
    frontier = Stack()
    frontier.push((problem.getStartState(), []))

    while not frontier.isEmpty():
        state, path = frontier.pop()

        if state in closed:
            continue

        # check if we reached goal state
        if (problem.isGoalState(state)):
            return path

        closed.add(state)

        # get successor-states and insert them, if not in closed set
        successors = problem.getSuccessors(state)
        for succ_state, succ_action, succ_cost in successors:
            if succ_state in closed:
                continue
            # new path is the parent-node's one + current successor's action
            new_path = list(path)
            new_path.append(succ_action)
            frontier.push((succ_state, new_path))
    return []
    
def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    
    from util import Queue

    # keep track of visited states in a closed set
    closed = set()

    # frontier consists of tuples of (state, path_to_this_state)
    frontier = Queue()
    frontier.push((problem.getStartState(), []))

    while not frontier.isEmpty():
        state, path = frontier.pop()

        if state in closed:
            continue

        # check if we reached goal state
        if (problem.isGoalState(state)):
            return path

        closed.add(state)

        # get successor-states and insert them, if not in closed set
        successors = problem.getSuccessors(state)
        for succ_state, succ_action, succ_cost in successors:
            if succ_state in closed:
                continue
            # new path is the parent-node's one + current successor's action
            new_path = list(path)
            new_path.append(succ_action)
            frontier.push((succ_state, new_path))
    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    
    from util import PriorityQueue

    # keep track of visited states in a closed set
    closed = set()

    # frontier consists of tuples of (state, path_to_this_state) and state's priority
    frontier = PriorityQueue()
    frontier.push((problem.getStartState(), []), 0)

    while not frontier.isEmpty():
        state, path = frontier.pop()

        if state in closed:
            continue

        # check if we reached goal state
        if (problem.isGoalState(state)):
            return path

        closed.add(state)

        # get successor-states and insert them, if not in closed set
        successors = problem.getSuccessors(state)
        for succ_state, succ_action, succ_cost in successors:
            if succ_state in closed:
                continue
            # new path is the parent-node's one + current successor's action
            new_path = list(path)
            new_path.append(succ_action)
            frontier.update((succ_state, new_path), problem.getCostOfActions(new_path)) # use of update method in case a shorter path to this node is found
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    from util import PriorityQueue

    # keep track of visited states in a closed set
    closed = set()

    # frontier consists of tuples of (state, path_to_this_state) and state's priority
    frontier = PriorityQueue()
    frontier.push((problem.getStartState(), []), 0)

    while not frontier.isEmpty():
        state, path = frontier.pop()

        if state in closed:
            continue

        # check if we reached goal state
        if (problem.isGoalState(state)):
            return path

        closed.add(state)

        # get successor-states and insert them, if not in closed set
        successors = problem.getSuccessors(state)
        for succ_state, succ_action, succ_cost in successors:
            if succ_state in closed:
                continue
            # new path is the parent-node's one + current successor's action
            new_path = list(path)
            new_path.append(succ_action)
            frontier.update((succ_state, new_path), problem.getCostOfActions(new_path) + heuristic(succ_state, problem)) # use of update method in case a shorter path to this node is found
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
