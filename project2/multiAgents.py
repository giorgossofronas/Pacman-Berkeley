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

        # check for win or loss
        if successorGameState.isWin():
            return float("inf")
        elif successorGameState.isLose():
            return float("-inf")

        # score is initialized as the game score
        score = successorGameState.getScore()

        # penalty for stopping is subtracting game-score
        if action == Directions.STOP:
            score -= abs(score)

        # add manhattan distances of ghosts to score
        for ghost_pos in successorGameState.getGhostPositions():
            distance = manhattanDistance(newPos, ghost_pos)
            if (distance == 1): # if ghost is super close, avoid it
                return float("-inf")
            score += manhattanDistance(newPos, ghost_pos)

        # successor game state is better if there are scared times, thus add them to score
        if len(successorGameState.getCapsules()) > 0:
            for ghost_scared_val in newScaredTimes:
                score += ghost_scared_val

        # subtract from score the smallest food distance
        min_food_dist = float("inf")
        for food in newFood.asList():
            if successorGameState.hasFood(food[0], food[1]):
                md = manhattanDistance(newPos, food)
                if (md < min_food_dist):
                    min_food_dist = md
        score -= min_food_dist

        return score

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
        best_action = None
        best_eval = float("-inf")
        # for every possible Pac-Man (MAX player) action, find and return the best/highest-valued one
        for action in gameState.getLegalActions(0):
            # start depth is 0 and MIN plays next, so as agent we input number one to examine all ghost agents (0 = Pac-Man & 1-... = ghosts)
            succ_eval = self.minimax(gameState.generateSuccessor(0, action), 0, 1)
            if succ_eval > best_eval:
                best_eval = succ_eval
                best_action = action
        return best_action

    # simple minimax implementation returning the value of the successor that MAX or MIN should choose
    def minimax(self, gameState: GameState, depth: int, agent: int) -> float:
        # check if game state is terminal
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # pacman agent, our MAX player
        if agent == 0:
            max_val = float("-inf")
            for action in gameState.getLegalActions(0):
                succ = gameState.generateSuccessor(0, action)
                max_val = max(max_val, self.minimax(succ, depth, 1))
            return max_val
        # last ghost agent
        elif agent == (gameState.getNumAgents() - 1):
            min_val = float("inf")
            for action in gameState.getLegalActions(agent):
                succ = gameState.generateSuccessor(agent, action)
                # after last ghost's move, the depth is increased and it is Pac-Man's (MAX) turn
                min_val = min(min_val, self.minimax(succ, depth + 1, 0))
            return min_val
        # any other ghost agent != last
        else:
            min_val = float("inf")
            for action in gameState.getLegalActions(agent):
                succ = gameState.generateSuccessor(agent, action)
                # a single search ply is considered to be one Pac-Man move and all the ghosts’ responses
                # therefore, in order to examine all ghosts, call minimax() in the next ghost agent in the same depth
                min_val = min(min_val, self.minimax(succ, depth, agent + 1))
            return min_val

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        best_action = None
        best_eval = float("-inf")

        # alpha and beta are initialized as (-00, +00)
        alpha = float("-inf")
        beta = float("inf")

        # for every possible Pac-Man (MAX player) action, find and return the best/highest-valued one
        for action in gameState.getLegalActions(0):
            # start depth is 0 and MIN plays next, so as agent we input number one to examine all ghost agents (0 = Pac-Man & 1-... = ghosts)
            succ_eval = self.alphabeta(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if succ_eval > best_eval:
                best_eval = succ_eval
                best_action = action
            alpha = max(alpha, succ_eval) # update alpha
        return best_action

    # simple minimax with alpha-beta pruning implementation returning the value of the successor that MAX or MIN should choose
    def alphabeta(self, gameState: GameState, depth: int, agent: int, alpha: float, beta: float) -> float:
        # check if game state is terminal
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # pacman agent, our MAX player
        if agent == 0:
            max_val = float("-inf")
            for action in gameState.getLegalActions(0):
                succ = gameState.generateSuccessor(0, action)
                max_val = max(max_val, self.alphabeta(succ, depth, 1, alpha, beta))
                if max_val > beta:
                    return max_val
                alpha = max(alpha, max_val)
            return max_val
        # last ghost agent
        elif agent == (gameState.getNumAgents() - 1):
            min_val = float("inf")
            for action in gameState.getLegalActions(agent):
                succ = gameState.generateSuccessor(agent, action)
                # after last ghost's move, the depth is increased and it is Pac-Man's (MAX) turn
                min_val = min(min_val, self.alphabeta(succ, depth + 1, 0, alpha, beta))
                if min_val < alpha:
                    return min_val
                beta = min(beta, min_val)
            return min_val
        # any other ghost agent != last
        else:
            min_val = float("inf")
            for action in gameState.getLegalActions(agent):
                succ = gameState.generateSuccessor(agent, action)
                # a single search ply is considered to be one Pac-Man move and all the ghosts’ responses
                # therefore, in order to examine all ghosts, call alphabeta() in the next ghost agent in the same depth
                min_val = min(min_val, self.alphabeta(succ, depth, agent + 1, alpha, beta))
                if min_val < alpha:
                    return min_val
                beta = min(beta, min_val)
            return min_val

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
        best_action = None
        best_eval = float("-inf")
        # for every possible Pac-Man (MAX player) action, find and return the best/highest-valued one
        for action in gameState.getLegalActions(0):
            # start depth is 0 and MIN plays next, so as agent we input number one to examine all ghost agents (0 = Pac-Man & 1-... = ghosts)
            succ_eval = self.expectiminimax(gameState.generateSuccessor(0, action), 0, 1)
            if succ_eval > best_eval:
                best_eval = succ_eval
                best_action = action
        return best_action

    def expectiminimax(self, gameState: GameState, depth: int, agent: int) -> float:
        # check if game state is terminal
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # pacman agent, our MAX player
        if agent == 0:
            max_val = float("-inf")
            for action in gameState.getLegalActions(0):
                succ = gameState.generateSuccessor(0, action)
                max_val = max(max_val, self.expectiminimax(succ, depth, 1))
            return max_val
        # for the next two cases, regarding ghost agent(s), instead of assuming that MIN plays optimally
        # we return the average of all possible legal actions as the expected-minimax value
        elif agent == (gameState.getNumAgents() - 1):
            val_sum = 0.0
            for action in gameState.getLegalActions(agent):
                succ = gameState.generateSuccessor(agent, action)
                # after last ghost's move, the depth is increased and it is Pac-Man's (MAX) turn
                val_sum += self.expectiminimax(succ, depth + 1, 0)
            return val_sum / len(gameState.getLegalActions(agent))
        else:
            val_sum = 0.0
            for action in gameState.getLegalActions(agent):
                succ = gameState.generateSuccessor(agent, action)
                # a single search ply is considered to be one Pac-Man move and all the ghosts’ responses
                # therefore, in order to examine all ghosts, call expectiminimax() in the next ghost agent in the same depth
                val_sum += self.expectiminimax(succ, depth, agent + 1)
            return val_sum / len(gameState.getLegalActions(agent))

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # check for win or loss
    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return float("-inf")

    pos = currentGameState.getPacmanPosition()
    # score is initialized as the game score
    score = currentGameState.getScore()

    # add manhattan distances of ghosts
    for ghost_pos in currentGameState.getGhostPositions():
        distance = manhattanDistance(pos, ghost_pos)
        if (distance == 1):
            return float("-inf")
        score += manhattanDistance(pos, ghost_pos)

    # subtract distance of closest food
    foodList = currentGameState.getFood().asList()
    min_food_dist = float("inf")
    for food in foodList:
        if currentGameState.hasFood(food[0], food[1]):
            md = manhattanDistance(pos, food)
            if (md < min_food_dist):
                min_food_dist = md
    score -= min_food_dist

    # subtract the average capsules' manhattan distance as penalty for not taking them
    num_of_caps = len(currentGameState.getCapsules())
    if num_of_caps > 0:
        sum_caps = 0
        for caps_pos in currentGameState.getCapsules():
            sum_caps += manhattanDistance(pos, caps_pos)
        score -= sum_caps / num_of_caps

    # add to score the rounded half of each scared time value
    scaredTimes = [ghost.scaredTimer for ghost in currentGameState.getGhostStates()]
    for i in scaredTimes:
        score += round(i / 2)

    return score

# Abbreviation
better = betterEvaluationFunction
