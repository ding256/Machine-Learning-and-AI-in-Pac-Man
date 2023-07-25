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
import game
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
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []

    from util import Stack
    frontier = Stack()
    frontier.push(problem.getStartState())
    goal = []
    visit = [problem.getStartState()]
    temp = problem.getSuccessors(frontier.pop())
    for a in temp:
        a += ((a[1],),)
        frontier.push(a)
    if_visit = False

    while not frontier.isEmpty():
        cur = frontier.pop()
        if problem.isGoalState(cur[0]):
            goal.append(cur[3])
            break
        for b in visit:
            if cur[0] == b:
                if_visit = True
                break
        if not if_visit:
            visit.append(cur[0])
            temp = problem.getSuccessors(cur[0])
        else:
            if_visit = False
            continue
        for a in temp:
            a += (cur[3] + (a[1],),)
            frontier.push(a)

    path = []
    for a in goal[0]:
        path.append(a)
    return path


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []

    from util import Queue
    frontier = Queue()
    frontier.push(problem.getStartState())
    goal = []
    visit = [problem.getStartState()]
    temp = problem.getSuccessors(frontier.pop())
    for a in temp:
        a += ((a[1],),)
        frontier.push(a)
    if_visit = False

    while not frontier.isEmpty():
        cur = frontier.pop()
        if problem.isGoalState((cur[0])):
            goal.append(cur[3])
            break
        for b in visit:
            if cur[0] == b[0]:
                if_visit = True
                break
            if cur[2] != 1:
                if cur[0] != b[0] or cur[1] != b[1]:
                    if_visit = False
                else:
                    if_visit = True
        if not if_visit:
            visit.append(cur)
            temp = problem.getSuccessors(cur[0])
        else:
            if_visit = False
            continue
        for a in temp:
            a += (cur[3] + (a[1],),)
            frontier.push(a)

    path = []
    for a in goal[0]:
        path.append(a)
    return path

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []

    from util import PriorityQueue
    frontier = PriorityQueue()
    frontier.push(problem.getStartState(), 0)
    goal = []
    visit = [problem.getStartState()]
    temp = problem.getSuccessors(frontier.pop())
    for a in temp:
        a += ((a[1],),)
        frontier.push(a, a[2])
    if_visit = False

    while not frontier.isEmpty():
        cur = frontier.pop()
        if problem.isGoalState((cur[0])):
            goal.append(cur[3])
            break
        for b in visit:
            if cur[0] == b:
                if_visit = True
                break
        if not if_visit:
            visit.append(cur[0])
            temp = problem.getSuccessors(cur[0])
        else:
            if_visit = False
            continue
        for a in temp:
            a += (cur[3] + (a[1],),)
            b = list(a)
            b[2] += cur[2]
            a = tuple(b)
            frontier.push(a, a[2])

    path = []
    for a in goal[0]:
        path.append(a)
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    if problem.isGoalState(problem.getStartState()):
        return []

    from util import PriorityQueue
    frontier = PriorityQueue()
    frontier.push(problem.getStartState(), 0)
    goal = []
    visit = [problem.getStartState()]
    temp = problem.getSuccessors(frontier.pop())
    for a in temp:
        a += ((a[1],),)
        frontier.push(a, a[2] + heuristic(a[0], problem))
    if_visit = False

    while not frontier.isEmpty():
        cur = frontier.pop()
        if problem.isGoalState((cur[0])):
            goal.append(cur[3])
            break
        for b in visit:
            if cur[0] == b:
                if_visit = True
                break
        if not if_visit:
            visit.append(cur[0])
            temp = problem.getSuccessors(cur[0])
        else:
            if_visit = False
            continue
        for a in temp:
            a += (cur[3] + (a[1],),)
            b = list(a)
            b[2] += cur[2]
            a = tuple(b)
            frontier.push(a, a[2] + heuristic(a[0], problem))

    path = []
    for a in goal[0]:
        path.append(a)
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
