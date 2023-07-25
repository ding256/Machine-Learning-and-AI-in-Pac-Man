# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
            Run the value iteration
        """
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for i in range(0, self.iterations):
            new_values = dict(self.values)
            for state in states:
                v_star = 0
                actions = self.mdp.getPossibleActions(state)
                q_values = []
                for action in actions:
                    q_values.append(self.computeQValueFromValues(state, action))
                if len(q_values) != 0:
                    v_star = max(q_values)
                    new_values[state] = v_star
                else:
                    new_values[state] = self.values[state]
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transition_states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        for i in range(0, len(transition_states_and_probs)):
            prob = transition_states_and_probs[i][1]
            reward = self.mdp.getReward(state, action, transition_states_and_probs[i][0])
            v_star = self.getValue(transition_states_and_probs[i][0])
            q_value += prob * (reward + self.discount * v_star)
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        q_values = [self.computeQValueFromValues(state, action) for action in actions]
        best_q_value = max(q_values)
        best_index = [index for index in range(0, len(q_values)) if q_values[index] == best_q_value]
        return actions[best_index[0]]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        counter = 0
        for i in range(0, self.iterations):
            new_values = self.values.copy()
            if counter == len(states):
                counter = 0
            state = states[counter]
            if self.mdp.isTerminal(state):
                counter += 1
                continue
            v_star = 0
            actions = self.mdp.getPossibleActions(state)
            q_values = []
            for action in actions:
                q_values.append(self.computeQValueFromValues(state, action))
            if len(q_values) != 0:
                v_star = max(q_values)
                new_values[state] = v_star
            else:
                new_values[state] = self.getValue(state)
            self.values = new_values
            counter += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = {}

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            successors_of_state = set()
            for action in actions:
                successor_states_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
                potential_successors_of_state = [elements[0] for elements in successor_states_and_probs]
                probs = [elements[1] for elements in successor_states_and_probs]
                for i in range(len(potential_successors_of_state)):
                    if probs[i] != 0:
                        successors_of_state.add(potential_successors_of_state[i])

            for state2 in successors_of_state:
                if state2 in predecessors:
                    predecessors[state2].add(state)
                else:
                    predecessors[state2] = {state}

        priority_queue = util.PriorityQueue()

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            values = [self.computeQValueFromValues(state, action) for action in actions]
            q_value = max(values)
            diff = self.values[state] - q_value
            if diff > 0:
                diff *= -1
            priority_queue.update(state, diff)

        for i in range(self.iterations):
            if priority_queue.isEmpty():
                break
            state = priority_queue.pop()
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            values = [self.computeQValueFromValues(state, action) for action in actions]
            self.values[state] = max(values)

            for predecessor in predecessors[state]:
                if self.mdp.isTerminal(predecessor):
                    continue
                pre_actions = self.mdp.getPossibleActions(predecessor)
                pre_values = [self.computeQValueFromValues(predecessor, action) for action in pre_actions]
                diff = abs(self.values[predecessor] - max(pre_values))
                if diff > self.theta:
                    priority_queue.update(predecessor, -diff)