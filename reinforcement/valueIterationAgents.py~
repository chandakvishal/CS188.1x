# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    ## get dict of transitions for all (state, action, destination):
    self.T = dict()
    self.R = dict()
    self.D = dict()
    
    for state in self.mdp.getStates():
        if self.mdp.isTerminal(state):
            continue
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            for (destination, prob ) in mdp.getTransitionStatesAndProbs(state, action):
                self.T[(state, action, destination)] = prob
                self.R[(state, action, destination)] = mdp.getReward(state, action, destination)
                if (state,action) not in self.D:
                    self.D[(state, action)] = [destination]
                else:
                    self.D[(state, action)] += [destination]
    #print "----------T:\n", self.T, '\n-------R:\n', self.R, '\n------D:\n', self.D
    for _ in range(iterations):
        #print "\n----------SELF.VALUES\n", self.values
        copyVals = util.Counter() 
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)

            bestVals = []
            #print state , mdp.getPossibleActions(state)
            for action in actions:
                #for dest in self.D[state,action]:
                    #print "(state,action,dest):", (state, action, dest) , "T:", self.T[(state, action, dest)], "R:",self.R[(state, action, dest)], "k_val:", self.values[dest]
                    #bestVals += [ sum( [ self.T[state, action, dest] * ( self.R[state, action, dest] + self.discount*self.values[dest] ) ] ) ]
                bestVals += [self.getQValue(state,action)]
                #bestVals += [ sum( [ T[state, action, dest] * ( R[state, action, dest] + self.discount*self.values[state] )  
                #                    for  dest in D[state,action] ] ) ]
                
            #print "state", state, "bestvals", bestVals
            copyVals[state] = max(bestVals)
            #self.values[state] = max(bestVals)
            
        self.values = copyVals.copy()
        #print "----------self.values----------\n", self.values    
    #print self.values, discount
        
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    return sum([ self.T[state,action,dest]*(self.R[state,action,dest] + self.discount*self.values[dest])  
                for dest in self.D[state,action] ] )

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    if self.mdp.isTerminal(state):
        return None
    return max([(self.getQValue(state,action),action) for action in self.mdp.getPossibleActions(state)])[1]
    

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  

