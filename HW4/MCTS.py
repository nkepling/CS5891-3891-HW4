import gymnasium as gym
from copy import deepcopy
import random
from collections import defaultdict
from utils import type_checker
import math


"""
MCTS Implemenation. This implementation uses a global table to store the Q values and visit counts for state-action pairs and states. Compatible with Gymnasium environments.
"""

class Node:
    """
    Nodes of the search tree, labeled by a (state,actions) pair.
    """
    def __init__(self, parent, state, action, is_terminal, reward):
        """
        Args:
            parent (Node): The parent node of this node.
            state (Union[int,np.ndarray,tuple]): Environment state.
            action (int): Action taken to reach this state.
            is_terminal (bool): Is the state terminal.
            reward (float): Immediate reward for reaching this state.

        Attributes:
            children (list): List of child nodes. Index is the action taken to reach the child node.

        """
        self.parent = parent # parent node
        self.state = state # state of the environment (observation)
        self.action = action # action taken to reach this state
        self.is_terminal = is_terminal # is the state terminal (done)
        self.reward = reward # immediate reward for reaching this state
        self.children = [] # list of child nodes, initially empty, index is the action taken to reach the child node

    def is_leaf(self):
        return not self.children
    
    def is_root(self):
        return self.parent is None
    
    def is_terminal(self):
        return self.is_terminal






class MCTS:
    """Vanilla MCTS; Compatible with Gymnasium environments.
        Selection and expansion are combined into the "treepolicy method"
        The rollout/simluation is the "default" policy. 

        For reference see:
        
        A Survey of Monte Carlo Tree Search Methods Browne et al. 2012
        
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6145622


        You will implement the following methods:

        1. search()
        2. _tree_policy()
        3. _default_policy()
        4. _selection()
        5. _expand()
        6. _backpropagation()
        7. update_node()
        9. best_action()


        NOTE: To be compatable with NS-Gym be sure to pass all observations and rewards into the type_checker() util function. 

        For example:

        ```python
        observation, reward, done, truncated,info = env.step(action)
        observation, reward = type_checker(observation, reward)

        ###################### or ######################

        observation, _  = type_checker(observation, None)

        ###################### or ######################

        _, reward = type_checker(None, reward)
        ```


    """
    def __init__(self,env:gym.Env,state,d,m,c,gamma) -> None:
        """
        Args:
            env (gym.Env): The environment to run the MCTS on.
            state (Union[int, np.ndarray,tuple]): The state to start the MCTS from.
            d (int): The rollout depth of the MCTS.
            m (int): The number of simulations to run.
            c (float): The UCT exploration constant.
            gamma (float): The discount factor.

        Attributes:
            root (Node): The root node of the tree.
            possible_actions (list): List of possible actions in the environment.
            Qsa (dict): Dictionary to store Q values for state-action pairs.
            Nsa (dict): Dictionary to store visit counts for state-action pairs.
            Ns (dict): Dictionary to store visit counts for states.

        """
        self.env = env # gym environment
        self.d = d # rollout depth 
        self.m = m # number of simulations
        self.c = c # UCT exploration constant
        
        self.possible_actions = [x for x in range(env.action_space.n)] # possible actions in the environment
        self.gamma = gamma # discount factor

        ##### Global tables for storing Q values and visit counts #####

        self.Qsa = defaultdict(float)  # stores Q values for s,a pairs, defaults to Qsa of 0
        self.Nsa = defaultdict(float)  # stores visit counts for s,a pairs, default to Nsa of 0
        self.Ns = defaultdict(float) # stores visit counts for states, default to Ns of 0

        ##############################################################

        # Root node

        state,_ = type_checker(state,None)
        self.root = Node(None,state,None,False,0) 

    def search(self):
        """Do the MCTS by doing m simulations from the current state s. 
        After doing m simulations we simply choose the action with the most visits.

        Returns:
            best_action(int): best action to take
            action_values(list): list of Q values for each action.
        """
        for k in range(self.m):
            self.sim_env = deepcopy(self.env) # make a deepcopy of of the original environment for simulation. 
            # YOUR CODE HERE

        # return best_action,action_values


    def _tree_policy(self, node:Node):
        """Tree policy for MCTS. Traverse the tree from the root node to a leaf node or terminal state.
        Args:
            node (Node): The root node of the tree.
        Returns:
            leaf_node (Node): The leaf node reached by the tree policy.
        """
        # YOUR CODE HERE
     
        #return node

    def _default_policy(self,node:Node):
        """Simulate/Playout step 
        While state is non-terminal or rollout depth is less than depth limit, choose an action uniformly at random and transition to new state. 
        Return the discounted cummlative reward collected during the rollout.
        Args:
            node (Node): The node to start the simulation from.
        Returns:
            tot_reward (float): The total discounted reward from the rollout.
        """  
        tot_reward = 0
        terminated = False
        truncated = False
        depth = 0

        # YOUR CODE HERE

        return tot_reward

    def _selection(self,node:Node):
        """Pick the next node to go down in the search tree based on UTC formula.
        Use the values stored in the global tables Qsa, Nsa, Ns to calculate the UTC value.
        """
        # YOUR CODE HERE

        best_node = None

        return best_node

    def _expand(self,node:Node):
        """Expand the tree by adding a new nodes to the tree.
        Given the leaf node expand all actions at once then reutrn a newly added child node at random. 
        Args:
            node (Node): The node to expand.
        Returns: 
            Node: The new node added to the tree.
        """
        # YOUR CODE HERE
        # NOTE: You may need to make another deepcopy of the simulation environment to get the new state for each child node.

        pass
            

    def _backpropagation(self,R,node:Node):
        """Backtrack to update the number of times a node has beenm visited and the value of a node until we reach the root node. 
        We update the global tables Qsa, Nsa, Ns.

        Args:
            R (float): The discounted reward from the rollout.
            node (Node): The node to start the backpropagation from.
        """
        # YOUR CODE HERE
        pass


    def best_action(self,node:Node):
        """Get the best action to take from the root node based on visit counts.
        Args:
            node (Node): The root node.
        """
        #YOUR CODE HERE
        pass

    

if __name__ == "__main__":
    pass



