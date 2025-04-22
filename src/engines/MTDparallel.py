import chess
from typing import Dict, List, Optional, Tuple
import math
import random
import numpy as np
from typing import Callable, TypeAlias
from scipi.stats import norm
import heapq

class Node:
    """Represents a node in the Monte Carlo Search Tree with distribution-based minimax values."""

    def __init__(self, parent: Optional['Node'], move: Optional[chess.Move], board: chess.Board, maximizing_player: bool, policy: float = 0.0, action_value: float = 0.0):
        """
        Initializes a Node.

        Args:
            parent: The parent node. None for the root node.
            move: The chess move that led from the parent to this node. None for the root node.
            board: The chess board state for this node.
            prior_policy: The prior probability (P) of selecting the move leading to this node.
        """
        #setup variables
        #
        #
        self.parent: Optional['Node'] = parent
        self.move: Optional[chess.Move] = move
        self.board: chess.Board = board  # Assumes board state is correctly passed
        self.is_terminal: bool = False
        self.terminal_value: Optional[float] = None # Outcome if terminal (1=win, -1=loss, 0=draw for player *at this node*)
        self.maximizing_player: bool = maximizing_player
        self.expanded: bool = False

        #NN from parent variables
        #
        #
        self.prior_policy: float = policy
        #self.action_value: float = action_value

        #MTD variables
        #
        #
        self.upper: float = 1
        self.lower: float = -1
        self.weight: float = 0 #lower is better, is sum 1-(interval)/(total interval a,b)
        
        #children related variables
        #
        #
        self.legal_moves: Optional[List[chess.Move]] = None 
        self.children: Dict[chess.Move, 'Node'] = {}
        self.edge_policy_values: Dict[chess.Move, float] = {}

    def is_leaf(self) -> bool:
        """Checks if the node is a leaf node (i.e., has not been expanded)."""
        return not self.expanded

    def set_weight(self,alpha,beta) -> bool:
        """ sets weight, returns True if the node needs to be deleted"""
        if not self.parent:
            self.weight = 0
            return False
        maxi = min(self.parent.upper,beta)
        mini = max(self.parent.lower,alpha)
        denomenator = self.parent.upper - self.parent.lower
        if self.maximizing_player: #parent is a minimizer
            numerator = self.parent.upper-self.lower
            if numerator <= 0:
                return True
        else: #parent is a maximizer
            numerator = self.upper - self.parent.lower
            if numerator <= 0:
                return True
        self.weight = self.parent.weight+(1-numerator/denomenator)
        return False
    
    def NN_update(self, value: float, variance: float, edge_policy: Dict[chess.Move, float] = {}):
        self.lower = value - 2*np.sqrt(variance)
        self.upper = value + 2*np.sqrt(variance)
        perplexity = 1
        for prob in edge_policy.values():
            perplexity *= prob**(-prob)
        for move in edge_policy.keys():
            if edge_policy[move]*perplexity<.05:
                continue
            self.edge_policy_values[move] = edge_policy[move]
            self.children[move] = Node(self,move,self.board.copy().push(move),!(self.maximizing_player),edge_policy[move])

    def delete(self):
        #TODO get rid of edge to parent

        #if no more edges to parent delete parent

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the node."""
        move_str = self.move.uci() if self.move else "ROOT"
        return f"Node(move={move_str}, N={self.visit_count}, V={self.minimax_value:.3f}, P={self.prior_policy:.3f})"

# Define the type hint for the neural network evaluation function
# It takes a board and returns a tuple: (policy_dict, value)
# policy_dict maps chess.Move to float (probability)
# value is a float representing the board evaluation [-1, 1]
EvaluateNN: TypeAlias = Callable[[chess.Board], Tuple[Dict[chess.Move, float], float]]

class MTD:
    """Manages the MTD process with weights determined by above."""
    def __init__(self, initial_board: chess.Board, nn_evaluate_func: EvaluateNN):
        """
        Initializes the MCTS search.

        Args:
            initial_board: The starting board state.
            nn_evaluate_func: A function that takes a chess.Board and returns
                              a tuple (policy_dict, value).
        """
        self.nn_evaluate = nn_evaluate_func
        # Create the root node - no parent, no move leading to it
        self.root = Node(parent=None, move=None, board=initial_board.copy(), prior_policy=0.0)
        self.min_heap = []
        self.GPU_feed_number

    def search(self):
        """Runs the MTD search for a given number of simulations."""
        best_move = None
        while not best_move:
            self._build_minheap(self.root,-1,1)
            self._expand()
            self._backup(self.root)
            best_move = self.stopping_condition()
        return best_move

    def stopping_condition(self) -> Node: 
        #returns a node if its the best move and we are done searching otherwise returns none
        #TODO

    def _build_minheap(self, node: Node, alpha: float, beta: float):
        #traverse tree, updating weights 
        #deletes nodes where the weight function returns zero or where alpha/beta pruning accors 
        #add leaves to minheap
        if node.is_leaf():
            if not node.is_terminal:
                min_heap.append((node.parent.weight,node.prior_policy,node))
            return
        if node.set_weight(alpha, beta):
            node.delete()
            return
        alpha = max(alpha,node.lower)
        beta = min(beta,node.upper) #should never be lower
        for child in node.children.values():
            self._build_minheap(child,alpha,beta)
        return

    def _expand(self):
        #pops a number of nodes in the min_heap and runs the NN on them
        leaves = len(self.heap)
        if leaves < self.GPU_feed_number:
            #TODO feed all leaves to GPU
            heap = []
            return
        heapq.heapify(self.heap)
        GPU_feed = []
        for _ in range(self.GPU_feed_number)
            GPU_feed.append(heapq.heappop(self.heap))
        #TODO feed GPU_feed to GPU
        heap = []
        return

    def _backup(self, node: Node) -> Tuple(float): #TODO fix tuple notation
        #updates upper and lower through the tree.
        if node.is_leaf():
            if node.is_terminal():
                return (node.terminal_value,node.terminal_value)
            return (node.parent.lower,node.parent.upper)#until searched just return parents old value
        children_LU = []
        for child in node.children.values():
            children_LU.append(self._backup(child))
        if node.maximizing_player:
            node.lower = max(children_LU[:,0])
            node.upper = max(children_LU[:,1])
        else:
            node.lower = min(children_LU[:,0])
            node.upper = min(children_LU[:,1])
        return (node.lower, node.upper)