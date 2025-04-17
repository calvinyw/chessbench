import chess
from typing import Dict, List, Optional, Tuple
import math
import random
import numpy as np
from typing import Callable, TypeAlias
from scipi.stats import norm

global move_policy_cutoff = 0.02
global uncertainty_threshold = .001

class Node:
    """Represents a node in the Monte Carlo Search Tree with distribution-based minimax values."""

    def __init__(self, parent: Optional['Node'], move: Optional[chess.Move], board: chess.Board, prior_policy: float = 0.0):
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
        self.expanded = False
        self.reach_prob: float = 1.0
        self.branch_entropy: float = 1.0
        self.player_turn: bool = board.turn

        #variables for edge coming into
        #
        #
        self.prior_policy: float = prior_policy #policy value for parent->this node (maybe can remove since the edge has the info)

        #children related variables
        #
        #
        self.legal_moves: Optional[List[chess.Move]] = None 
        self.children: Dict[chess.Move, 'Node'] = {}
        self.edge_policy_values: Dict[chess.Move, float] = {}
        self.edge_prob: Dict[chess.Move float] = {}
        self.sum_edge_prob: float = 0.0
        self.expanded_children_policy_total: float = 0.0
        self.child_edge_entropy: float = 1.0

        #main variables
        #
        #
        self.visit_count: int = 0
        self.uncertainty: float = parent.uncertainty if parent else 0.0
        # Initialize with a uniform distribution
        self.num_bins = 81
        self.bin_edges = np.linspace(-1.0, 1.0, self.num_bins)
        self.val_cdf = np.linspace(0.0, 1.0, self.num_bins)  # Uniform CDF
        self.value: float = parent.value
        #self.draw_cdf = np.linspace(0.0, 1.0, self.num_bins)  # Uniform CDF


        #Neural Net variables
        #
        #
        self.NN_val_cdf = np.linspace(0.0, 1.0, self.num_bins)
        self.NN_draw_cdf = np.linspace(0.0, 1.0, self.num_bins)



        #Minimax variables
        #
        #
        self.minimax_val_cdf = np.ones(self.num_bins)
        #self.minimax_draw_cdf = np.linspace(0.0, 1.0, self.num_bins)

    def is_leaf(self) -> bool:
        """Checks if the node is a leaf node (i.e., has not been expanded)."""
        return not self.children

    def get_value(self) -> float:
        """
        Returns the minimax value of this node.
        Represents the predicted outcome from the perspective of the player whose turn it is at this node.

        Returns:
            The minimax value, or 0.0 if the node has not been visited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value

    def set_NN_distributions(self, value: float, variance: float, draw: float, draw_variance: float):
        """
        Sets the node's distribution based on a single value and variance.
        
        Args:
            value: The evaluation value (-1 to 1)
            certainty: How certain we are about this value (0 to 1)
        """
        cdf_values = norm.cdf(self.bin_edges, loc=value, scale=np.sqrt(variance))
        cdf_values = cdf_values / cdf_values[-1] if cdf_values[-1] > 0 else np.linspace(0.0, 1.0, self.num_bins)
    
        self.NN_val_cdf = cdf_values

        cdf_values = norm.cdf(self.bin_edges, loc=draw, scale=np.sqrt(draw_variance))
        cdf_values = cdf_values / cdf_values[-1] if cdf_values[-1] > 0 else np.linspace(0.0, 1.0, self.num_bins)
    
        self.NN_draw_cdf = cdf_values

    def update_minimax_value(self, child_new_val_cdf: np.ndarray, child_old_val_cdf: np.ndarray):
        """
        Updates the node's distribution based on the new child's distribution.
        Args:
            new_child_dist: The new distribution of the child.
            old_child_dist: The old distribution of the child.
        """
        child_new_val_cdf = 1.0 -child_new_val_cdf[::-1]
        child_old_val_cdf = 1.0 -child_old_val_cdf[::-1]
        self.minimax_val_cdf %= child_old_val_cdf
        self.minimax_val_cdf *= child_new_val_cdf
        #right now not messing with draw update but eventually will. Will need to do something over every child which is a pain
        
    def update_main_distributions(self):
        self.val_cdf = (1-self.expanded_children_policy_total)*self.NN_val_cdf+ self.expanded_children_policy_total*self.minimax_val_cdf
        #self.draw_cdf = (1-self.expanded_children_policy_total)*self.NN_draw_cdf+ self.expanded_children_policy_total*self.minimax_draw_cdf

    def update_value_and_uncertainty(self):
        pdf = np.diff(np.append(0, self.val_cdf))
        self.value = np.sum(pdf * self.bin_edges)
        #version one: no draw involved
        self.uncertainty = np.sum(pdf * (self.bin_edges - self.value)**2)


    def first_NN_update(self, value: float, variance: float, draw: float, draw_variance: float):
        """
        Updates the node's distribution based on a single value and variance.

        Args:
            value: The evaluation value (-1 to 1)
            certainty: How certain we are about this value (0 to 1)
        """
        self.visit_count = 1 
        self.set_NN_distributions(value, variance, draw, draw_variance)
        self.val_cdf = self.NN_val_cdf
        self.draw_cdf = self.NN_draw_cdf
        self.update_value_and_uncertainty()

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the node."""
        move_str = self.move.uci() if self.move else "ROOT"
        return f"Node(move={move_str}, N={self.visit_count}, V={self.minimax_value:.3f}, P={self.prior_policy:.3f})"

# Define the type hint for the neural network evaluation function
# It takes a board and returns a tuple: (policy_dict, value)
# policy_dict maps chess.Move to float (probability)
# value is a float representing the board evaluation [-1, 1]
EvaluateNN: TypeAlias = Callable[[chess.Board], Tuple[Dict[chess.Move, float], float]]

class MinimaxDistMCTS:
    """Manages the Monte Carlo Tree Search process with distribution-based minimax values."""
    def __init__(self, initial_board: chess.Board, nn_evaluate_func: EvaluateNN, c_puct: float = 1.41, fpu_reduction: float = 0.5):
        """
        Initializes the MCTS search.

        Args:
            initial_board: The starting board state.
            nn_evaluate_func: A function that takes a chess.Board and returns
                              a tuple (policy_dict, value).
            c_puct: The exploration constant (Cp in the PUCT formula).
            fpu_reduction: Value subtracted from parent's value for FPU when selecting unvisited nodes.
        """
        self.nn_evaluate = nn_evaluate_func
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        # Create the root node - no parent, no move leading to it
        self.root = Node(parent=None, move=None, board=initial_board.copy(), prior_policy=0.0)
        self.tree_size = 1

    def search(self):
        """Three steps which repeat
        Step 0: make a max-heap of the posible moves sorted by (reach prob * uncertainty)
        
        Step 1: Tree expandsion
            store max-heap of nodes sorted by (reach prob * uncertainty)
            pop the top node (or several) to expand those subtrees

            output: none
        
        Step 2: alpha-beta tree backup
            for each node, compute the minimax distribution
                but if during the step, we find that the probability that dist < alpha is less than some threshold*, then break, and  cut parent node
            for each node, update the value distribution and uncertainty

            output: distribution, cut_bool
        
        Step 3: recompute reach probabilities
            run through the tree, and update the reach probabilities, chop any edge with small prob (i.e. prob<.05). Make a max-heap with all the unexpanded leaves with the appropriate (reach prob * uncertainty)
            cut if the reach-threshold* is too low
        repeat 1-3

        threshold*: .05*(reach_prob)
        or better: .05*Prod(vertices v above p) (1+entropy of the v)^{-1}. entropy is an approximation of the 


        when to end search:
            if expected error = value(root)-value(move) < length of a bucket (2/81) then we should stop and take the best move.
            if the max reach prob*uncertainty < something then we should stop since doing more updates will hardly change our uncertainty
            so            
            remaining updates ~ (value(root)-max_children value(move)-length(bucket))/(max_{leaves} reach prob*uncertainty)
            update time ~ sqrt(tree_size) but maybe we will just use tree size
            so we should stop when
            C*remaining updates*update time < (remaining time)/2 if increment then (remaining time)/2 + incrememnt
        """

        while not self.stopping_condition():
            #step 3:
            self._update_reach_probabilities(self.root)
            #step 1:
            self._expand_tree()
            #step 2:
            self._backup_alpha_beta()

    def stopping_condition(self, uncertainty_delta: float) -> bool:
        """
        Checks if the search should stop.
        if expected error = value(root)-value(move) < length of a bucket (2/81) then we should stop and take the best move.
            if the max reach prob*uncertainty < something then we should stop since doing more updates will hardly change our uncertainty
            so            
            remaining updates ~ (value(root)-max_children value(move)-length(bucket))/(max_{leaves} reach prob*uncertainty)
            update time ~ sqrt(tree_size) but maybe we will just use tree size
            so we should stop when
            C*remaining updates*update time < (remaining time)/2 if increment then (remaining time)/2 + incrememnt
        """
        best_move_value = [max(child.value for child in self.root.children)]
        expected_error=self.root.value-best_move_value
        if expected_error <= 2/81:#2/81 is the length of a bucket. Should probably change to a variable.
            return True
            
        remaining_updates = expected_error/uncertainty_delta
        if remaining_updates*np.sqrt(self.tree_size) < (constant)*self.remaining_time():
            return True
        return False

    def _update_reach_probabilities(self, parent: Node):
        """
        Updates the reach probabilities of the nodes in the tree and the entropies
        """
        if parent.is_leaf():
            #add to the max heap for expand
            return
        
        parent.sum_edge_prob  = 0
        #compute new edge probabilities
        for child in node.children:
            alpha = np.sqrt(parent.number_of_visits)/(np.sqrt(parent.number_of_visits)+child.number_of_visits/self.c_puct)
            parent.edge_prob[child.move] = alpha*parent.edge_policy_values[child.move] + (1-alpha) * self.prob_child_maximizes(child.val_cdf,parent.val_cdf)
            parent.sum_edge_prob += parent.edge_prob[child.move]

        #normalize and compute entropy
        parent.child_edge_entropy = 1.0
        for child in parent.children:
            parent.edge_prob[child.move] /= parent.sum_edge_prob
            if parent.edge_prob[child.move] <0.05:
                #TODO delete the edge and child.
            else:
                parent.child_edge_entropy -= parent.edge_prob[child.move]*np.ln(parent.edge_prob[child.move])
        parent.entropy *= parent.child_edge_entropy

        #recurse on children
        for child in parent.children:
            child.reach_prob = parent.reach_prob * parent.edge_prob[child.move]
            child.entropy = parent.entropy
            self._update_reach_probabilities(child)


    def _expand_tree(self, N_bound: float, u_bound: float):#whole function TODO
        """
        Expands the tree by selecting by leaves via a max-heap for uncertainty*reach_prob

        at most N_bound number of nodes are added to the tree.
        halts if the uncertainty*reach_prob that is popped is less than u_bound
        """
        for _ in range(N_bound):
            #pop the head of the max-heap

            if blah < u_bound:
                return
            
            #expands the popped node
            new_leaves = self._expand_node(popped_node)
            self.tree_size +=1

            #adds the new leaves to the max-heap

    def _backup_alpha_beta(self, node: Node, alpha: float, beta: float) -> None:
        """
        alpha,beta represent the median result that one can guarantee
        """ 
        if node.is_leaf() or node.is_terminal():
            return node.val_cdf
        minimax = np.ones(self.num_bins)
        for child in node.children:
            #update expanded if necessary
            if not child.expanded:
                node.expanded_children_policy_total += parent.edge_prob[child.move]
                child.expanded = True
            
            #do alpha beta minimax computation
            child_cdf, delete = self._backup_alpha_beta(child, alpha, beta)
            if delete:
#TODO                #TODO delete the node and the minimax

#TODO                #TODO delete the edge from the parent!!!!!!

                return np.zeroes(self.num_bins), False
            child_cdf = 1.0 - child_cdf[::-1]
            minimax *= child_cdf
            if node.player_turn:
                alpha = max(alpha, median(minimax))
                if probability_less_than(minimax, 1-beta, .025*node.branch_entropy):
                    return np.ones(self.num_bins), True
            else:
                beta = max(beta, median(minimax))
                if probability_less_than(minimax, 1-alpha, .025*node.branch_entropy):
                    return np.ones(self.num_bins), True
        #update parent distribution
        node.minimax_val_cdf = minimax
        node.update_main_distributions()

        return node.val_cdf, False
            
        

    def _expand_node(self, node: Node) -> List[Node]:
        """
        Expands a node with depth 0 search
        """
        if node.legal_moves is None:
            node.legal_moves = list(node.board.legal_moves)

        # Check for terminal state
        if node.board.is_game_over(claim_draw=False): # Simplified draw check
            node.is_terminal = True
            result = node.board.result(claim_draw=False)
            if result == '1-0': # White won
                node.terminal_value = 1.0
            elif result == '0-1': # Black won
                node.terminal_value = -1.0
            else: # Draw
                node.terminal_value = 0.0

            # Adjust terminal value based on whose turn it is at the node
            # If it's White's turn and White won (1.0), value is 1.0
            # If it's Black's turn and White won (1.0), value is -1.0
            if node.board.turn == chess.BLACK:
                node.terminal_value *= -1.0

            # For terminal nodes, create a very certain distribution
            if node.terminal_value == 0.0:
                node.first_NN_update(0,0,1,0)
            else:
                node.first_NN_update(node.terminal_value,0,0,0)
            return None

        # If not terminal, expand using the neural network
        # The NN value is from the perspective of the current player at node.board
        policy_dict, value, variance, draw, draw_variance = self.nn_evaluate(node.board)
        
        node.first_NN_update(node.value,node.variance,node.draw,node.draw_variance)

        # Populate children based on legal moves and policy network output
        for move in node.legal_moves:
            move_policy = policy_dict.get(move, 0.0) # Use get for safety, default to 0 if move not in policy
            if move_policy > move_policy_cutoff:
                # Create the child board state
                child_board = node.board.copy()
                try:
                    child_board.push(move)
                    # Create the child node and add it to the parent's children
                    node.children[move] = Node(parent=node, move=move, board=child_board, prior_policy=move_policy)
                    node.edge_policy_values[move] = move_policy
                    node.edge_prob[move] = move_policy
                except Exception as e:
                    print(f"Error pushing move {move.uci()} on board {node.board.fen()}: {e}")
                    # Decide how to handle invalid moves predicted by policy, e.g., skip
                    continue
        
        
        favorite_move = max(node.edge_prob, key=node.edge_prob.get)



