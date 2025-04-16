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
        self.draw_cdf = np.linspace(0.0, 1.0, self.num_bins)  # Uniform CDF


        #Neural Net variables
        #
        #
        self.NN_val_cdf = np.linspace(0.0, 1.0, self.num_bins)
        self.NN_draw_cdf = np.linspace(0.0, 1.0, self.num_bins)



        #Minimax variables
        #
        #
        self.minimax_val_cdf = np.ones(self.num_bins)
        self.minimax_draw_cdf = np.linspace(0.0, 1.0, self.num_bins)

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
        return self.value #WARNING MIGHT BE APPENDING ON THE WRONG SIDE!

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
        self.val_cdf = (1-self.explored_children_policy_total)*self.NN_val_cdf+ self.explored_children_policy_total*self.minimax_val_cdf
        self.draw_cdf = (1-self.explored_children_policy_total)*self.NN_draw_cdf+ self.explored_children_policy_total*self.minimax_draw_cdf

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
        """Runs the MCTS search for a given number of simulations."""
        while self.root.uncertainty > uncertainty_threshold*np.log(self.tree_size) or threshold_delta*1000 < uncertainty_threshold*np.log(self.tree_size):
            leaf_node, reach_prob = self._select(self.root)
            threshold_delta = reach_prob*leaf_node.uncertainty #this is roughly proportional to the maximum we can hope to change the uncertainty in a single search.
            # Value returned by expand is from the perspective of the leaf_node
            new_leaf_node = self._expand(leaf_node, 1, threshold = (leaf_node.uncertainty + uncertainty_threshold)/2)#try to half the amount of unvertainty
            self._backup(new_leaf_node)

    def _select(self, node: Node) -> Node:
        """Selects a leaf node starting from the given node using PUCT.

        A leaf node is one that has not been expanded yet (node.children is empty).
        Descends the tree by selecting the child with the highest PUCT score until a leaf is reached.
        """
        current_node = node
        reach_prob = 1.0
        while not current_node.is_leaf():
            best_score = -float('inf')
            best_child = None
            current_node.visit_count +=1

            # Ensure legal moves are available (should be populated during expansion)
            if current_node.legal_moves is None:
                # This should ideally not happen if the node was expanded correctly.
                current_node.legal_moves = list(current_node.board.legal_moves)
                if not current_node.legal_moves:
                    # If no legal moves, it's effectively terminal, treat as leaf
                    break

            # Iterate through existing children to find the best one according to PUCT
            for move in current_node.children:
                child_node = current_node.children[move]

                score = current_node.edge_prob[move]*child_node.uncertainty

                if score > best_score:
                    best_score = score
                    best_child = child_node

            if best_child is None:
                # This case should not be hit if the node is not a leaf,
                # as a non-leaf node must have children.
                # If it happens, it might indicate an issue in expansion or selection logic.
                # For robustness, we can treat it as a leaf.
                break
            reach_prob = reach_prob*current_node.edge_prob[best_child.move]/current_node.sum_edge_prob
            current_node = best_child # Move down the tree

        # Return the node that is determined to be a leaf in this path
        return current_node, reach_prob

    def _expand(self, node: Node, reach_prob: float, threshold: float) -> Tuple[float, np.ndarray]:
        """Sucsessively expands leaf nodes until the uncertainty is below a threshold.
        Args:
            node: The leaf node to expand.

        Returns:
            node: the last node we expanded
        """
        uncertainty, favorite_child = self._expand_helper(node)
        #really the helper builds a while loop so probably should change to a while loop at some point

        if uncertainty <threshold*np.log(1/reach_prob):
            return node
        if favorite_child:
            return self._expand(favorite_child, reach_prob * node.edge_prob[favorite_child.move], threshold)
        else:
            return node

    def _expand_helper(self, node: Node) -> float:
        """Expands a leaf node: computes legal moves, checks terminal state, evaluates with NN.

        Args:
            node: The leaf node to expand.

        Returns:
            uncertainty of the node we expanded
        """
        # Generate legal moves if not already done (e.g., for root node initially)
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
            return 0, None

        # If not terminal, expand using the neural network
        # The NN value is from the perspective of the current player at node.board
        policy_dict, value, variance, draw, draw_variance = self.nn_evaluate(node.board)
        
        node.first_NN_update(node.value,node.variance,node.draw,node.draw_variance)

        # Populate children based on legal moves and policy network output
        for move in node.legal_moves:
            move_policy = policy_dict.get(move, 0.0) # Use get for safety, default to 0 if move not in policy
            if move_policy > move_policy_cutoff:
                self.tree_size +=1
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
        
        # Return the value and CDF from the NN (perspective of the current node's player)
        return node.uncertainty, node.children[favorite_move]

    def _backup(self, child: Node):
        """Backpropagates the value and distribution up the tree from the expanded node. Really is updating the parents information and then the edge parent to child before moving onto the parent of the parent etc.

        Args:
            child: The node from where the backup starts (a leaf).
        """
        parent = child.parent
        old_val_cdf = np.ones(self.num_bins)
        while parent is not None:


            #upate parents minimmax
            parent.update_minimax_value(child.val_cdf, old_val_cdf)

            #update parents expanded
            if child.expanded == False:
                parent.expanded_children_policy_total += parent.edge_policy_values[child.move]
                child.expanded = True

            #update parents main distributions
            if parent.expanded == False:
                old_val_cdf_2 = parent.val_cdf.copy()
            else:
                old_val_cdf_2 = np.ones(self.num_bins)
            parent.update_main_distributions()

            #update edge weight and parent total sum edge weights:
            parent.sum_edge_prob -= parent.edge_prob[child.move]

            alpha = np.sqrt(parent.number_of_visits)/(np.sqrt(parent.number_of_visits)+child.number_of_visits/self.c_puct)
            parent.edge_prob[child.move] = alpha*parent.edge_policy_values[child.move] + (1-alpha) * self.prob_child_maximizes(child.val_cdf,parent.val_cdf)

            parent.sum_edge_prob += parent.edge_prob[child.move]

            #pass to parent
            child = parent
            parent = child.parent
            old_val_cdf = old_val_cdf_2



    def prob_child_maximizes(child_cdf: np.ndarray,parent_cdf: np.ndarray)->float:
        """
        Helper function to calculate how often child_dist is <= parent_dist assuming independence.
        """
        # Calculate the minimizer of the two distributions
        child_cdf_swapped = 1.0-child_dist[::-1]
        parent_cdf_wo_child = parent_cdf/child_cdf
        child_pdf_swapped = np.diff(np.append(0, child_cdf_swapped))
        return np.dot(np.append(child_pdf_swapped,1),(1.0 - np.append(0,parent_cdf_wo_child)))

    def get_best_move(self) -> Optional[chess.Move]:
        """Selects the best move from the root node's children.

        Args:
            temperature: Controls the randomness of selection.
                         0: deterministic (choose based on visits and minimax value).
                         >0: sample based on visit counts raised to 1/temperature.

        Returns:
            The selected chess.Move, or None if the root has no children.
        """
        if not self.root.children:
            # If root hasn't been expanded or has no legal moves initially.
            print("Warning: Root node has no children, cannot select best move.")
            if self.root.legal_moves is None:
                self.root.legal_moves = list(self.root.board.legal_moves)
            return random.choice(self.root.legal_moves) if self.root.legal_moves else None

        available_children = self.root.children

        if temperature == 0.0:
            # Deterministic: Choose the move based on visit count and minimax value
            # First check for terminal positions (wins/losses/draws)
            terminal_wins = []
            for move, child in available_children.items():
                if child.is_terminal and child.terminal_value == 1.0:
                    terminal_wins.append((move, child))
            
            # If we have winning moves, choose the one with shortest path to mate (highest visit count)
            if terminal_wins:
                return max(terminal_wins, key=lambda x: x[1].visit_count)[0]
                
            # Otherwise use values to choose the best move
            best_move = max(available_children.keys(), 
                           key=lambda move: (available_children[move].value))
            return best_move
        else:
            # Probabilistic sampling based on visit counts raised to 1/temperature
            moves = list(available_children.keys())
            visit_counts = [available_children[m].visit_count for m in moves]

            if not visit_counts or all(v == 0 for v in visit_counts):
                 # If no visits recorded, or during early search, sample uniformly
                 if not moves: return None
                 return random.choice(moves)

            # Apply temperature
            try:
                powered_visits = [v**(1.0 / temperature) for v in visit_counts]
                total_power = sum(powered_visits)
            except OverflowError:
                # Handle potential overflow with large visit counts and low temperature
                # Fallback to choosing the max visit count move
                max_visits = -1
                best_move = None
                for i, m in enumerate(moves):
                    if visit_counts[i] > max_visits:
                        max_visits = visit_counts[i]
                        best_move = m
                return best_move

            if total_power == 0:
                # Avoid division by zero if all powered visits are zero (e.g., high temp on low visits)
                return random.choice(moves)

            probabilities = [p / total_power for p in powered_visits]

            # Sample a move based on the calculated probabilities
            try:
                chosen_move = random.choices(moves, weights=probabilities, k=1)[0]
                return chosen_move
            except ValueError:
                # Handle potential issue if probabilities don't sum to 1 (shouldn't happen if total_power > 0)
                print("Warning: Error during weighted random choice. Falling back to max visit count.")
                best_move = max(available_children.keys(), key=lambda move: available_children[move].visit_count)
                return best_move

# --- Placeholder for MCTS class later ---
