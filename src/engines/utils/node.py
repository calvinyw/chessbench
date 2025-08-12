from collections import defaultdict
from typing import Optional, List, Tuple, cast, Dict
import chess
from collections.abc import Sequence
from searchless_chess.src.engines.utils.nnutils import reduced_fen


class TTEntry:
    """
    Transposition table entry that stores comprehensive search information for a position.
    
    For each position (reduced_fen), we store:
    - Static evaluation (value and policy) for node creation
    - Exact score + depth (for PV nodes where alpha < score < beta)
    - Upper bound score + depth (for nodes that fail low, score <= alpha)
    - Lower bound score + depth (for nodes that fail high, score >= beta)
    """
    
    def __init__(self, static_value: float, policy: List[Tuple[chess.Move, float]], U: float = 0.0, bin_centers: Optional[List[float]] = None, hl_probs: Optional[List[float]] = None):
        """
        Initialize a transposition table entry.
        
        Args:
            static_value: The static evaluation of the position
            policy: List of (move, probability) tuples for move ordering
        """
        self.static_value = static_value
        self.policy = policy
        self.U = U
        self.bin_centers = bin_centers if bin_centers is not None else []
        self.hl_probs = hl_probs if hl_probs is not None else []
        # Exact score entries: (score, depth)
        self.exact_score: Optional[Tuple[float, float]] = None
        
        # Upper bound entries: (score, depth) - score <= alpha (fail low)
        self.upper_bound: Optional[Tuple[float, float]] = None
        
        # Lower bound entries: (score, depth) - score >= beta (fail high)
        self.lower_bound: Optional[Tuple[float, float]] = None
    
    def store_exact(self, score: float, depth: float) -> None:
        """Store an exact score (PV node result)."""
        if self.exact_score is None or depth >= self.exact_score[1]:
            self.exact_score = (score, depth)
    
    def store_upper_bound(self, score: float, depth: float) -> None:
        """Store an upper bound (fail low result)."""
        if self.upper_bound is None or depth >= self.upper_bound[1]:
            self.upper_bound = (score, depth)
    
    def store_lower_bound(self, score: float, depth: float) -> None:
        """Store a lower bound (fail high result)."""
        if self.lower_bound is None or depth >= self.lower_bound[1]:
            self.lower_bound = (score, depth)
    
    def query(self, alpha: float, beta: float, depth: float) -> Optional[float]:
        """
        Query the transposition table entry for a usable score.
        
        Returns a score if we have sufficient information to answer the alpha-beta query,
        None otherwise.
        
        Args:
            alpha: The alpha bound for the search
            beta: The beta bound for the search
            depth: The current search depth
            
        Returns:
            The score if we can answer the query, None otherwise
        """
        # Check for exact score with sufficient depth
        if (self.exact_score is not None and 
            self.exact_score[1] >= depth):
            return self.exact_score[0]
        
        # Check for upper bound that causes fail low
        if (self.upper_bound is not None and 
            self.upper_bound[1] >= depth and 
            self.upper_bound[0] <= alpha):
            return self.upper_bound[0]
        
        # Check for lower bound that causes fail high
        if (self.lower_bound is not None and 
            self.lower_bound[1] >= depth and 
            self.lower_bound[0] >= beta):
            return self.lower_bound[0]
        
        return None
    
    def __str__(self) -> str:
        """String representation of the TT entry."""
        exact_str = f"Exact: {self.exact_score}" if self.exact_score else "Exact: None"
        upper_str = f"Upper: {self.upper_bound}" if self.upper_bound else "Upper: None"
        lower_str = f"Lower: {self.lower_bound}" if self.lower_bound else "Lower: None"
        return f"TTEntry(Value={self.static_value:.4f}, {exact_str}, {upper_str}, {lower_str})"


class Node:
    """
    Represents a node in a chess search tree.
    Used for algorithms like negamax and alpha-beta pruning.
    """
    
    def __init__(self, 
                 board: chess.Board,
                 parent: Optional['Node'] = None,
                 value: float = 0.0,
                 policy: Optional[List[Tuple[chess.Move, float]]] = None,
                 U: float = 0.0,
                 bin_centers: Optional[List[float]] = None,
                 hl_probs: Optional[List[float]] = None,
                 terminal: bool = False):
        """
        Initialize a new Node in the search tree.
        
        Args:
            board: The chess board state at this node
            parent: The parent node (None if this is the root)
            value: Static evaluation of this position
            policy: List of (move, probability) tuples in decreasing probability order
            U: Uncertainty/variance of the position
            bin_centers: List of bin center values for the distribution
            hl_probs: List of probabilities corresponding to each bin center
            terminal: Whether this node is a terminal state (checkmate, stalemate, etc.)
        """
        self.board = board.copy()
        self.moves = list(self.board.generate_legal_moves())
        self.parent = parent
        self.value = value
        # Convert policy to new format: (move, probability, child_node_or_none, metadata_dict)
        if policy is not None:
            self.policy: List[Tuple[chess.Move, float, Optional['Node'], Dict[str, float]]] = [(move, prob, None, metadata) for move, prob, metadata in policy]
        else:
            self.policy: List[Tuple[chess.Move, float, Optional['Node'], Dict[str, float]]] = []
        self.terminal = terminal
        self.U = U
        self.bin_centers = bin_centers if bin_centers is not None else []
        self.hl_probs = hl_probs if hl_probs is not None else []
        
    def is_root(self) -> bool:
        """Check if this node is the root of the tree (has no parent)."""
        return self.parent is None
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (has no expanded children)."""
        return not any(child is not None for _, _, child, _ in self.policy)

    def is_terminal(self) -> bool:
        """Check if this node is a terminal state (checkmate, stalemate, etc.)."""
        return self.terminal
    
    def is_fully_expanded(self) -> bool:
        """
        Check if all possible moves from this position have been expanded.
        This checks if all policy entries have corresponding child nodes.
        """
        return all(child is not None for _, _, child, _ in self.policy)
    
    def get_children(self) -> List['Node']:
        """Get all expanded child nodes."""
        return [child for _, _, child, _ in self.policy if child is not None]
    
    def get_child_count(self) -> int:
        """Get the number of expanded children."""
        return sum(1 for _, _, child, _ in self.policy if child is not None)
    
    def add_child(self, child: 'Node', move: chess.Move) -> None:
        """Add a child node for the specified move."""
        for i, (policy_move, prob, existing_child, metadata) in enumerate(self.policy):
            if policy_move == move:
                if existing_child is None:
                    self.policy[i] = (policy_move, prob, child, metadata)
                    return
                else:
                    raise ValueError(f"Child already exists for move {move}")
        raise ValueError(f"Move {move} not found in policy")
        
    def get_best_child(self, negamax: bool = True) -> Optional['Node']:
        """Get the child with the highest value (from opponent's perspective)."""
        children = self.get_children()
        if not children:
            return None
        # For negamax/alphabeta, we want the minimum value from opponent's perspective
        return min(children, key=lambda child: child.value) if negamax else max(children, key=lambda child: child.value)
    
    def get_highest_policy_move(self) -> Optional[chess.Move]:
        """
        Get the move with the highest policy value.
        Returns None if no policy is available.
        """
        if len(self.policy) == 0:
            return None
        return self.policy[0][0]  # Return the move from the highest policy tuple
    
    def get_move_to_child(self, child: 'Node') -> Optional[chess.Move]:
        """Find the move that leads to a specific child node."""
        for move, _, node, _ in self.policy:
            if node is child:
                return move
        return None
    
    def get_child_for_move(self, move: chess.Move) -> Optional['Node']:
        """Get the child node for a specific move, if it exists."""
        for policy_move, _, child, _ in self.policy:
            if policy_move == move:
                return child
        return None
    
    def get_metadata_for_move(self, move: chess.Move) -> Optional[Dict[str, float]]:
        """Get the metadata dictionary for a specific move, if it exists."""
        for policy_move, _, _, metadata in self.policy:
            if policy_move == move:
                return metadata
        return None
    
    def update_metadata_for_move(self, move: chess.Move, metadata: Dict[str, float]) -> None:
        """Update the metadata dictionary for a specific move."""
        for i, (policy_move, prob, child, existing_metadata) in enumerate(self.policy):
            if policy_move == move:
                self.policy[i] = (policy_move, prob, child, metadata)
                return
        raise ValueError(f"Move {move} not found in policy")
    
    def __str__(self) -> str:
        """String representation of the node."""
        status = "Terminal" if self.terminal else "Internal"
        value_str = f"{self.value:.4f}"
        move_count = len(self.policy)
        child_count = self.get_child_count()
        return f"Node({self.board.fen()}, Value={value_str}, Moves={move_count}, Children={child_count}, policy: {[(m, p) for m, p, _, _ in self.policy]})"
        
    def print_lineage(self) -> str:
        """
        Print this node and all its ancestors in a tree-like structure.
        Each node is indented based on its depth in the tree.
        """
        # Collect all ancestors from root to current
        ancestors = []
        current = self
        while current:
            ancestors.insert(0, current)
            current = current.parent
            
        # Build the display string
        result = []
        for i, node in enumerate(ancestors):
            indent = "-" * (i + 1) if i > 0 else ""
            result.append(f"{indent}{node.__str__()}")
            
        return "\n".join(result)
    
    def print_tree(self) -> str:
        """
        Print the tree structure of this node and all its descendants.
        Each descendant is indented based on its depth below this node.
        """
        result = []
        history = defaultdict(int)
        
        def _print_tree_recursive(node: 'Node', depth: int = 0) -> None:
            indent = "-" * depth if depth > 0 else ""

            if history[reduced_fen(node.board)] == 0:
                result.append(f"{indent}{node.__str__()}")
            else:
                result.append(f"{indent}{node.__str__()} - REPEAT")
                return
            history[reduced_fen(node.board)] += 1
            
            for child in node.get_children():
                _print_tree_recursive(child, depth + 1)
        
        _print_tree_recursive(self)
        return "\n".join(result)
    
    def sort_and_normalize(self) -> None:
        """
        Sort and normalize the policy while maintaining the child correspondence.
        
        Sorts the policy by probability (descending) and normalizes all probabilities 
        to sum to 1. The child nodes and metadata move with their corresponding policy entries.
        """
        if not self.policy:
            return
        
        # Sort by probability in descending order
        self.policy.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize all probabilities to sum to 1
        total_prob = sum(prob for _, prob, _, _ in self.policy)
        if total_prob > 0:
            self.policy = [(move, prob / total_prob, child, metadata) for move, prob, child, metadata in self.policy]

class MCTSNode(Node):
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, value: float = 0.0, policy: Optional[List[Tuple[chess.Move, float]]] = None, terminal: bool = False):
        super().__init__(board, parent, value, policy, terminal=terminal)
        self.Q = value
        self.N = 1

    def get_value(self) -> float:
        """Get the value of this node."""
        return self.Q / self.N
    
    def avg_in(self, Q: float):
        self.Q += Q
        self.N += 1
    