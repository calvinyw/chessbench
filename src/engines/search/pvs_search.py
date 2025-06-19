import chess
import math
from typing import Optional, Dict
from collections import defaultdict
from enum import Enum, auto

from .base import SearchAlgorithm, SearchResult
from searchless_chess.src.engines.utils.node import Node, TTEntry
from searchless_chess.src.engines.utils.nnutils import reduced_fen
import searchless_chess.src.data_loader as data_loader
import numpy as np


class NodeType(Enum):
    PV_NODE = auto()
    CUT_NODE = auto()
    ALL_NODE = auto()


NULL_EPS = 0.0001


class PVSSearch(SearchAlgorithm):
    """Principal Variation Search with policy-based move ordering and depth extension."""
    
    def __init__(self):
        super().__init__("pvs")
        self.tt_hits = 0  # Track transposition table hits
        self.root_depth = 0.0  # Store the root depth

    def reset_metrics(self):
        """Reset engine metrics."""
        self.metrics = {
            'num_nodes': 0,
            'num_searches': 0,
            'bf': 0,
            'depth': 0,
            'parent_nodes': 0,
            'pv': 0,
        }
    
    def search(self, board: chess.Board, inference_func, batch_inference_func=None, depth=2.0, **kwargs) -> SearchResult:
        """
        Perform PVS search with iterative deepening.
        """
        num_nodes = kwargs.get('num_nodes', 400)
        
        # Store inference function for use in node creation
        self.inference_func = inference_func
        self.tt_hits = 0  # Reset TT hit counter
        self.root_depth = depth  # Store the initial depth
        
        # Create root node
        root = self._create_node(board, inference_func)
        history = defaultdict(int)
        tt = defaultdict(lambda: None)
        
        start_depth = 2.0
        node_count = self.metrics['num_nodes']
        current_depth = start_depth
        best_score = None
        best_move = None
        
        # Iterative deepening
        while self.metrics['num_nodes'] - node_count < num_nodes * 0.95 and current_depth < 20:
            f = best_score - NULL_EPS / 2 if best_score is not None else None
            score, move = self._pvs_root(root, current_depth, history, tt, f)
            
            if move is not None:
                best_score = score
                best_move = move
            
            current_depth += 0.2
        
        self.metrics['depth'] = current_depth
        
        if best_move is None:
            # Fallback - just pick the first move from policy
            if root.policy:
                best_move = root.policy[0][0]
            else:
                best_move = list(board.legal_moves)[0]

        self.metrics['bf'] = self.metrics['num_nodes'] / max(1, self.metrics['parent_nodes'])

        pv = root
        self.metrics['pv'] += 1
        while not pv.is_terminal() and not pv.is_leaf():
            self.metrics['pv'] += 1
            pv = pv.policy[0][2]
        
        return SearchResult(
            move=best_move,
            score=best_score if best_score is not None else 0.0,
            metadata={
                'depth': current_depth,
                'nodes': self.metrics['num_nodes'],
                'bf': self.metrics['bf'],
                'tt_hits': self.tt_hits,
                'tt_entries': len([entry for entry in tt.values() if entry is not None]),
                'pv': self.metrics['pv']
            }
        )
    
    def _pvs_root(self, root: Node, depth: float, history: Dict[str, int], tt: Dict[str, TTEntry], f=None) -> tuple[float, Optional[chess.Move]]:
        """Root level PVS call."""
        return self._pvs(root, depth, -1.0, 1.0, history, tt, NodeType.PV_NODE, 0)
    
    def _pvs(self, node: Node, depth: float, alpha: float, beta: float, 
             history: Dict[str, int], tt: Dict[str, TTEntry], 
             node_type: NodeType = NodeType.PV_NODE, rec_depth: int = 0, soft_create: bool = False) -> tuple[float, Optional[chess.Move]]:
        """
        PVS with policy-based move ordering and depth extension.
        """
        board = node.board
        position_key = reduced_fen(board)

        if node.is_terminal():
            return node.value, None
        
        if history[position_key] >= 1:
            # This position has been seen before, so we can return a draw
            return 0.0, None
        
        # Query transposition table
        if position_key in tt and tt[position_key] is not None:
            tt_score = tt[position_key].query(alpha, beta, depth)
            if tt_score is not None:
                self.tt_hits += 1
                return tt_score, None

        # Multiply likelihood with the variance of this node
        depth_reduction = -2 * math.log(node.U + 1e-6)

        #this is the window of the depths we would like to end in: i.e. from depth_reduction to depth_reduction + depth_window.
        depth_window = -.4
        #this is the window of the depths we would like to end in: i.e. from depth_reduction to depth_reduction + depth_window.
        #the width of this window is roughly probability 2/3. So it is pretty narrow.

        #reduction_error is the difference between the depth_reduction between the node and its children
        reduction_error = -depth_reduction/20
        #putting at 5% the current depth reduction


        #compare entropy before versus after expanding:
        entropy_before = (self.root_depth -depth)
        entropy_after = 0
        weight_divisor = 1.0
        new_depths = []
        count = 0
        total_move_weight = 0
        unexpanded_count = 0
        for i, (move, move_weight, child_node) in enumerate(node.policy):
            assert move_weight > 0.0
            total_move_weight += move_weight
            # Compute new depth with policy extension
            new_depth = depth + math.log(move_weight + 1e-6)  - 0.1
            if new_depth < depth_reduction + depth_window + reduction_error and child_node is None:
                new_board = board.copy()
                new_board.push(move)
                if self._create_node(new_board, parent=node, tt=tt, soft_create=True) is None:
                    weight_divisor -= move_weight
            else:
                if child_node is None:
                    new_board = board.copy()
                    new_board.push(move)
                    if self._create_node(new_board, parent=node, tt=tt, soft_create=True) is None:
                        unexpanded_count +=1
                entropy_after += max((self.root_depth- depth_reduction), self.root_depth - new_depth)*move_weight
                new_depths.append(new_depth)
                total_move_weight += move_weight
                count += 1
        
        # Leaf node evaluation
        if node.is_leaf():
            if depth <= depth_reduction or entropy_before>entropy_after:
                return node.value, None
        
        # Safety check against excessive recursion
        if rec_depth > 50:
            return node.value, None

        if node.is_leaf():
            self.metrics['parent_nodes'] += 1
        
        max_eval = -float('inf')
        history[position_key] += 1
        
        total_move_weight = 0
        best_move_depth = None
        best_move = None
        original_alpha = alpha
        
        for i, (move, move_weight, child_node) in enumerate(node.policy):
            assert move_weight > 0.0

            # Compute new depth with policy extension
            new_depth = depth + math.log(move_weight + 1e-6) - math.log(weight_divisor + 1e-6) - 0.1

            if best_move_depth is None:
                best_move_depth = new_depth
            
            # Skip low probability moves if depth is too low
            if i>count and child_node is None:
                new_board = board.copy()
                new_board.push(move)
                if self._create_node(new_board, parent=node, tt=tt, soft_create=True) is None:
                    total_move_weight += move_weight
                    continue
            
            # Create child node if needed
            if child_node is None:
                child_board = board.copy()
                child_board.push(move)
                child_node = self._create_node(child_board, inference_func=self.inference_func, parent=node, tt=tt)
                node.add_child(child_node, move)

            child_re_searches = 0
            RE_SEARCH_DEPTH = 0.2
            
            while True:
                # Use full window for first move, null window for others
                search_alpha = -beta if i == 0 else -alpha - NULL_EPS
                search_beta = -alpha
                
                # Determine child node type
                if i == 0 and node_type == NodeType.PV_NODE:
                    child_node_type = NodeType.PV_NODE
                elif node_type == NodeType.CUT_NODE:
                    child_node_type = NodeType.ALL_NODE
                else:
                    child_node_type = NodeType.CUT_NODE
                
                score, _ = self._pvs(
                    child_node, 
                    new_depth, 
                    search_alpha,
                    search_beta,
                    history,
                    tt,
                    child_node_type,
                    rec_depth + 1
                )
                score = -score  # Negate for current player's perspective

                if i > 0 and score > alpha:
                    # This move improved alpha
                    
                    if new_depth < best_move_depth + RE_SEARCH_DEPTH:
                        # Re-search with deeper depth
                        new_depth = min(new_depth + RE_SEARCH_DEPTH, best_move_depth + RE_SEARCH_DEPTH)
                        child_re_searches += 1
                        continue
                    else:
                        if node_type == NodeType.PV_NODE:
                            # Re-search with full window
                            score, _ = self._pvs(
                                child_node, 
                                new_depth, 
                                -beta, 
                                -alpha,
                                history,
                                tt,
                                NodeType.PV_NODE,
                                rec_depth + 1
                            )
                        else:
                            # Re-search with full window
                            score, _ = self._pvs(
                                child_node, 
                                new_depth, 
                                -beta, 
                                -alpha,
                                history,
                                tt,
                                child_node_type,
                                rec_depth + 1
                            )
                        score = -score
                break

            # Update policy if re-searches occurred
            if child_re_searches > 0:
                if node_type == NodeType.CUT_NODE and score > alpha:
                    # This is a CUT node and the move is best, so have to update the policy quickly
                    new_policy = node.policy[i][1] * math.exp(child_re_searches * RE_SEARCH_DEPTH)
                else:
                    # This is a PV or ALL node, so we can afford to do a slower update
                    new_policy = node.policy[i][1] + 0.1
                    if not (score > alpha):
                        # This is not the best move, so clip the policy
                        clip = max(node.policy[0][1] * 0.98, node.policy[i][1])
                        new_policy = min(new_policy, clip)
                
                node.policy[i] = (move, new_policy, child_node)
                node.sort_and_normalize()
                best_move_depth = max(best_move_depth, new_depth)

            if score > max_eval:
                max_eval = score
                best_move = move
            
            # Update alpha for pruning
            alpha = max(alpha, max_eval)
            
            # Beta cutoff
            if alpha >= beta:
                break

            total_move_weight += move_weight

        history[position_key] -= 1
        
        # Store result in transposition table
        if position_key in tt and tt[position_key] is not None:
            if max_eval <= original_alpha:
                # Fail low - upper bound
                tt[position_key].store_upper_bound(max_eval, depth)
            elif max_eval >= beta:
                # Fail high - lower bound
                tt[position_key].store_lower_bound(max_eval, depth)
            else:
                # Exact score
                tt[position_key].store_exact(max_eval, depth)
        
        return max_eval, best_move
    
    def _create_node(self, board: chess.Board, inference_func=None, parent: Optional[Node] = None, tt: Dict[str, TTEntry] = None, soft_create: bool = False) -> Node | None:
        """Create a node with static evaluation and policy."""
        position_key = reduced_fen(board)
        
        # Check for terminal conditions
        terminal_value = None
        if board.is_checkmate():
            terminal_value = -1.0
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            terminal_value = 0.0

        if terminal_value is not None:
            return Node(board=board, parent=parent, value=terminal_value, terminal=True)

        # Check transposition table for cached evaluation
        if tt is not None and position_key in tt and tt[position_key] is not None:
            tt_entry = tt[position_key]
            return Node(board=board, parent=parent, value=tt_entry.static_value, policy=tt_entry.policy, U=tt_entry.U)
        
        if soft_create:
            return None
        
        self.metrics['num_nodes'] += 1
        
        # Get model evaluation if inference function is available
        if inference_func:
            output = inference_func(board)
            value = output['value'][0, 0].item() * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
            
            # Get policy for move ordering
            from searchless_chess.src.engines.utils.nnutils import get_policy
            policies = output['hardest_policy'].float().cpu().numpy()
            policy, _, perplexity = get_policy(board, policies[0])

            D = output['draw'][0, 0].item()

            hl_logits = output['hl'].float().cpu().numpy()[0]  # Shape: (81,)
            hl_probs = np.exp(hl_logits - np.max(hl_logits))  # Softmax numerically stable
            hl_probs = hl_probs / np.sum(hl_probs)
            
            # Bin centers for NUM_BINS evenly spaced intervals in [0, 1]
            bin_centers = np.array([(2 * i + 1) / (2 * data_loader.NUM_BINS) for i in range(data_loader.NUM_BINS)])
            
            # Compute variance: E[X^2] - E[X]^2
            hl_mean = np.sum(hl_probs * bin_centers)
            hl_variance = np.sum(hl_probs * bin_centers**2) - hl_mean**2
            
            wdl_variance = math.sqrt(max(0, hl_variance * 4))
        else:
            # Default values for when inference_func is not available
            value = 0.0
            policy = [(move, 1.0/len(list(board.legal_moves))) for move in board.legal_moves]
        
        new_node = Node(board=board, parent=parent, value=value, policy=policy, U=wdl_variance)

        # Store static evaluation in transposition table
        if tt is not None:
            tt[position_key] = TTEntry(static_value=value, policy=policy, U=wdl_variance)

        return new_node 