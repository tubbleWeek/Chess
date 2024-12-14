import chess
import math
import random
import concurrent.futures
# from joblib import Parallel, delayed
import threading
import logging

logging.basicConfig(level=logging.INFO)

class Node:
    def __init__(self, state: chess.Board, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
    
    def is_fully_expanded(self):
        tried_moves = {child.state.peek() for child in self.children}
        all_moves = set(self.state.legal_moves)
        logging.debug(f"Legal moves: {all_moves}")
        logging.debug(f"Tried moves: {tried_moves}")
        is_expanded = all(move in tried_moves for move in all_moves)
        logging.debug(f"Is fully expanded: {is_expanded}")
        return is_expanded


    # def is_fully_expanded(self):
    #     tried_moves = {child.state.peek() for child in self.children}
    #     return all(move in tried_moves for move in self.state.legal_moves)

    def best_child(self, exploration_weight=1.41, temperature=1.0):
        if not self.children:
            # No children; cannot select the best child
            logging.error("No children found in best_child.")
            return None

        probabilities = [
            math.exp((child.wins / (child.visits + 1e-6)) / temperature) for child in self.children
        ]
        total = sum(probabilities)
        if total == 0:
            logging.debug("All children have zero visits.")
            return None  # Or choose a random child or other default behavior
        probabilities = [p / total for p in probabilities]
        return random.choices(self.children, probabilities)[0]


    def expand(self):
        tried_moves = {child.state.peek() for child in self.children}
        untried_moves = [move for move in self.state.legal_moves if move not in tried_moves]
        logging.debug(f"Untried moves: {untried_moves}")
        for move in untried_moves:
            new_state = self.state.copy()
            new_state.push(move)
            new_node = Node(new_state, self)
            self.children.append(new_node)
        logging.error(f"Children count after expansion: {len(self.children)}")
        return untried_moves


# def evaluate_material(board: chess.Board):
#     """Simple material evaluation heuristic."""
#     piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
#     value = 0

#     for square, piece in board.piece_map().items():
#         if piece.piece_type not in piece_values:
#             # logging.error(f"Unexpected piece type at square {square}: {piece}")
#             continue
#         value += piece_values[piece.piece_type] if piece.color == chess.WHITE else -piece_values[piece.piece_type]

#     return value
def evaluate_material(board: chess.Board):
    """Improved material evaluation heuristic."""
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    value = 0

    # Count material value
    for square, piece in board.piece_map().items():
        if piece.piece_type not in piece_values:
            logging.debug(f"Unexpected piece type at square {square}: {piece}")
            continue
        value += piece_values[piece.piece_type] if piece.color == chess.WHITE else -piece_values[piece.piece_type]

    # Additional evaluation: material advantage (e.g., piece mobility, king safety)
    # Placeholder for more sophisticated evaluation. Example: giving a small bonus for attacking moves.
    if board.is_checkmate():
        return 1 if board.turn == chess.WHITE else -1
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    else:
        return value

def heuristic_playout(state: chess.Board):
    board = state.copy()
    while not board.is_game_over():
        moves = list(board.legal_moves)
        if not moves:
            break  # No legal moves, end the playout
        
        move_scores = []
        for move in moves:
            board.push(move)
            score = evaluate_material(board)
            board.pop()
            move_scores.append((score, move))
        # logging.error(f"Legal moves: {moves}")
        # logging.error(f"Move scores: {move_scores}")
        if board.is_game_over():
            logging.error("Game over detected during playout.")
        _, best_move = max(move_scores, key=lambda x: x[0] if board.turn else -x[0])
        # logging.error(f"Playout selected move: {best_move}")
        board.push(best_move)
        
    result = board.result()
    logging.error(f"Playout result: {result}")
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    else:
        return 0.5


# def mcts(root: Node, num_simulations: int = 1000, num_threads: int = 4):
#     def single_simulation():
#         current_node = root
#         while current_node.is_fully_expanded() and current_node.children:
#             current_node = current_node.best_child()
#             if current_node is None:
#                 logging.error("Stopping simulation due to no valid child.")
#                 return

#         if not current_node.state.is_game_over():
#             untried_moves = current_node.expand()
#             if untried_moves:
#                 move = random.choice(untried_moves)
#                 new_state = current_node.state.copy()
#                 new_state.push(move)
#                 new_node = Node(new_state, current_node)
#                 current_node.children.append(new_node)
#                 current_node = new_node

#         result = heuristic_playout(current_node.state)
#         while current_node:
#             current_node.visits += 1
#             current_node.wins += result if current_node.state.turn == chess.WHITE else -result
#             current_node = current_node.parent

#     Parallel(n_jobs=num_threads)(delayed(single_simulation)() for _ in range(num_simulations))
#     best_child = root.best_child(exploration_weight=0)
#     if best_child is None:
#         logging.error("MCTS failed to find a valid move.")
#         return None
#     return best_child.state.peek()

root_lock = threading.Lock()

import concurrent.futures
import threading

# A lock to ensure thread-safety when updating shared statistics (visits, wins) in the root node.
root_lock = threading.Lock()

def mcts(root: Node, num_simulations: int = 1000, num_threads: int = 4):
    def single_simulation(root_copy):
        current_node = root_copy
        logging.debug(f"Starting simulation for node with state:\n{current_node.state}")
        
        # Expand until a fully expanded node is reached
        while current_node.is_fully_expanded() and current_node.children:
            current_node = current_node.best_child()
            logging.debug(f"Choosing best child node:\n{current_node.state}")

        if not current_node.state.is_game_over():
            untried_moves = current_node.expand()
            if untried_moves:
                logging.debug(f"Expanding node with {len(untried_moves)} untried moves.")
                move = random.choice(untried_moves)
                new_state = current_node.state.copy()
                new_state.push(move)
                new_node = Node(new_state, current_node)
                current_node.children.append(new_node)
                current_node = new_node
            else:
                logging.error(f"No untried moves available to expand further.")
            
        result = heuristic_playout(current_node.state)
        logging.debug(f"Playout result: {result}")
        
        # Update the root node's statistics in a thread-safe manner
        with root_lock:
            while current_node:
                current_node.visits += 1
                current_node.wins += result if current_node.state.turn == chess.WHITE else -result
                current_node = current_node.parent


    # Create copies of the root node for each thread to work with independently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(single_simulation, Node(root.state.copy())) for _ in range(num_simulations)]
        concurrent.futures.wait(futures)  # Wait for all threads to finish

    # After simulations, choose the best move from the root node
    best_child = root.best_child(exploration_weight=0)
    if best_child is None:
        logging.error("MCTS failed to find a valid move.")
        return None
    return best_child.state.peek()


if __name__ == "__main__":
    board = chess.Board()
    root = Node(board)

    best_move = mcts(root, num_simulations=100, num_threads=1)
    if best_move:
        print(f"Best Move: {best_move}")
    else:
        print("No valid move found.")