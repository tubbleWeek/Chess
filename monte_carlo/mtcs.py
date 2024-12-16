import chess
import math
import random
import concurrent.futures
import threading
import logging

# logging.basicConfig(level=logging.INFO)

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
        return all(move in tried_moves for move in all_moves)

    def best_child(self, exploration_weight=1.41, temperature=1.0):
        if not self.children:
            logging.error("No children found in best_child.")
            return None

        return max(
            self.children,
            key=lambda child: (
                child.wins / (child.visits + 1e-6)
                + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            )
        )

    def expand(self):
        tried_moves = {child.state.peek() for child in self.children}
        untried_moves = [move for move in self.state.legal_moves if move not in tried_moves]

        for move in untried_moves:
            new_state = self.state.copy()
            new_state.push(move)
            new_node = Node(new_state, self)
            self.children.append(new_node)

        return untried_moves


def evaluate_material(board: chess.Board):
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    value = 0

    for square, piece in board.piece_map().items():
        if piece.piece_type not in piece_values:
            continue
        value += piece_values[piece.piece_type] if piece.color == chess.WHITE else -piece_values[piece.piece_type]

    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0.5
    return value


def heuristic_playout(state: chess.Board):
    board = state.copy()
    while not board.is_game_over():
        moves = list(board.legal_moves)
        if not moves:
            break

        # Prioritize checks and captures to increase aggression
        prioritized_moves = []
        for move in moves:
            board.push(move)
            score = evaluate_material(board)
            prioritized_moves.append((score, move))
            board.pop()

        # Select the best aggressive move or fallback to random
        prioritized_moves.sort(key=lambda x: x[0], reverse=board.turn == chess.WHITE)
        _, best_move = prioritized_moves[0]
        board.push(best_move)

    result = board.result()
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    return 0.5


root_lock = threading.Lock()


def mcts(root: Node, num_simulations: int = 1000, num_threads: int = 4):
    def single_simulation():
        current_node = root
        while current_node.is_fully_expanded() and current_node.children:
            current_node = current_node.best_child()

        if not current_node.state.is_game_over():
            untried_moves = current_node.expand()
            if untried_moves:
                move = random.choice(untried_moves)
                new_state = current_node.state.copy()
                new_state.push(move)
                new_node = Node(new_state, current_node)
                current_node.children.append(new_node)
                current_node = new_node

        result = heuristic_playout(current_node.state)
        with root_lock:
            while current_node:
                current_node.visits += 1
                current_node.wins += result if current_node.state.turn == chess.WHITE else -result
                current_node = current_node.parent

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(single_simulation) for _ in range(num_simulations)]
        concurrent.futures.wait(futures)

    best_child = root.best_child(exploration_weight=0)
    if best_child is None:
        logging.error("MCTS failed to find a valid move.")
        return None
    return best_child.state.peek()


# if __name__ == "__main__":
#     board = chess.Board()
#     root = Node(board)

#     best_move = mcts(root, num_simulations=100, num_threads=4)
#     if best_move:
#         print(f"Best Move: {best_move}")
#     else:
#         print("No valid move found.")
