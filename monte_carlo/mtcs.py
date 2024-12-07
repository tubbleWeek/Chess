import chess
import chess.engine
import random
from collections import defaultdict
import math

class Node:
    def __init__(self, state: chess.Board, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.state.legal_moves))

    def best_child(self, exploration_weight=1.41):
        # UCB1 formula: value = wins/visits + exploration_weight * sqrt(log(parent.visits)/visits)
        return max(
            self.children,
            key=lambda child: (child.wins / (child.visits + 1e-6)) +
                              exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
        )


def random_playout(state: chess.Board):
    """Simulate a random playout from the current state."""
    board = state.copy()
    while not board.is_game_over():
        move = random.choice(list(board.legal_moves))
        board.push(move)
    result = board.result()  # e.g., "1-0", "0-1", "1/2-1/2"
    if result == "1-0":
        return 1  # White wins
    elif result == "0-1":
        return -1  # Black wins
    else:
        return 0  # Draw


def mcts(root: Node, num_simulations: int = 1000):
    """Perform Monte Carlo Tree Search starting from the root node."""
    for _ in range(num_simulations):
        # Selection
        current_node = root
        while current_node.is_fully_expanded() and current_node.children:
            current_node = current_node.best_child()

        # Expansion
        if not current_node.state.is_game_over():
            untried_moves = [move for move in current_node.state.legal_moves
                             if move not in [child.state.peek() for child in current_node.children]]
            if untried_moves:
                move = random.choice(untried_moves)
                new_state = current_node.state.copy()
                new_state.push(move)
                new_node = Node(new_state, current_node)
                current_node.children.append(new_node)
                current_node = new_node

        # Simulation
        result = random_playout(current_node.state)

        # Backpropagation
        while current_node:
            current_node.visits += 1
            current_node.wins += result if current_node.state.turn == chess.WHITE else -result
            current_node = current_node.parent

    # Return the best move
    return root.best_child(exploration_weight=0).state.peek()


def play_mcts_game(num_simulations=1000):
    """Play a game where both sides use the MCTS engine."""
    board = chess.Board()
    while not board.is_game_over():
        print(board)
        print("\nThinking...")
        root = Node(board)
        best_move = mcts(root, num_simulations=num_simulations)
        board.push(best_move)
        print(f"\nMove chosen: {best_move}\n")
    print(board)
    print(f"Game Over. Result: {board.result()}")


# Play a game
play_mcts_game(num_simulations=500)
