import chess
import importlib
from alpha_beta import stockFish
from monte_carlo import mtcs
from q_learn import select_move, q_network, board_to_tensor


def evaluate(board):
    """Simple evaluation function for alpha-beta pruning."""
    if board.is_checkmate():
        return 1000 if board.turn == chess.BLACK else -1000
    return len(list(board.legal_moves)) if board.turn == chess.WHITE else -len(list(board.legal_moves))


def play_game(engine1, engine2, max_moves=100):
    """
    Play a game between two engines.
    :param engine1: Function representing engine1's move selection.
    :param engine2: Function representing engine2's move selection.
    :param max_moves: Maximum number of moves to avoid infinite games.
    :return: Result string (e.g., "1-0", "0-1", "1/2-1/2").
    """
    board = chess.Board()
    engines = [engine1, engine2]
    turn = 0

    print("Starting new game...")
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        print(board)
        print(f"\nTurn {board.fullmove_number}: {'White' if board.turn else 'Black'} to move.")

        move = engines[turn % 2](board)
        if move is None:
            print("No legal moves available!")
            break

        print(f"Engine {'1' if turn % 2 == 0 else '2'} plays: {move}")
        board.push(move)
        turn += 1

    print(board)
    print(f"Game Over. Result: {board.result()}")
    return board.result()


def alpha_beta_engine(board):
    """Alpha-beta pruning engine."""
    _, best_move = stockFish.alpha_beta_pruning(board, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=board.turn, evaluate=evaluate)
    return best_move


def mcts_engine(board):
    """MCTS engine."""
    root = mtcs.Node(board)
    return mtcs.mcts(root, num_simulations=500)


def q_learning_engine(board):
    """Q-learning engine."""
    return select_move(board, epsilon=0)  # Use greedy policy (epsilon = 0) for evaluation


if __name__ == "__main__":
    # Matchups: StockFish vs MCTS, StockFish vs Q-learning, MCTS vs Q-learning
    print("Match 1: Alpha-Beta Pruning vs MCTS")
    play_game(alpha_beta_engine, mcts_engine)

    print("\nMatch 2: Alpha-Beta Pruning vs Q-Learning")
    play_game(alpha_beta_engine, q_learning_engine)

    print("\nMatch 3: MCTS vs Q-Learning")
    play_game(mcts_engine, q_learning_engine)
