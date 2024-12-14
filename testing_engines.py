import chess
import importlib
import torch
# from IPython.display import display, SVG
from alpha_beta import stockFish
import logging
from monte_carlo import mtcs
# from r_learning.q_learn import select_move, q_network, board_to_tensor

logging.basicConfig(
    filename="chess_game_log.txt",
    filemode="w",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)
NUM_GAMES = 50
def evaluate_board(board):
    """
    Combined heuristic function for evaluating a chess position.
    :param board: chess.Board object representing the current position.
    :return: Evaluation score (positive for White's advantage, negative for Black's).
    """
    # Material values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King value is handled separately (e.g., king safety).
    }

    # Piece-square tables (simplified example for pawns)
    PAWN_TABLE = [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10,-20,-20, 10, 10,  5,
        5, -5,-10,  0,  0,-10, -5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5,  5, 10, 25, 25, 10,  5,  5,
       10, 10, 20, 30, 30, 20, 10, 10,
       50, 50, 50, 50, 50, 50, 50, 50,
        0,  0,  0,  0,  0,  0,  0,  0
    ]

    # Initialize score
    score = 0

    # Material evaluation
    for piece_type in piece_values.keys():
        # Count pieces for both sides
        white_pieces = len(board.pieces(piece_type, chess.WHITE))
        black_pieces = len(board.pieces(piece_type, chess.BLACK))
        score += piece_values[piece_type] * (white_pieces - black_pieces)

    # Positional evaluation (piece-square tables for pawns as an example)
    for square in board.pieces(chess.PAWN, chess.WHITE):
        score += PAWN_TABLE[square]
    for square in board.pieces(chess.PAWN, chess.BLACK):
        score -= PAWN_TABLE[chess.square_mirror(square)]  # Mirror for Black's pawns

    # Mobility evaluation
    white_mobility = len(list(board.legal_moves))
    board.turn = not board.turn  # Switch to Black
    black_mobility = len(list(board.legal_moves))
    board.turn = not board.turn  # Switch back to White
    score += 0.1 * (white_mobility - black_mobility)

    # King safety (penalize exposed kings)
    if not board.is_checkmate():
        white_king_safety = len(board.attackers(chess.BLACK, board.king(chess.WHITE)))
        black_king_safety = len(board.attackers(chess.WHITE, board.king(chess.BLACK)))
        score -= 0.3 * white_king_safety
        score += 0.3 * black_king_safety

    # Passed pawns (bonus for pawns that can promote without enemy pawns blocking)
    for square in board.pieces(chess.PAWN, chess.WHITE):
        if not any(board.pieces(chess.PAWN, chess.BLACK) & chess.SquareSet.ray(square, chess.H8)):
            score += 0.5
    for square in board.pieces(chess.PAWN, chess.BLACK):
        if not any(board.pieces(chess.PAWN, chess.WHITE) & chess.SquareSet.ray(square, chess.A1)):
            score -= 0.5

    return score


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
    # logging.info("Starting new game...")
    while not board.is_game_over() and board.fullmove_number <= max_moves:
        # print(board)
        # display(SVG(chess.svg.board(board)))  # Final board state
        print(f"\nTurn {board.fullmove_number}: {'White' if board.turn else 'Black'} to move.")

        move = engines[turn % 2](board)
        if move is None:
            print("No legal moves available!")
            logging.info("No legal moves available!")
            break

        print(f"Engine {'1' if turn % 2 == 0 else '2'} plays: {move}")
        logging.info(f"Engine {'1' if turn % 2 == 0 else '2'} plays: {move}")
        board.push(move)
        turn += 1
    # display(SVG(chess.svg.board(board)))  # Final board state
    # print(board)
    print(f"Game Over. Result: {board.result()}")
    logging.info(f"Final Board {board.board_fen()}")
    logging.info(f"Game Over. Result: {board.result()}")
    return board.result()


def alpha_beta_engine(board):
    """Alpha-beta pruning engine."""
    _, best_move = stockFish.alpha_beta_pruning(board, depth=4, alpha=float('-inf'), beta=float('inf'), maximizing_player=board.turn, evaluate=evaluate_board)
    return best_move


def mcts_engine(board):
    """MCTS engine."""
    root = mtcs.Node(board)
    return mtcs.mcts(root, num_simulations=1000)


# def q_learning_engine(board):
#     """Q-learning engine."""
#     # model = torch.load('./r_learning/q_learning_model_final.pth')
#     return select_move(board, epsilon=0)  # Use greedy policy (epsilon = 0) for evaluation


if __name__ == "__main__":
    # Matchups: StockFish vs MCTS, StockFish vs Q-learning, MCTS vs Q-learning
    # print("Match 1: Alpha-Beta Pruning vs MCTS")
    # play_game(alpha_beta_engine, mcts_engine)

    # print("\nMatch 2: Alpha-Beta Pruning vs Q-Learning")
    # logging.info("Match 2: Alpha-Beta Pruning vs Q-Learning")
    # play_game(alpha_beta_engine, q_learning_engine)
    # play_game(q_learning_engine, q_learning_engine)

    # print("\nMatch 3: MCTS vs Q-Learning")
    # play_game(mcts_engine, q_learning_engine)

    logging.info(f"MCTS vs Alpha-Beta")
    logging.info(f"Games Played: {NUM_GAMES}")

    logging.info("MCTS as white and AB as black")
    for i in range(0, 24):
        play_game(mcts_engine, alpha_beta_engine)

    logging.info("MCTS as white and AB as black")
    for i in range(24,NUM_GAMES):
        play_game(alpha_beta_engine, mcts_engine)

