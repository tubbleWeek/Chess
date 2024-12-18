import chess
import importlib
from alpha_beta import stockFish
import logging
from monte_carlo import mtcs
import r_learning.q_learn as q_learn

logging.basicConfig(
    filename="chess_game_log.txt",
    filemode="w",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)
NUM_GAMES = 50
def evaluate_board(board):
    """
    Advanced heuristic function for evaluating a chess position.
    :param board: chess.Board object representing the current position.
    :return: Evaluation score (positive for White's advantage, negative for Black's).
    """
    # Material values (including slight tweaks to emphasize piece importance)
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000  # King is invaluable, but this helps in endgame heuristics.
    }

    # Positional tables for various pieces
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

    KNIGHT_TABLE = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ]

    BISHOP_TABLE = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,   5,  10,  10,  10,  10,   5, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ]

    # Initialize score
    score = 0

    # Material evaluation
    for piece_type in piece_values.keys():
        white_pieces = board.pieces(piece_type, chess.WHITE)
        black_pieces = board.pieces(piece_type, chess.BLACK)
        score += piece_values[piece_type] * (len(white_pieces) - len(black_pieces))

        # Positional evaluation
        if piece_type == chess.PAWN:
            for square in white_pieces:
                score += PAWN_TABLE[square]
            for square in black_pieces:
                score -= PAWN_TABLE[chess.square_mirror(square)]
        elif piece_type == chess.KNIGHT:
            for square in white_pieces:
                score += KNIGHT_TABLE[square]
            for square in black_pieces:
                score -= KNIGHT_TABLE[chess.square_mirror(square)]
        elif piece_type == chess.BISHOP:
            for square in white_pieces:
                score += BISHOP_TABLE[square]
            for square in black_pieces:
                score -= BISHOP_TABLE[chess.square_mirror(square)]

    # Mobility evaluation
    white_mobility = len(list(board.legal_moves))
    board.turn = not board.turn
    black_mobility = len(list(board.legal_moves))
    board.turn = not board.turn
    score += 10 * (white_mobility - black_mobility)

    # King safety
    if not board.is_checkmate():
        white_king_safety = len(board.attackers(chess.BLACK, board.king(chess.WHITE)))
        black_king_safety = len(board.attackers(chess.WHITE, board.king(chess.BLACK)))
        score -= 30 * white_king_safety
        score += 30 * black_king_safety

    # Passed pawns
    for square in board.pieces(chess.PAWN, chess.WHITE):
        if not board.attackers(chess.BLACK, square):
            score += 50
    for square in board.pieces(chess.PAWN, chess.BLACK):
        if not board.attackers(chess.WHITE, square):
            score -= 50

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

model = q_learn.q_learning_engine("./r_learning/models/puzzle_model_1.pth")

def q_learning_engine(board):
    """Q-learning engine."""
    # model = torch.load('./r_learning/q_learning_model_final.pth')
    return model.select_move(board, epsilon=0)  # Use greedy policy (epsilon = 0) for evaluation



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

    # logging.info(f"MCTS vs Alpha-Beta")
    # logging.info(f"Games Played: {NUM_GAMES}")

    # logging.info("MCTS as white and AB as black")
    # for i in range(0, 24):
    #     play_game(mcts_engine, alpha_beta_engine)

    # logging.info("MCTS as black and AB as white")
    # for i in range(24,NUM_GAMES):
    #     play_game(alpha_beta_engine, mcts_engine)
    # logging.info(f"Q_Learn vs Alpha-Beta")
    # logging.info(f"Games Played: {NUM_GAMES}")

    # logging.info("Q_Learn as black and AB as white")
    # for i in range(0, 24):
    #     play_game(q_learning_engine, alpha_beta_engine)

    # logging.info("Q_Learn as white and AB as black")
    # for i in range(24,NUM_GAMES):
    #     play_game(alpha_beta_engine, q_learning_engine)

    logging.info(f"Q_Learn vs MCTS")
    logging.info(f"Games Played: {NUM_GAMES}")

    logging.info("Q_Learn as white and MCTS as black")
    for i in range(0, 24):
        play_game(q_learning_engine, mcts_engine)

    logging.info("Q_Learn as balck and MCTS as black")
    for i in range(24,NUM_GAMES):
        play_game(mcts_engine, q_learning_engine)
