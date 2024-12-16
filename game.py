import chess
import chess.engine
from alpha_beta.stockFish import alpha_beta_pruning
import torch
from r_learning.q_learn import select_move, q_network, board_to_tensor

# # new evalutation function for alpha beta pruning
# def evaluate(board):
#     """Simple evaluation function for alpha-beta pruning."""
#     if board.is_checkmate():
#         return 1000 if board.turn == chess.BLACK else -1000
#     return len(list(board.legal_moves)) if board.turn == chess.WHITE else -len(list(board.legal_moves))

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

def play_human_vs_ai():
    board = chess.Board()
    # model = torch.load('./r_learning/q_learning_model_final.pth')
    while not board.is_game_over():
        print(board)
        if board.turn:
            # Human plays
            move = input("Your move: ")
            try:
                board.push_san(move)
            except ValueError:
                print("Invalid move. Try again.")
        else:
            # AI plays
            _, ai_move = alpha_beta_pruning(board, 3, float('-inf'), float('inf'), maximizing_player=board.turn, evaluate=evaluate)
            board.push(ai_move)
            print(f"AI plays: {ai_move}")
    print("Game Over!")
    print(f"Result: {board.result()}")


if __name__ == "__main__":
    play_human_vs_ai()
