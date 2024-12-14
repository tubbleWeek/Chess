import chess
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



def alpha_beta_pruning(board, depth, alpha, beta, maximizing_player, evaluate):
    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = alpha_beta_pruning(board, depth - 1, alpha, beta, False, evaluate)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = alpha_beta_pruning(board, depth - 1, alpha, beta, True, evaluate)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move