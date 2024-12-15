import chess
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
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5,  10, 25, 25, 10,  5,  5,
        0,  0,  0,  20, 20,  0,  0,  0,
        5, -5, -10, 0,  0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
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