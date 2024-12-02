import chess

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