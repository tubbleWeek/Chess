import chess
import chess.engine
from alpha_beta.stockFish import alpha_beta_pruning

def evaluate_board(board):
    """
    Simple evaluation function. Assigns material points.
    """
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    score = 0
    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return score

def play_human_vs_ai():
    board = chess.Board()
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
            _, ai_move = alpha_beta_pruning(board, 3, float('-inf'), float('inf'), True, evaluate_board)
            board.push(ai_move)
            print(f"AI plays: {ai_move}")
    print("Game Over!")
    print(f"Result: {board.result()}")

def play_ai_vs_ai(engine_path):
    """
    Plays alpha-beta AI against an external engine.
    """
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    while not board.is_game_over():
        print(board)
        if board.turn:
            # Alpha-beta AI's turn
            _, ai_move = alpha_beta_pruning(board, 3, float('-inf'), float('inf'), True, evaluate_board)
            board.push(ai_move)
            print(f"Alpha-Beta AI plays: {ai_move}")
        else:
            # External engine's turn
            result = engine.play(board, chess.engine.Limit(time=1))
            board.push(result.move)
            print(f"External Engine plays: {result.move}")

    print("Game Over!")
    print(f"Result: {board.result()}")
    engine.quit()

if __name__ == "__main__":
    mode = input("Choose mode: (1) Human vs AI, (2) AI vs AI: ")
    if mode == "1":
        play_human_vs_ai()
    elif mode == "2":
        engine_path = input("Path to external engine (e.g., stockfish): ")
        play_ai_vs_ai(engine_path)
    else:
        print("Invalid choice.")