import pandas as pd
import chess
import ast

# dataset_path = "./chess_data/high_elo_opening.csv"
# # data = pd.read_csv(dataset_path)
# output_path = "./chess_data/filtered_openings.csv"

dataset_path = "./chess_data/lichess_puzzle_transformed.csv"
output_path = "./chess_data/filtered_chess_puzzle_large.csv"

data = pd.read_csv(dataset_path)

# data = data[['winner', 'moves']]
# uci_moves_list = []

# for idx, row in data.iterrows():
#     moves = row["moves"].split()
#     board = chess.Board()
#     uci_moves = []
#     for move in moves:
#         # Convert shorthand move to UCI
#         uci_move = board.parse_san(move).uci()
#         uci_moves.append(uci_move)
#         board.push_uci(uci_move)
#     uci_moves_list.append(" ".join(uci_moves))
#     # print(row)

# data["moves"] = uci_moves_list
# data.to_csv(output_path, index=True, header=True)









# data = pd.read_csv(output_path)
# uci_moves_list = []

# for idx, move_list in data.iterrows():

#     moves = ast.literal_eval(move_list["moves_list"])

#     board = chess.Board()
#     uci_moves = []
    

#     for move in moves:
#         try:
#             move = move.split('.')[-1]
#             # Convert shorthand move to UCI
#             uci_move = board.parse_san(move).uci()
#             uci_moves.append(uci_move)
#             board.push_uci(uci_move)
#             # board.push_uci(uci_move)
#         except:
#             break
#     uci_moves_list.append(" ".join(uci_moves))

# uci_data = pd.DataFrame({"uci_moves": uci_moves_list})
# uci_data.to_csv(output_path, index=True, header=True)

    
# data = pd.read_csv(output_path)
# holder = []
# for idx, move_list in data.iterrows():
#     # moves = ast.literal_eval(move_list["uci_moves"])
#     # print(moves)
#     # break
#     moves = str(move_list["uci_moves"])
#     print(moves)
#     if moves == "nan":
#         continue
#     else:
#         holder.append(moves)
# # print(holder)
# data = pd.DataFrame({"uci_moves": holder})
# data.to_csv(output_path, index=True, header=True)






# data = data[['moves_list']]
# , 'move1w', 'move1b', 'move2w', 'move2b', 'move3w', 'move3b', 'move4w', 'move4b', 'perc_white_win', 'perc_black_win'
# for idx, game in data.iterrows():
#     if idx == 1:
#         break
#     print(game)
# data.to_csv(output_path, index=True)
# print(f"Filtered dataset saved to {output_path}")



# Keep only the necessary columns
data = data[['Unnamed: 0', 'FEN', 'Moves']]

# Optionally, rename 'Unnamed: 0' to 'Index'
data.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)

# Save the cleaned dataset to a new CSV file
# output_path = "./chess_data/filtered_chess_data_large.csv"
subset_data = data.head(1000000)
subset_data.to_csv(output_path, index=False)
