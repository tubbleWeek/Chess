import pandas as pd

dataset_path = "./chess_data/high_elo_opening.csv"
data = pd.read_csv(dataset_path)
output_path = "./chess_data/filtered_openings.csv"
data = data[['moves_list', 'move1w', 'move1b', 'move2w', 'move2b', 'move3w', 'move3b', 'move4w', 'move4b', 'perc_white_win', 'perc_black_win']]

# for idx, game in data.iterrows():
#     if idx == 1:
#         break
#     print(game)
data.to_csv(output_path, index=True)
print(f"Filtered dataset saved to {output_path}")



# Keep only the necessary columns
# data = data[['Unnamed: 0', 'FEN', 'Moves']]

# Optionally, rename 'Unnamed: 0' to 'Index'
# data.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)

# Save the cleaned dataset to a new CSV file
# output_path = "./chess_data/filtered_chess_data.csv"
# subset_data = data.head(50000)
# subset_data.to_csv(output_path, index=False)
