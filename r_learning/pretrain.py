import pandas as pd
import chess
import torch
from q_learn import ChessQNetwork, board_to_tensor

# datasets
OPENING_DATASET = "./chess_data/filtered_openings.csv"
PUZZLE_DATASET = "./chess_data/filtered_chess_puzzle.csv"
GAME_DATASET = "./chess_data/filtered_games.csv"

# Pretraining parameters
PRETRAIN_EPOCHS = 10
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Q-network and optimizer
pretrain_q_network = ChessQNetwork().to(DEVICE)
optimizer = torch.optim.Adam(pretrain_q_network.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

def train_model_puzzle(dataset_path):
    data = pd.read_csv(dataset_path)
    print("Dataset loaded. Number of games:", len(data))
    for epoch in range(PRETRAIN_EPOCHS):
        total_loss = 0  # Initialize total loss for the epoch
        for idx, game in data.iterrows():
            try:
                # Initialize board from FEN
                board = chess.Board(game["FEN"])
                moves = game["Moves"].split()
                reward = 1  # Reward for solving the puzzle

                states, next_states, rewards = [], [], []
                for i, move in enumerate(moves):
                    if board.is_game_over():
                        break

                    try:
                        # Convert shorthand to UCI
                        uci_move = board.parse_san(move).uci()
                    except ValueError:
                        print(f"Invalid move {move}. Skipping game.")
                        break

                    # Record the state and rewards
                    states.append(board_to_tensor(board).to(DEVICE))
                    rewards.append(reward if i > 0 else -reward)  # Penalize the starting state

                    # Push the move to the board
                    board.push_uci(uci_move)
                    next_states.append(board_to_tensor(board).to(DEVICE))

                # Train the Q-network on this game's data
                for i in range(len(states)):
                    state = states[i]
                    next_state = next_states[i]
                    reward = rewards[i]

                    # Compute Q-value target
                    with torch.no_grad():
                        target_q_value = reward
                        if i < len(states) - 1:  # No next state for the last move
                            next_q_value = pretrain_q_network(next_state).item()
                            target_q_value += DISCOUNT_FACTOR * next_q_value

                    # Forward pass
                    predicted_q_value = pretrain_q_network(state)
                    loss = loss_fn(predicted_q_value, torch.tensor([target_q_value], dtype=torch.float32).to(DEVICE))

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            except Exception as e:
                print(f"Skipping game due to error: {e}")

        torch.save(pretrain_q_network.state_dict(), "puzzle_model_"+ f"{epoch+1}" +".pth")
        print(f"Epoch {epoch + 1} completed. Average loss: {total_loss / len(data):.4f}")
        

def train_model_openings(dataset_path):
    data = pd.read_csv(dataset_path)
    print("Dataset loaded. Number of games:", len(data))

    #pretrained model
    pretrain_q_network = ChessQNetwork().to(DEVICE)
    pretrain_q_network.load_state_dict(torch.load("puzzle_model_complete.pth"))


    for epoch in range(PRETRAIN_EPOCHS):
        total_loss = 0  # Initialize total loss for the epoch
        update_count = 0
        for idx, game in data.iterrows():
            try:
                # Initialize board from FEN
                board = chess.Board()
                moves = game["uci_moves"].split()
                reward = 1  # Reward for solving the puzzle

                states, next_states, rewards = [], [], []
                for i, move in enumerate(moves):
                    if board.is_game_over():
                        break

                    # Record the state and rewards
                    states.append(board_to_tensor(board).to(DEVICE))
                    rewards.append(reward)  # Assume opening is always optimal

                    # Push the move to the board
                    board.push_uci(move)
                    next_states.append(board_to_tensor(board).to(DEVICE))

                # Train the Q-network on this game's data
                if len(states) > 0:
                    state_batch = torch.stack(states).to(DEVICE)
                    next_state_batch = torch.stack(next_states).to(DEVICE)
                    reward_batch = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

                    # Compute Q-values and targets
                    with torch.no_grad():
                        target_q_values = reward_batch
                        next_q_values = pretrain_q_network(next_state_batch).squeeze()
                        target_q_values[:-1] += DISCOUNT_FACTOR * next_q_values[:-1]

                    predicted_q_values = pretrain_q_network(state_batch).squeeze()

                    # Compute loss and backpropagate
                    loss = loss_fn(predicted_q_values, target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    update_count += len(states)

            except Exception as e:
                print(f"Skipping game at index {idx} due to error: {e}")
                
        if (epoch + 1) % 5 == 0 or epoch == PRETRAIN_EPOCHS - 1:
            torch.save(pretrain_q_network.state_dict(), f"opening_model_epoch_{epoch+1}.pth")
        print(f"Epoch {epoch + 1} completed. Average loss: {total_loss / len(data):.4f}")
    torch.save(pretrain_q_network.state_dict(), "opening_learning_model.pth")
    print("Model saved as 'opening_learning_model.pth'")

def train_model_games(dataset_path):
    """
    Pretrain the Q-learning model using a chess dataset.
    :param dataset_path: Path to the chess dataset CSV file.
    """
    # Load dataset
    data = pd.read_csv(dataset_path)
    print("Dataset loaded. Number of games:", len(data))

    for epoch in range(PRETRAIN_EPOCHS):
        print(f"Epoch {epoch + 1}/{PRETRAIN_EPOCHS}")
        total_loss = 0
        for idx, game in data.iterrows():
            try:
                moves = game["moves"].split()  # Get the sequence of shorthand moves
                winner = game["winner"]  # "white", "black", or "draw"
                reward = 1 if winner == "white" else -1 if winner == "black" else 0

                # Reconstruct the game
                board = chess.Board()
                states, next_states, rewards, dones = [], [], [], []

                for move in moves:
                    if board.is_game_over():
                        break

                    # Convert shorthand to UCI using `parse_san`
                    uci_move = move

                    # Record the state before the move
                    states.append(board_to_tensor(board).to(DEVICE))
                    rewards.append(reward if board.turn == chess.WHITE else -reward)

                    # Push the move to the board
                    board.push_uci(uci_move)
                    next_states.append(board_to_tensor(board).to(DEVICE))
                    dones.append(board.is_game_over())

                # Train the Q-network on this game's data
                for i in range(len(states)):
                    state = states[i]
                    next_state = next_states[i]
                    reward = rewards[i]
                    done = dones[i]

                    # Compute Q-value target
                    with torch.no_grad():
                        target_q_value = reward
                        if not done:
                            next_q_value = pretrain_q_network(next_state).item()
                            target_q_value += DISCOUNT_FACTOR * next_q_value

                    # Forward pass
                    predicted_q_value = pretrain_q_network(state)
                    loss = loss_fn(predicted_q_value, torch.tensor([target_q_value], dtype=torch.float32).to(DEVICE))

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            except Exception as e:
                # Log and skip any malformed games/moves
                print(f"Skipping game due to error: {e}")
        torch.save(pretrain_q_network.state_dict(), "q_learning_model"+ f"{epoch+1}" +".pth")
        print(f"Epoch {epoch + 1} completed. Average loss: {total_loss / len(data):.4f}")

    # Save the trained model
    torch.save(pretrain_q_network.state_dict(), "q_learning_model.pth")
    print("Model saved as 'q_learning_model.pth'")


if __name__ == "__main__":
    dataset_path = "./chess_data/games.csv"
    # train_model_puzzle(PUZZLE_DATASET)
    train_model_openings(OPENING_DATASET)
    # train_model_games(dataset_path)