import pandas as pd
import chess
import torch
from q_learn import ChessQNetwork, board_to_tensor

# Pretraining parameters
PRETRAIN_EPOCHS = 10
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Q-network and optimizer
pretrain_q_network = ChessQNetwork().to(DEVICE)
optimizer = torch.optim.Adam(pretrain_q_network.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

def pretrain_model(dataset_path):
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

        for _, game in data.iterrows():
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
                    uci_move = board.parse_san(move).uci()

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
        torch.save(pretrain_q_network.state_dict(), "q_learning_model_cuda"+ f"{epoch+1}" +".pth")
        print(f"Epoch {epoch + 1} completed. Average loss: {total_loss / len(data):.4f}")

    # Save the trained model
    torch.save(pretrain_q_network.state_dict(), "q_learning_model_cuda.pth")
    print("Model saved as 'q_learning_model_cuda.pth'")


if __name__ == "__main__":
    dataset_path = "./chess_data/games.csv"
    pretrain_model(dataset_path)