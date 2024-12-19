import chess
import chess.engine
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper functions for chess encoding
def board_to_tensor(board: chess.Board):
    """Encodes the board into a tensor."""
    board_tensor = np.zeros(773)  # 64 squares * 12 pieces + 5 metadata
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Encode piece type and color
            piece_offset = (piece.piece_type - 1) + (6 if piece.color == chess.BLACK else 0)
            board_tensor[square * 12 + piece_offset] = 1
    # Add metadata
    board_tensor[-5] = board.turn  # 1 if white's turn, else 0
    board_tensor[-4] = board.has_kingside_castling_rights(chess.WHITE)
    board_tensor[-3] = board.has_queenside_castling_rights(chess.WHITE)
    board_tensor[-2] = board.has_kingside_castling_rights(chess.BLACK)
    board_tensor[-1] = board.has_queenside_castling_rights(chess.BLACK)
    return torch.tensor(board_tensor, dtype=torch.float32)

# Define the neural network for Q-learning
class ChessQNetwork(nn.Module):
    def __init__(self, input_size=773, hidden_size=2048, output_size=1):
        super(ChessQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # Output Q-value for the state
        return x



class q_learning_engine():
    def __init__(self, engine_path):
        self.q_network = ChessQNetwork().to(DEVICE)
        self.q_network.load_state_dict(torch.load(engine_path, map_location=DEVICE))

    def select_move(self, board, epsilon, temperature=1.0):
        """Select a move using a softmax policy for better exploration-exploitation balance."""
        legal_moves = list(board.legal_moves)

        if random.random() < epsilon:
            # Choose a random move
            return random.choice(legal_moves)
        else:
            # Evaluate all moves
            board_tensors = []
            for move in legal_moves:
                board_copy = board.copy()
                board_copy.push(move)
                board_tensors.append(board_to_tensor(board_copy).to(DEVICE))

            # Batch process Q-values
            board_tensors = torch.stack(board_tensors)
            q_values = self.q_network(board_tensors).detach().cpu().numpy()

            # Softmax sampling for exploration
            probabilities = np.exp(q_values / temperature) / np.sum(np.exp(q_values / temperature))
            return random.choices(legal_moves, probabilities)[0]
        

        
# # Initialize the Q-network, optimizer, and memory


# q_network.load_state_dict(torch.load("./r_learning/puzzle_model_1.pth"))



# def train_q_network(batch):
#     """Train the Q-network using a batch of experiences."""
#     states, actions, rewards, next_states, dones = zip(*batch)

#     states = torch.stack(states)
#     rewards = torch.tensor(rewards, dtype=torch.float32)
#     next_states = torch.stack(next_states)
#     dones = torch.tensor(dones, dtype=torch.float32)

#     # Compute Q(s, a) and target Q-values
#     q_values = q_network(states).squeeze()
#     with torch.no_grad():
#         next_q_values = q_network(next_states).squeeze()
#         target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

#     # Compute loss and backpropagate
#     loss = nn.MSELoss()(q_values, target_q_values)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


# # RL Training Loop
# epsilon = EPSILON_START
# for episode in tqdm.tqdm(range(NUM_EPISODES)):
#     board = chess.Board()
#     state = board_to_tensor(board)
#     done = False

#     while not done:
#         # Select a move
#         move = select_move(board, epsilon)
#         board.push(move)

#         # Observe reward and next state
#         reward = 0
#         if board.is_checkmate():
#             reward = 1 if board.turn == chess.BLACK else -1  # Current player loses
#         elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
#             reward = 0  # Draw
#         next_state = board_to_tensor(board)
#         done = board.is_game_over()

#         # Store experience in memory
#         memory.append((state, move, reward, next_state, done))

#         # Sample a batch and train
#         if len(memory) >= BATCH_SIZE:
#             batch = random.sample(memory, BATCH_SIZE)
#             train_q_network(batch)

#         state = next_state

#     # Decay epsilon
#     epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

# print("Training complete. The Q-learning chess engine is ready!")
