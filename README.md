# Chess AI Project

This project was developed for the **UMN CSCI4511w** course. The task was to create a rational agent for a game, and for this project, we chose **Chess**. The repository contains three different chess AI agents that use different algorithms to make decisions:

- **Alpha-Beta Pruning**
- **Monte Carlo Tree Search (MCTS)**
- **Reinforcement Learning (RL)**

The project is built using the [python-chess](https://python-chess.readthedocs.io/en/latest/) package, which allows us to represent chess boards, generate legal moves, and interact with the game programmatically.

## AI Ranking

Based on our tests, we found the following performance ranking for the AI agents:

1. **Alpha-Beta Pruning**
2. **Reinforcement Learning**
3. **Monte Carlo Tree Search**

## Features

- **Alpha-Beta Pruning**: A classical search algorithm that uses pruning to optimize the evaluation of game trees, making it ideal for deterministic environments like chess.
- **Monte Carlo Tree Search (MCTS)**: A probabilistic algorithm that simulates random games to evaluate possible moves and is particularly useful in complex or uncertain scenarios.
- **Reinforcement Learning**: This AI learns by interacting with the environment, improving over time through self-play, training on chess puzzles, and game logs.

## Usage

### Play Against the Alpha-Beta AI

To play against the **Alpha-Beta Pruning** AI, run the `game.py` script. You can input moves using either **shorthand** notation (e.g., `e4`) or the **UCI** (Universal Chess Interface) format (e.g., `e2e4`).

```bash
python game.py
```
### Play AI vs AI

In addition to playing against the **Alpha-Beta Pruning** AI, you can also test different AI algorithms by having them play against each other. To do this, run the `testing_engines.py` script. This script will simulate **50 games** between two AI engines, with each engine playing 25 games as White and 25 games as Black.

The game results, including the moves made by each AI, the timestamps for each move, and the final game state, will be logged for further analysis. These logs provide valuable data on the performance of each algorithm in various scenarios.

To run AI vs AI matches, use the following command:

```bash
python testing_engines.py
```
