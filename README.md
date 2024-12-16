# Chess

This is a project for umn CSCI4511w. We were tasked to create a rational agent for a game.  For the game we chose chess, and in this repo there are three different agents: Alpha-Beta pruning, Monte-Carlo Tree Search, and Reinforcement leaning. This was built using the chess package from python.

The rankings for our chess AI are Alpha-Beta > Reinforcement learning > Monte-Carlo Tree Search

# Use
To play against the Alpha-Beta pruning AI you can run the game.py script. Input moves can be shorthand or UCI format.


To play AIs against each other you schold run the testing engines script. Currently it will play games with AIs face against each other for 50 games; 25 as white and 25 as black. The final board state and moves are logged using the logging package.