Models are stored in models folder

Iteration 1: "q_leaning_model_cuda.pth"
input -> 1024 -> output


Iteration 2: "puzzle_model_complete.pth", "opening_learning_model.pth", "q_learning_model_final.pth"
input -> 1024 -> 512 -> output

"puzzle_model_complete.pth": trained on 50000 puzzles from kaggle dataset
"opening_learning_model.pth": trained on 1881 openings and 50000 puzzles 
"q_learning_model_final.pth":" trained on 20057 chess games, 1881 openings, 50000 puzzles

Extra info:
training on games
Epoch 1 completed. Average loss: 31.6136
Epoch 2 completed. Average loss: 31.6136
Epoch 3 completed. Average loss: 31.6136
Epoch 4 completed. Average loss: 31.6136
Epoch 5 completed. Average loss: 31.6136
Epoch 6 completed. Average loss: 31.6136
Epoch 7 completed. Average loss: 31.6136
Epoch 8 completed. Average loss: 31.6136
Epoch 9 completed. Average loss: 31.6136
did not learn from the games 