# SAI Co-op Puzzle Competition: Reinforcement Learning and Behavioural Cloning

Run `./train.sh` to replicate my solution.
The script performs two steps:

1. Collect expert demonstrations from a sequence of shortest path algorithms
that complete the game perfectly.
2. Train a PPO reinforcement learning algorithm to replicate the expert policy
using behavioural cloning.

You might need to run this script a few times to get the shortest path algorithm
to work.