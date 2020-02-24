# q-learning with ocbo

This code applies concepts from OCBO ([Offline Contextual Bayesian Optimization](https://papers.nips.cc/paper/8711-offline-contextual-bayesian-optimization)) to reinforcement learning by choosing start states for each episode in a "smart" fashion: each episode starts at an "interesting" state that is expected to give high improvement, rather than choosing start states randomly.

The environment is tabular (discrete actions, discrete states), and a customizable Grid-World environment is implemented.
