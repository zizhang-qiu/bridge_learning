The directory contains the learning environment of the game of Contract Bridge and several learning algorithms for bidding and playing phase.

# Overview
[bridge_lib](./bridge_lib/) consists of the learning environment of  the game of Contract Bridge written in C++. The code is mostly inspired by [OpenSpiel](https://github.com/google-deepmind/open_spiel) and [hanabi-learning-environment](https://github.com/google-deepmind/hanabi-learning-environment).

[pybridge](./pybridge/) consists of the pybind file which makes it available to use C++ code in python.

[pysrc](./pysrc/) consists of python code including PyTorch models and other algorithms for Contract Bridge.

[rela  (REinforcement Learning Assemly) ](./rela/) is a library to is a set of tools for efficient batched neural network inference written in C++ with multi-threading from [off-belief-learning](https://github.com/facebookresearch/off-belief-learning).

[rlcc](./rlcc/) consists of some reinforcement learning items.

[playcc](./playcc/) consists some algorithms for card playing (PIMC and alpha_mu).

# Details of Bridge Learning Environment
The bridge learning environment is made up mainly by following components
  
- BridgeGame
- BridgeCard
- BridgeHand
- BridgeMove
- BridgeState
- BridgeObservation
- Encoder

The BridgeGame is a class which creates specific game with parameters including

- vulnerability
- dealer
- random seed
  
Once you created a BridgeGame instance, you can use it to create a BridgeState, e.g.
```
auto game = std::make_shared<BridgeGame>({});
auto state = BridgeState(game);
```

Each move in bridge including deal, bidding and playing is an instance of BridgeMove. 

The moves are divided into 2 main categories: chances (deal) and general moves (bidding and playing). 

For deal and playing moves, the uids are indexed as follows:

0-C2, 1-D2, 2-H2, 3-S2, 4-C3, ...., 51-SA.

The only difference is when you get the move, you should choose between `GetChanceOutcome()` and `GetMove()`.

The uids of bidding moves are:
52-Pass, 53-Double, 54-Redouble, 55-1C, 56-1D, 57-1H, 58-1S, 59-1NT, ..., 89-7NT.

You should use `GetMove()` to get bidding moves and `ApplyMove()` to apply a move.

A full example:

```C++
auto game = std::make_shared<BridgeGame>({});
auto state = BridgeState(game);
while state.IsChanceNode(){
    state.ApplyRandomChance(); // Deal random cards.
}
// Bid 1C, Pass, Pass, Pass.
for(const int uid: {55, 52, 52, 52}){
    auto move = state.GetMove(uid);
    state.ApplyMove(move);
}
// Apply the first move (card to play) until game ends.
while(!state.IsTerminal()){
    const auto legal_moves = state.LegalMoves();
    state.ApplyMove(legal_moves[0]);
}
```

