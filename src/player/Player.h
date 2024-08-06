#ifndef PLAYER_H
#define PLAYER_H

#include "../game/Game.h"

namespace Backgammon {
    class CheckerMove;
    class State;
    class Dice;

    typedef std::multiset<CheckerMove> Move;
    typedef std::vector<Move> Moves;

    class Player {
    public:
        virtual int choose_move(const State& state, const Dice& dice, const Moves& moves) { assert(false); }
        virtual void new_game() { assert(false); }
        virtual void no_moves(const State& state) { assert(false); }
        virtual void game_over(const State& state, int player) { assert(false); }
    };
}

#endif
