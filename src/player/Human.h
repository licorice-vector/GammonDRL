#ifndef HUMAN_H
#define HUMAN_H

#include "Player.h"

namespace Backgammon {
    class Human : public Player {
    public:
        std::string name;
        Human(std::string name);
        int choose_move(const State& state, const Dice& dice, const Moves& moves);
        void new_game();
        void no_moves(const State& state);
        void game_over(const State& state, int player);
    };
}

#endif
