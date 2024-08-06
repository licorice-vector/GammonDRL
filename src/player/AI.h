#ifndef AI_H
#define AI_H

#include "Player.h"
#include "../model/Model.h"

namespace Backgammon {
    class AI : public Player {
    public:
        std::string name;
        std::shared_ptr<Model> model;
        AI(std::string name, std::shared_ptr<Model> model);
        int choose_move(const State& state, const Dice& dice, const Moves& moves);
        void new_game();
        void no_moves(const State& state);
        void game_over(const State& state, int player);
    };
}

#endif
