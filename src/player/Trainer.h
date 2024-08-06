#ifndef TRAINER_H
#define TRAINER_H

#include "Player.h"
#include "../model/Model.h"

namespace Backgammon {
    class Trainer : public Player {
    public:
        std::string name;
        std::shared_ptr<Model> model;
        Trainer(std::string name, std::shared_ptr<Model> model);
        int choose_move(const State& state, const Dice& dice, const Moves& moves);
        void new_game();
        void no_moves(const State& state);
        void game_over(const State& state, int player);
    };
}

#endif
