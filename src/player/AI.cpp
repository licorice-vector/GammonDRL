#include "AI.h"

namespace Backgammon {
    AI::AI(std::string name, std::shared_ptr<Model> model) : name(name), model(model) {}
    
    int AI::choose_move(const State& state, const Dice& dice, const Moves& moves) {
        int index = model->choose_move(state, dice, moves);
        assert(index < (int)moves.size());
        return index;
    }

    void AI::new_game() {}

    void AI::no_moves(const State& state) {}
    
    void AI::game_over(const State& state, int player) {}
}
