#include "Trainer.h"

namespace Backgammon {
    Trainer::Trainer(std::string name, std::shared_ptr<Model> model) : name(name), model(model) {}

    int Trainer::choose_move(const State& state, const Dice& dice, const Moves& moves) {
        int index = model->choose_move(state, dice, moves);
        assert(index < (int)moves.size());
        model->update(state, moves[index]);
        return index;
    }

    void Trainer::new_game() {
        model->new_game();
    }

    void Trainer::no_moves(const State& state) {}
    
    void Trainer::game_over(const State& state, int player) {}
}
