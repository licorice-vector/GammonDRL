#ifndef MODEL_H
#define MODEL_H

#include "../RevGrad/model/Model.h"
#include "../player/Player.h"

namespace Backgammon {
    class NeuralNetwork : public RevGrad::Model {
    public:
        RevGrad::Linear l1;
        RevGrad::Linear l2;
        NeuralNetwork();
        RevGrad::Tensor forward(RevGrad::Tensor x);
    };

    class Model {
    public:
        NeuralNetwork nn;
        std::vector<float> momentum;
        Model();
        void save(std::string filename);
        void load(std::string filename);
        RevGrad::Tensor tensor_from_state(const State& state);
        int choose_move(const State& state, const Dice& dice, const Moves& moves);
        void new_game();
        RevGrad::Tensor predict(const State& state);
        void update(const State& state, const Move& move);
    };
}

#endif
