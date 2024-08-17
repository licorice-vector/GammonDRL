#include "Model.h"

#define INPUT_FEATURES 201

namespace Backgammon {
    NeuralNetwork::NeuralNetwork(int hidden_units) {
        l1 = RevGrad::Linear(this, INPUT_FEATURES, hidden_units);
        l2 = RevGrad::Linear(this, hidden_units, 1);
    }

    RevGrad::Tensor NeuralNetwork::forward(RevGrad::Tensor x) {
        x = l1(x);
        x = RevGrad::Tensor::relu(x);
        x = l2(x);
        x = RevGrad::Tensor::sigmoid(x);
        return x;
    }

    Model::Model(int hidden_units) : nn(NeuralNetwork(hidden_units)) {}

    void Model::save(std::string filename) { nn.save_parameters(filename); }
    void Model::load(std::string filename) { nn.load_parameters(filename); }

    RevGrad::Tensor Model::tensor_from_state(const State& state) {
        std::vector<float> values;
        values.reserve(INPUT_FEATURES);
        for (int player = 0; player <= 1; player++) {
            for (int point = 0; point < BOARD_SIZE; point++) {
                int n = state.on[player][point];
                if (n == 0) {
                    values.push_back(0.0f);
                    values.push_back(0.0f);
                    values.push_back(0.0f);
                    values.push_back(0.0f);
                    continue;
                }
                if (n == 1) {
                    values.push_back(1.0f);
                    values.push_back(0.0f);
                    values.push_back(0.0f);
                    values.push_back(0.0f);
                    continue;
                }
                if (n == 2) {
                    values.push_back(1.0f);
                    values.push_back(1.0f);
                    values.push_back(0.0f);
                    values.push_back(0.0f);
                    continue;
                }
                values.push_back(1.0f);
                values.push_back(1.0f);
                values.push_back(1.0f);
                values.push_back((n - 3.0f) / 2.0f);
            }
            values.push_back(state.compute_pip(player) / 375.0f);
        }
        for (int player = 0; player <= 1; player++) {
            values.push_back(state.on[player][BAR] / 2.0f);
        }
        for (int player = 0; player <= 1; player++) {
            values.push_back(state.on[player][OUT] / 15.0f);
        }
        if (state.turn == WHITE) {
            values.push_back(1.0f);
            values.push_back(0.0f);
        } else {
            values.push_back(0.0f);
            values.push_back(1.0f);
        }
        values.push_back(state.race());
        assert((int)values.size() == INPUT_FEATURES);
        return RevGrad::Tensor(RevGrad::Shape({INPUT_FEATURES, 1}), values);
    }

    int Model::choose_move(const State& state, const Dice& dice, const Moves& moves) {
        int index = 0;
        float best_probability = state.turn == WHITE ? 
            std::numeric_limits<float>::lowest() : 
            std::numeric_limits<float>::max();
        State s = state;
        for (int i = 0; i < (int)moves.size(); i++) {
            s.make_move(moves[i]);
            s.turn = !s.turn;
            float probability = predict(s).values()[0];
            s.undo_move();
            s.turn = !s.turn;
            if (
                (s.turn == WHITE && best_probability < probability) ||
                (s.turn == BLACK && best_probability > probability)
            ) {
                best_probability = probability;
                index = i;
            }
        }
        return index;
    }

    void Model::new_game() {}

    RevGrad::Tensor Model::predict(const State& state) {
        return nn.forward(tensor_from_state(state));
    }

    void Model::update(const State& state, const Move& move) {
        float alpha = 0.1f;
        float error = 0.0f;
        State next = state;
        next.make_move(move);
        next.turn = !next.turn;
        RevGrad::Tensor prediction = predict(state);
        if (next.on[WHITE][OUT] == 15 || next.on[BLACK][OUT] == 15) {
            Outcome outcome = next.outcome(WHITE);
            if (
                outcome == Outcome::WON_SINGLE_GAME ||
                outcome == Outcome::WON_GAMMON ||
                outcome == Outcome::WON_BACKGAMMON
            ) {
                error = 1 - prediction.values()[0];
            } else {
                error = 0 - prediction.values()[0];
            }
        } else {
            error = predict(next).values()[0] - prediction.values()[0];
        }
        // Zero the gradients
        for (auto param : nn.get_params()) {
            for (auto& grad : param.grads()) {
                grad = 0.0f;
            }
        }
        // Compute the gradients
        prediction.backward();
        // Update the values
        for (auto param : nn.get_params()) {
            for (int i = 0; i < param.size(); i++) {
                param.values()[i] += alpha * error * param.grads()[i];
            }
        }
    }
}
