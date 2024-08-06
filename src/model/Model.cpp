#include "Model.h"
#include "../RevGrad/utill/Print.h"

namespace Backgammon {
    NeuralNetwork::NeuralNetwork() {
        l1 = RevGrad::Linear(this, 200, 40);
        l2 = RevGrad::Linear(this, 40, 1);
    }

    RevGrad::Tensor NeuralNetwork::forward(RevGrad::Tensor x) {
        RevGrad::Tensor y = x;
        y = l1(y);
        y = RevGrad::Tensor::relu(y);
        y = l2(y);
        y = RevGrad::Tensor::sigmoid(y);
        return y;
    }

    Model::Model() : nn(NeuralNetwork()) {
        int size = 0;
        for (auto param : nn.get_params()) {
            size += param.size();
        }
        momentum = std::vector<float>(size);
    }

    void Model::save(std::string filename) { nn.save_parameters(filename); }
    void Model::load(std::string filename) { nn.load_parameters(filename); }

    RevGrad::Tensor Model::tensor_from_state(const State& state) {
        std::vector<float> values;
        values.reserve(200);
        for (int player = 0; player <= 1; player++) {
            int pip_cnt = 0;
            for (int point = 0; point < BOARD_SIZE; point++) {
                pip_cnt += state.on[player][point] * abs(point - (player == WHITE ? -1 : 24));
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
            values.push_back(pip_cnt / 540.0f);
        }
        for (int player = 0; player <= 1; player++) {
            int n = state.on[player][BAR];
            values.push_back(n / 2.0f);
        }
        for (int player = 0; player <= 1; player++) {
            int n = state.on[player][REMOVED];
            values.push_back(n / 15.0f);
        }
        if (state.turn == WHITE) {
            values.push_back(1.0f);
            values.push_back(0.0f);
        } else {
            values.push_back(0.0f);
            values.push_back(1.0f);
        }
        return RevGrad::Tensor(RevGrad::Shape({200, 1}), values);
    }

    int Model::choose_move(const State& state, const Dice& dice, const Moves& moves) {
        int index = 0;
        float best_prediction = state.turn == WHITE ? 
            std::numeric_limits<float>::lowest() : 
            std::numeric_limits<float>::max();
        State s = state;
        for (int i = 0; i < (int)moves.size(); i++) {
            s.make_move(moves[i]);
            s.turn = !s.turn;
            float prediction = predict(s).value({0});
            if (
                (state.turn == WHITE && best_prediction < prediction) ||
                (state.turn == BLACK && best_prediction > prediction)
            ) {
                best_prediction = prediction;
                index = i;
            }
            s.undo_move();
            s.turn = !s.turn;
        }
        return index;
    }

    void Model::new_game() {
        int size = 0;
        for (auto param : nn.get_params()) {
            size += param.size();
        }
        momentum = std::vector<float>(size);
    }

    RevGrad::Tensor Model::predict(const State& state) {
        return nn.forward(tensor_from_state(state));
    }

    void Model::update(const State& state, const Move& move) {
        State next = state;
        next.make_move(move);
        next.turn = !next.turn;
        RevGrad::Tensor next_prediction = predict(next);
        int reward = 0;
        if (next.on[WHITE][REMOVED] == 15 || next.on[BLACK][REMOVED] == 15) {
            Outcome outcome = next.outcome(WHITE);
            if (outcome == Outcome::WON_SINGLE_GAME) {
                reward = 1;
            } else if (outcome == Outcome::WON_GAMMON) {
                reward = 2;
            } else if (outcome == Outcome::WON_BACKGAMMON) {
                reward = 3;
            } else if (outcome == Outcome::LOST_SINGLE_GAME) {
                reward = -1;
            } else if (outcome == Outcome::LOST_GAMMON) {
                reward = -2;
            } else if (outcome == Outcome::LOST_BACKGAMMON) {
                reward = -3;
            }
            std::cout << "Prediction: " << next_prediction.value({0}) << std::endl;
            std::cout << "Reward: " << reward << std::endl;
        }
        float gamma = 0.7;
        float learning_rate = 0.1;
        RevGrad::Tensor prediction = predict(state);
        float error = reward + gamma * (next_prediction.value({0}) - prediction.value({0}));
        // Zero the gradients
        for (auto param : nn.get_params()) {
            for (auto& grad : param.grads()) {
                grad = 0.0f;
            }
        }
        // Compute the gradients
        prediction.backward();
        // Update the values
        int j = 0;
        for (auto param : nn.get_params()) {
            for (int i = 0; i < param.size(); i++) {
                momentum[j] = gamma * momentum[j] + param.grads()[i];
                param.values()[i] += learning_rate * error * momentum[j];
                j++;
            }
        }
    }
}
