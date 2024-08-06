#include "Model.h"

namespace RevGrad {
    std::vector<Tensor> Model::get_params() {
        std::vector<Tensor> params;
        for (auto& param : parameters) {
            params.push_back(param);
        }
        return params;
    }

    Tensor Model::operator()(Tensor x) {
        return forward(x);
    }

    void Model::save_parameters(const std::string& filename) {
        std::ofstream file(filename);
        assert(file.is_open());
        for (const auto& param : parameters) {
            std::vector<float> values = param.values();
            int size = (int)values.size();
            file << size << "\n";
            for (int i = 0; i < size; i++) {
                file << values[i];
                if (i + 1 == size) {
                    file << "\n";
                } else {
                    file << ",";
                }
            }
        }
        file.close();
    }

    void Model::load_parameters(const std::string& filename) {
        std::ifstream file(filename);
        assert(file.is_open());
        int index = 0;
        std::string line;
        while (std::getline(file, line)) {
            int size = std::stoi(line);
            assert(size == (int)parameters[index].values().size());
            assert(std::getline(file, line));
            std::vector<float> values;
            std::stringstream ss(line);
            std::string s;
            while (std::getline(ss, s, ',')) {
                values.push_back(std::stof(s));
            }
            assert((int)values.size() == size);
            parameters[index].values() = values;
            index++;
        }
        file.close();
    }

    Linear::Linear(Model* parent_model, int in_features, int out_features) 
        : in_features(in_features),
          out_features(out_features),
          weights(Tensor::random(Shape({out_features, in_features}), in_features)), 
          bias(Tensor(Shape({out_features, 1})))
    {
        parent_model->parameters.push_back(weights);
        parent_model->parameters.push_back(bias);
    }

    Tensor Linear::forward(Tensor x) {
        return Tensor::matmul(weights, x) + bias;
    }
}
