#ifndef REVGRAD_MODEL_H
#define REVGRAD_MODEL_H

#include <functional>
#include <cassert>

#include "../tensor/Tensor.h"

namespace RevGrad {
    class Model {
    public:
        std::vector<Tensor> parameters;
        Model() {}
        std::vector<Tensor> get_params();
        Tensor operator()(Tensor x);
        virtual Tensor forward(Tensor x) = 0;
        void save_parameters(const std::string& filename);
        void load_parameters(const std::string& filename);
    };

    class Linear : public Model {
    public:
        int in_features;
        int out_features;
        Tensor weights;
        Tensor bias;
        Linear() {}
        Linear(Model* parent_model, int in_features, int out_features);
        /*
            @param x tensor of shape (features, batch size)
        */
        Tensor forward(Tensor x) override;
    };
}

#endif
