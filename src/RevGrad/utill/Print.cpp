#include "Print.h"

namespace RevGrad {
    std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << "(";
        for (int i = 0; i < (int)shape.size(); i++) {
            os << shape[i];
            if (i < (int)shape.size() - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }

    std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "Tensor(shape=" << tensor.shape() << ", data=";
        const Values& values = tensor.values();
        Indices indices(tensor.shape().size());
        std::function<void(int)> print_tensor = [&] (int level) {
            if (level == (int)tensor.shape().size()) {
                os << tensor.value(indices);
            } else {
                os << "[";
                for (indices[level] = 0; indices[level] < tensor.shape()[level]; indices[level]++) {
                    if (indices[level] > 0) os << ", ";
                    print_tensor(level + 1);
                }
                os << "]";
            }
        };
        print_tensor(0);
        os << ")";
        return os;
    }
}
