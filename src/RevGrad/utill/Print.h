#ifndef REVGRAD_PRINT_H
#define REVGRAD_PRINT_H

#include "../tensor/Tensor.h"

namespace RevGrad {
    std::ostream& operator<<(std::ostream& os, const Shape& shape);
    std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
}

#endif
