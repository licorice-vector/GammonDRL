#include "Tensor.h"

namespace RevGrad {
    namespace ViewUtill {
        int shape_size(Shape shape) {
            return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        }

        Strides strides_from_shape(Shape shape) {
            int d = shape.size();
            Strides strides(d);
            int stride = 1;
            for (int i = d - 1; i >= 0; i--) {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }

        Shape broadcast_shape(const Shape& a, const Shape& b) {
            int a_size = a.size();
            int b_size = b.size();
            int n = std::max(a_size, b_size);
            Shape c(n);
            int i = a_size - 1;
            int j = b_size - 1;
            int k = n - 1;
            while (i >= 0 && j >= 0) {
                assert(a[i] == b[j] || a[i] == 1 || b[j] == 1);
                c[k] = std::max(a[i], b[j]);
                i--, j--, k--;
            }
            while (i >= 0) {
                c[k] = a[i];
                i--, k--;
            }
            while (j >= 0) {
                c[k] = b[j];
                j--, k--;
            }
            return c;
        }

        Indices unravel(int index, const Shape& shape, const Strides& strides) {
            int size = shape.size();
            Indices indices(size);
            for (int i = 0; i < size; i++) {
                indices[i] = (index / strides[i]) % shape[i];
            }
            return indices;
        }

        int ravel(const Indices& indices, const Strides& strides) {
            int offset = 0;
            for (int i = 0; i < indices.size(); ++i) {
                offset += indices[i] * strides[i];
            }
            return offset;
        }

        Indices reshape_indices(const Indices& indices, const Shape& shape) {
            int size = shape.size();
            Indices reshaped_indices(size);
            int size_delta = indices.size() - size;
            for (int i = 0; i < size; i++) {
                reshaped_indices[i] = indices[size_delta + i] % shape[i];
            }
            return reshaped_indices;
        }
    }

    Node::Node(float value) 
        : shape(Shape(1, 1))
    {
        values = Values(1, value);
        strides = ViewUtill::strides_from_shape(shape);
        grads = Gradients(1);
    }

    Node::Node(Shape shape, float value) 
        : shape(shape), 
          strides(ViewUtill::strides_from_shape(shape))
    {
        int size = ViewUtill::shape_size(shape);
        values = Values(size, value);
        grads = Gradients(size);
    }

    Node::Node(Shape shape, Values values) 
        : values(values),
          shape(shape), 
          strides(ViewUtill::strides_from_shape(shape)),
          grads(Gradients((int)values.size()))
    {
        assert(ViewUtill::shape_size(shape) == (int)values.size());
    }

    namespace TensorUtill {
        Tensor addition(const Tensor& u, const Tensor& v) {
            Shape shape = ViewUtill::broadcast_shape(u.shape(), v.shape());
            Tensor w(shape);
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                float u_value = u.value(ViewUtill::reshape_indices(indices, u.shape()));
                float v_value = v.value(ViewUtill::reshape_indices(indices, v.shape()));
                w.values()[i] = u_value + v_value;
            }
            w.add_edge(u), w.add_edge(v);
            return w;
        }

        void addition_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 2);
            Tensor u = w.edges()[0];
            Tensor v = w.edges()[1];
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                Indices u_indices = ViewUtill::reshape_indices(indices, u.shape());
                Indices v_indices = ViewUtill::reshape_indices(indices, v.shape());
                u.grad(u_indices) += w.grads()[i];
                v.grad(v_indices) += w.grads()[i];
            }
        }

        Tensor subtraction(const Tensor& u, const Tensor& v) {
            Shape shape = ViewUtill::broadcast_shape(u.shape(), v.shape());
            Tensor w(shape);
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                float u_value = u.value(ViewUtill::reshape_indices(indices, u.shape()));
                float v_value = v.value(ViewUtill::reshape_indices(indices, v.shape()));
                w.values()[i] = u_value - v_value;
            }
            w.add_edge(u), w.add_edge(v);
            return w;
        }

        void subtraction_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 2);
            Tensor u = w.edges()[0];
            Tensor v = w.edges()[1];
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                Indices u_indices = ViewUtill::reshape_indices(indices, u.shape());
                Indices v_indices = ViewUtill::reshape_indices(indices, v.shape());
                u.grad(u_indices) += w.grads()[i];
                v.grad(v_indices) -= w.grads()[i];
            }
        }

        Tensor multiplication(const Tensor& u, const Tensor& v) {
            Shape shape = ViewUtill::broadcast_shape(u.shape(), v.shape());
            Tensor w(shape);
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                float u_value = u.value(ViewUtill::reshape_indices(indices, u.shape()));
                float v_value = v.value(ViewUtill::reshape_indices(indices, v.shape()));
                w.values()[i] = u_value * v_value;
            }
            w.add_edge(u), w.add_edge(v);
            return w;
        }

        void multiplication_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 2);
            Tensor u = w.edges()[0];
            Tensor v = w.edges()[1];
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                Indices u_indices = ViewUtill::reshape_indices(indices, u.shape());
                Indices v_indices = ViewUtill::reshape_indices(indices, v.shape());
                u.grad(u_indices) += w.grads()[i] * v.value(v_indices);
                v.grad(v_indices) += w.grads()[i] * u.value(u_indices);
            }
        }

        Tensor division(const Tensor& u, const Tensor& v) {
            Shape shape = ViewUtill::broadcast_shape(u.shape(), v.shape());
            Tensor w(shape);
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                float u_value = u.value(ViewUtill::reshape_indices(indices, u.shape()));
                float v_value = v.value(ViewUtill::reshape_indices(indices, v.shape()));
                w.values()[i] = u_value / v_value;
            }
            w.add_edge(u), w.add_edge(v);
            return w;
        }

        void division_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 2);
            Tensor u = w.edges()[0];
            Tensor v = w.edges()[1];
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                Indices u_indices = ViewUtill::reshape_indices(indices, u.shape());
                Indices v_indices = ViewUtill::reshape_indices(indices, v.shape());
                float u_value = u.value(u_indices);
                float v_value = v.value(v_indices);
                u.grad(u_indices) += w.grads()[i] * (1.0f / v_value);
                v.grad(v_indices) += w.grads()[i] * (-u_value / (v_value * v_value));
            }
        }

        Tensor sum(const Tensor& u, int axis) {
            if (axis == -1) {
                float sum = 0.0f;
                for (int i = 0; i < u.size(); i++) {
                    sum += u.values()[i];
                }
                Tensor w(sum);
                w.meta_data()["axis"] = axis;
                w.add_edge(u);
                return w;
            }
            Shape shape = u.shape();
            shape.erase(shape.begin() + axis);
            Tensor w(shape);
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                indices.insert(indices.begin() + axis, 0);
                float sum = 0.0f;
                for (int j = 0; j < u.shape()[axis]; j++) {
                    indices[axis] = j;
                    float value = u.value(indices);
                    sum += value;
                }
                indices.erase(indices.begin() + axis);
                w.values()[ViewUtill::ravel(indices, w.strides())] = sum;
            }
            if (shape.size() == 0) {
                assert(w.size() == 1);
                w.shape() = Shape({1});
                w.strides() = ViewUtill::strides_from_shape(w.shape());
            }
            w.meta_data()["axis"] = axis;
            w.add_edge(u);
            return w;
        }

        void sum_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 1);
            Tensor u = w.edges()[0];
            assert(w.meta_data().count("axis"));
            int axis = w.meta_data().at("axis");
            if (axis == -1) {
                assert(w.size() == 1);
                for (int i = 0; i < u.size(); i++) {
                    u.grads()[i] += w.grads()[0];
                }
                return;
            }
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                indices.insert(indices.begin() + axis, 0);
                for (int j = 0; j < u.shape()[axis]; j++) {
                    indices[axis] = j;
                    u.grad(indices) += w.grads()[i];
                }
            }
        }

        Tensor max(const Tensor& u, int axis) {
            Shape shape = u.shape();
            shape.erase(shape.begin() + axis);
            Tensor w(shape);
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                indices.insert(indices.begin() + axis, 0);
                float max_value = std::numeric_limits<float>::lowest();
                for (int j = 0; j < u.shape()[axis]; j++) {
                    indices[axis] = j;
                    float value = u.value(indices);
                    if (value > max_value) {
                        max_value = value;
                    }
                }
                indices.erase(indices.begin() + axis);
                w.values()[ViewUtill::ravel(indices, w.strides())] = max_value;
            }
            if (shape.size() == 0) {
                assert(w.size() == 1);
                w.shape() = Shape({1});
                w.strides() = ViewUtill::strides_from_shape(w.shape());
            }
            w.meta_data()["axis"] = axis;
            w.add_edge(u);
            return w;
        }

        void max_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 1);
            assert(w.meta_data().count("axis"));
            int axis = w.meta_data().at("axis");
            Tensor u = w.edges()[0];
            for (int i = 0; i < w.size(); i++) {
                Indices indices = ViewUtill::unravel(i, w.shape(), w.strides());
                indices.insert(indices.begin() + axis, 0);
                float max_value = w.values()[i];
                float cnt = 0;
                for (int j = 0; j < u.shape()[axis]; j++) {
                    indices[axis] = j;
                    if (u.value(indices) == max_value) {
                        cnt++;
                    }
                }
                for (int j = 0; j < u.shape()[axis]; j++) {
                    indices[axis] = j;
                    if (u.value(indices) == max_value) {
                        u.grad(indices) += w.grads()[i] / cnt;
                    }
                }
            }
        }

        Tensor exp(const Tensor& u) {
            Tensor w(u.shape());
            for (int i = 0; i < u.size(); i++) {
                w.values()[i] = std::exp(u.values()[i]);
            }
            w.add_edge(u);
            return w;
        }

        void exp_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 1);
            Tensor u = w.edges()[0];
            for (int i = 0; i < u.size(); i++) {
                u.grads()[i] += w.grads()[i] * std::exp(u.values()[i]);
            }
        }

        Tensor log(const Tensor& u) {
            Tensor w(u.shape());
            for (int i = 0; i < u.size(); i++) {
                float value = u.values()[i];
                assert(!(value != value)); // nan
                assert(value != 0.0f);
                assert(value > 0.0f);
                w.values()[i] = std::log(value);
            }
            w.add_edge(u);
            return w;
        }

        void log_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 1);
            Tensor u = w.edges()[0];
            for (int i = 0; i < u.size(); i++) {
                u.grads()[i] += w.grads()[i] * (1 / u.values()[i]);
            }
        }

        Tensor relu(const Tensor& u) {
            Tensor w(u.shape());
            for (int i = 0; i < u.size(); i++) {
                w.values()[i] = std::max(0.0f, u.values()[i]);
            }
            w.add_edge(u);
            return w;
        }

        void relu_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 1);
            Tensor u = w.edges()[0];
            for (int i = 0; i < u.size(); i++) {
                u.grads()[i] += w.grads()[i] * (u.values()[i] > 0.0f ? 1.0f : 0.0f);
            }
        }

        Tensor sigmoid(const Tensor& u) {
            Tensor w(u.shape());
            for (int i = 0; i < u.size(); i++) {
                float value = u.values()[i];
                if (0 < value) {
                    w.values()[i] = 1.0f / (1.0f + std::exp(-value));
                } else {
                    float exp_value = std::exp(value);
                    w.values()[i] = exp_value / (1.0f + exp_value);
                }
            }
            w.add_edge(u);
            return w;
        }

        void sigmoid_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 1);
            Tensor u = w.edges()[0];
            for (int i = 0; i < u.size(); i++) {
                float value = w.values()[i];
                u.grads()[i] += w.grads()[i] * (value * (1 - value));
            }
        }

        Tensor softmax(const Tensor& u) {
            Tensor mx = Tensor::max(u);
            Tensor exp = Tensor::exp(u - mx);
            Tensor s = exp / Tensor::sum(exp, 0);
            Tensor w(u.shape());
            w.values() = s.values();
            w.add_edge(u);
            return w;
        }

        void softmax_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 1);
            Tensor u = w.edges()[0];
            Shape shape = u.shape();
            assert((int)shape.size() == 2); // {features, batch_size}
            for (int k = 0; k < shape[1]; k++) { // batch_size
                for (int i = 0; i < shape[0]; i++) { // features
                    float value_i = w.values()[i * shape[1] + k];
                    for (int j = 0; j < shape[0]; j++) { // features
                        float value_j = w.values()[j * shape[1] + k];
                        u.grads()[i * shape[1] + k] += w.grads()[j * shape[1] + k] * (value_i * ((i == j) - value_j));
                    }
                }
            }
        }

        Tensor log_softmax(const Tensor& u) {
            Tensor mx = Tensor::max(u);
            Tensor s = u - mx - Tensor::log(Tensor::sum(Tensor::exp(u - mx), 0));
            Tensor w(u.shape());
            assert(s.shape() == w.shape());
            w.values() = s.values();
            w.add_edge(u);
            return w;
        }

        void log_softmax_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 1);
            Tensor u = w.edges()[0];
            Tensor mx = Tensor::max(u);
            Tensor exp = Tensor::exp(u - mx);
            Tensor s = exp / Tensor::sum(exp, 0);
            Shape shape = u.shape();
            assert((int)shape.size() == 2);
            for (int k = 0; k < shape[1]; k++) { // batch_size
                for (int i = 0; i < shape[0]; i++) { // features
                    float value_i = s.values()[i * shape[1] + k];
                    for (int j = 0; j < shape[0]; j++) { // features
                        u.grads()[i * shape[1] + k] += w.grads()[j * shape[1] + k] * ((i == j) - value_i);
                    }
                }
            }
        }

        Tensor matmul(const Tensor& u, const Tensor& v) {
            Shape w_shape = Shape({u.shape()[0], v.shape()[1]});
            Tensor w(w_shape);
            std::vector<float> w_values = w.values();
            std::vector<float> u_values = u.values();
            std::vector<float> v_values = v.values();
            Shape u_shape = u.shape();
            Shape v_shape = v.shape();
            //#pragma omp parallel for
            for (int i = 0; i < w_shape[0]; i++) {
                for (int j = 0; j < w_shape[1]; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < u_shape[1]; k++) {
                        sum += u_values[i * u_shape[1] + k] * v_values[k * v_shape[1] + j];
                    }
                    w_values[i * w_shape[1] + j] = sum;
                }
            }
            w.values() = w_values;
            w.add_edge(u), w.add_edge(v);
            return w;
        }

        void matmul_backward_fn(const Tensor& w) {
            assert((int)w.edges().size() == 2);
            Tensor u = w.edges()[0];
            Tensor v = w.edges()[1];
            std::vector<float> u_values = u.values();
            std::vector<float> v_values = v.values();
            std::vector<float> w_grads = w.grads();
            std::vector<float> u_grads = u.grads();
            std::vector<float> v_grads = v.grads();
            Shape w_shape = w.shape();
            Shape u_shape = u.shape();
            Shape v_shape = v.shape();
            //#pragma omp parallel for
            for (int i = 0; i < w_shape[0]; i++) {
                for (int k = 0; k < u_shape[1]; k++) {
                    for (int j = 0; j < w_shape[1]; j++) {
                        u_grads[i * u_shape[1] + k] += (
                            w_grads[i * w_shape[1] + j] * 
                            v_values[k * v_shape[1] + j]
                        );
                        v_grads[k * v_shape[1] + j] += (
                            w_grads[i * w_shape[1] + j] * 
                            u_values[i * u_shape[1] + k]
                        );
                    }
                }
            }
            u.grads() = u_grads;
            v.grads() = v_grads;
        }
    }

    std::random_device Tensor::rd = std::random_device();
    std::mt19937 Tensor::rng = std::mt19937(rd());
    std::vector<float> Tensor::random_vector(int n, int in_degree) {
        std::vector<float> r(n);
        std::normal_distribution<float> he_dist(0.0f, std::sqrt(2.0f / in_degree));
        for (int i = 0; i < n; i++) {
            r[i] = he_dist(rng);
        }
        return r;
    }

    Tensor::Tensor(float value) : _data(std::make_shared<Node>(value)) {}
    Tensor::Tensor(Shape shape, float value) : _data(std::make_shared<Node>(shape, value)) {}
    Tensor::Tensor(Shape shape, Values values) : _data(std::make_shared<Node>(shape, values)) {}

    Tensor Tensor::from_csv(const std::string& filename) {
        std::ifstream file(filename);
        assert(file.is_open());
        std::vector<std::vector<float>> rows;
        std::string line;    
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string s;
            std::vector<float> row;
            while (std::getline(ss, s, ',')) {
                row.push_back(std::stoi(s));
            }
            rows.push_back(row);
        }
        file.close();
        for (int i = 1; i < (int)rows.size(); i++) {
            assert(rows[i].size() == rows[0].size());
        }
        std::vector<float> values;
        for (int i = 0; i < (int)rows.size(); i++) {
            for (auto value : rows[i]) {
                values.push_back(value);
            }
        }
        return Tensor(Shape({(int)rows.size(), (int)rows[0].size()}), values);
    }

    Tensor Tensor::random(Shape shape, int in_degree) {
        return Tensor(shape, random_vector(ViewUtill::shape_size(shape), in_degree));
    }

    const Data& Tensor::data() const { return _data; }

    Tensor Tensor::clone() {
        Tensor tensor;
        tensor.values() = values();
        tensor.shape() = shape();
        tensor.strides() = strides();
        tensor.grads() = grads();
        tensor.edges() = edges();
        tensor.backward_fn() = backward_fn();
        return tensor;
    }

    Values& Tensor::values() { return _data->values; }
    const Values& Tensor::values() const { return _data->values; }
    Shape& Tensor::shape() { return _data->shape; }
    const Shape& Tensor::shape() const { return _data->shape; }
    Strides& Tensor::strides() { return _data->strides; }
    const Strides& Tensor::strides() const { return _data->strides; }
    Gradients& Tensor::grads() { return _data->grads; }
    const Gradients& Tensor::grads() const { return _data->grads; }
    Edges& Tensor::edges() { return _data->edges; }
    const Edges& Tensor::edges() const { return _data->edges; }
    BackwardFn& Tensor::backward_fn() { return _data->backward_fn; }
    const BackwardFn& Tensor::backward_fn() const { return _data->backward_fn; }
    MetaData& Tensor::meta_data() { return _data->meta_data; }
    const MetaData& Tensor::meta_data() const { return _data->meta_data; }

    float& Tensor::value(const Indices& indices) {
        return values()[ViewUtill::ravel(indices, strides())];
    }
    const float& Tensor::value(const std::vector<int>& indices) const {
        return values()[ViewUtill::ravel(indices, strides())];
    }
    float& Tensor::grad(const Indices& indices) {
        return grads()[ViewUtill::ravel(indices, strides())];
    }
    const float& Tensor::grad(const std::vector<int>& indices) const {
        return grads()[ViewUtill::ravel(indices, strides())];
    }

    void Tensor::add_edge(const Tensor& tensor) { _data->edges.push_back(tensor); }

    bool Tensor::operator<(const Tensor& other) const { return _data < other._data; }
    
    Tensor operator+(const Tensor& u, const Tensor& v) {
        Tensor w = TensorUtill::addition(u, v);
        w.backward_fn() = TensorUtill::addition_backward_fn;
        return w;
    }

    Tensor operator-(const Tensor& u, const Tensor& v) {
        Tensor w = TensorUtill::subtraction(u, v);
        w.backward_fn() = TensorUtill::subtraction_backward_fn;
        return w;
    }

    Tensor operator*(const Tensor& u, const Tensor& v) {
        Tensor w = TensorUtill::multiplication(u, v);
        w.backward_fn() = TensorUtill::multiplication_backward_fn;
        return w;
    }

    Tensor operator/(const Tensor& u, const Tensor& v) {
        Tensor w = TensorUtill::division(u, v);
        w.backward_fn() = TensorUtill::division_backward_fn;
        return w;
    }

    Tensor& Tensor::operator+=(const Tensor& other) { return *this = *this + other; }
    Tensor& Tensor::operator-=(const Tensor& other) { return *this = *this - other; }
    Tensor& Tensor::operator*=(const Tensor& other) { return *this = *this * other; }
    Tensor& Tensor::operator/=(const Tensor& other) { return *this = *this / other; }
    Tensor Tensor::operator-() const { return Tensor() - *this; }

    Tensor Tensor::sum(const Tensor& u, int axis) {
        Tensor w = TensorUtill::sum(u, axis);
        w.backward_fn() = TensorUtill::sum_backward_fn;
        return w;
    }

    Tensor Tensor::mean(const Tensor& u) {
        return Tensor::sum(u) / float(u.size());
    }

    Tensor Tensor::max(const Tensor& u, int axis) {
        assert(axis >= 0 && axis < (int)u.shape().size());
        Tensor w = TensorUtill::max(u, axis);
        w.backward_fn() = TensorUtill::max_backward_fn;
        return w;
    }

    Tensor Tensor::exp(const Tensor& u) {
        Tensor w = TensorUtill::exp(u);
        w.backward_fn() = TensorUtill::exp_backward_fn;
        return w;
    }

    Tensor Tensor::log(const Tensor& u) {
        Tensor w = TensorUtill::log(u);
        w.backward_fn() = TensorUtill::log_backward_fn;
        return w;
    }

    Tensor Tensor::relu(const Tensor& u) {
        Tensor w = TensorUtill::relu(u);
        w.backward_fn() = TensorUtill::relu_backward_fn;
        return w;
    }
    
    Tensor Tensor::sigmoid(const Tensor& u) {
        Tensor w = TensorUtill::sigmoid(u);
        w.backward_fn() = TensorUtill::sigmoid_backward_fn;
        return w;
    }

    Tensor Tensor::softmax(const Tensor& u) {
        assert((int)u.shape().size() == 2); // {features, batch_size}
        Tensor w = TensorUtill::softmax(u);
        w.backward_fn() = TensorUtill::softmax_backward_fn;
        return w;
    }

    Tensor Tensor::log_softmax(const Tensor& u) {
        assert((int)u.shape().size() == 2); // {features, batch_size}
        Tensor w = TensorUtill::log_softmax(u);
        w.backward_fn() = TensorUtill::log_softmax_backward_fn;
        return w;
    }

    Tensor Tensor::matmul(const Tensor& u, const Tensor& v) {
        assert((int)u.shape().size() == 2 && (int)v.shape().size() == 2);
        assert(u.shape()[1] == v.shape()[0]);
        Tensor w = TensorUtill::matmul(u, v);
        w.backward_fn() = TensorUtill::matmul_backward_fn;
        return w;
    }

    int Tensor::size() const {
        return ViewUtill::shape_size(shape());
    }

    void Tensor::reshape(const Shape& shape) {
        this->shape() = shape;
        strides() = ViewUtill::strides_from_shape(this->shape());
    }

    void Tensor::flatten() {
        Shape shape({size()});
        this->shape() = shape;
        strides() = ViewUtill::strides_from_shape(this->shape());
    }

    void Tensor::transpose() {
        assert((int)this->shape().size() == 2);
        Shape shape = {this->shape()[1], this->shape()[0]};
        int n = size();
        Values values(n);
        Gradients grads(n);
        Strides strides = ViewUtill::strides_from_shape(shape);
        for (int i = 0; i < this->shape()[0]; i++) {
            for (int j = 0; j < this->shape()[1]; j++) {
                float value =this->values()[i * this->strides()[0] + j * this->strides()[1]];
                float grad =this->grads()[i * this->strides()[0] + j * this->strides()[1]];
                values[j * strides[0] + i * strides[1]] = value;
                grads[j * strides[0] + i * strides[1]] = grad;
            }
        }
        this->shape() = shape;
        this->values() = values;
        this->grads() = grads;
        this->strides() = strides;
    }

    Tensor Tensor::slice(const std::vector<std::pair<int, int>>& ranges) const {
        int n = this->shape().size();
        assert(n == (int)ranges.size());
        Shape shape(n);
        for (int i = 0; i < n; i++) {
            auto [start, end] = ranges[i];
            assert(start < this->shape()[i] && end <= this->shape()[i] && start < end);
            shape[i] = end - start;
        }
        Tensor tensor(shape);
        int size = tensor.size();
        for (int i = 0; i < size; i++) {
            Indices indices = ViewUtill::unravel(i, tensor.shape(), tensor.strides());
            for (int j = 0; j < n; j++) {
                auto [start, _] = ranges[j];
                indices[j] += start;
            }
            tensor.values()[i] = value(indices);
        }
        return tensor;
    }

    void Tensor::backward() {
        grads() = std::vector<float>(grads().size(), 1.0f);
        std::map<Tensor, std::vector<Tensor>> adj;
        std::queue<Tensor> Q;
        Q.push(*this);
        std::set<Tensor> S;
        while (!Q.empty()) {
            Tensor u = Q.front();
            Q.pop();
            if (S.count(u)) {
                continue;
            }
            S.insert(u);
            for (Tensor v : u.edges()) {
                adj[u].push_back(v);
                Q.push(v);
            }
        }
        S.clear();
        std::vector<Tensor> order;
        std::function<void(Tensor)> topsort;
        topsort = [&topsort, &adj, &S, &order] (Tensor u) -> void {
            S.insert(u);
            for (auto v : adj[u]) {
                if (!S.count(v)) {
                    topsort(v);
                }
            }
            order.push_back(u);
        };
        topsort(*this);
        std::reverse(order.begin(), order.end());
        for (Tensor u : order) {
            if (u.backward_fn()) {
                u.backward_fn()(u);
            }
        }
    }
}
