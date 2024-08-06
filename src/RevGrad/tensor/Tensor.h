#ifndef REVGRAD_TENSOR_H
#define REVGRAD_TENSOR_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cassert>
#include <functional>
#include <numeric>
#include <algorithm>
#include <random>
#include <memory>
#include <queue>
#include <set>
#include <map>
#include <omp.h>

namespace RevGrad {
    class Node;
    class TensorData;
    class Tensor;

    typedef std::vector<float> Values;
    typedef std::vector<float> Gradients;
    typedef std::vector<int> Shape;
    typedef std::vector<int> Strides;
    typedef std::vector<int> Indices;
    typedef std::shared_ptr<Node> Data;
    typedef std::vector<Tensor> Edges;
    typedef std::function<void(const Tensor&)> BackwardFn;
    typedef std::map<std::string, int> MetaData;

    namespace ViewUtill {
        int shape_size(Shape shape);
        Strides strides_from_shape(Shape shape);
        Shape broadcast_shape(const Shape& a, const Shape& b);
        Indices unravel(int index, const Shape& shape, const Strides& strides);
        int ravel(const Indices& indices, const Strides& strides);
        Indices reshape_indices(const Indices& indices, const Shape& shape);
    }

    class Node {
        public:
        Values values;
        Shape shape;
        Strides strides;
        Gradients grads;
        Edges edges;
        BackwardFn backward_fn;
        MetaData meta_data;
        Node(float value = 0.0f);
        Node(Shape shape, float value = 0.0f);
        Node(Shape shape, Values values);
    };

    namespace TensorUtill {
        Tensor addition(const Tensor& u, const Tensor& v);
        void addition_backward_fn(const Tensor& w);
        Tensor subtraction(const Tensor& u, const Tensor& v);
        void subtraction_backward_fn(const Tensor& w);
        Tensor multiplication(const Tensor& u, const Tensor& v);
        void multiplication_backward_fn(const Tensor& w);
        Tensor division(const Tensor& u, const Tensor& v);
        void division_backward_fn(const Tensor& w);
        Tensor sum(const Tensor& u, int axis);
        void sum_backward_fn(const Tensor& w);
        Tensor max(const Tensor& u, int axis);
        void max_backward_fn(const Tensor& w);
        Tensor exp(const Tensor& u);
        void exp_backward_fn(const Tensor& w);
        Tensor log(const Tensor& u);
        void log_backward_fn(const Tensor& w);
        Tensor relu(const Tensor& u);
        void relu_backward_fn(const Tensor& w);
        Tensor sigmoid(const Tensor& u);
        void sigmoid_backward_fn(const Tensor& w);
        Tensor softmax(const Tensor& u);
        void softmax_backward_fn(const Tensor& w);
        Tensor log_softmax(const Tensor& u);
        void log_softmax_backward_fn(const Tensor& w);
        Tensor matmul(const Tensor& u, const Tensor& v);
        void matmul_backward_fn(const Tensor& w);
    }

    class Tensor {
        Data _data;
        static std::random_device rd;
        static std::mt19937 rng;
        static std::vector<float> random_vector(int n, int in_degree);
    public:
        Tensor(float value = 0.0f);
        Tensor(Shape shape, float value = 0.0f);
        Tensor(Shape shape, Values values);
        static Tensor from_csv(const std::string& filename);
        static Tensor random(Shape shape, int in_degree);
        Tensor clone();
        const Data& data() const;
        Values& values();
        const Values& values() const;
        Shape& shape();
        const Shape& shape() const;
        Strides& strides();
        const Strides& strides() const;
        Gradients& grads();
        const Gradients& grads() const;
        Edges& edges();
        const Edges& edges() const;
        BackwardFn& backward_fn();
        const BackwardFn& backward_fn() const;
        MetaData& meta_data();
        const MetaData& meta_data() const;
        float& value(const Indices& indices);
        const float& value(const std::vector<int>& indices) const;
        float& grad(const Indices& indices);
        const float& grad(const std::vector<int>& indices) const;
        void add_edge(const Tensor& tensor);
        bool operator<(const Tensor& other) const;
        friend Tensor operator+(const Tensor& u, const Tensor& v);
        friend Tensor operator-(const Tensor& u, const Tensor& v);
        friend Tensor operator*(const Tensor& u, const Tensor& v);
        friend Tensor operator/(const Tensor& u, const Tensor& v);
        Tensor& operator+=(const Tensor& other);
        Tensor& operator-=(const Tensor& other);
        Tensor& operator*=(const Tensor& other);
        Tensor& operator/=(const Tensor& other);
        Tensor operator-() const;
        static Tensor sum(const Tensor& u, int axis = -1);
        static Tensor mean(const Tensor& u);
        static Tensor max(const Tensor& u, int axis = 0);
        static Tensor exp(const Tensor& u);
        static Tensor log(const Tensor& u);
        static Tensor relu(const Tensor& u);
        static Tensor sigmoid(const Tensor& u);
        static Tensor softmax(const Tensor& u);
        static Tensor log_softmax(const Tensor& u);
        static Tensor matmul(const Tensor& u, const Tensor& v);
        int size() const;
        void reshape(const Shape& shape);
        void flatten();
        void transpose();
        Tensor slice(const std::vector<std::pair<int, int>>& ranges) const;
        void backward();
    };
}

#endif
