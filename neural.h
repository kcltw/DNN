#ifndef DNN_NEURAL_H
#define DNN_NEURAL_H
#include"matrix.h"
#include"loss.h"

template<typename T>
matrix<T> sigmoid(matrix<T> &z, matrix<T> &w)
{
    matrix<T> result;
    result = w.transpose() * z;

    for (auto &i:result.element)
    {
        for (auto &j:i)
        {
            j = 1 / (1 + exp(-1 * j));
        }
    }
    return result;

}

template<typename T>
matrix<T> softmax(matrix<T> &z, matrix<T> &w)
{
    matrix<T> result;
    double sum;
    result = w.transpose() * z;
    for (int i = 0; i < result.col; i++)
    {
        sum = 0;
        for (int j = 0; j < result.row; j++)
        {
            sum = exp(result.element[j][i]) + sum;
        }
        for (int j = 0; j < result.row; j++)
        {
            result.element[j][i] = exp(result.element[j][i]) / sum;
        }

    }
    return result;
}

template<typename T>
class neural
{

public:
    int num_layer;
    double learning_rate;
    int batch_size;
    std::vector<int> layer_size;
    loss<T> *batch_error;
    std::vector<matrix<T>> z;
    std::vector<matrix<T>> w;
    neural(int n_layer, double lr, int bs, std::vector<int> l_size) : num_layer(n_layer),
                                                                      learning_rate(lr),
                                                                      batch_size(bs),
                                                                      layer_size(l_size)
    {
        std::cout << "num of layer = " << num_layer << std::endl;
        std::cout << "learning rate = " << learning_rate << std::endl;
        std::cout << "batch size = " << batch_size << std::endl;
        std::vector<int>::iterator it_i;

        for (it_i = layer_size.begin(); it_i != layer_size.end(); ++it_i)
            std::cout << "layer " << std::distance(layer_size.begin(), it_i) << " sizes = " << *it_i << std::endl;
        for (int i = 0; i < num_layer; i++)
        {
            z.push_back(matrix<T>(layer_size[i], batch_size));
            if (i < num_layer - 1)
                z[i].add_bias();
        }

        for (int i = 0; i < num_layer - 1; i++)
        {
            w.push_back(matrix<T>(layer_size[i] + 1, layer_size[i + 1]));

            //initial weights
            for (auto &j:w[i].element)
            {
                for (auto &value:j)
                {
                    value = (dis(gen)-0.5)*2*4*sqrt((double)6/(w[i].row+w[i].col));
                }
            }
        }

    };

    neural(int n_layer, double lr, int bs, std::vector<int> l_size, int size) : num_layer(n_layer),
                                                                                learning_rate(lr),
                                                                                batch_size(size),
                                                                                layer_size(l_size)
    {

        for (int i = 0; i < num_layer; i++)
        {
            z.push_back(matrix<T>(layer_size[i], batch_size));
            w.push_back(matrix<T>(layer_size[i] + 1, layer_size[i + 1]));
            if (i < (num_layer - 1))
            {
                z[i].add_bias();
            }

        }
    }

    void feedforward(std::vector<std::vector<T>> train_x, std::vector<std::vector<T>> train_y)
    {
        z[0].setMatrix(train_x);
        batch_error = new loss<double>();

        for (int i = 1; i < num_layer; i++)
        {
            z[i] = sigmoid(z[i - 1], w[i - 1]);
            z[i].add_bias();
        }
        z[num_layer - 1] = softmax(z[num_layer - 2], w[num_layer - 2]);

        batch_error->eval_error(train_y, z[num_layer - 1]);
    }

    void backpro()
    {
        matrix<T> *grad, local_grad, *dW;
        grad = new matrix<T>[num_layer];
        dW = new matrix<T>[num_layer];
        for (int i = 0; i < num_layer - 1; i++)
        {
            dW[i] = matrix<T>(w[i].row, w[i].col);
        }
        grad[num_layer - 1] = matrix<T>(batch_error->er->row, batch_error->er->col);
        grad[num_layer - 1].assign(*batch_error->er);
        delete batch_error;

        for (int i = num_layer - 2; i > 0; i--)
        {
            local_grad = matrix<T>(z[i].row, z[i].col);
            for (int j = 0; j < z[i].row; j++)
            {
                for (int k = 0; k < z[i].col; k++)
                {
                    local_grad.element[j][k] = 0;
                    local_grad.element[j][k] = z[i].element[j][k] * (1 - z[i].element[j][k]);
                }
            }
            if (i == num_layer - 2)
            {
                grad[i] = (w[i]) * (grad[i + 1]).transpose();
                for (int j = 0; j < z[i].row; j++)
                {
                    for (int k = 0; k < z[i].col; k++)
                    {
                        grad[i].element[j][k] = grad[i].element[j][k] * local_grad.element[j][k];
                    }
                }
            } else
            {
                grad[i + 1].de_bias();
                grad[i] = (w[i]) * (grad[i + 1]);
                for (int j = 0; j < grad[i].row; j++)
                {
                    for (int k = 0; k < grad[i].col; k++)
                    {
                        grad[i].element[j][k] = grad[i].element[j][k] * local_grad.element[j][k];
                    }
                }
            }
        }
        for (int i = 0; i < num_layer - 1; i++)
        {
            if (i == num_layer - 2)
            {
                dW[i].assign( z[i] * grad[i + 1]);
            } else
            {
                if (i == 0)
                {
                    grad[i + 1].de_bias();
                }
                dW[i].assign(z[i] * (grad[i + 1].transpose()));
            }
            for (int j = 0; j < dW[i].row; j++)
            {
                for (int k = 0; k < dW[i].col; k++)
                {
                    dW[i].element[j][k] = learning_rate * dW[i].element[j][k] / batch_size;
                }
            }
            w[i].assign(w[i] + dW[i]);
        }
        delete[] grad;
        delete[] dW;
    }
};

#endif //DNN_NEURAL_H

