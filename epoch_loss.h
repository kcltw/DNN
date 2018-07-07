#ifndef DNN_EPOCH_LOSS_H
#define DNN_EPOCH_LOSS_H

#include<iostream>
#include<vector>
#include<string>
#include<stdio.h>
#include<stdlib.h>
#include<iomanip>
#include<random>
#include<vector>
#include<math.h>
#include"matrix.h"
#include"neural.h"

template<typename T>
class epoch_loss
{
public:
    double miss_rate;
    double cross_entropy;

    epoch_loss()
    {};

    ~epoch_loss()
    {};

    void eval_loss(neural<T> nn, std::vector<std::vector<double> > &x, std::vector<std::vector<double> > &y)
    {
        int count = 0;
        int z_max_idx = 0;
        int y_max_idx = 0;
        double z_max = 0;
        double y_max = 0;
        auto loss_nn = neural<T>(nn.num_layer, nn.learning_rate, nn.batch_size, nn.layer_size, x.size());
        (loss_nn.w) = (nn.w);
        loss_nn.feedforward(x, y);


        cross_entropy = 0;
        for (int i = 0; i < x.size(); i++)
        {
            z_max = loss_nn.z[loss_nn.num_layer - 1].element[0][i];
            y_max = y.at(i).at(0);
            z_max_idx = 0;
            y_max_idx = 0;
            for (int j = 0; j < loss_nn.z[loss_nn.num_layer - 1].row; j++)
            {
                if (z_max < (loss_nn.z[loss_nn.num_layer - 1].element[j][i]))
                {
                    z_max_idx = j;
                    z_max = loss_nn.z[loss_nn.num_layer - 1].element[j][i];
                }
                if (y_max < y.at(i).at(j))
                {
                    y_max_idx = j;
                    y_max = y.at(i).at(j);
                }
            }
            if (z_max_idx != y_max_idx)
            {
                count++;
            }
        }
        miss_rate = (double) count / x.size();
        cross_entropy = loss_nn.batch_error->cross_entropy;
    }
};

#endif //DNN_EPOCH_LOSS_H
