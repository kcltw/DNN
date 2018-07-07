#ifndef DNN_LOSS_H
#define DNN_LOSS_H

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

template<typename T>
class loss
{
public:
    double cross_entropy;
    matrix<T> *er;
    loss()
    {};

    ~loss()
    {};


    void eval_error(std::vector<std::vector<T> > &y, matrix<T> &z)
    {
        er = new matrix<T>(y.size(), y.at(0).size());
        for (int i = 0; i < y.size(); i++)
        {
            for (int j = 0; j < y.at(i).size(); j++)
            {
                er->element[i][j] = y.at(i).at(j) - (z.element[j][i]);
            }
        }
        cross_entropy=0;
        auto temp_m = matrix<T>(y.size(),y.at(0).size());
        auto log_z = matrix<T>(z.row,z.col);
        log_z.assign(z);
        log_z.matrix_log();

        for(int i=0;i<y.size();i++)
        {
            for(int j=0;j<y.at(i).size();j++)
            {
                temp_m.element[i][j]=y.at(i).at(j)*(log_z.element[j][i]);
                cross_entropy=cross_entropy+temp_m.element[i][j];
            }
        }
        cross_entropy=-1*cross_entropy/y.size();
    }

};

#endif //DNN_LOSS_H
