#ifndef DNN_MATRIX_H
#define DNN_MATRIX_H

#include<iostream>
#include<iomanip>
#include<stdio.h>
#include<random>
#include<vector>
#include<thread>
#include<math.h>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);

template<typename T>
class matrix
{

public:
    int row;
    int col;
    std::vector<std::vector<T>> element;

    matrix() {}
    matrix(int r, int c) : row(r),
                           col(c),
                           element(r, std::vector<T>(c, 0)) {}

    void show()
    {
        for (auto &i:element)
        {
            for (auto &j:i)
            {
                std::cout << std::setprecision(4) << j << " ";
            }
            std::cout << std::endl;
        }
    }

    matrix<T> operator*(const matrix<T> &m)
    {
        int num_thread = 4;
        int tasks_thread = row / num_thread;
        int rest = row % num_thread;
        std::vector<int> lbs(num_thread+1,0);
        lbs.at(num_thread) = row;
        for(int idx =0; idx<num_thread;idx++)
        {
            int tmp =idx*tasks_thread;
            lbs[idx] = tmp;
        }
        std::vector<std::thread> t;
        matrix<T> result(row, m.col);

        for (int idx = 0; idx < num_thread; idx++)
        {
            t.emplace_back([&](int lb,int ub)
                           {
                                   for (int i = lb; i < ub; i++)
                                   {
                                       for (int j = 0; j < m.col; j++)
                                       {
                                           for (int k = 0; k < col; k++)
                                           {
                                               result.element[i][j] += element[i][k] * m.element[k][j];
                                           }
                                       }
                                   }

                           },lbs.at(idx),lbs.at(idx+1));

        }

        for (int idx = 0; idx < num_thread ;idx++)
        {
            if(t[idx].joinable())
            t[idx].join();
        }

//        matrix<T> result(row, m.col);
//        for (int i = 0; i < row; i++)
//        {
//            for (int j = 0; j < m.col; j++)
//            {
//                for (int k = 0; k < col; k++)
//                {
//                    result.element[i][j] += element[i][k] * m.element[k][j];
//                }
//                //std::cout<<result.element[i][j]<<std::endl;
//            }
//        }

        return result;
    }

    matrix<T> &operator=(const matrix<T> &m)
    {
        row = m.row;
        col = m.col;
        this->element = std::move(m.element);
        return *this;
    }

    void assign(matrix<T> m)
    {
//        row = m.row;
//        col = m.col;
//        this->element = std::move(m.element);
        int num_thread = 4;
        int tasks_thread = row / num_thread;
        int rest = row % num_thread;
        int low_bound;
        std::vector<std::thread> t;
        for (int idx = 0; idx < num_thread; idx++)
        {
            low_bound = idx * tasks_thread;
            t.emplace_back([=]
                           {
                               if (idx != num_thread - 1)
                               {
                                   for (int i = low_bound; i < low_bound + tasks_thread; i++)
                                   {
                                       for (int j = 0; j < col; j++)
                                       {
                                           element[i][j] = m.element[i][j];
                                       }
                                   }
                               } else
                               {
                                   for (int i = low_bound; i < low_bound + tasks_thread + rest; i++)
                                   {
                                       for (int j = 0; j < col; j++)
                                       {
                                           element[i][j] = m.element[i][j];
                                       }
                                   }
                               }
                           });
        }
        for (std::thread &worker: t)
            worker.join();
    }

    matrix<T> operator+(const matrix<T> &m)
    {
        matrix<T> result(row, col);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                result.element[i][j] = element[i][j] + m.element[i][j];
            }
        }
        return result;
    }

    matrix<T> operator-(const matrix<T> &m)
    {
        matrix<T> result(row, col);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                result.element[i][j] = element[i][j] - m.element[i][j];
            }
        }
        return result;
    }

    void setMatrix(std::vector<std::vector<T>> &input)
    {

        for(int i=0;i<input.size();i++) //batch_size
        {
            for(int j=0;j<input.at(i).size();j++) //input dim
            {
                this->element[j][i]=input.at(i).at(j); //有加bias 所以 index要注意
            }
        }
        //this->show();

    }

    void add_bias()
    {
        row += 1;
        element.push_back(std::vector<T>(col, 1));
    }

    void de_bias()
    {
        row -= 1;
        element.pop_back();
    }

    void matrix_exp()
    {
        for (auto &i:element)
        {
            for (auto &j:i)
            {
                j = exp(j);
            }
        }
    }

    void matrix_log()
    {
        for (auto &i:element)
        {
            for (auto &j:i)
            {
                j = log(j);
            }
        }
    }

    void one_over_element()
    {
        for (auto &i:element)
        {
            for (auto &j:i)
            {
                j = 1 / j;
            }
        }
    }

    void multi_by_element(matrix &m1, matrix &m2)
    {
        for (int i = 0; i < m1.row; i++)
        {
            for (int j = 0; j < m1.col; j++)
            {
                element[i][j] = m1.element[i][j] * m2.element[i][j];
            }
        }
    }

    matrix transpose()
    {
        matrix result(col, row);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                result.element[j][i] = element[i][j];
            }
        }
        return result;
    }

    void scale(double c)
    {
        for (auto &i:element)
        {
            for (auto &j:i)
            {
                j = c * j;
            }
        }

    }

};

#endif //DNN_MATRIX_H
