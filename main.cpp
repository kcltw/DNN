#include <iostream>
#include <vector>
#include <fstream>
#include<string>
#include <cstdlib>
#include"matrix.h"
#include"epoch_loss.h"
#include <ctime>
#include <chrono>

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
void ReadMNIST(std::string img_path,int num_img,int dim_img,std::vector<std::vector<double>> &img,
               std::string label_path,int num_label,int dim_label,std::vector<std::vector<double>> &label )
{
    img.resize(num_img,std::vector<double>(dim_img));
    label.resize(num_label,std::vector<double>(dim_label));

    std::ifstream img_file(img_path,std::ios::binary);
    if(img_file.is_open())
    {
        int header = 0;
        int count = 0;
        int n_rows = 0;
        int n_cols = 0;
        img_file.read((char*)&header,sizeof(header));
        header = ReverseInt(header);
        img_file.read((char*)&count,sizeof(count));
        count = ReverseInt(count);
        //std::cout<<count<<std::endl;
        if(header!=2051)
        {
            std::cerr<<"Invalid image file header";
            exit(0);
        }
        if(count<num_img)
        {
            std::cerr<<"too many images";
            exit(0);
        }
        img_file.read((char*)&n_rows,sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        img_file.read((char*)&n_cols,sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i =0 ; i< num_img ;i++)
        {
            for(int r =0; r< n_rows ; r++)
            {
                for(int c=0; c< n_cols; c++)
                {
                    unsigned char temp = 0;
                    img_file.read((char*)&temp,sizeof(temp));
                    img[i][(n_rows*r)+c]= (double)temp/255;
                }
            }
        }
    }
    else
    {
        std::cout<<"file can't open"<<std::endl;
    }
    std::ifstream label_file (label_path,std::ios::binary);
    if (label_file.is_open())
    {
        int header=0;
        int count=0;
        label_file.read((char*)&header,sizeof(header));
        header= ReverseInt(header);
        label_file.read((char*)&count,sizeof(count));
        count= ReverseInt(count);
        if(header!=2049)
        {
            std::cerr<<"Invalid label file header";
            exit(0);
        }
        if(count<num_img)
        {
            std::cerr<<"too many images";
            exit(0);
        }
        for(int i=0;i<num_label;i++)
        {
            unsigned char temp=0;
            label_file.read((char*)&temp,sizeof(temp));
            for(int r=0;r<10;r++)
            {
                label[i][r]=0;
            }
            label[i][(int)temp]= 1;
        }
    }
    else
    {
        std::cerr<<"Error file not open";
    }

}

void loadbatches(int numofbatch,int batch_size,std::vector< std::vector<double> > &batch_x,std::vector< std::vector<double> > &batch_y,
                 std::vector< std::vector<double> > train_x,std::vector< std::vector<double> > train_y)
{
    std::copy(train_x.begin()+batch_size*numofbatch,train_x.begin()+batch_size*(numofbatch+1),batch_x.begin());
    std::copy(train_y.begin()+batch_size*numofbatch,train_y.begin()+batch_size*(numofbatch+1),batch_y.begin());
}
int main()
{
    int num_layer=4;
    double learning_rate=0.1;
    int epochs=10;
    int batch_size=100;
    int num_train_data = 500;
    int num_test_data = 500;
    std::vector<int> ls;
    int input=0;
    ls.push_back(784);
    ls.push_back(300);
    ls.push_back(100);
    ls.push_back(10);

//    std::cout<<"Enter the <num_layer> <learning_rate>  <epochs> <batch_size>"<<std::endl;
    //std::cin>> num_layer>> learning_rate >> epochs >> batch_size;
    //std::cout<<num_layer<<" "<<learning_rate<<" "<<epochs<<" "<<batch_size<<" "<<std::endl;

//    for(int i=0;i<num_layer;i++)
//    {
//        int size=0;
//        cin>>size;
//        ls.push_back(size);
//    }

    std::string train_x_filename = "C:\\Users\\KCL\\CLionProjects\\DNN\\train-images.idx3-ubyte";
    std::string train_y_filename = "C:\\Users\\KCL\\CLionProjects\\DNN\\train-labels.idx1-ubyte";
    std::string test_x_filename = "C:\\Users\\KCL\\CLionProjects\\DNN\\t10k-images.idx3-ubyte";
    std::string test_y_filename = "C:\\Users\\KCL\\CLionProjects\\DNN\\t10k-labels.idx1-ubyte";

    std::vector< std::vector<double> > train_x,train_y,test_x,test_y;
    ReadMNIST(train_x_filename,num_train_data,784,train_x,train_y_filename,num_train_data,10,train_y);
    ReadMNIST(test_x_filename,num_test_data,784,test_x,test_y_filename,num_test_data,10,test_y);

    //neural<double>* nn;
    epoch_loss<double> train_loss,test_loss;

    auto nn = neural<double>(num_layer,learning_rate,batch_size,ls);

    int num_batches=num_train_data/batch_size;
    std::vector<std::vector<double>> batch_x,batch_y;
    batch_x.resize(batch_size,std::vector<double>(784));
    batch_y.resize(batch_size,std::vector<double>(10));
    //timer start
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // process of training
    for(int i=0;i<epochs;i++)
    {
        std::cout<<"epoch =========> "<<i+1<<std::endl;
        for(int j=0;j<num_batches;j++)
        {
            loadbatches(j,batch_size,batch_x,batch_y,train_x,train_y);
            nn.feedforward(batch_x,batch_y);
            nn.backpro();
        }
        train_loss.eval_loss(nn,train_x,train_y);
        std::cout << "epoch loss = " << train_loss.cross_entropy <<std::endl;
        std::cout << "training accuracy = " <<1-train_loss.miss_rate<<std::endl;
    }
    // process of testing
    test_loss.eval_loss(nn,test_x,test_y);
    std::cout<<"testing accuracy = "<<1-test_loss.miss_rate<<std::endl;

    //timer end
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::cout << "Printing took "
              << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
              << "s.\n";


    return 0;
}