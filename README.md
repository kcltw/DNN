# DNN
## This project is implemented by C++11 and paralleled with threading
* **Main.cpp** : load data and execute the main algorithm
* **Matrix.h** : implement all matrix operation included parallization
* **Neural.h** : construct the whole networks and implement its function like activation functions , feed-forward and error-backpropagation
* **Loss.h** : implement a function that calculate the cross-entropy and the error needed to be backpropagated
* **epoch_loss.h** : implement a function that calculate the miss_rate

## Which part be paralled?

* Matrix Multiplication – A * B
* Matrix Assignment – A = B

*Assign each thread has its own (A.row / numthread) tasks and the last thread with (A.row /numthread)+(A.row % numthread) tasks.*
