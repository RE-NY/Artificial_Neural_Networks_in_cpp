#include<iostream>
#include<ctime>
#define NN_IMPLEMENTATION_ // also imports the implementation for the nn.hpp
#include "nn.hpp" // by default i.e. without #define NN_IMPLEMENTATION it only acts as header file
                 // and gives only declarations
using namespace std;

#define MAT_PRINT(m) m.matrix_print(#m, 0);
#define ARR_SIZE(arr) sizeof(arr)/sizeof(arr[0]);


int main(){
    srand(69);
    
    //Taking input matrix from the user
    size_t row, col;
    cin >> row >> col;
    Matrix input(row, col);
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            cin >> input.mat[i][j];
        }
    }
    //Taking output matrix from user
    Matrix output(row, 1);
    for(int i=0; i<row; i++) cin >> output.mat[i][0];

    //Input and output matrices created

    float epsilon = 1e-1;
    float learning_rate = 1e-2;
    size_t num_iterations = 1e3;

    //Making of architecture
    size_t arr[] = {2,10,5, 1};
    size_t arrSize = ARR_SIZE(arr);
    NN nn(arr, arrSize);
    NN gradient(arr, arrSize);
    nn.make_random(-1, 1);

    //Training
    cout << "cost before training : " << nn_cost(nn, input, output) << endl;

    for(int i=0; i<num_iterations; i++){
        nn_learn(nn, gradient, input, output, learning_rate, epsilon);
    }
    //final cost value
    cout << "cost after training : " << nn_cost(nn, input, output) << endl;


    
    // cout << "Prediction : \n";
    // for(int i=0; i<2; i++){
    //     for(int j=0; j<2; j++){
    //         nn.as[0]->mat[0][0] = i;
    //         nn.as[0]->mat[0][1] = j;
    //         nn_forwardPass(nn);
    //         cout << i << " ^ " << j << NN_OUTPUT(nn)->mat[0][0];
    //     }
    // }


    // NN_PRINT(nn);
    // nn_learn(nn, gradient, input, output, learning_rate, epsilon);
    // cout << "cost : " << nn_cost(nn, input, output) << endl;
    // NN_PRINT(gradient);




    return 0;
    //*****************************///Making the XOR model///**************************

    //All required memory allocated after the below in run
    Matrix a0(1,2); // represents one training example
    a0.mat[0][0] = 1.00;
    a0.mat[0][1] = 1.00;
    Matrix w1(2,2);
    Matrix b1(1,2);

    Matrix a1(1,2);
    Matrix w2(2,1);
    Matrix b2(1,1);

    Matrix a2(1,1);

    //Below are the parameters...
    w1.make_random(-1, 1);
    b1.make_random(-1, 1);
    w2.make_random(-1, 1);
    b2.make_random(-1, 1);

    //forward pass for first layer
    matrix_mul(a1, a0, w1);
    matrix_sum(a1, b1);
    matrix_sigmoidf(a1);
    
    //forward pass for second layer
    matrix_mul(a2, a1, w2);
    matrix_sum(a2, b2);
    matrix_sigmoidf(a2);

    //a2 is the final prediction for this one example!.
    MAT_PRINT(a0);
    MAT_PRINT(w1);
    MAT_PRINT(b1);
    MAT_PRINT(a1);
    cout << "-------------------------------------------" << endl;
    MAT_PRINT(w2);
    MAT_PRINT(b2);
    MAT_PRINT(a2);


}