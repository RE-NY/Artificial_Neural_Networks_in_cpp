#ifndef NN_H_
#define NN_H_
//contains the header part..declarations etc.

#include<cstddef>
#include<cstdlib>
#include<cassert>
#include<cmath>
using namespace std;


#define MAT_PRINT(m) m.matrix_print(#m, 0);

#define NN_PRINT(nn) (nn).nn_print(#nn);
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).layers_count]




class Matrix{
    public:
        size_t rows;
        size_t cols;
        float** mat;

        Matrix();
        Matrix(size_t rows, size_t cols); // will also act as matrix_alloc() i.e. to allocate memory

        void matrix_print(string s, size_t indent);
        void make_random(float a, float b);
};

float randFloatf(float a, float b);
float sigmoidf(float x);
void matrix_sigmoidf(Matrix& m);
void matrix_sum(Matrix& dest, Matrix& m2);
void matrix_mul(Matrix& dest, Matrix& m1, Matrix& m2);
void matrix_copy(Matrix& dst, Matrix& src);
Matrix matrix_getRow(Matrix m, size_t row);
Matrix matrix_getCol(Matrix m, size_t col);




class NN{ 
// NN is imagined abstractly as ordered sequence of matrices which are store in ws, bs and as
//in an order(defined by the architecture) and can be efficiently utilized.
    public:
        size_t layers_count;
        Matrix** ws; // array of weights 
        Matrix** bs; // array of biases
        Matrix** as; // array of activations(Its size is 1+count, to accomodate a0 : this input example)

        NN(size_t* archArr, size_t archSize);
        void nn_print(string s);
        void make_random(float a, float b);
};


void nn_forwardPass(NN& nn);
float nn_cost(NN nn, Matrix in, Matrix out);
void nn_finiteDifference(NN& nn, NN& gradient, float& epsilon, Matrix& in, Matrix& out);
void nn_gradinetDescent(NN& nn, NN& gradient, float& alpha);
void nn_learn(NN& nn, NN& gradient, Matrix& in, Matrix& out, float& alpha, float epsilon);









#endif //NN_H_

#ifdef NN_IMPLEMENTATION_
//contains the c++ part..use of above declarations

float randFloatf(float a, float b){
    return ((float)rand()/(float)RAND_MAX)*(b-a) + a;
}
float sigmoidf(float x){
    return 1.f/(1.f+expf(-1*x));
}
void matrix_sigmoidf(Matrix& m){
    for(size_t i=0; i<m.rows; i++){
        for(size_t j=0; j<m.cols; j++){
            float val = sigmoidf(m.mat[i][j]);
            m.mat[i][j] = val;
        }
    }    
}


Matrix::Matrix(){
    this->rows = 0;
    this->cols = 0;
    float** arr = new float*[rows];
    for(size_t i=0; i<rows; i++) arr[i] = new float[cols];
    this->mat = arr;

}
Matrix::Matrix(size_t rows, size_t cols){
    this->rows = rows;
    this->cols = cols;
    float** arr = new float*[rows];
    for(size_t i=0; i<rows; i++) arr[i] = new float[cols];
    this->mat = arr;
}

void Matrix::matrix_print(string s, size_t indent){
    for(size_t k=0; k<indent/2; k++) cout << " ";
    cout << s << " = [\n";
    for(size_t i=0; i<rows; i++){
        for(size_t k=0; k<indent; k++) cout << " ";
        for(size_t j=0; j<cols; j++){
            cout << mat[i][j] << "   ";
        }
        cout << endl;
    }
    for(size_t k=0; k<indent/2; k++) cout << " ";
    cout << "]\n";
}
void Matrix::make_random(float a, float b){
    for(size_t i=0; i<rows; i++){
        for(size_t j=0; j<cols; j++){
            mat[i][j] = randFloatf(a, b);
        }
    }
}

void matrix_sum(Matrix& dest, Matrix& m2){ // we have not made new matrix because of memory allocation....
    assert(dest.rows == m2.rows);
    assert(dest.cols == m2.cols);
    for(size_t i=0; i<dest.rows; i++){
        for(size_t j=0; j<dest.cols; j++){
            dest.mat[i][j] += m2.mat[i][j];
        }
    }
}
void matrix_mul(Matrix& dest, Matrix& m1, Matrix& m2){
    assert(m1.cols == m2.rows); //we will do dest = m1 X m2
    assert(dest.rows == m1.rows);
    assert(dest.cols == m2.cols);

    for(size_t i=0; i<m1.rows; i++){
        for(size_t j=0; j<m2.cols; j++){
            for(size_t k=0; k < m1.cols; k++){
                dest.mat[i][j] += ((m1.mat[i][k])*(m2.mat[k][j]));
            }
        }
    }
}
void matrix_copy(Matrix& dst, Matrix& src){

    assert(dst.rows == src.rows);
    assert(dst.cols == src.cols);

    for(size_t i=0; i<src.rows; i++){
        for(size_t j=0; j<src.cols; j++){
            dst.mat[i][j] = src.mat[i][j];
        }
    }
}
Matrix matrix_getRow(Matrix m, size_t row){ // row is the index of row so it should be from 0 to m.row-1
    assert(row >= 0);
    assert(row < m.rows);

    Matrix dest(1, m.cols); 
    for(int j=0; j<m.cols; j++){
        dest.mat[0][j] = m.mat[row][j];
    }

    return dest;   
}
Matrix matrix_getCol(Matrix m, size_t col){ // col is the index of row so it should be from 0 to m.col-1
    assert(col >= 0);
    assert(col < m.cols);

    Matrix dest(m.rows, 1); 
    for(int i=0; i<m.rows; i++){
        dest.mat[i][col] = m.mat[i][col];
    }

    return dest;
}





//-----------------NN class implemetation------------------------//

NN::NN(size_t* archArr, size_t archSize){
    //note that architecture array has size 1 more that layer count as it also contain "input layer"
    assert(archSize > 0);
    layers_count = archSize-1;
    
    ws = new Matrix*[layers_count];
    bs = new Matrix*[layers_count];
    as = new Matrix*[archSize];

    //archArr = {2,4,5,2,1};

    as[0] = new Matrix(1, archArr[0]);
    for(size_t i=1; i<archSize; i++){
        as[i] = new Matrix(1, archArr[i]);
        ws[i-1] = new Matrix(as[i-1]->cols, archArr[i]);
        bs[i-1] = new Matrix(1, archArr[i]);
    }

}
void NN::nn_print(string s){
    cout << s << " = [\n";
    for(size_t i=0; i<layers_count; i++){
        ws[i]->matrix_print("w"+to_string(i+1), 4);
        bs[i]->matrix_print("b"+to_string(i+1), 4);
    }
    cout << "]\n";
}
void NN::make_random(float a, float b){
    for(size_t i=0; i<layers_count; i++){
        ws[i]->make_random(a, b);
        bs[i]->make_random(a, b);
    }
}

void nn_forwardPass(NN& nn){
    for(size_t i=0; i < nn.layers_count; i++){
        matrix_mul(*nn.as[i+1], *nn.as[i], *nn.ws[i]);
        matrix_sum(*nn.as[i+1], *nn.bs[i]);
        matrix_sigmoidf(*nn.as[i+1]);
    }
}
float nn_cost(NN nn, Matrix in, Matrix out){ // computes cost for all training samples for one pass.
    //notice that in in the input matrix not including the last column!(It is in the out matrix).
    assert(in.rows == out.rows);
    //assert(nn.as[0]->cols + NN_OUTPUT(nn)->cols == out.cols);

    float cost = 0;
    size_t sampleSize = in.rows;
    for(size_t i=0; i<sampleSize; i++){
        Matrix ith_sample = matrix_getRow(in, i);
        nn.as[0] = &ith_sample;
        nn_forwardPass(nn);
        //NN_OUTPUT(nn);
        size_t features = out.cols;
        for(int j=0; j<features; j++){
            float d = NN_OUTPUT(nn)->mat[0][j] - out.mat[0][j];
            cost += d*d;
        }
    }
    cost/=sampleSize;
    return cost;
}
void nn_finiteDifference(NN& nn, NN& gradient, float& epsilon, Matrix& in, Matrix& out){
    //updates the gradient NN which has same shape and size as nn NN.
    float saved = 0.f; // to temporarely store a value
    float costBefore = nn_cost(nn, in, out);

    //cout << "cost : " << costBefore << endl;

    for(size_t i=0; i<nn.layers_count; i++){
        // for weight parameters
        for(size_t j=0; j<nn.ws[i]->rows; j++){
            for(size_t k=0; k<nn.ws[i]->cols; k++){
                saved = nn.ws[i]->mat[j][k];
                nn.ws[i]->mat[j][k] += epsilon;
                float costAfterWiggle = nn_cost(nn, in, out);
                gradient.ws[i]->mat[j][k] = (costAfterWiggle - costBefore)/epsilon ;
                nn.ws[i]->mat[j][k] = saved;
            }
        }
        //for bias parameters
        for(size_t j=0; j<nn.bs[i]->rows; j++){
            for(size_t k=0; k<nn.bs[i]->cols; k++){
                saved = nn.bs[i]->mat[j][k];
                nn.bs[i]->mat[j][k] += epsilon;
                float costAfterWiggle = nn_cost(nn, in, out);
                gradient.bs[i]->mat[j][k] = (costAfterWiggle - costBefore)/epsilon ;
                nn.bs[i]->mat[j][k] = saved;
            }
        }
    }
}
void nn_gradinetDescent(NN& nn, NN& gradient, float& alpha){
    //updates the weight of nn NN using the gradient NN which has same shape and size as nn NN.
    for(size_t i=0; i<nn.layers_count; i++){
        // for weight parameters
        for(size_t j=0; j<nn.ws[i]->rows; j++){
            for(size_t k=0; k<nn.ws[i]->cols; k++){
                nn.ws[i]->mat[j][k] -= alpha*(gradient.ws[i]->mat[j][k]);
            }
        }
        //for bias parameters
        for(size_t j=0; j<nn.bs[i]->rows; j++){
            for(size_t k=0; k<nn.bs[i]->cols; k++){
                nn.bs[i]->mat[j][k] -= alpha*(gradient.bs[i]->mat[j][k]);
            }
        }
    }
}

void nn_learn(NN& nn, NN& gradient, Matrix& in, Matrix& out, float& alpha, float epsilon){
    nn_finiteDifference(nn, gradient, epsilon, in, out);
    nn_gradinetDescent(nn, gradient, alpha);
}











#endif //NN_IMPLEMENTATION_

//#endif //NN_H_
