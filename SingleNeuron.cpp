#include<iostream>
#include<vector>
#include<ctime> //to set new seed everytime to PRNG
#include<cmath>
using namespace std;

vector<vector<float>> data = {
                        {1,2},
                        {2,4},
                        {3,6},
                        {4,8},
                        {5,10}
                    };
//We are trying to make a simple model to predict for a given input what should be the output...
//Basic idea of this prediction is boiled down in ML as optimizing some "parameters" for an optimization problem
// In this case may be we can try y = w*x where w is a "parameter". and we will maximize the "loss" function

float get_random(){
    return (float)rand()/RAND_MAX ;
}

vector<float> forwardPass(vector<vector<float>>& data, float w){
    vector<float> ans;
    for(vector<float> example : data){
        ans.push_back(w*example[0]);
    }
    return ans;
}
float calculateCost(vector<vector<float>>& data, vector<float>& prediction){
    float val = 0.0;
    for(int i=0; i<data.size(); i++){
        val += pow((data[i][1]-prediction[i]), 2);
    }
    return val/data.size();
}

float computeDerivativeApprox(vector<vector<float>>& data, float w, float epsilon){
    vector<float> predictionBefore = forwardPass(data, w);
    vector<float> predictionAfter = forwardPass(data, w+epsilon);
    // basically derivative formula
    float dw = (calculateCost(predictionAfter) - calculateCost(predictionBefore))/epsilon;
    return dw;

}
float computeDerivativeFormula(vector<vector<float>>& data, vector<float>& prediction, float w){
    float ans = 0;
    for(int i=0; i<data.size(); i++){
        ans += (2*data[i][0]*(w*data[i][0] - data[i][1]));
    }
    return ans/data.size();
}

float paramUpdate(float derivative, float w, float alpha){
    w -= (alpha*derivative);
    return w;
}
void oneUpdate(vector<vector<float>>& data, float& initial_w, float alpha, int i){
    if(i%10 == 1) cout << "w at start : " << initial_w  << " ";
    vector<float> inference = forwardPass(data, initial_w);
    if(i%10 == 1) cout << "Cost function value : " << calculateCost(data, inference) << " ";
    float derivative = computeDerivativeFormula(data, inference, initial_w);
    initial_w = paramUpdate(derivative, initial_w, alpha);
    //cout << "final w now : " << initial_w << endl;
    if(i%10 == 1) cout << endl;
}

int main(){

    vector<vector<float>> data = {
                        {1,2},
                        {2,4},
                        {3,6},
                        {4,8},
                        {5,10}
                    };

    //srand(time(0));
    srand(69);
    float w = get_random()*10; // gives random number less than 10.
    //defining the hyperparameters...
    float learning_rate = 0.01;
    float num_iterations = 101;
    int i=1;
    while(i<=num_iterations){
        if(i%10 == 1) cout << "********* " << i << " <-number of iteration ***********" << endl;
        oneUpdate(data, w, learning_rate, i);
        if(i%10 == 1) cout << "----------------------------------------------" << endl;
        i++;
    }
    cout << w << endl;
    
}