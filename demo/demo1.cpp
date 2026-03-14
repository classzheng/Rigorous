#include "rigorous.h"
using namespace Rigorous;
using namespace std;
int main() {
	srand(time(nullptr));
	std::vector<int> arg={2, 4, 1};
	NeuralNetwork::NetworkUnion<double> nn(arg, 0.1,
	   [](double x){return std::tanh(x);}, // hidden layer: tanh
	   [](double x){return 1.0 - std::tanh(x)*std::tanh(x);} // tanh derivative
    );
	std::vector<std::vector<double>> inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
	std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};
	
	for(int epoch = 0; epoch < 10000; epoch++) {
		nn.train(std::make_pair(inputs,targets));
	}
	cout << "\nResult: " << endl;
	for(auto& in : inputs) {
		auto res = nn.forward(in);
		cout << in[0] << " XOR " << in[1] << " = " << res[0] << endl;
	}
	
	return 0;
}
