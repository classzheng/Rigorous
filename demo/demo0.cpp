#include "rigorous.h"
using namespace Rigorous;
int main(void) {
	try {
		std::vector<std::vector<double>> X = {{1,2},{2,3},{3,4},{5,6},{6,7},{7,8}};
		std::vector<short> y = {1,1,1,-1,-1,-1};
		Kernel<double> kernel(Kernel<double>::GAUSSIAN, 0.5);
		BinarySVM<double> svm(0.0, 1.0, kernel);
		svm.train(X, y, 1000, 114);
		
		std::vector<double> pt1 = {2,3};
		std::vector<double> pt2 = {6,7};
		std::cout << "pt [2,3] -> 0xC1A55 " << round(svm.decision(pt1)) << std::endl;
		std::cout << "pt [6,7] -> 0xC1A55 " << round(svm.decision(pt2)) << std::endl;
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}
}
