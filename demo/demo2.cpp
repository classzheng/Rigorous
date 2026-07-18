#include <iostream>
#include "randomforest.hpp"
int main(void) {
    using namespace RandomForest;
    BinaryDecisionTree<double> tree;

    std::vector<std::vector<double>> X={{0.3,0.7},{-0.2,0.9},{0.2,-0.5},{21.0,35.0},{44.0,19.0}};
    std::vector<bool> Y={0,0,0,1,1};

    tree.train(X, Y, 5);

    std::cout << "Tree structure:\n";
    tree.vision();

    std::vector<std::vector<double>> tests = {
        {0.1, 0.2},
        {0.5, 0.4},
        {18.0, 37.0},
        {46.0, 77.0}
    };

    std::cout << "\nPredictions:\n";
    for(auto &v: tests){
        bool pred = tree.decision(v);
        std::cout << "pt (" << v[0] << "," << v[1] << ") -> " << (pred ? "class1" : "class0") << "\n";
    }

    std::cout << "\nPredictions:\n";
    for(auto &v: X){
        bool pred = tree.decision(v);
        std::cout << "pt (" << v[0] << "," << v[1] << ") -> " << (pred ? "class1" : "class0") << "\n";
    }

    return 0;
}
