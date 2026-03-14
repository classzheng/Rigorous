/******************************************************************************
 * Rigorous/NeuralNetwork: A c++11 implementation of BP-ANN                   *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.20 (latest upd)                                              *
 * @Description: A c++11 implementation of BP-ANN                             *
 * @Modules: {                                                                *
 *    NeuralNetwork::UnitLayer, NeuralNetwork::NetworkUnion                   *
 * }                                                                          *
 ******************************************************************************/

#include "mathlib.hpp"

 #pragma once
 #pragma GCC optimize (2)

namespace NeuralNetwork {
	 
	using namespace Mathlib;
	 
	using initType =  double;
	template<typename _Type> class UnitLayer;
	template<typename _Type> class NetworkUnion;
	template<typename _Type> using Matrix = std::vector<std::vector<_Type>>;
	template<typename _Type> using Function = std::function<_Type(const _Type)>;
	
	template<typename _Type> class UnitLayer {
		public: Matrix<_Type> weights;
		public: std::vector<_Type> inputs;
		public: std::vector<_Type> outputs;
		public: std::vector<_Type> mtbias;
		public: std::vector<_Type> mtdelta;
		public: const Function<_Type> &actfunction;
		public: const Function<_Type> &drvfunction;
	public: UnitLayer(int dim1, int dim2, const Function<_Type>& f, const Function<_Type>& g):
		inputs(dim1), outputs(dim2), actfunction(f), drvfunction(g), mtbias(dim2), mtdelta(dim2) {
			weights.resize(dim2,std::vector<_Type>(dim1));
			for(int i = 0; i < dim2; i++) {
				for(int j = 0;  j < dim1; j++) {
					weights[i][j]=_Type(genrandom());
				}
				mtbias[i]=genrandom();
			}
		}
		
		public: std::vector<_Type> forward(const std::vector<_Type>& tin) {
			inputs=tin;
			for(int i = 0; i < outputs.size(); i++) {
				_Type sigma = mtbias[i];
				for(int j = 0; j < inputs.size(); j++) sigma+=inputs[j]*weights[i][j];
				outputs[i]=actfunction(sigma);
			}
			return outputs;
		}
	};
	
	template<typename _Type> class NetworkUnion {
		public: std::vector<UnitLayer<_Type>> hidden;
		public: const Function<_Type> &actfunction;
		public: const Function<_Type> &drvfunction;
		public: _Type ln_rate;
	public: NetworkUnion(std::vector<int> arguments, const _Type lnr, const Function<_Type>& f, const Function<_Type>& g):
		ln_rate(lnr), actfunction(f), drvfunction(g) {
			for(int i = 1; i < arguments.size(); i++) 
				hidden.push_back(UnitLayer<_Type>(arguments[i-1],arguments[i],actfunction,drvfunction));
		}
		public: std::vector<_Type> forward(const std::vector<_Type>& input) {
			std::vector<_Type> tempi=input;
			for(auto&is:hidden) tempi=is.forward(tempi);
			return tempi;
		}
		public: void backward(std::vector<_Type>& target) {
			for(int i = hidden.size()-1; i>=0; i--) {
				UnitLayer<_Type> &current = hidden[i];
				if(i==hidden.size()-1) {
					for(int j = 0; j < current.outputs.size(); j++) {
						current.mtdelta[j] = (target[j]-current.outputs[j]) * current.drvfunction(current.outputs[j]);
					}
				} else {
					for(int j = 0; j < current.outputs.size(); j++) {
						_Type delta = {0};
						for(int l = 0; l < hidden[i+1].outputs.size(); l++) 
							delta += hidden[i+1].weights[l][j]*hidden[i+1].mtdelta[l];
						current.mtdelta[j]=delta*current.drvfunction(current.outputs[j]);
					}
				}
			}
			for(int i = 0; i < hidden.size(); i++) {
				UnitLayer<_Type> &current = hidden[i];
				const std::vector<_Type> &inputs = (i==0) ? current.inputs : hidden[i-1].outputs;
				for(int j = 0; j < current.outputs.size(); j++) {
					for(int l = 0; l < inputs.size(); l++) 
						current.weights[j][l]+=ln_rate*current.mtdelta[j]*inputs[l];
					current.mtbias[j]+=ln_rate*current.mtdelta[j];
				}
			}
		}
		public: void train(std::pair<Matrix<_Type>, Matrix<_Type>> data) {
			for(int i = 0; i < data.first.size(); i++) {
				this->forward(data.first[i]);
				this->backward(data.second[i]);
			}
		}
	};
};
