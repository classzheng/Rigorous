/******************************************************************************
 * Rigorous/HiddenMarkovModel: A c++11 implementation of HMM.                 *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.7.16 (latest upd)                                              *
 * @Description: A c++11 implementation of HMM.                               *
 * @Modules: {                                                                *
 *    NeuralNetwork::UnitLayer, NeuralNetwork::NetworkUnion                   *
 * }                                                                          *
 ******************************************************************************/

#include "mathlib.hpp"

// #pragma once

namespace HiddenMarkovModel {
	using namespace Mathlib;
	using initType =  double;
	template<typename _Type> class HMM {
		public: std::vector<std::vector<_Type>> observeprob;
		public: std::vector<std::vector<_Type>> emittingprob;
		public: std::vector<_Type> hiddenstatus;
		public: std::vector<_Type> observestatus;
		public: HMM(void) = default;
		public: ~HMM(void) = default;
		public: _Type forward(const std::vector<int>& obs) const {
			if(hiddenstatus.size()==0 || obs.empty()) return _Type(0);
			std::vector<_Type> alpha(hiddenstatus.size());
			for(int i=0;i<hiddenstatus.size();i++) alpha[i]=hiddenstatus[i]*observeprob[i][obs[0]];
			for(int t=1;t<obs.size();t++){
				std::vector<_Type> next(hiddenstatus.size());
				for(int j=0;j<hiddenstatus.size();j++){
					_Type s=0;
					for(int i=0;i<hiddenstatus.size();i++) s+=alpha[i]*emittingprob[i][j];
					next[j]=observeprob[j][obs[t]]*s;
				}
				alpha.swap(next);
			}
			_Type prob = 0;
			for(int i=0;i<hiddenstatus.size();i++) prob+=alpha[i];
			return prob;
		}
		public: void train(const std::vector<int> data, const std::vector<int> tag) {
			;
			return ;
		}
	};
}
