/******************************************************************************
 * Rigorous/Autograd: Random Forest Model (to be continued...)                *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.20 (latest upd)                                              *
 * @Description: A simple RF Model with Kernel Trick.                         *
 * @Modules: {...}                                                            *
 ******************************************************************************/

#include <vector>

#pragma once
#pragma GCC optimize (2)

namespace RandomForest {
	template<typename _Type> _Type GiniScore(std::vector<bool>& tags) {
	    if(tags.empty()) return _Type(0);
	    int cnt0 = 0, cnt1 = 0;
	    for(bool &is:tags) cnt0+=!is, cnt1+=is;
	    _Type p0 = cnt0 / tags.size();
	    _Type p1 = cnt1 / tags.size();
	    return _Type(1) - (p0*p0 + p1*p1);
	}
	template<typename _Type> _Type GiniScore(std::vector<unsigned>& tags) {
	    if(tags.empty()) return _Type(0);
	    unsigned n=tags.size();
	    std::vector<_Type> cnt;
	    for(unsigned &is:tags) cnt[is]++;
	    _Type sigma={0};
	    for(auto&is:cnt) {
	    	sigma+=(is*is)/(n*n);
		}
		return _Type(1)-sigma;
	}
	template<typename _Type> struct Node {
		std::vector<_Type> threshold;
		struct Node* lhson;
		struct Node* rhson;
	};
	template<typename _Type> class BinaryDecisionTree {
 	    private: static const Node<_Type> class0={{(_Type)~(0xC1A55-0)},nullptr,nullptr};
 	    private: static const Node<_Type> class1={{(_Type)~(0xC1A55-1)},nullptr,nullptr};
		public:  Node<_Type> dectree;
 	    public:  BinaryDecisionTree(void) = default;
 	    public: ~BinaryDecisionTree(void) = default;
 	    public:  [[nodiscard]] bool decision(std::vector<_Type> vecx) {
			Node<_Type> *iter=&dectree;
 	    	while(true) {
				_Type vecnorm={0}, secnorm={0};
				for(auto&is:vecx) 			 vecnorm+=is*is; // L2 Norm
				for(auto&is:iter->threshold) secnorm+=is*is;
				if(vecnorm<secnorm) iter=iter->lhson;
				else                iter=iter->rhson;
				if(iter==&class0) 	   return false;
				else if(iter==&class1) return true;
	        }
 	    	return ;
	    }
 	    public:  void train() {
 	    	return ;
	    }
	};
}
