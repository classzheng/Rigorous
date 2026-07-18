/******************************************************************************
 * Rigorous/Autograd: Random Forest Model (to be continued...)                *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.7.18 (latest upd)                                              *
 * @Description: A simple RF Model with Kernel Trick.                         *
 * @Modules: {...}                                                            *
 ******************************************************************************/

#include "mathlib.hpp"
#include <algorithm>

#pragma once
#pragma GCC optimize (2)

namespace RandomForest {
	using namespace Mathlib;
	namespace Handle {
		template<typename _Type> _Type GiniScore(const std::vector<bool>& tags) {
		    if(tags.empty()) return _Type(0);
		    _Type cnt0 = 0, cnt1 = 0;
		    for(const bool &is:tags) cnt0+=!is, cnt1+=is;
		    _Type p0 = cnt0 / tags.size();
		    _Type p1 = cnt1 / tags.size();
		    return _Type(1) - (p0*p0 + p1*p1);
		}
		template<typename _Type> _Type GiniScore(const std::vector<unsigned>& tags) {
		    if(tags.empty()) return _Type(0);
		    unsigned n=tags.size();
		    std::vector<_Type> cnt(1145,0);
		    for(const unsigned &is:tags) cnt[is]++;
		    _Type sigma={0};
		    for(auto&is:cnt) {
		    	sigma+=(is*is)/(n*n);
			}
			return _Type(1)-sigma;
		}
	    template<typename _Type> _Type L2Square(std::vector<_Type> vecx) {
			_Type s={0};
			for(auto&is:vecx) s+=is*is;
			return s;
		}
	}
	template<typename _Type> struct Node {
		std::vector<_Type> threshold;
		struct Node* lhson = nullptr;
		struct Node* rhson = nullptr;
		bool isLeaf = false;
		bool leaftag = false;
	};
	template<typename _Type> class BinaryDecisionTree {
 	    private: Node<_Type> class0, class1;
		public:  Node<_Type> dectree;
 	    public:  BinaryDecisionTree(void) {
			class0.threshold={(_Type)~(0xC1A55-0)}, class0.isLeaf=true, class0.leaftag=false;
			class1.threshold={(_Type)~(0xC1A55-1)}, class1.isLeaf=true, class1.leaftag=true;
			return ;
		}
 	    public: ~BinaryDecisionTree(void) {
			std::function<void(Node<_Type>*)> release = [&](Node<_Type> *iter) {
				if(!iter) return ;
				if(iter->lhson && iter->lhson!=&class0 && iter->lhson!=&class1) {
					release(iter->lhson);
					delete iter->lhson;
				}
				if(iter->rhson && iter->rhson!=&class0 && iter->rhson!=&class1) {
					release(iter->rhson);
					delete iter->rhson;
				}
				return ;
			};
			release(&dectree);
			return ;
		}
 	    public:  [[nodiscard]] bool decision(std::vector<_Type> vecx) {
			using namespace Handle;
			Node<_Type> *iter=&dectree;
 	    	while(true) {
				if(L2Square(vecx)<L2Square(iter->threshold)) iter=iter->lhson;
				else                						 iter=iter->rhson;
				if(iter==&class0) 	   return false;
				else if(iter==&class1) return true;
				else if(iter==nullptr) return false;
	        }
	    }
 	    public:  void train(const std::vector<std::vector<_Type>> &vecx, const std::vector<bool> &tagy, const int maxdepth) {
			using namespace Handle;
			if(vecx.empty() || tagy.empty()) {
				dectree.lhson = &class0;
				dectree.rhson = &class0;
				return ;
			}
			int n = (int)vecx.size();
			std::function<void(Node<_Type>*, const std::vector<int>&, int)> build;
			build = [&](Node<_Type>* node, const std::vector<int>& idx, int depth)->void {
				if(idx.empty()) {
					node->lhson = &class0;
					node->rhson = &class0;
					return ;
				}
				bool allsame = true;
				for(size_t i=1;i<idx.size();++i) if(tagy[idx[i]]!=tagy[idx[0]]) { allsame=false; break; }
				if(depth>=maxdepth || allsame) {
					int cnt=0;
					for(int id:idx) if(tagy[id]) cnt++;
					node->lhson = node->rhson = (cnt*2>=(int)idx.size() ? &class1 : &class0);
					return ;
				}
				std::vector<_Type> norms;
				for(auto &id:idx) {
					_Type s=0;
					for(auto &v: vecx[id]) s+=v*v;
					norms.push_back(s);
				}
				std::vector<_Type> tmp = norms;
				std::sort(tmp.begin(), tmp.end());
				node->threshold.clear(); node->threshold.push_back((_Type)std::sqrt((double)tmp[tmp.size()/2]));
				std::vector<int> left, right;
				for(int k=0;k<idx.size();++k) {
					if(norms[k]<tmp[tmp.size()/2]) left.push_back(idx[k]);
					else 						   right.push_back(idx[k]);
				}
				if(left.empty()) node->lhson = &class0;
				else {
					Node<_Type> *l = new Node<_Type>();
					node->lhson=l;
					build(l, left, depth+1);
				}
				if(right.empty()) node->rhson=&class1;
				else {
					Node<_Type>*r=new Node<_Type>();
					node->rhson=r;
					build(r,right,depth+1);
				}
				return ;
			};
			std::vector<int> all(n);
			for(int i=0;i<n;++i) all[i]=i;
			build(&dectree, all, 0);
 	    	return ;
	    }
		public:  void vision(void) {
   			std::function<void(Node<_Type>*, int)> render = [&](Node<_Type> *iter, int depth) {
				for(int k = 0; k < depth; k++) std::cout << "  ";
			    if(iter==&class0) {
				    std::cout << "0xC1A55-0.\n";
					return ;
				}else if(iter==&class1) {
				    std::cout << "0xC1A55-1.\n";
					return ;
				}
			    std::cout<<depth<<" - [ ";
				for(auto&is:iter->threshold) std::cout<<is<<" ";
				std::cout << "].\n";
				depth++;
				render(iter->lhson,depth), render(iter->rhson,depth);
				return ;
			};
			render(&dectree,0);
			return ;
		}
	};
	template<typename _Type> class RandomForest {
		public: RandomForest(void) = default;
		public: ~RandomForest(void) = default;
	};
}
