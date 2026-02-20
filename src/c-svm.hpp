/******************************************************************************
 * Rigorous/C-SVM: A c++11 implementation of Binary-SVM & Multiple-SVM        *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.20 (latest upd)                                              *
 * @Description: A c++11 implementation of Binary-SVM & Multiple-SVM          *
 * @Modules: {                                                                *
 *     SVMPackage::Kernel, SVMPackage::BinarySVM, SVMPackage::Multiple-SVM    *
 * }                                                                          *
 ******************************************************************************/

#include <stdexcept>
#include <vector>
#include <cmath>
#include <ctime>
#include <random>

#pragma once
#pragma GCC optimize (2)

namespace SVMPackage {
	template<typename _Type> class Kernel {
	    public:    enum Type { LINEAR, RBF } type = LINEAR;
	    protected: _Type gamma = 0.5;

	    public: Kernel(Type t = LINEAR, _Type g = 0.5L) : type(t), gamma(g) {}
	    public: ~Kernel(void) {}

	    public: _Type operator()(const std::vector<_Type>& a, const std::vector<_Type>& b) const {
	        if (type == LINEAR) {
	            _Type s = {0};
	            for (int i = 0; i < a.size(); ++i) s += a[i] * b[i];
	            return s;
	        } else if(type == RBF) {
	            _Type sq = {0};
	            for (int i = 0; i < a.size(); ++i) {
	                _Type d = a[i] - b[i];
	                sq += d * d;
	            }
	            return std::exp(-gamma * sq);
	        } else {
	        	return -65536;
			}
	    }
	};

	template<typename _Type> class BinarySVM {
		public: Kernel<_Type> kernel;
		public: _Type b={0}, C={0};
		public: std::vector<_Type> alplist;
		public: std::vector<std::vector<_Type>> K; // kernel matrix
		public: BinarySVM(_Type argb, _Type argC, Kernel<_Type> argk): b(argb), C(argC), kernel(argk) {}
		public: ~BinarySVM(void) {}
		public: _Type decision(const std::vector<_Type>& x,
	 				           const std::vector<std::vector<_Type>>& vecx,
							   const std::vector<short>& tagy) {
			_Type r = {0};
			for(int j = 0; j < alplist.size(); j++) r+=alplist[j]*tagy[j]*kernel(vecx[j],x);
			return r+b;
		}
		public: _Type decision(const volatile register int index,
	 				           const std::vector<std::vector<_Type>>& vecx,
							   const std::vector<short>& tagy) {
			_Type r = {0};
			for(int j = 0; j < alplist.size(); j++) r+=alplist[j]*tagy[j]*K[j][index];
			return r+b;
		}
		public: void train(const std::vector<std::vector<_Type>>& vecx,
						   const std::vector<short>& tagy, const int expects, const int limit) {
			int N=vecx.size(), iter=0;
			K.resize(N,std::vector<_Type>(N,_Type{0}));
			alplist.resize(N,0.L);
			if(C<=0) throw std::invalid_argument("s.t. C > 0!");
			for(int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
					K[i][j]=kernel(vecx[i], vecx[j]);
			std::mt19937 rng(time(nullptr));
			while(iter < expects && iter < limit) {
				int nc=0;
				for(int i = 0, j = 0; i < N; i++, j=i) {
					while(j==i) j=rng()%N;
					_Type Ei=decision(i,vecx, tagy)-tagy[i], Ej=decision(j,vecx, tagy)-tagy[j];
					_Type at=K[i][i]-_Type(2)*K[i][j]+K[j][j], alop=alplist[i], alon=alplist[j];
					if(at<1e-6) continue;
					// Clipping multipliers {
	                    _Type L={0}, H={0};
	                    if (tagy[i] != tagy[j]) {
	                        L = std::max(_Type(0), alplist[j]-alplist[i]);
	                        H = std::min(C, C+alplist[j]-alplist[i]);
	                    } else {
	                        L = std::max(_Type(0), alplist[i]+alplist[j]-C);
	                        H = std::min(C, alplist[i]+alplist[j]);
	                    }
	                    if (L == H) continue;
						alplist[j]=alon+(Ei-Ej)/at*tagy[j];
					    alplist[j]=std::min(H,std::max(L,alplist[j]));
					// }

					if(std::abs(alplist[j]-alon)<1e-6) continue;

					alplist[i]=alop+tagy[i]*tagy[j]*(alon-alplist[j]);
					_Type b1 = b - Ei - tagy[i]*(alplist[i]-alop)*K[i][i] - tagy[j]*(alplist[j]-alon)*K[i][j];
	                _Type b2 = b - Ej - tagy[i]*(alplist[i]-alop)*K[i][j] - tagy[j]*(alplist[j]-alon)*K[j][j];

	                if (_Type{0} < alplist[i] && alplist[i] < C) 	  b = b1;
	                else if (_Type{0} < alplist[j] && alplist[j] < C) b = b2;
	                else 										 	  b = _Type{0.5}*(b1 + b2);

					nc++;
				}
				if(!nc) iter++;
				else    iter=0;
			}
			return ;
		}
	};

	template<typename _Type> class MultipleSVM {
		public: Kernel<_Type> kernel;
		public: std::vector<_Type> b;
		public: _Type C={0};
		public: unsigned tagamt;
		public: std::vector<std::vector<_Type>> alplist;
		public: std::vector<std::vector<_Type>> K; // kernel matrix
		public: MultipleSVM(std::vector<_Type> argb, _Type argC, Kernel<_Type> argk):
				  b(argb), C(argC), kernel(argk), tagamt(argb.size()) {}
		public: ~MultipleSVM(void) {}
		public: int predict(const std::vector<_Type>& x,
	 				           const std::vector<std::vector<_Type>>& vecx,
							   const std::vector<short>& tagy) {
			_Type r = {-1e9};
			int c=-1;
			for(int i = 0; i < tagamt; i++) {
				_Type vote = b[i];
			    for(int j = 0; j < alplist.size(); j++) vote+=alplist[j][i]*kernel(vecx[j],x);
			    if(vote>r) r=vote, c=i;
			}
			return c;
		}
		public: _Type decision(const volatile register int index,
	 				           const std::vector<_Type>& x,
	 				           const std::vector<std::vector<_Type>>& vecx) {
			_Type r = b[index];
			for(int j = 0; j < alplist.size(); j++) r+=alplist[j][index]*kernel(vecx[j],x);
			return r;
		}
		public: void train(const std::vector<std::vector<_Type>>& vecx,
						   const std::vector<short>& tagy, const int expects, const int limit) {
			int N=vecx.size(), iter=0;
			K.resize(N,std::vector<_Type>(N,_Type{0}));
			alplist.resize(N,std::vector<_Type>(tagamt,_Type{0}));
			if(C<=0) throw std::invalid_argument("s.t. C > 0!");
			for(int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
					K[i][j]=kernel(vecx[i], vecx[j]);
			std::mt19937 rng(time(nullptr));
			while(iter < expects && iter < limit) {
				int nc=0;
				for(int i = 0, j=-1; i < N; i++, j=-1) {
		            _Type maxsc = -1e9;
	            	std::vector<_Type> score(tagamt, _Type{0});
	            	for(int k = 0; k < tagamt; k++) score[k]=decision(k,vecx[i],vecx);
		            for(int c=0; c<tagamt; c++) {
		                if(c != tagy[i] && score[c] > maxsc) maxsc=score[c], j=c;
		            }
		            if(j == -1) continue;

					_Type loss=score[j]-score[tagy[i]]+1;
					_Type at=K[i][i]-_Type(2)*K[i][j]+K[j][j],
						  alop=alplist[i][tagy[i]],
						  alon=alplist[i][j];

					if(loss<1e-6) continue;
					if(at<1e-6) continue;

		            alplist[i][tagy[i]] += loss/at;
		            alplist[i][j] -= loss/at;

		            alplist[i][tagy[i]] = std::min(C, std::max(_Type{0}, alplist[i][tagy[i]]));
		            alplist[i][j] = std::min(C, std::max(_Type{0}, alplist[i][j]));

		            b[tagy[i]] += (alplist[i][tagy[i]] - alop) * K[i][i];
		            b[j] += (alplist[i][j] - alon) * K[i][i];

					nc++;
				}
				if(!nc) iter++;
				else    iter=0;
			}
			return ;
		}
	};
}
