/******************************************************************************
 * Rigorous/C-SVM: A c++11 implementation of Binary-SVM                       *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.26 (latest upd)                                              *
 * @Description: A c++11 implementation of Binary-SVM                         *
 ******************************************************************************/

#pragma once
#pragma GCC optimize (2)

namespace SVMPackage {
	
	using namespace Mathlib;
	
	template<typename _Type> class BinarySVM {
		public: Kernel<_Type> kernel;
		public: _Type b={0}, C={0};
		public: std::vector<_Type> alplist;
		public: std::vector<short> tagy;
		public: std::vector<std::vector<_Type>> vecx;
		public: std::vector<std::vector<_Type>> K; // kernel matrix
		public: BinarySVM(_Type argb, _Type argC, Kernel<_Type> argk): b(argb), C(argC), kernel(argk) {}
		public: ~BinarySVM(void) {}
		public: [[nodiscard]] _Type decision(const std::vector<_Type>& x) {
			_Type r = {0};
			for(int j = 0; j < alplist.size(); j++) r+=alplist[j]*tagy[j]*kernel(vecx[j],x);
			return r+b;
		}
		public: [[nodiscard]] _Type decision(const int index) {
			_Type r = {0};
			for(int j = 0; j < alplist.size(); j++) r+=alplist[j]*tagy[j]*K[j][index];
			return r+b;
		}
		public: void train(const std::vector<std::vector<_Type>>& vx,
						   const std::vector<short>& ty,
						   const int expects,
						   const int limit) {
			vecx=vx, tagy=ty;
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
					_Type Ei=decision(i)-tagy[i], Ej=decision(j)-tagy[j];
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
}
