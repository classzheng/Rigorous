/******************************************************************************
 * Rigorous/C-SVM: A c++11 implementation of LSSVM                            *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.26 (latest upd)                                              *
 * @Description: A c++11 implementation of LSSVM                              *
 ******************************************************************************/

#pragma once
#pragma GCC optimize (2)

namespace SVMPackage {
	
	using namespace Mathlib;
	
	template<typename _Type> class LSSVM {
		public: Kernel<_Type> kernel;
		public: _Type b={0}, C={0};
		public: std::vector<_Type> alplist;
		public: std::vector<short> tagy;
		public: std::vector<std::vector<_Type>> vecx;
		public: std::vector<std::vector<_Type>> K; // kernel matrix
		public: LSSVM(_Type argb, _Type argC, Kernel<_Type> argk): b(argb), C(argC), kernel(argk) {}
		public: ~LSSVM(void) {}
		public: [[nodiscard]] _Type decision(const std::vector<_Type>& x) {
			_Type r = {0};
			for(int j = 0; j < alplist.size(); j++) r+=alplist[j]*tagy[j]*kernel(vecx[j],x);
			return r+b;
		}
		public: void optimize(const std::vector<std::vector<_Type>>& vx,
							  const std::vector<short>& ty) {
			vecx=vx, tagy=ty;
			int N=vecx.size(), iter=0;
			K.resize(N,std::vector<_Type>(N,_Type{0}));
			alplist.resize(N,0.L);
			if(C<=0) throw std::invalid_argument("s.t. C > 0!");
			for(int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
					K[i][j]=kernel(vecx[i], vecx[j]);
			std::mt19937 rng(time(nullptr));
			std::vector<std::vector<_Type>> Kp=K; // Updating matrix
			for(int i = 0; i < N; i++) {
				for(int j = 0; j < N; j++) {
					Kp[i][j]*=tagy[i]*tagy[j];
					if(i==j) Kp[i][j]+=_Type(1.L)/C;
				}
			}
			std::vector<std::vector<_Type>> A(N+1,std::vector<_Type>(N+1,_Type{0}));
			for(int i = 0; i < N; i++) A[0][i+1]=_Type(tagy[i]);
			for(int i = 0; i < N; i++) A[i+1][0]=_Type(tagy[i]);
			for(int i = 1; i < N+1; i++) {
				for(int j = 1; j < N+1; j++) {
					A[i][j]=Kp[i-1][j-1];
				}
			}
			if(!GaussianEliminate<_Type>(A)) throw "A is not inversable.";
			for(int i = 0; i < N+1; i++) {
				for(int j = 0; j < N+1; j++) {
					A[i][j] = A[i][j + N + 1];
				}
			}
			std::vector<_Type> fnvec(N+1,0);
			for(int i = 0; i < N+1; i++) {
				for(int j = 0; j < N+1; j++){
					fnvec[i]+=_Type((bool)j)*A[i][j];
				}
			}
			b=fnvec[0];
			for(int i = 1; i < N+1; i++) alplist[i-1]=fnvec[i];
			return ;
		}
	};	
}

