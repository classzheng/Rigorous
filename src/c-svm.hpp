/******************************************************************************
 * Rigorous/C-SVM: A c++11 implementation of Binary-SVM & Multiple-SVM        *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.26 (latest upd)                                              *
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
	
	static double epsilon = 1e-12;
	template<typename T> bool GaussianEliminate(std::vector<std::vector<T>>& a) {
		int n = a.size();
		for (int i = 0; i < n; ++i) {
			a[i].resize(2*n);
			for(int j = n; j < 2*n; ++j) {
				a[i][j] = T(j == n+i);
			}
		}
		
		for (int i = 0; i < n; ++i) {
			int maxr = i;
			for (int j = i; j < n; ++j) {
				if (std::abs(a[j][i]) - std::abs(a[maxr][i]) > epsilon) {
					maxr = j;
				}
			}
			
			if (maxr != i) {
				std::swap(a[i], a[maxr]);
			}
			
			if (std::abs(a[i][i]) <= epsilon) {
				return false;
			}
			
			T pivot = a[i][i];
			for (int k = i; k < 2 * n; ++k) {
				a[i][k] /= pivot;
			}
			
			for (int j = 0; j < n; ++j) {
				if (j != i && std::abs(a[j][i]) > epsilon) {
					T delta = a[j][i];
					for (int k = i; k < 2 * n; ++k) {
						a[j][k] -= delta * a[i][k];
					}
				}
			}
		}
		return true;
	}
	
	template<typename _Type> class Kernel {
		public:    enum Type {
					LINEAR,
					GAUSSIAN,
					LAPLACIAN,
					CAUCHY,
					COSINE,
					POLYNOMIAL,
					TRIANGULAR  } type = LINEAR;
		protected: _Type gamma = 0.5L, theta = 3.1415L;  // gamma=1/(2*sigma^2)
		protected: int degree = 1;
		
		public: Kernel(Type t = LINEAR, _Type eg = 1.0L) : type(t), gamma(eg), theta(eg), degree((int)eg) {}
		public: ~Kernel(void) {}
		
		public: _Type operator()(const std::vector<_Type>& a, const std::vector<_Type>& b) const {
			_Type sq = {0};
			switch(type) {
				case LINEAR:
					for (int i = 0; i < a.size(); ++i) sq += a[i] * b[i];
					return sq;
				case GAUSSIAN:
					for (int i = 0; i < a.size(); ++i) {
						_Type d = a[i] - b[i];
						sq += d * d;
					}
					return std::exp(-gamma * sq);
				case LAPLACIAN:
					for (int i = 0; i < a.size(); ++i) {
						sq += fabs(a[i] - b[i]);
					}
					return std::exp(-sqrt(_Type(2)*gamma) * sq);
				case CAUCHY:
					for (int i = 0; i < a.size(); ++i) {
						_Type d = a[i] - b[i];
						sq += d * d;
					}
					return _Type(1)/(_Type(1)+sq*(_Type(2)*gamma));
				case COSINE:
					for (int i = 0; i < a.size(); ++i) sq += (a[i] - b[i])*(a[i] - b[i]);
					return std::cos(theta*std::sqrt(sq));
				case POLYNOMIAL:
					for (int i = 0; i < a.size(); ++i) sq += a[i] * b[i];
					return std::pow(sq,degree);
				case TRIANGULAR:
					sq=_Type(1);
					for (int i = 0; i < a.size(); ++i)
						sq *= std::max(_Type(1)-std::fabs(a[i] - b[i])*sqrt(_Type(2)*gamma),_Type(0));
					return sq;
				default:
					return -65536;
			};
		}
	};
	
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
	
	template<typename _Type> class MultipleSVM {
		public: Kernel<_Type> kernel;
		public: std::vector<_Type> b;
		public: _Type C={0};
		public: unsigned tagamt;
		public: std::vector<std::vector<_Type>> alplist;
		public: std::vector<std::vector<_Type>> vecx;
		public: std::vector<_Type> tagy;
		public: std::vector<std::vector<_Type>> K; // kernel matrix
	public: MultipleSVM(std::vector<_Type> argb, _Type argC, Kernel<_Type> argk):
		b(argb), C(argC), kernel(argk), tagamt(argb.size()) {}
		public: ~MultipleSVM(void) {}
		public: [[nodiscard]] int predict(const std::vector<_Type>& x) {
			_Type r = {-1e9};
			int c=-1;
			for(int i = 0; i < tagamt; i++) {
				_Type vote = b[i];
				for(int j = 0; j < alplist.size(); j++) vote+=alplist[j][i]*kernel(vecx[j],x);
				if(vote>r) r=vote, c=i;
			}
			return c;
		}
		public: [[nodiscard]] _Type decision(const int index, const std::vector<_Type>& x) {
			_Type r = b[index];
			for(int j = 0; j < alplist.size(); j++) r+=alplist[j][index]*kernel(vecx[j],x);
			return r;
		}
		public: void train(const std::vector<std::vector<_Type>>& vx,
						   const std::vector<short>& ty, const int expects, const int limit) {
			vecx=vx, tagy=ty;
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
					for(int k = 0; k < tagamt; k++) score[k]=decision(k,vecx[i]);
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
