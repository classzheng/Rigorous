/******************************************************************************
 * Rigorous/MathLib: The header-file of mathmatic modules                     *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.26 (latest upd)                                              *
 * @Description: The header-file of mathmatic modules                         *
 ******************************************************************************/

#include <stdexcept>
#include <vector>
#include <cmath>
#include <ctime>
#include <random>
#include <iostream>
#include <functional>
#include <cstdlib>

#pragma once
#pragma GCC optimize (2)


namespace Mathlib {
	
	static double epsilon = 1e-12;
	template<typename _Type> bool GaussianEliminate(std::vector<std::vector<_Type>>& a) {
		int n = a.size();
		for (int i = 0; i < n; ++i) {
			a[i].resize(2*n);
			for(int j = n; j < 2*n; ++j) {
				a[i][j] = _Type(j == n+i);
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
			
			_Type pivot = a[i][i];
			for (int k = i; k < 2 * n; ++k) {
				a[i][k] /= pivot;
			}
			
			for (int j = 0; j < n; ++j) {
				if (j != i && std::abs(a[j][i]) > epsilon) {
					_Type delta = a[j][i];
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
	
	inline double genrandom(void) {
		return ( double)(rand()%1000)/500.L-1.L;
	}
	
	template<typename _Type> _Type sigmoid(const _Type x) {
		return 1.0 / (1.0 + exp(-x));
	}
	
	template<typename _Type> _Type tanh_derivate(const double x) {
		return 1 - (std::tanh(x)) * (std::tanh(x));
	}
	
	template<typename _Type> _Type sigmoid_derivate(const _Type x) {
		return x * (1 - x);
	}
}
