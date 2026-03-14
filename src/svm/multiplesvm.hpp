/******************************************************************************
 * Rigorous/C-SVM: A c++11 implementation of Multiple-SVM                     *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.26 (latest upd)                                              *
 * @Description: A c++11 implementation of Multiple-SVM                       *
 ******************************************************************************/

#pragma once
#pragma GCC optimize (2)

namespace SVMPackage {
	
	using namespace Mathlib;
	
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
}

