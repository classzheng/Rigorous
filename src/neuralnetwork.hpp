/******************************************************************************
 * Rigorous/NeuralNetwork: A c++11 implementation of BP-ANN                   *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.20 (latest upd)                                              *
 * @Description: A c++11 implementation of BP-ANN                             *
 * @Modules: {                                                                *
 *    NeuralNetwork::UnitLayer, NeuralNetwork::NetworkUnion                   *
 * }                                                                          *
 ******************************************************************************/

#include "autograd.hpp"

#pragma once
#pragma GCC optimize (2)

inline long double gendata(void) {
    return (long double)(rand() % 1145) / 1145.0L;
}

inline long double tanh_derivate(long double x) {
	return 1 - (std::tanh(x)) * (std::tanh(x));
}

namespace NeuralNetwork {
	using initType = long double;
	template<typename _Type> class UnitLayer;
	template<typename _Type> class NetworkUnion;
	template<typename _Type> using AgFloat = BackwardGrad::Autograd<_Type>;
	template<typename _Type> using Matrix = std::vector<std::vector<AgFloat<_Type>&>>;

	template<typename _Type> class UnitLayer {
		public: static const _Type zero=_Type{0.f};
	    public: std::vector<_Type> &gradlist;
	    public: BackwardGrad::Automem<_Type> &mempool;
		public: std::vector<AgFloat<_Type>&> elements;
		public: Matrix<_Type> weights;

	    public: UnitLayer(unsigned n, std::vector<_Type> &mount, BackwardGrad::Automem<_Type> &peepool)
	        		  : gradlist(mount), mempool(peepool) {
			[[maybe_unused]] unsigned original = n;
			while(n--) {
			    AgFloat<_Type> *node = new AgFloat<_Type>(_Type(gendata()), _Type(0), gradlist, mempool);
			    mempool.push(node);
			    elements.push_back(node);
			}
			for (unsigned i = 0; i < elements.size(); i++) {
				weights.push_back({});
				for (unsigned j = 0; j < elements.size(); j++) {
					AgFloat<_Type> *w = new AgFloat<_Type>(_Type(gendata()), _Type(1), gradlist, mempool);
					mempool.push(*w);
					weights.back().push_back(*w);
				}
			}
			return ;
	    }
		public: ~UnitLayer(void) {}

	    public: [[nodiscard]] std::vector<AgFloat<_Type>&> forward(void) {
	    	std::vector<AgFloat<_Type>&> result;
	    	for(int i = 0; i < weights.size(); i++) {
	    		AgFloat<_Type> *sigma = new AgFloat<_Type>(0.f,1.f,gradlist,mempool);
		    	for(int j = 0; j < weights[0].size(); j++) {
		    		sigma=&((*sigma)+elements[j]*weights[i][j]);
				}
				result.push_back(AgTanh(*sigma));
			}
			return result;
		}
	};

	template<typename _Type> class NetworkUnion {
	    public: std::vector<_Type> gradlist;
	    public: BackwardGrad::Automem<_Type> mempool;
		public: std::vector<UnitLayer<AgFloat<_Type>>> network;
	    public: NetworkUnion(unsigned n, std::vector<unsigned> arguments) {
			for(int i = 0; i < n; i++)
			    network.push_back(UnitLayer<AgFloat<_Type>>(arguments[i],gradlist,mempool));
			return ;
	    }
	    public: ~NetworkUnion(void) {}

	    public: [[nodiscard]] UnitLayer<AgFloat<_Type>>& forward(UnitLayer<AgFloat<_Type>> &xspvec) {
	    	network[0].elements = xspvec.elements;
	    	for(int i = 1; i < network.size(); i++) {
	    		network[i].elements = network[i-1].forward();
			}
			return network[network.size()-1];
		}

	    public: [[nodiscard]] AgFloat<_Type>& mseloss (
			  UnitLayer<AgFloat<_Type>> &xspvec,
			  UnitLayer<AgFloat<_Type>> &yspvec,
			  unsigned batchsz) {
	    	AgFloat<_Type> *loss=new AgFloat<_Type>(0.f,0.f);
	    	UnitLayer<AgFloat<_Type>> &y_hat=forward(xspvec);
	    	for(int i = 0; i < y_hat.elements.size(); i++)
			    loss = &((*loss) + (y_hat.elements[i]-yspvec.elements[i])*(y_hat.elements[i]-yspvec.elements[i])*batchsz);
   	    	if (!loss) {
   	    	    AgFloat<_Type> *z = new AgFloat<_Type>(_Type(0), _Type(0), gradlist, mempool);
   	    	    mempool.push(*z);
   	    	    loss = z;
   	    	}
   	    	return *loss;
		}

		public: void train (
			  std::vector<UnitLayer<AgFloat<_Type>>> &xspset,
			  std::vector<UnitLayer<AgFloat<_Type>>> &yspset,
			  unsigned batchsz,
			  long double ln_rate=0.03f) {
			for(unsigned j = 0; j < batchsz; j++) {
				AgFloat<_Type> &loss = mseloss(xspset[j], yspset[j], batchsz);
				loss.resetgrad();
				loss.updgrad();
				for(auto &layer: network) {
					for(auto &row: layer.weights) {
						for(auto &wptr: row) {
							wptr.data -= _Type(ln_rate) * wptr->grad;
						}
					}
				}
			}
			return ;
		}
	};
};
