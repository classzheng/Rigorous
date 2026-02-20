/******************************************************************************
 * Rigorous/Autograd: Implementation of Autograd Datatype                     *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.2.20 (latest upd)                                              *
 * @Description: Includes Autograd Datatype & Automatic memory pool.          *
 * @Modules: {                                                                *
 *    BackwardGrad::Autograd, BackwardGrad::Automem                           *
 * }                                                                          *
 ******************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <functional>

#pragma once
#pragma GCC optimize (2)

namespace BackwardGrad {
	unsigned globalid = 0;
	template <typename _Type> class Automem;
	template <typename _Type> class Autograd {
	    public: _Type data;
	    public: _Type grad;
	    public: std::vector<_Type> &gradlist;
	    public: std::vector<Autograd<_Type>*> parents;
	    public: std::function<void(Autograd<_Type>&)> backward;
	    public: Automem<_Type> &mempool;
	    public: unsigned gradid = 0;

	    public: Autograd(_Type dt, _Type gd, std::vector<_Type> &mount, Automem<_Type> &peepool)
	        			: data(dt), grad(gd), gradlist(mount), mempool(peepool), gradid(globalid++) {
	        gradlist.push_back(grad);
	    }

	    public: void updgrad(void) {
	        if (backward) backward(*this);
	        for (auto *p : parents) if (p) p->updgrad();
	    }

	    public: void resetgrad(void) {
	        grad = _Type(0);
	        for (auto *p : parents) if (p) p->resetgrad();
	    }

	    public: Autograd<_Type>& operator+(Autograd<_Type> &rt) {
	        Autograd<_Type> *t = new Autograd<_Type>(this->data + rt.data, _Type(0), gradlist, mempool);
	        mempool.push(t);
	        t->parents.clear();
	        t->parents.push_back(this);
	        t->parents.push_back(&rt);
	        t->backward = [](Autograd<_Type> &self) -> void {
	            Autograd<_Type> *a = self.parents[0];
	            Autograd<_Type> *b = self.parents[1];
	            if (a) a->grad += self.grad;
	            if (b) b->grad += self.grad;
	        };
	        return *t;
	    }

	    public: Autograd<_Type>& operator-(Autograd<_Type> &rt) {
	        Autograd<_Type> *t = new Autograd<_Type>(this->data - rt.data, _Type(0), gradlist, mempool);
	        mempool.push(t);
	        t->parents.clear();
	        t->parents.push_back(this);
	        t->parents.push_back(&rt);
	        t->backward = [](Autograd<_Type> &self) -> void {
	            Autograd<_Type> *a = self.parents[0];
	            Autograd<_Type> *b = self.parents[1];
	            if (a) a->grad += self.grad;
	            if (b) b->grad -= self.grad;
	        };
	        return *t;
	    }

	    public: Autograd<_Type>& operator*(Autograd<_Type> &rt) {
	        Autograd<_Type> *t = new Autograd<_Type>(this->data * rt.data, _Type(0), gradlist, mempool);
	        mempool.push(t);
	        t->parents.clear();
	        t->parents.push_back(this);
	        t->parents.push_back(&rt);
	        t->backward = [](Autograd<_Type> &self) -> void {
	            Autograd<_Type> *a = self.parents[0];
	            Autograd<_Type> *b = self.parents[1];
	            if (a && b) {
	                a->grad += self.grad * b->data;
	                b->grad += self.grad * a->data;
	            }
	        };
	        return *t;
	    }

	    public: Autograd<_Type>& operator~(void) {
	        Autograd<_Type> *t = new Autograd<_Type>(this->data * this->data, _Type(0), gradlist, mempool);
	        mempool.push(t);
	        t->parents.clear();
	        t->parents.push_back(this);
	        t->backward = [](Autograd<_Type> &self) -> void {
	            Autograd<_Type> *a = self.parents[0];
	            if (a) a->grad += self.grad * (_Type(2) * a->data);
	        };
	        return *t;
	    }
	};

	template <typename _Type> Autograd<_Type>& AgSin(Autograd<_Type> &arg) {
	    Autograd<_Type> *t = new Autograd<_Type>(std::sin(arg.data), _Type(0), arg.gradlist, arg.mempool);
	    arg.mempool.push(t);
	    t->parents.clear();
	    t->parents.push_back(&arg);
	    t->backward = [](Autograd<_Type> &self) -> void {
	        Autograd<_Type> *a = self.parents[0];
	        if (a) a->grad += self.grad * std::cos(a->data);
	    };
	    return *t;
	}

	template <typename _Type> Autograd<_Type>& AgCos(Autograd<_Type> &arg) {
	    Autograd<_Type> *t = new Autograd<_Type>(std::cos(arg.data), _Type(0), arg.gradlist, arg.mempool);
	    arg.mempool.push(t);
	    t->parents.clear();
	    t->parents.push_back(&arg);
	    t->backward = [](Autograd<_Type> &self) -> void {
	        Autograd<_Type> *a = self.parents[0];
	        if (a) a->grad -= self.grad * std::sin(a->data);
	    };
	    return *t;
	}

	template <typename _Type> Autograd<_Type>& AgTanh(Autograd<_Type> &arg) {
	    Autograd<_Type> *t = new Autograd<_Type>(std::tanh(arg.data), _Type(0), arg.gradlist, arg.mempool);
	    arg.mempool.push(t);
	    t->parents.clear();
	    t->parents.push_back(&arg);
	    t->backward = [](Autograd<_Type> &self) -> void {
	        Autograd<_Type> *a = self.parents[0];
	        if (a) {
	            _Type tv = std::tanh(a->data);
	            a->grad += self.grad * ( (_Type)1 - tv * tv );
	        }
	    };
	    return *t;
	}

	template <typename _Type> class Automem {
	    public: std::vector<Autograd<_Type>*> pool;
	    public: ~Automem(void) {
	        clear();
	    }
	    public: void clear(void) {
	        for (auto &is : pool) delete is;
	        pool.clear();
	    }
	    public: void push(Autograd<_Type> *s) {
	        pool.push_back(s);
	    }
	};

};
