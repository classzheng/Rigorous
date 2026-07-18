/******************************************************************************
 * Bakaford/HoTT Reasoner: An abstraction of Homotopy Type Theory             *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.7.10 (latest upd)                                              *
 * @Description: Based on Homotopy Type Theory (HoTT).                        *
 ******************************************************************************/

#include <random>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <iostream>
#include <ctime>
#include <chrono>
#include <thread>

#pragma once
#pragma GCC optimize (2)

namespace Bakaford {

	const static std::string PICHAR("Π"), SIGCHAR("Σ"), LAMCHAR("λ"), ARWCHAR("→"), CRSCHAR("×");
	
	using ElementType = enum {
		CustomType,
		Variable,
		Constant,
		Lambda,
		CartesianPair,
		CoproductPair,
		CartproType,
		CoproType,
	};
	class Element;
	class Autoptr {
		public: bool hascleared=false;
		public: std::vector<Element*> table;
		public: Autoptr(void) = default;
		public: ~Autoptr(void) {
			if(hascleared) return ;
			for (auto p : table) {
				if(p!=nullptr) delete p, p=nullptr;
			}
			table.clear();
			hascleared=true;
			return ;
		}
		public: inline void push(Element* e) { table.push_back(e); }
		public: void clear(void) {
			if(hascleared) return ;
			for (auto p : table) {
				if(p!=nullptr) delete p, p=nullptr;
			}
			table.clear();
			hascleared=true;
			return ;
		}
	} mainpool;
	class Element {
		public: std::string var;
		public: Element *type;
		public: Element *first, *second;
		public: ElementType et;
		
		public: Element(void):
					var(""), type(nullptr), first(nullptr), second(nullptr), et(Constant) {}
		public: ~Element(void) = default;
		public: Element(std::string v, Element *tp, ElementType t=Constant) : var(v), type(tp), et(t) {}
		public: Element(Element *a, Element *c, float) : first(a), second(c), et(Lambda)
					{mainpool.push(a), mainpool.push(c);}
		public: Element(Element *f, Element *s, ElementType t=CartesianPair) : first(f), second(s), et(t) 
					{mainpool.push(f), mainpool.push(s);}
		
		public: Element(const Element& other) : 
					var(other.var), type(nullptr),
					first(other.first), second(other.second),
					et(other.et) {
			if (other.type != nullptr) {
				type = new Element(other.type->var, nullptr);
				// mainpool.push(type);
			}
		}
		
		public: Element& operator=(const Element& other) {
			if (this != &other) {
				var = other.var;
				et = other.et;
				first = other.first;
				second = other.second;
				if (other.type) {
					if (type) {
						type->var = other.type->var;
					} else {
						type = new Element(other.type->var, nullptr);
						mainpool.push(type);
					}
				} else {
					type = nullptr;
				}
			}
			return *this;
		}
		
		public: bool operator==(const Element& rhs) {
			if(type==nullptr||rhs.type==nullptr) {
				return et==rhs.et && var==rhs.var;
			} else {
				return et==rhs.et && var==rhs.var && (*type)==(*rhs.type);
			}
		}
		
		public: Element copy(void) {
			return std::move(*this);
		}
		
		public: void betareduce(const Element& alpha, const Element& beta) {
			if (type == nullptr) return;
			int t = 0;
			while ((t = var.find(alpha.var, t)) != -1) {
				var.replace(t, alpha.var.length(), beta.var);
				t += beta.var.length();
			}
			if (type) type->betareduce(alpha, beta); // recursion
			return;
		}
		
	}   Universe("TypeUniverse",nullptr,CustomType),
		TypeZero("0",&Universe,CustomType), TypeOne("1",&Universe,CustomType),
		Asterisk("*",&TypeOne,CustomType), NilType("nil",&Universe,CustomType);
	
	class Judgment {  // Judgment of Element
		public: Element ele;
		
		public: Judgment(std::string var, Element *tp) {
			ele.var = var;
			if (tp != nullptr) {
				if(tp==&Universe)
					ele.type = new Element(tp->var, &NilType,CustomType);
				else
					ele.type = new Element(tp->var, &NilType);
			} else {
				ele.type = &NilType;
			}
		}
		
		public: Judgment(Element e) : ele(e) {}
		
		public: ~Judgment(void) {
			if (ele.type != nullptr && ele.type != &NilType) {
				delete ele.type;
			}
			ele.type = &NilType;
		}
		
		public: inline Element& operator() (void) {
			return ele;
		}
		
		public: inline std::string literal(void) {
			return ele.var + ":" + ele.type->var;
		}
	};
	
	class Reference {  // Constructor of Element
		public: Element a, b;
		
		public: Reference(void) = default;
		
		public: Reference(const Element &e1, const Element &e2) : a(e1), b(e2) {}
		
		public: ~Reference(void) = default;
		
		public: inline Element& alpha(void) {
			return a;
		}
		
		public: inline Element& beta(void) {
			return b;
		}
		
		public: inline std::string literal(void) {
			return a.var + ":=" + b.var;
		}
	};
	
	class LambdaAbst {  // Constructor of Axioms
		public: Element antecedent;
		public: Element consequence;
		public: Element lambda;
		
		public: LambdaAbst(void) : antecedent(), consequence(), lambda() {}
		public: ~LambdaAbst(void) {}
		
		public: LambdaAbst(Element a, Element c) : antecedent(a), consequence(c), lambda() {
			std::string typeStr = a.type->var + ARWCHAR + c.type->var;
			Element *tmpType=new Element(typeStr, &Universe,CustomType);
			Judgment af(LAMCHAR+"(" + Judgment(antecedent).literal() + ")." + c.var, tmpType);
			lambda = af();
			lambda.et=Lambda;
			lambda.first = new Element(antecedent);
			lambda.second = new Element(consequence);
			lambda.first->et=Lambda;
			lambda.second->et=Lambda;
			mainpool.push(lambda.first);
			mainpool.push(lambda.second);
			return ;
		}
		
		public: Element Appl(Element var) {  // Apply function by Beta-Reduction
			Element applresult = consequence;
			applresult.betareduce(antecedent, var);
			applresult.et=Lambda;
			return applresult;
		}
		
		public: inline Element Abst(void) {  // Abstract function by Eta-Expansion
			lambda.et=Lambda;
			return lambda;
		}
		
		public: inline Element operator() (void) {
			lambda.et=Lambda;
			return lambda;
		};
		
		public: inline std::string literal(void) {
			return antecedent.type->var + ARWCHAR + consequence.type->var;
		}
	};
	
	class PiType {  // Constructor of Forall logic
		public: Element A, B, con;
		
		public: PiType(void) : A(), B() {}
		public: ~PiType(void) {}
		
		public: PiType(Element a, Element b) :  // A:U, B:A->U 
		A(a), B(b), con("("+PICHAR+"_(" + A.var + ") " + B.var + "(-))", &Universe) {}
		
		public: Element operator() (void) {
			return con;
		}
		
		public: inline std::string literal(void) {
			return "("+PICHAR+"_(" + A.var + ") " + B.var + "(-))";
		}
	};
	
	class SigmaType {  // Constructor of Exist logic
		public: Element A, B, con;
		
		public: SigmaType(void) : A(), B() {}
		public: ~SigmaType(void) {}
		
		public: SigmaType(Element a, Element b) :  // A:U, B:A->U 
		A(a), B(b), con("("+SIGCHAR+"_(" + A.var + ") " + B.var + "(-))", &Universe) {}
		
		public: Element operator() (void) {
			return con;
		}
		
		public: inline std::string literal(void) {
			return SIGCHAR+"_(" + A.var + ") " + B.var + "(-)";
		}
	};
	
	class CartesianProduct {
		public: Element A, B;
		public: CartesianProduct(void) = default;
		public: CartesianProduct(Element a, Element b): A(a), B(b) {};
		public: ~CartesianProduct(void) = default;
		
		public: Element Currying(void) {
			// return LambdaAbst(Element(A.var,A.type),Element(B.var,B.type));
			Element *temp=new Element(A.type->var+ARWCHAR+B.type->var,&Universe);
			mainpool.push(temp);
			return Element(A.var+ARWCHAR+B.var,temp);
		}
		
		public: inline std::string literal(void) {
			return A.var+CRSCHAR+B.var;
		}
		
		public: inline Element operator() (void) {
			Element lit(this->literal(),&Universe);
			lit.et=CartproType;
			lit.first = new Element(A);
			lit.second = new Element(B);
			mainpool.push(lit.first);
			mainpool.push(lit.second);
			return lit;
		}
		
		public: Element constructor(Element *a, Element *b) {
			Element *type = new Element((*this)());
			Element con("("+a->var+","+b->var+")_"+CRSCHAR, type, CartesianPair);
			con.et=CartesianPair;
			con.first=new Element(*a);
			con.second=new Element(*b);
			mainpool.push(con.first);
			mainpool.push(con.second);
			mainpool.push(type);
			return con;
		};
		
		public: [[deprecated]] Element recursor_org(Element C, Element g, Element pair) {  // g:A->B->C, pair:(A*B)
			LambdaAbst abst(*g.first,*g.second),
			appl0=LambdaAbst(abst.Appl(*pair.first), Element(B.var+ARWCHAR+C.var, nullptr));
			return appl0.Appl((*pair.second));
		};
		
		public: [[deprecated]] Element inductor_org(Element C, Element g, Element pair) {
			// C:(A*B)->U, g:A->B->(C(x) forall(x:A*B)), pair:(A*B), ind(-,-,-):(C(x) forall(x:A*B))
			LambdaAbst abst(*g.first,*g.second),
			appl0=LambdaAbst(abst.Appl(*pair.first), Element(B.var+ARWCHAR+C.var, nullptr));
			return appl0.Appl((*pair.second));
		};
		
		public: Element recursor(Element C, Element g, Element pair) {  // g:A->B->C, pair:(A*B)
			Element A = *(pair.first);
			Element B = *(pair.second);
			Element Cparam(C.var, &Universe,CustomType);
			std::string ab2c = "(" + A.var + ARWCHAR + B.var + ARWCHAR + Cparam.var + ")";
			std::string pair2c = "(" + A.var + CRSCHAR + B.var + ARWCHAR + Cparam.var + ")";
			std::string whole = ab2c + ARWCHAR + pair2c;
			Element wholeElem(whole, &Universe,CustomType);
			PiType ft(Cparam, wholeElem);
			return ft();
		};
		
		public: Element inductor(Element C, Element g, Element pair, unsigned markindex) {
			Element A = *pair.first;
			Element B = *pair.second;
			std::string CdomStr = "("+C.var+"_"+std::to_string(markindex)+":(" + A.var + CRSCHAR + B.var + ")"+ARWCHAR+"TypeUniverse)";
			Element CdomElem(CdomStr, &Universe,CustomType);
			std::string gsig = "("+PICHAR+"_(a:" + A.var + ") ("+PICHAR+"_(b:" + B.var + ") "+C.var+"_"+std::to_string(markindex)+"(-)))";
			std::string outsig = "(" + A.var + CRSCHAR + B.var + ARWCHAR + C.var+"_"+std::to_string(markindex)+"(-))";
			std::string whole = gsig + ARWCHAR + outsig;
			Element wholeElem(whole, &Universe,CustomType);
			PiType ft(CdomElem, wholeElem);
			return ft();
		};
	};
	
	class DisjointCoproduct {
		public: Element A, B;
		public: DisjointCoproduct(void) = default;
		public: DisjointCoproduct(Element a, Element b): A(a), B(b) {};
		public: ~DisjointCoproduct(void) = default;
		
		public: inline std::string literal(void) {
			return A.var+"+"+B.var;
		}
		
		public: inline Element operator() (void) {
			Element lit(this->literal(),&Universe);
			lit.et=CoproType;
			lit.first = new Element(A);
			lit.second = new Element(B);
			mainpool.push(lit.first);
			mainpool.push(lit.second);
			return lit;
		}
		
		public: Element inl(Element *i) {  // Left injection
			Element *type = new Element((*this)());
			Element con("(0,"+i->var+")_+", type, CoproductPair);
			con.et = CoproductPair;
			con.first = &TypeZero;
			con.second = new Element(*i);
			mainpool.push(con.first);
			mainpool.push(con.second);
			mainpool.push(type);
			return con;
		}
		
		public: Element inr(Element *i) {  // Right injection
			Element *type = new Element((*this)());
			Element con("(1,"+i->var+")_+", type, CoproductPair);
			con.et = CoproductPair;
			con.first = &TypeOne;
			con.second = new Element(*i);
			mainpool.push(con.first);
			mainpool.push(con.second);
			mainpool.push(type);
			return con;
		}
	};
	
	namespace Handle {
	
		std::pair<Element,Reference> nil=std::make_pair(NilType,Reference());
		
		std::pair<Element,Element> abstlambda(Element &A, Element &C, unsigned markindex=0) {
			if (A.et != Variable) return std::make_pair(NilType,NilType);
			if (C.et != Variable) return std::make_pair(NilType,NilType);
			
			LambdaAbst lab(A, C);
			Element lambda = lab.Abst();
			Element type(A.type->var + ARWCHAR + C.type->var, &Universe, CustomType);
			
			return std::make_pair(lambda,type);
		}
		
		std::pair<Element,Reference> appllambda(Element &lambda, Element &appl, unsigned markindex=0) {
			if (lambda.et != Lambda) return nil;
			if (!lambda.first) return nil;
			
			Element arg = *lambda.first;
			if (!appl.type || !arg.type) return nil;
			// if (!(*appl.type == *arg.type)) return nil;  // 何意味
			if (appl.et != Constant && appl.et != Variable) return nil;
			
			LambdaAbst func(*lambda.first, *lambda.second);
			Element ret = func.Appl(arg);
			
			Reference ref(ret, Element(lambda.var + "[" + (*lambda.first).var + ":=" + appl.var + "]", func.consequence.type));
			return std::make_pair(ret,ref);
		}
		
		std::pair<Element,Reference> currying(Element &cartprotype, unsigned markindex=0) {
			// if (cartprotype.et != CartproType) return nil;
			if (!cartprotype.first || !cartprotype.second) return nil;
			
			CartesianProduct cart(*(cartprotype.first), *(cartprotype.second));
			// Element curried = cart.Currying().Abst();
			Element curried = cart.Currying();
			
			return std::make_pair(curried,Reference());
		}
		
		std::vector<Element> emplace(const Reference &ref, const std::vector<Element> gamma, unsigned markindex=0) {
			std::vector<Element> sigma=gamma;
			Element alpha=ref.a, beta=ref.b;
			if (alpha.var == "" || beta.var.find(alpha.var) != -1) return sigma;
			
			for (auto &is : sigma) {
				std::string temp = is.var;
				std::string refa = alpha.var;
				std::string refb = beta.var;
				if (temp.find(refa) != std::string::npos) {
					size_t pos = temp.find(refa);
					if (pos == -1) continue;
					while (pos != -1) {
						temp.replace(pos, refa.length(), refb);
						pos = temp.find(refa, pos+refb.length());
					}
					sigma.push_back(Element(temp, is.type, is.et));
				}
			}
			return sigma;
		}
		
		std::pair<Element,Reference> recursion(Element &cartpair, unsigned markindex) {
			if (cartpair.et != CartesianPair) return nil;
			if (!cartpair.first || !cartpair.second) return nil;
			
			Element A = *(cartpair.first);
			Element B = *(cartpair.second);
			Element C("rec_C"+std::to_string(markindex)+"(" + cartpair.var + ")", &Universe);
			Element g("rec_g"+std::to_string(markindex)+"(" + A.var + "," + B.var + ARWCHAR + C.var + ")", &Universe);
			
			CartesianProduct cart(A, B);
			Element rec = cart.recursor(C, g, cartpair);
			
			return std::make_pair(rec,Reference(cartpair, rec));
		}
		
		std::pair<Element,Reference> induction(Element &cartpair, unsigned markindex) {
			if (cartpair.et != CartesianPair) return nil;
			if (!cartpair.first || !cartpair.second) return nil;
			
			Element A = *(cartpair.first);
			Element B = *(cartpair.second);
			Element C("pred_C"+std::to_string(markindex)+"(" + cartpair.var + ")", &Universe, CustomType);
			Element g("ind_g"+std::to_string(markindex)+"(" + A.var + "," + B.var + ARWCHAR + C.var +"(-))", &Universe, CustomType);
			
			CartesianProduct cart(A, B);
			Element ind = cart.inductor(C, g, cartpair, markindex);
			
			return std::make_pair(ind,Reference(cartpair, ind));
		}
		
		Element constructproduct(Element &A, Element &B, unsigned markindex=0) {
			if (A.et != CustomType || B.et != CustomType) return NilType;
			
			CartesianProduct cart(A, B);
			return cart();
		}
		
		Element constructcoproduct(Element &A, Element &B, unsigned markindex=0) {
			if (A.et != CustomType || B.et != CustomType) return NilType;
			DisjointCoproduct copr(A, B);
			return copr();
		}
		
		std::pair<Element,Reference> cartcouple(Element &v1, Element &v2, unsigned markindex=0) {
			if (!v1.type || !v2.type) return nil;
			
			CartesianProduct cart(*(v1.type), *(v2.type));
			Element pair = cart.constructor(&v1, &v2);
			
			return std::make_pair(pair,Reference());
		}
		
		std::pair<Element,Reference>  coprocouple(Element &lv, Element &rv, bool cl, unsigned markindex=0) {
			if (!lv.type || !rv.type) return nil;
			
			DisjointCoproduct cop(*(lv.type), *(rv.type));
			Element inj = cl ? cop.inl(&lv) : cop.inr(&rv);
			return std::make_pair(inj,Reference());
		}
	};

	class Reasoner {
	  // Provide chain methods
		public: std::vector<Element> pool;
		public: std::vector<Reference> ref;
		public: unsigned index=0u;
		public: Reasoner(void) = default;
		public: ~Reasoner(void) {
			mainpool.clear();
			return ;
		}
		public: inline Reasoner& intro(Element &arg1) {
			pool.push_back(arg1);
			return (*this);
		}
		public: inline Reasoner& declare(std::string v, Element *tp, ElementType t=Constant, Element *ref=nullptr) {
			if(ref!=nullptr) *ref=Element(v,tp,t), pool.push_back(*ref);
			else			 pool.push_back(Element(v,tp,t));
			return (*this);
		}
		public: inline Reasoner& define(const Element &arg1, const Element &arg2) {
			ref.push_back(Reference(arg1,arg2));
			return (*this);
		}
		public: Reasoner& abst(Element &arg1, Element &arg2) {
			std::pair<Element,Element> dist=Handle::abstlambda(arg1,arg2);
			pool.push_back(dist.second);
			pool.push_back(dist.first);
			return (*this);
		}
		public: Reasoner& appl(Element &arg1, Element &arg2) {
			std::pair<Element,Reference> dist=Handle::appllambda(arg1,arg2);
			pool.push_back(dist.first);
			ref.push_back(dist.second);
			return (*this);
		}
		public: Reasoner& currying(Element &arg1) {
			std::pair<Element,Reference> dist=Handle::currying(arg1);
			pool.push_back(dist.first);
			// ref.push_back(dist.second);
			return (*this);
		}
		public: Reasoner& emplace(Reference &arg1) {
			pool = Handle::emplace(arg1,pool);
			return (*this);
		}
		public: Reasoner& emplace(void) {
			pool = Handle::emplace((*this)[0],pool);
			return (*this);
		}
		public: Reasoner& rec(Element &arg1) {
			std::pair<Element,Reference> dist=Handle::recursion(arg1,index++);
			pool.push_back(dist.first);
			// ref.push_back(dist.second);
			return (*this);
		}
		public: Reasoner& ind(Element &arg1) {
			std::pair<Element,Reference> dist=Handle::induction(arg1,index++);
			pool.push_back(dist.first);
			// ref.push_back(dist.second);
			return (*this);
		}
		public: Reasoner& make_pair(Element &arg1, Element &arg2) {
			std::pair<Element,Reference> dist=Handle::cartcouple(arg1,arg2);
			pool.push_back(dist.first);
			return (*this);
		}
		public: Reasoner& make_copair(Element &arg1, Element &arg2, const bool inz) {
			std::pair<Element,Reference> dist=Handle::coprocouple(arg1,arg2,inz);
			pool.push_back(dist.first);
			return (*this);
		}
		public: inline Reasoner& eq(Element &n) {
			n=(*this)(0);
			return (*this);
		}
		public: inline void qed(void) {
		    for(auto& is:pool)
		        std::cout << is.var << " : " << is.type->var <<"\n";
		    for(auto& is:ref)
		        std::cout << is.literal() <<"\n";
			return ;
		}
		public: inline Element& operator() (const size_t& index) {
			return pool[std::max(pool.size()-index-1,(size_t)0)];
		}
		public: inline Reference& operator[] (const size_t& index) {
			return ref[std::max(ref.size()-index-1,(size_t)0)];
		}
	};
}
