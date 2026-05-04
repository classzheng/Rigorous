/******************************************************************************
 * Rigorous/TheoremOperator: Automatic Theorem Prover templates.              *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.5.4 (latest upd)                                               *
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

namespace TheoremOperator {
	
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
		public: std::vector<Element*> table;
		public: Autoptr(void) = default;
		public: ~Autoptr(void) {
			for(auto&is:table) delete is;
		}
		public: inline void push(Element* e) { table.push_back(e); }
		public: inline void clear(void) {
			for (auto p : table) {
				delete p;
			}
			table.clear();
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
	
	class AtomicFormula {  // Judgment of Element
		public: Element ele;
		
		public: AtomicFormula(std::string var, Element *tp) {
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
		
		public: AtomicFormula(Element e) : ele(e) {}
		
		public: ~AtomicFormula(void) {
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
			std::string typeStr = a.type->var + "->" + c.type->var;
			Element *tmpType=new Element(typeStr, &Universe,CustomType);
			AtomicFormula af("lambda^(" + AtomicFormula(antecedent).literal() + ")." + c.var, tmpType);
			lambda = af();
			lambda.et=Lambda;
			lambda.first = new Element(antecedent);
			lambda.second = new Element(consequence);
			lambda.first->et=Lambda;
			lambda.second->et=Lambda;
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
			return antecedent.type->var + "->" + consequence.type->var;
		}
	};
	
	class ForallType {  // Constructor of Forall Logic
		public: Element A, B, con;
		
		public: ForallType(void) : A(), B() {}
		public: ~ForallType(void) {}
		
		public: ForallType(Element a, Element b) :  // A:U, B:A->U 
		A(a), B(b), con("(Pi^(" + A.var + ")," + B.var + "(-))", &Universe) {}
		
		public: Element operator() (void) {
			return con;
		}
		
		public: inline std::string literal(void) {
			return "(Pi^(" + A.var + ")," + B.var + "(-))";
		}
	};
	
	class ExistType {  // Constructor of Exist Logic
		public: Element A, B, con;
		
		public: ExistType(void) : A(), B() {}
		public: ~ExistType(void) {}
		
		public: ExistType(Element a, Element b) :  // A:U, B:A->U 
		A(a), B(b), con("(Sigma^(" + A.var + ")," + B.var + "(-))", &Universe) {}
		
		public: Element operator() (void) {
			return con;
		}
		
		public: inline std::string literal(void) {
			return "(Sigma^(" + A.var + ")," + B.var + "(-))";
		}
	};
	
	class CartesianProduct {
		public: Element A, B;
		public: CartesianProduct(void) = default;
		public: CartesianProduct(Element a, Element b): A(a), B(b) {};
		public: ~CartesianProduct(void) = default;
		
		public: LambdaAbst Currying(void) {
			return LambdaAbst(Element(A.var,&A),Element(B.var,&B));
		}
		
		public: inline std::string literal(void) {
			return "("+A.var+")*("+B.var+")";
		}
		
		public: inline Element operator() (void) {
			Element lit(this->literal(),&Universe);
			lit.et=CartproType;
			lit.first = new Element(A);
			lit.second = new Element(B);
			return lit;
		}
		
		public: Element constructor(Element *a, Element *b) {
			Element *type = new Element((*this)());
			Element con("*("+a->var+","+b->var+")", type, CartesianPair);
			con.et=CartesianPair;
			con.first=new Element(*a);
			con.second=new Element(*b);
			return con;
		};
		
		public: [[deprecated]] Element recursor_org(Element C, Element g, Element pair) {  // g:A->B->C, pair:(A*B)
			LambdaAbst abst(*g.first,*g.second),
			appl0=LambdaAbst(abst.Appl(*pair.first), Element(B.var+"->"+C.var, nullptr));
			return appl0.Appl((*pair.second));
		};
		
		public: [[deprecated]] Element inductor_org(Element C, Element g, Element pair) {
			// C:(A*B)->U, g:A->B->(C(x) forall(x:A*B)), pair:(A*B), ind(-,-,-):(C(x) forall(x:A*B))
			LambdaAbst abst(*g.first,*g.second),
			appl0=LambdaAbst(abst.Appl(*pair.first), Element(B.var+"->"+C.var, nullptr));
			return appl0.Appl((*pair.second));
		};
		
		public: Element recursor(Element C, Element g, Element pair) {  // g:A->B->C, pair:(A*B)
			Element A = *(pair.first);
			Element B = *(pair.second);
			Element Cparam(C.var, &Universe,CustomType);
			std::string ab2c = "(" + A.var + "->" + B.var + "->" + Cparam.var + ")";
			std::string pair2c = "(" + A.var + "*" + B.var + "->" + Cparam.var + ")";
			std::string whole = ab2c + "->" + pair2c;
			Element wholeElem(whole, &Universe,CustomType);
			ForallType ft(Cparam, wholeElem);
			return ft();
		};
		
		public: Element inductor(Element C, Element g, Element pair) {
			Element A = *pair.first;
			Element B = *pair.second;
			std::string CdomStr = "(C:(" + A.var + "*" + B.var + ")->U)";
			Element CdomElem(CdomStr, &Universe,CustomType);
			std::string gsig = "(Pi^(a:" + A.var + "),(Pi^(b:" + B.var + "),C(-)))";
			std::string outsig = "(" + A.var + "*" + B.var + "->C(-))";
			std::string whole = gsig + "->" + outsig;
			Element wholeElem(whole, &Universe,CustomType);
			ForallType ft(CdomElem, wholeElem);
			return ft();
		};
	};
	
	class DisjointCoproduct {
		public: Element A, B;
		public: DisjointCoproduct(void) = default;
		public: DisjointCoproduct(Element a, Element b): A(a), B(b) {};
		public: ~DisjointCoproduct(void) = default;
		
		public: inline std::string literal(void) {
			return "("+A.var+")+("+B.var+")";
		}
		
		public: inline Element operator() (void) {
			Element lit(this->literal(),&Universe);
			lit.et=CoproType;
			lit.first = new Element(A);
			lit.second = new Element(B);
			return lit;
		}
		
		public: Element inl(Element *i) {  // Left injection
			Element *type = new Element((*this)());
			Element con("+(0,"+i->var+")", type, CoproductPair);
			con.et = CoproductPair;
			con.first = &TypeZero;
			con.second = new Element(*i);
			return con;
		}
		
		public: Element inr(Element *i) {  // Right injection
			Element *type = new Element((*this)());
			Element con("+(1,"+i->var+")", type, CoproductPair);
			con.et = CoproductPair;
			con.first = &TypeOne;
			con.second = new Element(*i);
			return con;
		}
	};
	
	class Operator {
		
		public: Operator(void) =default;
		public: ~Operator(void) =default;
		
		public: void init(std::function<void(Operator&)> initfunc) {
			initfunc(*this);
			return;
		}
		
		std::pair<Element,Element> abstlambda(Element &A, Element &C, unsigned epoch) {
			if (A.et != Variable) return std::make_pair(NilType,NilType);
			if (C.et == CustomType) return std::make_pair(NilType,NilType);
			
			LambdaAbst lab(A, C);
			Element lambda = lab.Abst();
			Element type(A.type->var + "->" + C.type->var, &Universe, CustomType);
			
			return std::make_pair(lambda,type);
		}
		
		std::pair<Element,Reference> appllambda(Element &lambda, Element &appl, unsigned epoch) {
			std::pair<Element,Reference> nil=std::make_pair(NilType,Reference());
			if (lambda.et != Lambda) return nil;
			if (!lambda.first) return nil;
			
			Element arg = *lambda.first;
			if (!appl.type || !arg.type) return nil;
			if (!(*appl.type == *arg.type)) return nil;
			if (appl.et != Constant && appl.et != Variable) return nil;
			
			LambdaAbst func(*lambda.first, *lambda.second);
			Element ret = func.Appl(arg);
			
			Reference ref(ret, Element(lambda.var + "[" + (*lambda.first).var + ":=" + appl.var + "]", func.consequence.type));
			return std::make_pair(ret,ref);
		}
		
		std::pair<Element,Reference> currying(Element &cartprotype, unsigned epoch) {
			std::pair<Element,Reference> nil=std::make_pair(NilType,Reference());
			if (cartprotype.et != CartproType) return nil;
			if (!cartprotype.first || !cartprotype.second) return nil;
			
			CartesianProduct cart(*(cartprotype.first), *(cartprotype.second));
			Element curried = cart.Currying().Abst();
			
			return std::make_pair(curried,Reference(*cartprotype.first, curried));
		}
		
		std::vector<Element> replace(const Reference &ref, const std::vector<Element> gamma, unsigned epoch) {
			std::vector<Element> sigma=gamma;
			Element alpha=ref.a, beta=ref.b;
			if (alpha.var == "" || beta.var.find(alpha.var) != std::string::npos) return sigma;
			
			for (auto &is : sigma) {
				std::string temp = is.var;
				std::string refa = alpha.var;
				std::string refb = beta.var;
				if (temp.find(refa) != std::string::npos) {
					size_t pos = temp.find(refa);
					if (pos == std::string::npos) continue;
					while (pos != std::string::npos) {
						temp.replace(pos, refa.length(), refb);
						pos = temp.find(refa, pos + refb.length());
					}
					sigma.push_back(Element(temp, is.type, is.et));
				}
			}
			return sigma;
		}
		
		std::pair<Element,Reference> recursion(Element &cartpair, unsigned epoch) {
			std::pair<Element,Reference> nil=std::make_pair(NilType,Reference());
			if (cartpair.et != CartesianPair) return nil;
			if (!cartpair.first || !cartpair.second) return nil;
			
			Element A = *(cartpair.first);
			Element B = *(cartpair.second);
			Element C("rec^C"+std::to_string(epoch)+"(" + cartpair.var + ")", &Universe);
			Element g("rec^g"+std::to_string(epoch)+"(" + A.var + "," + B.var + "->" + C.var + ")", &Universe);
			
			CartesianProduct cart(A, B);
			Element rec = cart.recursor(C, g, cartpair);
			
			return std::make_pair(rec,Reference(cartpair, rec));
		}
		
		std::pair<Element,Reference> induction(Element &cartpair, unsigned epoch) {
			std::pair<Element,Reference> nil=std::make_pair(NilType,Reference());
			if (cartpair.et != CartesianPair) return nil;
			if (!cartpair.first || !cartpair.second) return nil;
			
			Element A = *(cartpair.first);
			Element B = *(cartpair.second);
			Element C("pred^C"+std::to_string(epoch)+"(" + cartpair.var + ")", &Universe, CustomType);
			Element g("ind^g"+std::to_string(epoch)+"(" + A.var + "," + B.var + "->C(-))", &Universe, CustomType);
			
			CartesianProduct cart(A, B);
			Element ind = cart.inductor(C, g, cartpair);
			
			return std::make_pair(ind,Reference(cartpair, ind));
		}
		
		Element constructproduct(Element &A, Element &B, unsigned epoch) {
			if (A.et != CustomType || B.et != CustomType) return NilType;
			
			CartesianProduct cart(A, B);
			return cart();
		}
		
		Element constructcoproduct(Element &A, Element &B, unsigned epoch) {
			if (A.et != CustomType || B.et != CustomType) return NilType;
			DisjointCoproduct copr(A, B);
			return copr();
		}
		
		std::pair<Element,Reference> cartcouple(Element &v1, Element &v2, unsigned epoch) {
			std::pair<Element,Reference> nil=std::make_pair(NilType,Reference());
			if (!v1.type || !v2.type) return nil;
			
			CartesianProduct cart(*(v1.type), *(v2.type));
			Element pair = cart.constructor(&v1, &v2);
			
			return std::make_pair(pair,Reference(cart(), pair));
		}
		
		std::pair<Element,Reference>  coprocouple(Element &lv, Element &rv, bool cl, unsigned epoch) {
			std::pair<Element,Reference> nil=std::make_pair(NilType,Reference());
			if (!lv.type || !rv.type) return nil;
			
			DisjointCoproduct cop(*(lv.type), *(rv.type));
			Element inj = cl ? cop.inl(&lv) : cop.inr(&rv);
			return std::make_pair(inj,Reference(cop(), inj));
		}
	};
}
