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

int main(void) {
	using namespace TheoremOperator;
	
	Operator op;
	op.init([&](Operator& self) -> void {
		Element *Human = new Element("Human", &Universe, Constant);
		Element *Death = new Element("Death", &Universe, Constant);
		Element *socrates = new Element("Socrates", Human, Constant);
		Element *human0 = new Element("human0", Human, Variable);
		Element *godie = new Element("godie(-)", Death, Variable);
		
		LambdaAbst abstFunc(*human0, *godie);
		Element willdie = abstFunc.Abst();
		
		std::cout << "Rigorous::MachineTheoremOperator initialized.\n\n";
		std::cout << "A simple deductive proof: " << abstFunc.literal() << ".\nProof: Socrates would die.\n";
		std::cout << "Applying to Socrates:" << std::endl;
		
		Element applResult = abstFunc.Appl(*socrates);
		std::cout << "  Result: " << applResult.var << "\n\n";
		
		return;
	});
	
	auto printElem = [&](const std::string &title, const Element &e) {
		std::cout << title << ": var=\"" << e.var << "\""
		<< "  type=" << (e.type ? e.type->var : "nil")
		<< "  et=" << static_cast<int>(e.et) << "\n";
	};
	
	unsigned epoch = 1;
	
// Setup base elements
	Element Human("Human", &Universe, Constant);
	Element Death("Death", &Universe, Constant);
	Element Socrates("Socrates", &Human, Constant);
	Element human0("human0", &Human, Variable);
	Element godie("godie(-)", &Death, Variable);
	LambdaAbst abstFunc(human0, godie);
	Element willdie = abstFunc.Abst();
	
// 1) abstlambda: build a lambda from human0 -> godie
	auto p_abst = op.abstlambda(human0, willdie /* use global from earlier? create local willdie */ , epoch++);
// If the Operator::abstlambda in your copy expects a consequence Element (C) we should pass an element typed by Death.
// Create a simple consequence element for demo:
	Element consequence("will_die", &Death, Constant);
	p_abst = op.abstlambda(human0, consequence, epoch++);
	
	if (p_abst.first.var != "nil") {
		printElem("abstlambda produced lambda", p_abst.first);
		printElem("abstlambda produced type", p_abst.second);
	} else {
		std::cout << "abstlambda failed\n";
	}
	std::cout << "\n";
	
// 2) appllambda: apply the produced lambda to Socrates (if we have a lambda)
	if (p_abst.first.var != "nil") {
		auto p_appl = op.appllambda(p_abst.first, Socrates, epoch++);
		if (p_appl.first.var != "nil") {
			printElem("appllambda result", p_appl.first);
			std::cout << "appllambda produced reference: " << p_appl.second.literal() << "\n";
		} else {
			std::cout << "appllambda failed\n";
		}
	} else {
		std::cout << "skipping appllambda because lambda missing\n";
	}
	std::cout << "\n";
	
// 3) constructproduct: build product type from TypeZero and TypeOne
	Element prod = op.constructproduct(TypeZero, TypeOne, epoch++);
	if (prod.var != "nil") {
		printElem("constructproduct produced", prod);
	} else {
		std::cout << "constructproduct failed\n";
	}
	std::cout << "\n";
	
// 4) currying: curry the cartesian product type
	if (prod.var != "nil") {
		auto p_curry = op.currying(prod, epoch++);
		if (p_curry.first.var != "nil") {
			printElem("currying produced curried", p_curry.first);
			std::cout << "currying produced reference: " << p_curry.second.literal() << "\n";
		} else {
			std::cout << "currying failed\n";
		}
	} else {
		std::cout << "skipping currying because product missing\n";
	}
	std::cout << "\n";
	
// 5) replace: demonstrate replacing alpha->beta over a gamma vector
// Prepare a gamma with an element containing the alpha.var substring
	std::vector<Element> gamma;
	gamma.emplace_back("foo0bar", &TypeZero, Constant);
	gamma.emplace_back("keep_me", &TypeOne, Constant);
	
// Use the reference from currying if available, otherwise craft a simple ref
	Reference ref;
	if (prod.var != "nil") {
		// currying reference uses cartprotype.first as alpha; create alpha/beta for demo:
		Element alpha("0", &Universe, Constant);          // pretend alpha.var == "0"
		Element beta("ZERO", &Universe, Constant);        // replacement
		ref = Reference(alpha, beta);
	} else {
		Element alpha("0", &Universe, Constant);
		Element beta("ZERO", &Universe, Constant);
		ref = Reference(alpha, beta);
	}
	
	auto gamma_after = op.replace(ref, gamma, epoch++);
	std::cout << "replace produced gamma of size " << gamma_after.size() << ":\n";
	for (size_t i = 0; i < gamma_after.size(); ++i) {
		printElem("  gamma[" + std::to_string(i) + "]", gamma_after[i]);
	}
	std::cout << "\n";
	
// 6) constructcoproduct: build coproduct type
	Element copr = op.constructcoproduct(TypeZero, TypeOne, epoch++);
	if (copr.var != "nil") {
		printElem("constructcoproduct produced", copr);
	} else {
		std::cout << "constructcoproduct failed\n";
	}
	std::cout << "\n";
	
// 7) coprocouple: create left/right injections
	Element left0("left0", &TypeZero, Constant);
	Element right1("right1", &TypeOne, Constant);
	auto p_inl = op.coprocouple(left0, right1, true, epoch++);
	if (p_inl.first.var != "nil") {
		printElem("coprocouple(inl) produced", p_inl.first);
		std::cout << "coprocouple produced reference: " << p_inl.second.literal() << "\n";
	} else {
		std::cout << "coprocouple(inl) failed\n";
	}
	auto p_inr = op.coprocouple(left0, right1, false, epoch++);
	if (p_inr.first.var != "nil") {
		printElem("coprocouple(inr) produced", p_inr.first);
		std::cout << "coprocouple produced reference: " << p_inr.second.literal() << "\n";
	} else {
		std::cout << "coprocouple(inr) failed\n";
	}
	std::cout << "\n";
	
// 8) cartcouple: pair Socrates and human0
	auto p_pair = op.cartcouple(Socrates, human0, epoch++);
	if (p_pair.first.var != "nil") {
		printElem("cartcouple produced pair", p_pair.first);
		std::cout << "cartcouple produced reference: " << p_pair.second.literal() << "\n";
	} else {
		std::cout << "cartcouple failed\n";
	}
	std::cout << "\n";
	
// 9) recursion & induction: run on the produced Cartesian pair (if any)
	if (p_pair.first.var != "nil") {
		auto p_recr = op.recursion(p_pair.first, epoch++);
		if (p_recr.first.var != "nil") {
			printElem("recursion produced", p_recr.first);
			std::cout << "recursion produced reference: " << p_recr.second.literal() << "\n";
		} else {
			std::cout << "recursion failed\n";
		}
		
		auto p_ind = op.induction(p_pair.first, epoch++);
		if (p_ind.first.var != "nil") {
			printElem("induction produced", p_ind.first);
			std::cout << "induction produced reference: " << p_ind.second.literal() << "\n";
		} else {
			std::cout << "induction failed\n";
		}
	} else {
		std::cout << "skipping recursion/induction because cart pair missing\n";
	}
	std::cout << "\n";
	
	mainpool.clear();
	return 0;
}
