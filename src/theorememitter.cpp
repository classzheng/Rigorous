/******************************************************************************
 * Rigorous/TheoremEmitter: Automatic Theorem Prover templates.               *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.3.27 (latest upd)                                              *
 * @Description: Based on Martin-Lof Type Theory (MLTT).                      *
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

namespace TheoremEmitter {
	
	using ElementType = enum {
		Variable,
		Constant,
		Lambda,
		Reference,
		CartPair
	};
	class Element {
		public: std::string var;
		public: Element *type;
		public: Element *ac, *co;
		public: Element *first, *second;
		public: ElementType et;
		
		public: Element(void) : var(""), type(nullptr) {}
		public: ~Element(void) = default;
		public: Element(std::string v, Element *tp) : var(v), type(tp), et(Constant) {}
		public: Element(Element *a, Element *c, float) : ac(a), co(c), et(Lambda) {}
		public: Element(Element *f, Element *s, long) : first(f), second(s), et(CartPair) {}
		
		public: Element(const Element& other) : var(other.var), type(nullptr), et(Reference) {
			if (other.type != nullptr) {
				type = new Element(other.type->var, nullptr);
			}
			return ;
		}
		
		public: Element& operator=(const Element& other) {
			if (this != &other) {
				var = other.var;
				type = other.type;
			}
			return *this;
		}
		
		public: void betaredc(Element alpha, Element beta) {
			if (type == nullptr) return;
			while(var.find(alpha.var)!=-1)
				var.replace(var.find(alpha.var),var.find(alpha.var)+alpha.var.length(),beta.var);
			type->betaredc(alpha, beta); // recursion
			return;
		}
	} Universe("TypeUniverse",nullptr);
	
	class AtomicFormula {  // Constructor of Element
		public: Element ele;
		
		public: AtomicFormula(std::string var, Element *tp) {
			ele.var = var;
			if (tp != nullptr) {
				ele.type = new Element(tp->var, nullptr);
			} else {
				ele.type = nullptr;
			}
		}
		
		public: AtomicFormula(Element e) : ele(e) {}
		
		public: ~AtomicFormula(void) {
			if (ele.type != nullptr) {
				ele.type = nullptr;
			}
		}
		
		public: inline Element& operator() (void) {
			return ele;
		}
		
		public: inline std::string literal(void) {
			return ele.var + ":" + ele.type->var;
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
			AtomicFormula af("lambda^(" + AtomicFormula(antecedent).literal() + ")." + c.var, new Element(typeStr, nullptr));
			lambda = af();
			lambda.et=Lambda;
			lambda.ac=&antecedent;
			lambda.co=&consequence;
			return ;
		}
		
		public: Element Appl(Element var) {  // Apply function by Beta-Reduction
			Element applresult = consequence;
			applresult.betaredc(antecedent, var);
			return applresult;
		}
		
		public: inline Element Abst(void) {  // Abstract function by Eta-Expansion
			return lambda;
		}
		
		public: inline Element operator() (void) {
			return lambda;
		};
		
		public: inline std::string literal(void) {
			return antecedent.var + "->" + consequence.type->var;
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
			return A.var+"*"+B.var;
		}
		
		public: inline Element operator() (void) {
			return Element(this->literal(),&Universe);
		}
		
		public: Element constructor(Element a, Element b) {
			Element type=(*this)();
			return Element("*("+a.var+","+b.var+")", &type);
		};
		
		public: Element recursor(Element C, Element g, Element pair) {  // g:A->B->C, pair:(A*B)
			LambdaAbst abst(*g.ac,*g.co),
			appl0=LambdaAbst(abst.Appl(*pair.first), Element(B.var+"->"+C.var, nullptr));
			return appl0.Appl((*pair.second));
		};
		
		public: Element inductor(Element C, Element g, Element pair) {
			// C:(A*B)->U, g:A->B->(C(x) forall(x:A*B)), pair:(A*B), ind(-,-,-):(C(x) forall(x:A*B))
			LambdaAbst abst(*g.ac,*g.co),
			appl0=LambdaAbst(abst.Appl(*pair.first), Element(B.var+"->"+C.var, nullptr));
			return appl0.Appl((*pair.second));
		};
	};
	
	class DisjointCoproduct {
		public: Element A, B;
		public: DisjointCoproduct(void) = default;
		public: DisjointCoproduct(Element a, Element b): A(a), B(b) {};
		public: ~DisjointCoproduct(void) = default;
		
		public: LambdaAbst Currying(void) {
			return LambdaAbst(Element(A.var,&A),Element(B.var,&B));
		}
		
		public: inline std::string literal(void) {
			return A.var+"*"+B.var;
		}
		
		public: inline Element operator() (void) {
			return Element(this->literal(),&Universe);
		}
		
		public: Element constructor(Element a, Element b) {
			Element type=(*this)();
			return Element("+("+a.var+","+b.var+")", &type);
		};
		
		public: Element recursor(Element C, Element g, Element pair) {  // g:A->B->C, pair:(A+B)
			LambdaAbst abst(*g.ac,*g.co),
			appl0=LambdaAbst(abst.Appl(*pair.first), Element(B.var+"->"+C.var, nullptr));
			return appl0.Appl((*pair.second));
		};
		
		public: Element inductor(Element C, Element g, Element pair) {
			// C:(A*B)->U, g:A->B->(C(x) forall(x:A*B)), pair:(A*B), ind(-,-,-):(C(x) forall(x:A*B))
			LambdaAbst abst(*g.ac,*g.co),
			appl0=LambdaAbst(abst.Appl(*pair.first), Element(B.var+"->"+C.var, nullptr));
			return appl0.Appl((*pair.second));
		};
	};
	
	using EmitOperation = enum {
		AbstLambda,
		ApplLambda, // todo: add more!
		Currying,
		Recursion,
		Induction,
		ConstructProduct,
		ConstructCoproduct,
	};
	const size_t operationamt=7;
	
	class Emitter {
		public: std::vector<Element> Gamma;
		
		public: Emitter(void) {}
		public: ~Emitter(void) {
			Gamma.clear();
			return ;
		}
		
		public: void init(std::function<void(Emitter&)> initfunc) {
			initfunc(*this);
			return;
		}
		
		public: void emitfor(int limit) {
			int epoch = 0;
			EmitOperation emtop;
			
			std::random_device rd;
			std::mt19937 mtrng(rd());
			
			while (epoch < limit) {
				emtop = (EmitOperation)(mtrng() % operationamt);
				
				switch (emtop) {
					case AbstLambda:
						// Construct a function here. 
						break;
					case ApplLambda:
						// Select a function and arguments.
						break;
					case Currying:
						// Convert a Cartesian product type.
						break;
					case Recursion:
						// Construct a recursor.
						break;
					case Induction:
						// Construct a inductor (Always to prove)
						break;
					case ConstructProduct:
						// Construct a product.
						break;
					case ConstructCoproduct:
						// Construct a coproduct.
						break;
					default:
						throw std::exception();
				}
				
				epoch++;
			}
			
			std::cout << "Have run for " << epoch << " epochs.\n";
			return;
		}
		
		public: void printSigma(void) {
			std::cout << "Sigma contents:" << std::endl;
			for (size_t i = 0; i < Gamma.size(); i++) {
				std::cout << "  Sigma[" << i << "]:= " << Gamma[i].var << " : " << Gamma[i].type->var << std::endl;
			}
			return ;
		}
	};
}

int main(void) {
	using namespace TheoremEmitter;
	
	Emitter emt;
	emt.init([&](Emitter& self) -> void {
		Element Human("Human", &Universe);
		Element Mortal("Mortal", &Human);
		Element socrates("Socrates", &Mortal);
		Element human0("human0", &Human);
		Element death("death", &Universe);
		Element godie("godie(-)", &death);
		
		LambdaAbst abstFunc(human0, godie);
		Element willdie = abstFunc.Abst();
		
		self.Gamma.push_back(Human);
		self.Gamma.push_back(Mortal);
		self.Gamma.push_back(socrates);
		self.Gamma.push_back(human0);
		self.Gamma.push_back(death);
		self.Gamma.push_back(willdie);
		
		std::cout << "Rigorous::MachineTheoremEmitter initialized.\n\n";
		std::cout << "A simple deductive proof: " << abstFunc.literal() << ".\nProof: Socrates would die.\n";
		std::cout << "Applying to Socrates:" << std::endl;
		
		Element applResult = abstFunc.Appl(socrates);
		std::cout << "  Result: " << applResult.var << "\n\n";
		
		return;
	});
	
	emt.emitfor(114);
	emt.printSigma();
	
	return 0;
}
