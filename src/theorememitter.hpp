/******************************************************************************
 * Rigorous/TheoremEmitter: Automatic Theorem Prover templates.               *
 * @Author: classzheng@github                                                 *
 * @Date: 2026.4.11 (latest upd)                                              *
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

namespace TheoremEmitter {
	
	using ElementType = enum {
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
			(*this).~Autoptr();
		}
	} mainpool;
	class Element {
		public: std::string var;
		public: Element *type;
		public: Element *first, *second;
		public: ElementType et;
		
		public: Element(void)
		: var(""), type(nullptr), first(nullptr), second(nullptr), et(Constant) {}
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
		
		public: Element* copy(void) {
//			Element *cp=new Element(*this);
//			cp->first=first;
//			cp->second=second;
			return this;
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
		
	} Universe("TypeUniverse",nullptr),
	TypeZero("0",&Universe), TypeOne("1",&Universe), Asterisk("*",&TypeOne), NilType("nil",&Universe);
	
	class AtomicFormula {  // Judgment of Element
		public: Element ele;
		
		public: AtomicFormula(std::string var, Element *tp) {
			ele.var = var;
			if (tp != nullptr) {
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
			Element *tmpType=new Element(typeStr, &Universe);
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
		
		
		// I've used GPT-5 mini to write the below code.
		public: Element recursor(Element C, Element g, Element pair) {  // g:A->B->C, pair:(A*B)
			Element A = *(pair.first);
			Element B = *(pair.second);
			Element Cparam(C.var, &Universe);
			std::string ab2c = "(" + A.var + "->" + B.var + "->" + Cparam.var + ")";
			std::string pair2c = "(" + A.var + "*" + B.var + "->" + Cparam.var + ")";
			std::string whole = ab2c + "->" + pair2c;
			Element wholeElem(whole, &Universe);
			ForallType ft(Cparam, wholeElem);
			return ft();
		};
		
		public: Element inductor(Element C, Element g, Element pair) {
			Element A = *pair.first;
			Element B = *pair.second;
			std::string CdomStr = "(C:(" + A.var + "*" + B.var + ")->U)";
			Element CdomElem(CdomStr, &Universe);
			std::string gsig = "(Pi^(a:" + A.var + "),(Pi^(b:" + B.var + "),C(-)))";
			std::string outsig = "(" + A.var + "*" + B.var + "->C(-))";
			std::string whole = gsig + "->" + outsig;
			Element wholeElem(whole, &Universe);
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
			Element type=(*this)();
			Element con("+(0,"+i->var+")", &type, CoproductPair);
			con.et=CoproductPair;
			con.first=&TypeZero;
			// minimal fix: make second a heap copy
			con.second=new Element(*i);
			return con;
		};
		
		public: Element inr(Element *i) {  // Right injection
			Element type=(*this)();
			Element con("+(1,"+i->var+")", &type, CoproductPair);
			con.et=CoproductPair;
			con.first=&TypeOne;
			con.second=new Element(*i);
			return con;
		};
		
		
		
		// CAUTION: I am still learning about Coproduct type, so it's not available now... 
		public: [[deprecated]] Element recursor(Element C, Element g, Element pair) {  // g:A->B->C, pair:(A+B)
			LambdaAbst abst(*g.first,*g.second),
			appl0=LambdaAbst(abst.Appl(*pair.first), Element(B.var+"->"+C.var, nullptr));
			return appl0.Appl((*pair.second));
		};
		
		public: [[deprecated]] Element inductor(Element C, Element g, Element pair) {
			// C:(A*B)->U, g:A->B->(C(x) forall(x:A*B)), pair:(A*B), ind(-,-,-):(C(x) forall(x:A*B))
			LambdaAbst abst(*g.first,*g.second),
			appl0=LambdaAbst(abst.Appl(*pair.first), Element(B.var+"->"+C.var, nullptr));
			return appl0.Appl((*pair.second));
		};
	};
	
	using EmitOperation = enum {
		AbstLambda,
		ApplLambda, // todo: add more!
		Currying,
		Replace,
		Recursion,
		Induction,
		ConstructProduct,
		ConstructCoproduct,
	};
	const size_t operationamt=8;
	
	class Emitter {
		public: std::vector<Element> Gamma;
		public: std::vector<Reference> mapping;
		
		public: Emitter(void) {}
		public: ~Emitter(void) {
//			for(auto& is:Gamma) delete (&is);
			Gamma.clear();
			return ;
		}
		
		public: void init(std::function<void(Emitter&)> initfunc) {
			initfunc(*this);
			return;
		}
		
		public: void emitfor(int limit) {
			int epoch = 0, applcnt=0;
			EmitOperation emtop;
			
			std::random_device rd;
			std::mt19937 mtrng(rd());
			
			auto rand_index = [&](int sz) -> int {
				if (sz == 0) return 0;
				std::uniform_int_distribution<int> dist(0, sz - 1);
				return dist(mtrng);
			};
			
			while (epoch < limit) {
				emtop = (EmitOperation)(mtrng() % operationamt);
				while(rand_index(Gamma.size()))  // Shuffle
					std::swap(Gamma[rand_index(Gamma.size())],Gamma[rand_index(Gamma.size())]);
				while(rand_index(mapping.size()))
					std::swap(mapping[rand_index(mapping.size())],mapping[rand_index(mapping.size())]);
				
				switch (emtop) {
					case AbstLambda: {
					// Construct a function here. 
					if (Gamma.size() < 2) break;
					int a=0, c=0;
					while(Gamma[a].et!=Variable) {
						a = rand_index(Gamma.size()), c = rand_index(Gamma.size());
						if (a == c) c = (c + 1) % Gamma.size();
					}
					
					LambdaAbst lab(Gamma[a], Gamma[c]);
					Element lambda = lab.Abst();
					
					std::cout << "(abst) Gamma:- " << lambda.var
					<< " : " << (lambda.type ? lambda.type->var : "nil") << ".\n";
					
					Gamma.push_back(lambda);
					break;
				}
					
					case ApplLambda: {
//						// Select a function and arguments.
						Element *lambda=nullptr;
						for(auto &is:Gamma) 
							if(is.et==Lambda) {lambda=&is; break;}
						if(lambda==nullptr) {
							break;
						}
						Element arg = (lambda->first ? *lambda->first : Element());
						Element *appl=nullptr;
						for(auto &is:Gamma) {
							if(is.type && arg.type && *is.type==*arg.type && is.et==Constant) {
								appl=&is;
								break;
							}
						}
						if(appl==nullptr) {
							break;
						}
						LambdaAbst func(*lambda->first, *lambda->second);
						Element ret = func.Appl(arg);
						std::cout << "(appl) Gamma, " << lambda->var << ":- ("
						<< (AtomicFormula(ret).literal()) << "):=" << lambda->var+"["+(*lambda->first).var+":="+appl->var+"]" << ".\n";
						Reference ref(ret, Element(lambda->var+"["+(*lambda->first).var+":="+appl->var+"]",func.consequence.type));
						mapping.push_back(ref);
						Gamma.push_back(ret);
						break;
					}
					
					case Currying: {
						// Convert a Cartesian product type.
						Element *cartprotype=nullptr;
						for(auto &is:Gamma) {
							if(is.et==CartproType) {
								cartprotype=&is;
								break;
							}
						}
						if(cartprotype==nullptr) {
							break;
						}
						CartesianProduct cart2pro(*(cartprotype->first),*(cartprotype->second));
						Element curried = cart2pro.Currying().Abst();
						std::cout << "(curry) Gamma:- " << cart2pro.literal() << ":=" << curried.var << ".\n";
						Reference ref(*cartprotype, curried);
						mapping.push_back(ref);
						Gamma.push_back(curried);
						break;
					}
					
					case Replace: {
						// Replace some elements.
						if(mapping.size()==0) break;
						Reference ref = mapping[0];
						if(ref.alpha().var==""||ref.beta().var.find(ref.alpha().var)!=-1) break;
						std::cout << "(repl) " << ref.literal() << ":- Gamma.\n";
						for(auto &is:Gamma) {
							std::string temp=is.var, refa=ref.alpha().var, refb=ref.beta().var;
							if(is.var.find(ref.alpha().var)!=-1) {
								int pos=temp.find(refa);
								if(pos==-1) continue;
								while(pos!=-1) {
									temp.replace(pos, (refa).length(), refb);
//									temp.erase(pos, (refa).length());
									pos=temp.find(refa);
								}
								Gamma.push_back(Element(temp,is.type));
							}
						}
						break;
					}
					
					case Recursion: {
						// Construct a recursor.
						break;
					}
					
					case Induction: {
						// Construct a inductor (Always to prove)
						break;
					}
					
					case ConstructProduct: {
						// Construct a product.
						Element *A=&Gamma[rand_index(Gamma.size())], *B=&Gamma[rand_index(Gamma.size())];
						CartesianProduct cart(*A,*B);
						std::cout << "(cart) " << AtomicFormula(*A).literal() << "," << AtomicFormula(*B).literal()
						<< ":- " << cart.literal() << " : TypeUniverse.\n";
						Gamma.push_back(cart());
						break;
					}
					
					case ConstructCoproduct: {
						// Construct a coproduct.
						Element *A=&Gamma[rand_index(Gamma.size())], *B=&Gamma[rand_index(Gamma.size())];
						DisjointCoproduct discop(*A,*B);
						std::cout << "(copr) " << AtomicFormula(*A).literal() << "," << AtomicFormula(*B).literal()
						<< ":-" << discop.literal() << " : TypeUniverse.\n";
						Gamma.push_back(discop());
						break;
					}
					
				default:
					throw std::exception();
				}
//				getchar();
				epoch++;
			}
			
			std::cout << "Have run for " << epoch+applcnt << " epochs.\n";
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
