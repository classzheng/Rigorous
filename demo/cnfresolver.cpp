#include <bits/stdc++.h>
using ClauseIndex = int;
class Clause {
    public: std::vector<ClauseIndex> lits;
    public: bool taut = false;  // Shrinking.

    public: Clause() = default;
    public: Clause(std::vector<ClauseIndex> v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (ClauseIndex &x : v) {
            if (std::binary_search(v.begin(), v.end(), -x)) { taut = true; lits.clear(); return; }
        }
        lits = std::move(v);
    }
    public: Clause(const ClauseIndex v1, const ClauseIndex v2, bool) {
        std::vector<ClauseIndex> v={-v1,v2};  // v1 -> v2:= ~v1 v v2
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (ClauseIndex &x : v) {
            if (std::binary_search(v.begin(), v.end(), -x)) { taut = true; lits.clear(); return; }
        }
        lits = std::move(v);
    }
    public: Clause(const ClauseIndex v1, const ClauseIndex v2, float) {
        std::vector<ClauseIndex> v={-v1,v2, -v2,v1};  // v1 ^ v2:= (~v1 v v2) v (~v2 v v1)
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (ClauseIndex &x : v) {
            if (std::binary_search(v.begin(), v.end(), -x)) { taut = true; lits.clear(); return; }
        }
        lits = std::move(v);
    }
    public: inline bool empty(void) const {
        return !taut&&lits.empty();
    }
    public: std::string literal(void) {
        if(taut)    return "(Tautology)";
        if(empty()) return "(Nil)";
        std::stringstream s;
        for (int i=0; i<lits.size(); i++) {
            ClauseIndex x = lits[i];
            if (x<0) s << "~";
            s << "x" << std::to_string(std::abs(x));
            if (i<lits.size()-1) s << " v ";
        }
        s << ")";
        return "("+s.str();
    }
    public: bool subset(const Clause &other) const {
        ClauseIndex i=0,j=0;
        while (i<lits.size() && j<other.lits.size()) {
            if (lits[i]==other.lits[j]) { ++i; ++j; }
            else if (lits[i]>other.lits[j]) ++j;
            else return false;
        }
        return i==lits.size();
    }
};

class Resolver {
    public: std::vector<Clause> axioms;
    public: Resolver(void) = default;
    public: ~Resolver(void) = default;
    public: void add(const Clause& cl) {
        if(cl.taut)    return ;
        axioms.push_back(cl);
        return ;
    }
    public: ClauseIndex resolve(ClauseIndex idx1, ClauseIndex idx2) {
        Clause& cl1 = axioms[idx1];
        Clause& cl2 = axioms[idx2];
        if(cl1.taut || cl2.taut) return -1;
        if(cl1.empty() || cl2.empty()) return -1;
        int pivot = 0;
        bool found = false;
        for(ClauseIndex lit:cl1.lits) {
            if(std::binary_search(cl2.lits.begin(), cl2.lits.end(), -lit)) {
                pivot = lit;
                found = true;
                break;
            }
        }
        if(!found) return -1;
        std::vector<ClauseIndex> newlits;
        for(ClauseIndex lit : cl1.lits) if(lit != pivot) newlits.push_back(lit);
        for(ClauseIndex lit : cl2.lits) if(lit != -pivot) newlits.push_back(lit);
        Clause result(newlits);
        if(result.taut) return -1;
        for (auto& cl : axioms) if (cl.subset(result)) return -1;
        std::cout << result.literal()
                  << ":= [" << cl1.literal() << "&" << cl2.literal() << "].\n";
        add(result);
        return axioms.size() - 1;
    }
    public: ClauseIndex run(ClauseIndex t) {
        int oldSize = 0;
        this->add(Clause({-t}));
        while(oldSize != axioms.size()) {
            oldSize = axioms.size();
            for(int i=0; i<axioms.size(); i++) {
                for(int j=i+1; j<axioms.size(); j++) {
                    ClauseIndex res = resolve(i, j);
                    if(res != -1 && axioms[res].empty()) return res;
                }
            }
        }
        return -1;
    }
};
int main(void) {
    Resolver solver;
    solver.add(Clause({1,2}));  // x1 v x2
    solver.add(Clause({-1,3})); // ~x1 v x3
    solver.add(Clause({-2,3})); // ~x2 v x3
    solver.add(Clause({-3}));   // ~x3
    std::cout << (solver.run(1) == -1 ? 0 : 1);
	return 0;
}
