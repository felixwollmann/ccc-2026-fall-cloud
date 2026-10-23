// C++ port of solver.py (level 6, Nanofabricator).
//
// Same algorithm as the Python version: a beam search over events (whole-line
// fills with blockers on the skirt cells, point events creating pattern cells
// directly, cleanups) with exact incremental simulation, plus fixed-order sweep
// variants; passes are repeated until the required count is reached.
//
// Build:  g++ -O2 -std=c++17 -pthread -o solver_cpp solver.cpp
// Usage:  ./solver_cpp input_files/in_level-6_1-small.txt [-o output_files] [-j 4]
//                     [--only N] [--beam B] [--nlines N] [--final F] [--scale S]

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std;

// ---------------------------------------------------------------------------
// bitmask over board cells (radius 12 has 397 interior cells)

constexpr int MW = 7;

struct Mask {
    uint64_t w[MW];
    Mask() { memset(w, 0, sizeof w); }
    void set(int i) { w[i >> 6] |= 1ULL << (i & 63); }
    bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }
    Mask operator&(const Mask& o) const { Mask r; for (int i = 0; i < MW; i++) r.w[i] = w[i] & o.w[i]; return r; }
    Mask operator|(const Mask& o) const { Mask r; for (int i = 0; i < MW; i++) r.w[i] = w[i] | o.w[i]; return r; }
    Mask andnot(const Mask& o) const { Mask r; for (int i = 0; i < MW; i++) r.w[i] = w[i] & ~o.w[i]; return r; }
    Mask& operator|=(const Mask& o) { for (int i = 0; i < MW; i++) w[i] |= o.w[i]; return *this; }
    Mask& operator&=(const Mask& o) { for (int i = 0; i < MW; i++) w[i] &= o.w[i]; return *this; }
    bool any() const { for (int i = 0; i < MW; i++) if (w[i]) return true; return false; }
    int count() const { int c = 0; for (int i = 0; i < MW; i++) c += __builtin_popcountll(w[i]); return c; }
    bool operator==(const Mask& o) const { return memcmp(w, o.w, sizeof w) == 0; }
    bool operator!=(const Mask& o) const { return !(*this == o); }
    template <class F> void forEach(F f) const {
        for (int i = 0; i < MW; i++) {
            uint64_t x = w[i];
            while (x) { int b = __builtin_ctzll(x); f(i * 64 + b); x &= x - 1; }
        }
    }
    vector<int> bits() const { vector<int> v; forEach([&](int i) { v.push_back(i); }); return v; }
    size_t hash() const {
        size_t h = 1469598103934665603ULL;
        for (int i = 0; i < MW; i++) { h ^= w[i]; h *= 1099511628211ULL; }
        return h;
    }
};

// ---------------------------------------------------------------------------
// geometry

const int FAMILY[6] = {0, 1, 2, 0, 1, 2};
const int PAIR[3][2] = {{0, 3}, {1, 4}, {2, 5}};
const int DIR_TO[6][2] = {{2, 0}, {1, 1}, {-1, 1}, {-2, 0}, {-1, -1}, {1, -1}};  // E SE SW W NW NE
const int FIRE_OV[6][2] = {{2, 0}, {1, 1}, {-1, 1}, {-2, 0}, {-1, -1}, {1, -1}};

static int lineKey(int f, int c, int r) { return f == 0 ? r : (f == 1 ? c - r : c + r); }

struct Geo {
    int R, radius, rows, cols, N, perim, maxoff;
    vector<pair<int, int>> cells;
    map<pair<int, int>, int> idx;
    vector<pair<int, int>> perimeter;
    int start[6];
    vector<Mask> ray[6];               // per perimeter index
    vector<int> lineOfCell[3];         // compact line index per cell
    vector<int> lineIds[3];            // compact index -> line id
    vector<Mask> lineMask[3];
    vector<int> offForLine[6];         // emitter -> compact line index (of its family) -> offset

    bool inGrid(int c, int r) const {
        if (r < 0 || r >= 2 * radius - 1) return false;
        if ((r + c + radius + 1) % 2 != 0) return false;
        if (r < radius) return c + r >= radius - 1 && c - r <= cols - radius + 1;
        return c + (2 * radius - 1 - r) >= radius - 1 && c + r - 2 * radius + 1 < cols - radius + 1;
    }
    char baseCell(int c, int r) const {
        if ((r + c + radius + 1) % 2 != 0) return '-';
        bool rRail = r == 0 || r == rows - 1;
        if (r < radius) {
            bool cRail = c == radius - 1 - r || c == cols - 1 - radius + 1 + r;
            if (c + r >= radius - 1 && c - r <= cols - radius + 1) return (rRail || cRail) ? '#' : 'O';
            return ':';
        }
        bool cRail = c == radius - 1 + r - 2 * radius + 2 || c == cols - 1 + radius - 1 - r;
        if (c + 2 * radius - 1 - r >= radius - 1 && c + r - 2 * radius + 1 < cols - radius + 1) return (rRail || cRail) ? '#' : 'O';
        return ':';
    }
    int pidx(int t, int o) const { return ((start[t] + o) % perim + perim) % perim; }
    const Mask& maskAt(int t, int o) const { return ray[t][pidx(t, o)]; }

    explicit Geo(int R_) : R(R_) {
        radius = R + 1;
        rows = 2 * radius - 1;
        cols = 4 * radius - 3;
        maxoff = R - 1;
        for (int c = 0; c < cols; c++)
            for (int r = 0; r < rows; r++)
                if (baseCell(c, r) == 'O') cells.push_back({c, r});
        sort(cells.begin(), cells.end());
        N = cells.size();
        for (int i = 0; i < N; i++) idx[cells[i]] = i;
        // perimeter clockwise from the left corner: NE, E, SE, SW, W, NW
        int order[6] = {5, 0, 1, 2, 3, 4};
        int c = 0, r = radius - 1;
        for (int d : order)
            for (int i = 0; i < radius - 1; i++) {
                perimeter.push_back({c, r});
                c += DIR_TO[d][0];
                r += DIR_TO[d][1];
            }
        perim = perimeter.size();
        for (int t = 0; t < 6; t++) start[t] = t * max(radius - 1, 1);
        for (int t = 0; t < 6; t++) {
            ray[t].assign(perim, Mask());
            for (int p = 0; p < perim; p++) {
                int cc = perimeter[p].first + FIRE_OV[t][0], rr = perimeter[p].second + FIRE_OV[t][1];
                while (inGrid(cc, rr)) {
                    auto it = idx.find({cc, rr});
                    if (it != idx.end()) ray[t][p].set(it->second);
                    cc += FIRE_OV[t][0];
                    rr += FIRE_OV[t][1];
                }
            }
        }
        for (int f = 0; f < 3; f++) {
            set<int> ids;
            for (auto& cell : cells) ids.insert(lineKey(f, cell.first, cell.second));
            map<int, int> compact;
            for (int id : ids) { compact[id] = lineIds[f].size(); lineIds[f].push_back(id); }
            lineMask[f].assign(lineIds[f].size(), Mask());
            lineOfCell[f].resize(N);
            for (int i = 0; i < N; i++) {
                int li = compact[lineKey(f, cells[i].first, cells[i].second)];
                lineOfCell[f][i] = li;
                lineMask[f][li].set(i);
            }
        }
        for (int t = 0; t < 6; t++) {
            int f = FAMILY[t];
            offForLine[t].assign(lineIds[f].size(), 0);
            int found = 0;
            for (int o = -maxoff; o <= maxoff; o++) {
                const Mask& m = maskAt(t, o);
                for (size_t li = 0; li < lineMask[f].size(); li++)
                    if (lineMask[f][li] == m) { offForLine[t][li] = o; found++; }
            }
            assert(found == 2 * R - 1);
        }
    }
};

// ---------------------------------------------------------------------------
// input

struct Case {
    int index, R, rows, cols, count;
    vector<string> pattern;
    set<pair<int, int>> matter, skirt;
};

vector<Case> parseInput(const string& path) {
    ifstream in(path);
    if (!in) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    vector<string> lines;
    string s;
    while (getline(in, s)) {
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
        lines.push_back(s);
    }
    size_t pos = 0;
    auto next = [&]() -> string {
        while (pos < lines.size() && lines[pos].find_first_not_of(" \t") == string::npos) pos++;
        if (pos >= lines.size()) { fprintf(stderr, "unexpected end of input\n"); exit(1); }
        return lines[pos++];
    };
    int n = stoi(next());
    vector<Case> cases;
    for (int i = 0; i < n; i++) {
        Case c;
        c.index = i + 1;
        istringstream h(next());
        h >> c.R >> c.rows >> c.cols >> c.count;
        for (int r = 0; r < c.rows; r++) {
            string ln = next();
            size_t a = ln.find_first_not_of(" \t"), b = ln.find_last_not_of(" \t");
            c.pattern.push_back(ln.substr(a, b - a + 1));
        }
        for (int r = 0; r < c.rows; r++)
            for (int x = 0; x < (int)c.pattern[r].size(); x++)
                if (c.pattern[r][x] == 'M') c.matter.insert({x, r});
        for (int r = 0; r < c.rows; r++)
            for (int x = 0; x < (int)c.pattern[r].size(); x++)
                if (c.pattern[r][x] == 'O')
                    for (auto& d : DIR_TO)
                        if (c.matter.count({x + d[0], r + d[1]})) { c.skirt.insert({x, r}); break; }
        cases.push_back(c);
    }
    return cases;
}

// ---------------------------------------------------------------------------
// plans

struct Req {
    bool set[6] = {false, false, false, false, false, false};
    int off[6] = {0, 0, 0, 0, 0, 0};
    bool act[6] = {false, false, false, false, false, false};
    void put(int t, int o, bool a) { set[t] = true; off[t] = o; act[t] = a; }
};

struct State {
    int off[6] = {0, 0, 0, 0, 0, 0};
    bool act[6] = {false, false, false, false, false, false};
    Mask matter;
};

struct StateKey {
    int off[6];
    bool act[6];
    Mask matter;
    bool operator==(const StateKey& o) const {
        return memcmp(off, o.off, sizeof off) == 0 && memcmp(act, o.act, sizeof act) == 0 && matter == o.matter;
    }
};
struct StateKeyHash {
    size_t operator()(const StateKey& k) const {
        size_t h = k.matter.hash();
        for (int t = 0; t < 6; t++) h = h * 1000003u + (size_t)(k.off[t] + 64) * 2 + k.act[t];
        return h;
    }
};

struct Plan {
    const Geo* geo;
    int T = 0;
    int off[6];
    bool act[6];
    int last[6] = {0, 0, 0, 0, 0, 0};
    Mask matter;
    int submitted = 0;
    vector<char> tracks[6];
    Mask filled;

    Plan(const Geo* g, const State& s) : geo(g) {
        memcpy(off, s.off, sizeof off);
        memcpy(act, s.act, sizeof act);
        matter = s.matter;
    }
    int need(int t, int o, bool a) const { return abs(o - off[t]) + (a != act[t] ? 1 : 0); }
    int earliest(const Req& req) const {
        int T1 = T + 1;
        for (int t = 0; t < 6; t++)
            if (req.set[t]) T1 = max(T1, last[t] + need(t, req.off[t], req.act[t]));
        return T1;
    }
    StateKey key() const {
        StateKey k;
        memcpy(k.off, off, sizeof off);
        memcpy(k.act, act, sizeof act);
        k.matter = matter;
        return k;
    }
    State state() const {
        State s;
        memcpy(s.off, off, sizeof off);
        memcpy(s.act, act, sizeof act);
        s.matter = matter;
        return s;
    }
    vector<array<char, 6>> lines() const {
        vector<array<char, 6>> out(T);
        for (int i = 0; i < T; i++)
            for (int t = 0; t < 6; t++) out[i][t] = i < (int)tracks[t].size() ? tracks[t][i] : '-';
        return out;
    }

    Plan add(const Req& req, const Mask& pmask, const Mask& smask) const {
        int T0 = T;
        int T1 = earliest(req);
        int n = T1 - T0;
        Plan p = *this;
        vector<Mask> c1(n), c2(n), c3(n);
        auto hit = [&](int k, const Mask& m) {
            c3[k] |= c2[k] & m;
            c2[k] |= c1[k] & m;
            c1[k] |= m;
        };
        for (int t = 0; t < 6; t++) {
            if (!req.set[t]) continue;
            int o1 = req.off[t];
            bool a1 = req.act[t];
            int o0 = off[t], lt = last[t];
            bool a0 = act[t];
            vector<char>& tr = p.tracks[t];
            tr.resize(max<size_t>(tr.size(), T1), '-');
            int d = o1 - o0;
            char step = d > 0 ? 'F' : 'B';
            int sgn = d > 0 ? 1 : -1;
            int nd = abs(d);
            if (a0 && a1) {
                for (int i = 0; i < nd; i++) tr[T0 + i] = step;
                for (int k = 0; k < n; k++) hit(k, geo->maskAt(t, o0 + sgn * min(k + 1, nd)));
            } else if (a1) {
                for (int i = 0; i < nd; i++) tr[lt + i] = step;
                tr[T1 - 1] = 'T';
                hit(n - 1, geo->maskAt(t, o1));
            } else if (a0) {
                tr[T0] = 'T';
                for (int i = 0; i < nd; i++) tr[T0 + 1 + i] = step;
            } else {
                for (int i = 0; i < nd; i++) tr[lt + i] = step;
            }
            p.off[t] = o1;
            p.act[t] = a1;
            p.last[t] = T1;
        }
        Mask m = matter;
        for (int k = 0; k < n; k++) {
            m = (m | c2[k].andnot(c3[k])).andnot(c3[k]);
            if ((m & pmask) == pmask && !(m & smask).any()) {
                m = m.andnot(pmask);
                p.submitted++;
            }
        }
        p.matter = m;
        p.T = T1;
        return p;
    }
};

// independent straightforward simulator for the final verification
int simulateProgram(const Geo& geo, const Mask& pmask, const Mask& smask, const vector<array<char, 6>>& prog) {
    int off[6] = {0, 0, 0, 0, 0, 0};
    bool act[6] = {false, false, false, false, false, false};
    Mask matter;
    int submitted = 0;
    for (auto& step : prog) {
        for (int t = 0; t < 6; t++) {
            if (step[t] == 'F') off[t]++;
            else if (step[t] == 'B') off[t]--;
            else if (step[t] == 'T') act[t] = !act[t];
        }
        Mask c1, c2, c3;
        for (int t = 0; t < 6; t++)
            if (act[t] && -geo.maxoff <= off[t] && off[t] <= geo.maxoff) {
                const Mask& m = geo.maskAt(t, off[t]);
                c3 |= c2 & m;
                c2 |= c1 & m;
                c1 |= m;
            }
        matter = (matter | c2.andnot(c3)).andnot(c3);
        if ((matter & pmask) == pmask && !(matter & smask).any()) {
            matter = matter.andnot(pmask);
            submitted++;
        }
    }
    return submitted;
}

// ---------------------------------------------------------------------------
// planner

struct Params {
    int beam = 8, final_ = 3, finalOrdered = 2, nlines = 4, hmax = 5;
    bool ordered = true, point = true;
    int smallBeam = 24, smallNlines = 8, smallFinal = 4;
    double scale = 1.0;
};

struct Planner {
    const Case& cs;
    const Geo& geo;
    Params P;
    int K;
    Mask pmask, smask;
    vector<int> activeLines[3];              // compact line indices with pattern or skirt cells
    vector<Mask> pOn[3], sOn[3];
    vector<vector<pair<int, int>>> variants;  // (f, li) sequences

    Planner(const Case& c, const Geo& g, Params prm) : cs(c), geo(g), P(prm), K(c.count) {
        if (geo.R <= 5) { P.beam = P.smallBeam; P.nlines = P.smallNlines; P.final_ = P.smallFinal; }
        P.beam = max(1, (int)lround(P.beam * P.scale));
        P.nlines = max(1, (int)lround(P.nlines * P.scale));
        for (auto& cell : c.matter) pmask.set(geo.idx.at(cell));
        for (auto& cell : c.skirt) smask.set(geo.idx.at(cell));
        for (int f = 0; f < 3; f++) {
            int n = geo.lineMask[f].size();
            pOn[f].resize(n);
            sOn[f].resize(n);
            for (int li = 0; li < n; li++) {
                pOn[f][li] = geo.lineMask[f][li] & pmask;
                sOn[f][li] = geo.lineMask[f][li] & smask;
                if (pOn[f][li].any() || sOn[f][li].any()) activeLines[f].push_back(li);
            }
        }
    }

    // -- cover selection ---------------------------------------------------
    vector<pair<int, int>> chooseFills(int primary) const {
        Mask uncovered = pmask;
        vector<pair<int, int>> fills;
        int fams[3] = {primary, (primary + 1) % 3, (primary + 2) % 3};
        if (fams[1] > fams[2]) swap(fams[1], fams[2]);
        for (int f : fams) {
            while (true) {
                int bestGain = -1, bestNeg = 0, bestL = -1;
                for (int li : activeLines[f]) {
                    if (!pOn[f][li].any() || sOn[f][li].count() > 4) continue;
                    if (find(fills.begin(), fills.end(), make_pair(f, li)) != fills.end()) continue;
                    int gain = (pOn[f][li] & uncovered).count();
                    if (gain == 0) continue;
                    int neg = -sOn[f][li].count();
                    if (gain > bestGain || (gain == bestGain && neg > bestNeg)) { bestGain = gain; bestNeg = neg; bestL = li; }
                }
                if (bestL < 0) break;
                fills.push_back({f, bestL});
                uncovered = uncovered.andnot(pOn[f][bestL]);
            }
        }
        return fills;
    }

    const vector<vector<pair<int, int>>>& orderedVariants() {
        if (variants.empty()) {
            for (int primary = 0; primary < 3; primary++) {
                auto fills = chooseFills(primary);
                for (int dir : {1, -1}) {
                    auto v = fills;
                    stable_sort(v.begin(), v.end(), [&](const pair<int, int>& a, const pair<int, int>& b) {
                        int fa = ((a.first - primary) % 3 + 3) % 3, fb = ((b.first - primary) % 3 + 3) % 3;
                        if (fa != fb) return fa < fb;
                        return dir * geo.lineIds[a.first][a.second] < dir * geo.lineIds[b.first][b.second];
                    });
                    variants.push_back(v);
                }
            }
        }
        return variants;
    }

    // -- event options -----------------------------------------------------
    Req offReq(const Plan& plan, const vector<int>& group) const {
        Req r;
        for (int t = 0; t < 6; t++)
            if (plan.act[t]) r.put(t, plan.off[t], find(group.begin(), group.end(), t) == group.end());
        return r;
    }

    int offCell(int t, int cell) const { return geo.offForLine[t][geo.lineOfCell[FAMILY[t]][cell]]; }

    // ordered k-selections of items (permutations of subsets)
    static void perms(const vector<int>& items, int k, vector<int>& cur, vector<bool>& used,
                      const function<void(const vector<int>&)>& cb) {
        if ((int)cur.size() == k) { cb(cur); return; }
        for (size_t i = 0; i < items.size(); i++) {
            if (used[i]) continue;
            used[i] = true;
            cur.push_back(items[i]);
            perms(items, k, cur, used, cb);
            cur.pop_back();
            used[i] = false;
        }
    }

    static void product(const vector<pair<int, vector<pair<int, bool>>>>& lists, size_t i, Req& r,
                        const function<void(const Req&)>& cb) {
        if (i == lists.size()) { cb(r); return; }
        for (auto& c : lists[i].second) {
            r.put(lists[i].first, c.first, c.second);
            product(lists, i + 1, r, cb);
        }
    }

    vector<Req> fillOptions(const Plan& plan, int f, int li, int limit = 16) const {
        int a = PAIR[f][0], b = PAIR[f][1];
        Req base;
        base.put(a, geo.offForLine[a][li], true);
        base.put(b, geo.offForLine[b][li], true);
        vector<int> holes = sOn[f][li].bits();
        vector<int> blockers;
        for (int t = 0; t < 6; t++) if (FAMILY[t] != f) blockers.push_back(t);
        const Mask& Lmask = geo.lineMask[f][li];
        vector<pair<int, Req>> opts;
        int k = min<int>(holes.size(), 4);
        // combinations of k holes
        vector<int> comb(k);
        for (int i = 0; i < k; i++) comb[i] = i;
        auto body = [&](const vector<int>& hs) {
            vector<int> cur;
            vector<bool> used(blockers.size(), false);
            perms(blockers, k, cur, used, [&](const vector<int>& perm) {
                Req req = base;
                for (int i = 0; i < k; i++) req.put(perm[i], offCell(perm[i], hs[i]), true);
                vector<pair<int, vector<pair<int, bool>>>> lists;
                for (int t : blockers) {
                    if (req.set[t] || !plan.act[t]) continue;
                    int cur_ = plan.off[t];
                    vector<pair<int, bool>> cs = {{cur_, false}};
                    int tries[5] = {cur_, cur_ - 1, cur_ + 1, cur_ - 2, cur_ + 2};
                    for (int o : tries) {
                        if (-geo.maxoff <= o && o <= geo.maxoff && !(geo.maskAt(t, o) & Lmask & pmask).any()) {
                            cs.push_back({o, true});
                            if (o != cur_) break;
                        }
                    }
                    lists.push_back({t, cs});
                }
                Req r = req;
                product(lists, 0, r, [&](const Req& rr) { opts.push_back({plan.earliest(rr), rr}); });
            });
        };
        if (k == 0) {
            body({});
        } else {
            while (true) {
                vector<int> hs(k);
                for (int i = 0; i < k; i++) hs[i] = holes[comb[i]];
                body(hs);
                int i = k - 1;
                while (i >= 0 && comb[i] == (int)holes.size() - k + i) i--;
                if (i < 0) break;
                comb[i]++;
                for (int j = i + 1; j < k; j++) comb[j] = comb[j - 1] + 1;
            }
        }
        stable_sort(opts.begin(), opts.end(), [](const pair<int, Req>& x, const pair<int, Req>& y) { return x.first < y.first; });
        vector<Req> out;
        for (int i = 0; i < (int)opts.size() && i < limit; i++) out.push_back(opts[i].second);
        return out;
    }

    vector<Req> pointOptions(const Plan& plan, int f, int li, int limit = 8) const {
        int a = PAIR[f][0], b = PAIR[f][1];
        const Mask& Lmask = geo.lineMask[f][li];
        vector<int> cells = pOn[f][li].andnot(plan.matter).bits();
        int g2 = (f + 1) % 3;
        stable_sort(cells.begin(), cells.end(), [&](int x, int y) { return geo.lineOfCell[g2][x] < geo.lineOfCell[g2][y]; });
        vector<int> secondaries;
        for (int t = 0; t < 6; t++) if (FAMILY[t] != f) secondaries.push_back(t);
        vector<pair<int, Req>> opts;
        for (int pass = 0; pass < 2; pass++) {
            int A = pass == 0 ? a : b, Ap = pass == 0 ? b : a;
            Req base;
            base.put(A, geo.offForLine[A][li], true);
            if (plan.act[Ap]) base.put(Ap, plan.off[Ap], false);
            int k = min<int>(4, cells.size());
            for (int lo = 0; lo + k <= (int)cells.size(); lo++) {
                vector<int> cur;
                vector<bool> used(secondaries.size(), false);
                perms(secondaries, k, cur, used, [&](const vector<int>& perm) {
                    Req req = base;
                    for (int i = 0; i < k; i++) req.put(perm[i], offCell(perm[i], cells[lo + i]), true);
                    vector<pair<int, vector<pair<int, bool>>>> lists;
                    for (int t : secondaries) {
                        if (req.set[t] || !plan.act[t]) continue;
                        int cur_ = plan.off[t];
                        vector<pair<int, bool>> cs = {{cur_, false}};
                        if (!(geo.maskAt(t, cur_) & Lmask & smask).any()) cs.push_back({cur_, true});
                        lists.push_back({t, cs});
                    }
                    Req r = req;
                    product(lists, 0, r, [&](const Req& rr) { opts.push_back({plan.earliest(rr), rr}); });
                });
            }
        }
        stable_sort(opts.begin(), opts.end(), [](const pair<int, Req>& x, const pair<int, Req>& y) { return x.first < y.first; });
        vector<Req> out;
        for (int i = 0; i < (int)opts.size() && i < limit; i++) out.push_back(opts[i].second);
        return out;
    }

    int estPointTime(const Plan& plan, int f, int li, const vector<int>& cells) const {
        int a = PAIR[f][0], b = PAIR[f][1];
        int est = plan.T + 1;
        est = max(est, min(plan.last[a] + plan.need(a, geo.offForLine[a][li], true),
                           plan.last[b] + plan.need(b, geo.offForLine[b][li], true)));
        int best = INT32_MAX;
        for (int c : cells) {
            int m = INT32_MAX;
            for (int t = 0; t < 6; t++)
                if (FAMILY[t] != f) m = min(m, plan.last[t] + plan.need(t, offCell(t, c), true));
            best = min(best, m);
        }
        return cells.empty() ? est : max(est, best);
    }

    int estFillTime(const Plan& plan, int f, int li) const {
        int est = plan.T + 1;
        for (int t : PAIR[f]) est = max(est, plan.last[t] + plan.need(t, geo.offForLine[t][li], true));
        sOn[f][li].forEach([&](int h) {
            int m = INT32_MAX;
            for (int t = 0; t < 6; t++)
                if (FAMILY[t] != f) m = min(m, plan.last[t] + plan.need(t, offCell(t, h), true));
            est = max(est, m);
        });
        return est;
    }

    double remainingEstimate(Mask undone) const {
        int n = 0;
        while (undone.any()) {
            int best = 0;
            const Mask* bl = nullptr;
            for (int f = 0; f < 3; f++)
                for (int li : activeLines[f]) {
                    if (sOn[f][li].count() > P.hmax) continue;
                    int g = (pOn[f][li] & undone).count();
                    if (g > best) { best = g; bl = &pOn[f][li]; }
                }
            if (best == 0) return n + undone.count() / 3.0;
            undone = undone.andnot(*bl);
            n++;
        }
        return n;
    }

    double key(const Plan& plan) const {
        if (plan.submitted) return plan.T;
        Mask undone = pmask.andnot(plan.matter);
        Mask dirty = plan.matter & smask;
        return plan.T + remainingEstimate(undone) + 1.5 * dirty.count() + 1.5 * (plan.filled & undone).count();
    }

    double orderedKey(const Plan& plan, const Mask& remaining, const pair<int, int>* nxt) const {
        if (plan.submitted) return plan.T;
        Mask dirty = (plan.matter & smask).andnot(remaining);
        Mask undone = (plan.filled & pmask).andnot(plan.matter).andnot(remaining);
        double look = nxt ? estFillTime(plan, nxt->first, nxt->second) - (plan.T + 1) : 0;
        return plan.T + 2.0 * (dirty.count() + undone.count()) + look;
    }

    // candidate bookkeeping: dedupe by state, keep the best key
    struct Cands {
        unordered_map<StateKey, size_t, StateKeyHash> index;
        vector<pair<double, Plan>> items;
        void offer(double k, Plan&& p) {
            StateKey sk = p.key();
            auto it = index.find(sk);
            if (it == index.end()) {
                index[sk] = items.size();
                items.push_back({k, std::move(p)});
            } else if (items[it->second].first > k) {
                items[it->second] = {k, std::move(p)};
            }
        }
        vector<Plan> top(int n) {
            vector<size_t> ord(items.size());
            for (size_t i = 0; i < ord.size(); i++) ord[i] = i;
            stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return items[a].first < items[b].first; });
            vector<Plan> out;
            for (size_t i = 0; i < ord.size() && (int)i < n; i++) out.push_back(std::move(items[ord[i]].second));
            return out;
        }
    };

    void addFillCands(const Plan& plan, int f, int li, Cands& cands,
                      const function<double(const Plan&)>& keyfn) const {
        int a = PAIR[f][0], b = PAIR[f][1];
        vector<Plan> starts;
        starts.push_back(plan);
        if (plan.act[a] || plan.act[b]) {
            vector<vector<int>> groups = {{a}, {b}, {a, b}, {0, 1, 2, 3, 4, 5}};
            for (auto& g : groups) {
                Req req = offReq(plan, g);
                bool anyOff = false;
                for (int t = 0; t < 6; t++) if (req.set[t] && !req.act[t]) anyOff = true;
                if (anyOff) starts.push_back(plan.add(req, pmask, smask));
            }
        }
        for (auto& sp : starts)
            for (auto& req : fillOptions(sp, f, li)) {
                Plan np = sp.add(req, pmask, smask);
                np.filled |= geo.lineMask[f][li];
                double k = keyfn(np);
                cands.offer(k, std::move(np));
            }
    }

    vector<Plan> runFillsOrdered(const State& state, const vector<pair<int, int>>& order) const {
        vector<Plan> beam;
        beam.emplace_back(&geo, state);
        vector<Mask> rem(order.size() + 1);
        for (int i = (int)order.size() - 1; i >= 0; i--) rem[i] = rem[i + 1] | geo.lineMask[order[i].first][order[i].second];
        for (size_t i = 0; i < order.size(); i++) {
            int f = order[i].first, li = order[i].second;
            const pair<int, int>* nxt = i + 1 < order.size() ? &order[i + 1] : nullptr;
            Cands cands;
            for (auto& plan : beam)
                addFillCands(plan, f, li, cands, [&](const Plan& p) { return orderedKey(p, rem[i + 1], nxt); });
            beam = cands.top(P.beam);
        }
        return beam;
    }

    vector<pair<int, Req>> fixReqs(const Plan& plan, int cell, bool cross) const {
        vector<pair<int, Req>> out;
        vector<vector<int>> famSets = cross ? vector<vector<int>>{{0, 1}, {0, 2}, {1, 2}} : vector<vector<int>>{{0, 1, 2}};
        for (auto& fs : famSets) {
            int n = fs.size();
            for (int mask = 0; mask < (1 << n); mask++) {
                Req req;
                for (int i = 0; i < n; i++) {
                    int t = PAIR[fs[i]][(mask >> i) & 1];
                    req.put(t, offCell(t, cell), true);
                }
                vector<int> rest;
                for (int t = 0; t < 6; t++) if (!req.set[t] && plan.act[t]) rest.push_back(t);
                for (int km = 0; km < (1 << rest.size()); km++) {
                    Req r = req;
                    for (size_t i = 0; i < rest.size(); i++) r.put(rest[i], plan.off[rest[i]], (km >> i) & 1);
                    out.push_back({plan.earliest(r), r});
                }
            }
        }
        stable_sort(out.begin(), out.end(), [](const pair<int, Req>& x, const pair<int, Req>& y) { return x.first < y.first; });
        return out;
    }

    double pointKey(const Plan& plan) const {
        if (plan.submitted) return plan.T;
        return plan.T + 1.0 * pmask.andnot(plan.matter).count() + 1.5 * (smask & plan.matter).count();
    }

    pair<double, Plan> extendCross(const Plan& plan, const Req& req, int T) const {
        Plan bestPlan = plan.add(req, pmask, smask);
        double bestK = pointKey(bestPlan);
        Req bestReq = req;
        bool improved = true;
        while (improved) {
            improved = false;
            Req curReq = bestReq;
            for (int t = 0; t < 6; t++) {
                if (curReq.set[t] && curReq.act[t]) continue;
                for (int o = -geo.maxoff; o <= geo.maxoff; o++) {
                    Req r = curReq;
                    r.put(t, o, true);
                    if (plan.earliest(r) > T) continue;
                    Plan np = plan.add(r, pmask, smask);
                    double k = pointKey(np);
                    if (k < bestK) { bestK = k; bestPlan = std::move(np); bestReq = r; improved = true; }
                }
            }
        }
        return {bestK, bestPlan};
    }

    optional<Plan> finish(Plan plan) const {
        int limit = cs.matter.size() + cs.skirt.size() + 10;
        for (int it = 0; it < limit; it++) {
            if (plan.submitted > 0) return plan;
            Mask undone = pmask.andnot(plan.matter);
            Mask dirty = smask & plan.matter;
            vector<tuple<int, int, Req>> cands;
            undone.forEach([&](int cell) {
                auto rs = fixReqs(plan, cell, true);
                for (int i = 0; i < (int)rs.size() && i < 2; i++) cands.push_back({rs[i].first, 0, rs[i].second});
            });
            dirty.forEach([&](int cell) {
                auto rs = fixReqs(plan, cell, false);
                for (int i = 0; i < (int)rs.size() && i < 1; i++) cands.push_back({rs[i].first, 1, rs[i].second});
                for (int f = 0; f < 3; f++) {
                    int li = geo.lineOfCell[f][cell];
                    if (sOn[f][li].count() <= 4)
                        for (auto& r : fillOptions(plan, f, li, 2)) cands.push_back({plan.earliest(r), 1, r});
                }
            });
            if (cands.empty()) return nullopt;
            stable_sort(cands.begin(), cands.end(), [](const tuple<int, int, Req>& x, const tuple<int, int, Req>& y) {
                return get<0>(x) < get<0>(y);
            });
            int Tmin = get<0>(cands[0]);
            bool have = false;
            double bestK = 0;
            Plan bestPlan = plan;
            for (auto& c : cands) {
                int T = get<0>(c);
                if (T > Tmin + 1) break;
                double k;
                Plan np = plan;
                if (get<1>(c) == 0) {
                    auto r = extendCross(plan, get<2>(c), T);
                    k = r.first;
                    np = std::move(r.second);
                } else {
                    np = plan.add(get<2>(c), pmask, smask);
                    k = pointKey(np);
                }
                if (!have || k < bestK) { have = true; bestK = k; bestPlan = std::move(np); }
            }
            plan = std::move(bestPlan);
        }
        return nullopt;
    }

    vector<Plan> runFills(const State& state) const {
        vector<Plan> beam;
        beam.emplace_back(&geo, state);
        vector<Plan> results;
        for (int step = 0; step < 60; step++) {
            Cands cands;
            for (auto& plan : beam) {
                Mask undone = pmask.andnot(plan.matter);
                Mask dirtyNow = plan.matter & smask;
                if (plan.submitted || (!undone.any() && !dirtyNow.any())) { results.push_back(plan); continue; }
                struct Cand { double score; int kind, f, li; };
                vector<Cand> lines;
                for (int f = 0; f < 3; f++)
                    for (int li : activeLines[f]) {
                        int g = (pOn[f][li] & undone).count();
                        if (g == 0 && !(sOn[f][li] & dirtyNow).any()) continue;
                        int hs = sOn[f][li].count();
                        if (hs <= P.hmax) {
                            int est = estFillTime(plan, f, li);
                            lines.push_back({est - 0.5 * g + 0.5 * max(0, hs - 4), 0, f, li});
                        }
                        if (P.point && g) {
                            vector<int> cells = (pOn[f][li] & undone).bits();
                            int est = estPointTime(plan, f, li, cells);
                            lines.push_back({est - 0.5 * min(4, g) + 0.5, 1, f, li});
                        }
                    }
                if (lines.empty()) { results.push_back(plan); continue; }
                if (dirtyNow.any()) {
                    vector<pair<int, Req>> cl;
                    dirtyNow.forEach([&](int cell) {
                        auto rs = fixReqs(plan, cell, false);
                        for (int i = 0; i < (int)rs.size() && i < 2; i++) cl.push_back(rs[i]);
                    });
                    stable_sort(cl.begin(), cl.end(), [](const pair<int, Req>& x, const pair<int, Req>& y) { return x.first < y.first; });
                    for (int i = 0; i < (int)cl.size() && i < 4; i++) {
                        Plan np = plan.add(cl[i].second, pmask, smask);
                        double k = key(np);
                        cands.offer(k, std::move(np));
                    }
                }
                stable_sort(lines.begin(), lines.end(), [&](const Cand& x, const Cand& y) {
                    if (x.score != y.score) return x.score < y.score;
                    if (x.kind != y.kind) return x.kind < y.kind;
                    if (x.f != y.f) return x.f < y.f;
                    return geo.lineIds[x.f][x.li] < geo.lineIds[y.f][y.li];
                });
                for (int i = 0; i < (int)lines.size() && i < P.nlines; i++) {
                    auto& c = lines[i];
                    if (c.kind == 1) {
                        for (auto& req : pointOptions(plan, c.f, c.li)) {
                            Plan np = plan.add(req, pmask, smask);
                            double k = key(np);
                            cands.offer(k, std::move(np));
                        }
                        continue;
                    }
                    addFillCands(plan, c.f, c.li, cands, [&](const Plan& p) { return key(p); });
                }
            }
            if (cands.items.empty()) break;
            beam = cands.top(P.beam);
        }
        for (auto& p : beam) results.push_back(p);
        stable_sort(results.begin(), results.end(), [&](const Plan& a, const Plan& b) { return key(a) < key(b); });
        return results;
    }

    optional<Plan> runPass(const State& state) {
        optional<Plan> best;
        auto consider = [&](const vector<Plan>& pool, int n) {
            for (int i = 0; i < (int)pool.size() && i < n; i++) {
                auto done = finish(pool[i]);
                if (done && (!best || done->T < best->T)) best = done;
            }
        };
        consider(runFills(state), P.final_);
        if (P.ordered)
            for (auto& order : orderedVariants()) consider(runFillsOrdered(state, order), P.finalOrdered);
        return best;
    }

    optional<vector<array<char, 6>>> solve() {
        State state;
        vector<array<char, 6>> prog;
        unordered_map<StateKey, optional<Plan>, StateKeyHash> cache;
        int submitted = 0;
        while (submitted < K) {
            StateKey sk;
            memcpy(sk.off, state.off, sizeof sk.off);
            memcpy(sk.act, state.act, sizeof sk.act);
            sk.matter = state.matter;
            auto it = cache.find(sk);
            if (it == cache.end()) it = cache.emplace(sk, runPass(state)).first;
            if (!it->second) return nullopt;
            const Plan& best = *it->second;
            auto ls = best.lines();
            prog.insert(prog.end(), ls.begin(), ls.end());
            submitted += best.submitted;
            state = best.state();
        }
        return prog;
    }
};

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    string input, outdir = "output_files";
    int jobs = thread::hardware_concurrency();
    int only = 0;
    Params prm;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto val = [&]() { return string(argv[++i]); };
        if (a == "-o" || a == "--outdir") outdir = val();
        else if (a == "-j" || a == "--jobs") jobs = stoi(val());
        else if (a == "--only") only = stoi(val());
        else if (a == "--beam") prm.beam = prm.smallBeam = stoi(val());
        else if (a == "--nlines") prm.nlines = prm.smallNlines = stoi(val());
        else if (a == "--final") prm.final_ = prm.smallFinal = stoi(val());
        else if (a == "--scale") prm.scale = stod(val());
        else if (a == "--no-ordered") prm.ordered = false;
        else if (a == "--hmax") prm.hmax = stoi(val());
        else if (a == "--final-ordered") prm.finalOrdered = stoi(val());
        else input = a;
    }
    if (input.empty()) { fprintf(stderr, "usage: solver_cpp <input> [-o dir] [-j n] [--only k]\n"); return 1; }
    auto t0 = chrono::steady_clock::now();
    vector<Case> cases = parseInput(input);
    if (only) {
        vector<Case> f;
        for (auto& c : cases) if (c.index == only) f.push_back(c);
        cases = f;
    }
    map<int, unique_ptr<Geo>> geos;
    for (auto& c : cases) if (!geos.count(c.R)) geos[c.R] = make_unique<Geo>(c.R);

    vector<optional<vector<array<char, 6>>>> results(cases.size());
    atomic<size_t> nextIdx{0};
    auto worker = [&]() {
        while (true) {
            size_t i = nextIdx++;
            if (i >= cases.size()) break;
            const Case& c = cases[i];
            Planner pl(c, *geos[c.R], prm);
            auto prog = pl.solve();
            if (prog && simulateProgram(*geos[c.R], pl.pmask, pl.smask, *prog) < c.count) prog = nullopt;
            results[i] = prog;
        }
    };
    vector<thread> ths;
    for (int j = 0; j < max(1, jobs); j++) ths.emplace_back(worker);
    for (auto& t : ths) t.join();

    long total = 0;
    vector<int> failed;
    string base = input.substr(input.find_last_of('/') + 1);
    string outName = base.rfind("in_", 0) == 0 ? "out_" + base.substr(3) : "out_" + base;
    string cmd = "mkdir -p '" + outdir + "'";
    if (system(cmd.c_str()) != 0) return 1;
    string outPath = outdir + "/" + outName;
    ofstream out(outPath);
    for (size_t i = 0; i < cases.size(); i++) {
        if (!results[i]) { failed.push_back(cases[i].index); }
        else {
            total += results[i]->size();
            for (auto& step : *results[i]) {
                for (int t = 0; t < 6; t++) out << (t ? " " : "") << step[t];
                out << "\n";
            }
        }
        if (i + 1 < cases.size()) out << "\n";
    }
    out.close();
    double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    printf("%s: %zu cases, score %ld, failed [", base.c_str(), cases.size(), total);
    for (size_t i = 0; i < failed.size(); i++) printf("%s%d", i ? ", " : "", failed[i]);
    printf("], %.1fs -> %s\n", secs, outPath.c_str());
    return 0;
}
