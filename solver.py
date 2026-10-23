#!/usr/bin/env python3
"""Solver for level 6 (Nanofabricator): generates short programs that produce
each pattern the required number of times.

Approach
--------
Every emitter rides the border rail and, while on its "effective" arc, fires
along one of the 2R-1 parallel lines of its direction family (rows for
emitters 1/4, the two diagonal families for 2/5 and 3/6).  Two opposite
emitters on the same line give every cell on it 2 hits, i.e. the whole line
is filled in one tick; each of the other four emitters crossing that line adds
a third hit on one cell, which keeps that cell empty.  So a line with at most
4 skirt cells (empty cells touching the pattern) can be produced in one tick.

A plan is a list of events (FILL a line with blockers on its skirt cells,
CROSS to create a single cell with two beams, CLEAN to destroy a cell with
three beams).  Events are scheduled as early as possible given that every
emitter moves one rail step per tick, compiled into instruction lines, and
verified with the exact simulator.  Several variants (primary family, sweep
direction, pass order for repetitions) are tried and the shortest valid
program is kept.

Usage:
    python3 solver.py input_files/in_level-6_1-small.txt [-o output_files] [-j 8]
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate import parse_input, Board, DIR_TO, FIRE_OV  # noqa: E402

# family of emitter t: 0 = rows (E/W), 1 = SE/NW diagonals, 2 = SW/NE diagonals
FAMILY = [0, 1, 2, 0, 1, 2]
PAIR = {0: (0, 3), 1: (1, 4), 2: (2, 5)}
LINE_KEY = [lambda c, r: r, lambda c, r: c - r, lambda c, r: c + r]


def popcount(x):
    return bin(x).count("1")


def bits(x):
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b


class Geo:
    """Line masks for one board radius."""

    _cache = {}

    @classmethod
    def get(cls, R):
        if R not in cls._cache:
            cls._cache[R] = cls(R)
        return cls._cache[R]

    def __init__(self, R):
        self.R = R
        board = Board(R)
        self.board = board
        self.cells = sorted(board.interior)
        self.idx = {cell: i for i, cell in enumerate(self.cells)}
        self.N = len(self.cells)
        self.perim = len(board.perimeter)
        self.start = list(board.start_indices)
        self.maxoff = R - 1

        # ray mask of emitter t at rail index p
        self.ray = [[0] * self.perim for _ in range(6)]
        for p in range(self.perim):
            for t in range(6):
                m = 0
                for cell in board.ray_cells[p][t]:
                    m |= 1 << self.idx[cell]
                self.ray[t][p] = m

        # line ids per family and cell
        self.line_of_cell = [[LINE_KEY[f](*cell) for cell in self.cells] for f in range(3)]
        self.line_mask = [{} for _ in range(3)]
        for f in range(3):
            for i, cell in enumerate(self.cells):
                self.line_mask[f].setdefault(self.line_of_cell[f][i], 0)
                self.line_mask[f][self.line_of_cell[f][i]] |= 1 << i

        # offset of emitter t that puts it on line id L of its family
        self.off_for_line = [{} for _ in range(6)]
        for t in range(6):
            f = FAMILY[t]
            for o in range(-self.maxoff, self.maxoff + 1):
                m = self.ray[t][self.pidx(t, o)]
                for L, lm in self.line_mask[f].items():
                    if lm == m:
                        self.off_for_line[t][L] = o
            assert len(self.off_for_line[t]) == 2 * R - 1, (t, len(self.off_for_line[t]))

        # neighbours per cell index
        self.nbr = []
        for cell in self.cells:
            m = 0
            for dc, dr in DIR_TO.values():
                n = (cell[0] + dc, cell[1] + dr)
                if n in self.idx:
                    m |= 1 << self.idx[n]
            self.nbr.append(m)

    def pidx(self, t, o):
        return (self.start[t] + o) % self.perim

    def mask_at(self, t, o):
        return self.ray[t][self.pidx(t, o)]


class State:
    __slots__ = ("off", "act", "matter", "submitted")

    def __init__(self, off=None, act=None, matter=0, submitted=0):
        self.off = list(off) if off else [0] * 6
        self.act = list(act) if act else [False] * 6
        self.matter = matter
        self.submitted = submitted

    def copy(self):
        return State(self.off, self.act, self.matter, self.submitted)


def simulate(geo, pmask, smask, lines, state):
    """Runs instruction lines from state (modified in place). Returns state."""
    maxoff = geo.maxoff
    off, act = state.off, state.act
    matter = state.matter
    for step in lines:
        for t in range(6):
            ins = step[t]
            if ins == "F":
                off[t] += 1
            elif ins == "B":
                off[t] -= 1
            elif ins == "T":
                act[t] = not act[t]
        c1 = c2 = c3 = 0
        for t in range(6):
            if act[t] and -maxoff <= off[t] <= maxoff:
                m = geo.ray[t][geo.pidx(t, off[t])]
                c3 |= c2 & m
                c2 |= c1 & m
                c1 |= m
        matter = (matter | (c2 & ~c3)) & ~c3
        if (matter & pmask) == pmask and not (matter & smask):
            matter &= ~pmask
            state.submitted += 1
    state.matter = matter
    return state


# ---------------------------------------------------------------------------
# scheduling

class Schedule:
    """Per-emitter constraint tracks, compiled to instruction lines."""

    def __init__(self, state, geo):
        self.geo = geo
        # last constraint per emitter: (tick, offset, active)
        self.last = [(0, state.off[t], state.act[t]) for t in range(6)]
        self.cons = [[] for _ in range(6)]   # list of (tick, offset, active)
        self.T = 0                           # tick of last event

    def need(self, t, off, act):
        lt, lo, la = self.last[t]
        return abs(off - lo) + (1 if act != la else 0)

    def earliest(self, req):
        T = self.T + 1
        for t, (o, a) in req.items():
            lt, lo, la = self.last[t]
            T = max(T, lt + self.need(t, o, a))
        return T

    def add(self, req, T=None):
        if T is None:
            T = self.earliest(req)
        assert T > self.T
        for t, (o, a) in req.items():
            self.cons[t].append((T, o, a))
            self.last[t] = (T, o, a)
        self.T = T
        return T

    def compile(self, state):
        n = self.T
        prog = [["-"] * 6 for _ in range(n)]
        for t in range(6):
            tick, o, a = 0, state.off[t], state.act[t]
            for (T1, o1, a1) in self.cons[t]:
                moves = o1 - o
                step = "F" if moves > 0 else "B"
                nm = abs(moves)
                # instruction at tick k lives in prog[k - 1]
                if a1 == a:
                    for i in range(nm):
                        prog[tick + i][t] = step
                elif a1:  # turn on: move first, toggle at T1
                    for i in range(nm):
                        prog[tick + i][t] = step
                    prog[T1 - 1][t] = "T"
                else:     # turn off: toggle at tick+1, then move
                    prog[tick][t] = "T"
                    for i in range(nm):
                        prog[tick + 1 + i][t] = step
                tick, o, a = T1, o1, a1
        return prog


# ---------------------------------------------------------------------------
# planning

class Planner:
    def __init__(self, case, geo):
        self.case = case
        self.geo = geo
        self.K = case.count
        self.pmask = sum(1 << geo.idx[c] for c in case.matter)
        self.smask = sum(1 << geo.idx[c] for c in case.skirt)
        self.pcells = set(geo.idx[c] for c in case.matter)
        # per family: line id -> (pmask on line, skirt mask on line)
        self.lines = [{} for _ in range(3)]
        for f in range(3):
            for L, lm in geo.line_mask[f].items():
                p = lm & self.pmask
                s = lm & self.smask
                if p or s:
                    self.lines[f][L] = (p, s)

    # -- cover selection ---------------------------------------------------
    def choose_fills(self, primary):
        """Greedy set cover of pattern cells with fillable lines (<=4 holes).
        Lines of the primary family are preferred; returns list of (f, L)."""
        geo = self.geo
        uncovered = self.pmask
        fills = []
        fam_order = [primary] + [f for f in range(3) if f != primary]
        # first take every fillable line of the primary family that has pattern cells
        for f in fam_order:
            while True:
                best = None
                for L, (p, s) in self.lines[f].items():
                    if not p or popcount(s) > 4 or (f, L) in fills:
                        continue
                    gain = popcount(p & uncovered)
                    if gain == 0:
                        continue
                    key = (gain, -popcount(s))
                    if best is None or key > best[0]:
                        best = (key, L)
                if best is None:
                    break
                fills.append((f, best[1]))
                uncovered &= ~self.lines[f][best[1]][0]
        return fills, uncovered

    # -- event construction -------------------------------------------------
    def fill_req(self, sched, f, L, others_off):
        """Requirement dict for filling line L of family f, picking blockers."""
        geo = self.geo
        a, b = PAIR[f]
        oa = geo.off_for_line[a][L]
        ob = geo.off_for_line[b][L]
        req = {a: (oa, True), b: (ob, True)}
        p, s = self.lines[f].get(L, (0, 0))
        holes = list(bits(s))
        blockers = [t for t in range(6) if FAMILY[t] != f]
        best = None

        def rec(i, assign, used):
            nonlocal best
            if i == len(holes):
                r = dict(req)
                for h, t in assign.items():
                    r[t] = (geo.off_for_line[t][geo.line_of_cell[FAMILY[t]][h]], True)
                # junk: crossings between blockers of different families
                junk = 0
                ts = list(assign.values())
                for x in range(len(ts)):
                    for y in range(x + 1, len(ts)):
                        tx, ty = ts[x], ts[y]
                        if FAMILY[tx] == FAMILY[ty]:
                            continue
                        c = geo.mask_at(tx, r[tx][0]) & geo.mask_at(ty, r[ty][0])
                        junk |= c
                junk &= ~geo.line_mask[f][L]
                bad = popcount(junk & self.smask)
                T = sched.earliest(r)
                key = (T, bad)
                if best is None or key < best[0]:
                    best = (key, r)
                return
            h = holes[i]
            for t in blockers:
                if t in used:
                    continue
                assign[h] = t
                used.add(t)
                rec(i + 1, assign, used)
                used.discard(t)
                del assign[h]

        rec(0, {}, set())
        r = best[1]
        if others_off:
            for t in range(6):
                if t not in r:
                    r[t] = (sched.last[t][1], False)
        return r

    def cross_req(self, sched, cell):
        """Create one cell with two beams; everything else off."""
        geo = self.geo
        best = None
        for f1 in range(3):
            for f2 in range(f1 + 1, 3):
                for t1 in PAIR[f1]:
                    for t2 in PAIR[f2]:
                        r = {t1: (geo.off_for_line[t1][geo.line_of_cell[f1][cell]], True),
                             t2: (geo.off_for_line[t2][geo.line_of_cell[f2][cell]], True)}
                        for t in range(6):
                            if t not in r:
                                r[t] = (sched.last[t][1], False)
                        T = sched.earliest(r)
                        if best is None or T < best[0]:
                            best = (T, r)
        return best[1]

    def clean_req(self, sched, cell):
        """Destroy one cell with three beams; everything else off."""
        geo = self.geo
        best = None
        for t0 in PAIR[0]:
            for t1 in PAIR[1]:
                for t2 in PAIR[2]:
                    r = {}
                    for t in (t0, t1, t2):
                        r[t] = (geo.off_for_line[t][geo.line_of_cell[FAMILY[t]][cell]], True)
                    for t in range(6):
                        if t not in r:
                            r[t] = (sched.last[t][1], False)
                    T = sched.earliest(r)
                    if best is None or T < best[0]:
                        best = (T, r)
        return best[1]

    # -- one pass -----------------------------------------------------------
    def build_pass(self, state, fills, extra_cells, others_off):
        """Schedules fills, then crossings for extra cells, then cleanups until
        the simulation submits the pattern. Returns (lines, new_state) or None."""
        geo = self.geo
        sched = Schedule(state, geo)
        for f, L in fills:
            sched.add(self.fill_req(sched, f, L, others_off))
        for cell in extra_cells:
            sched.add(self.cross_req(sched, cell))
        for _ in range(60):
            lines = sched.compile(state)
            st = simulate(geo, self.pmask, self.smask, lines, state.copy())
            if st.submitted > state.submitted:
                return lines, st
            undone = self.pmask & ~st.matter
            dirty = self.smask & st.matter
            if not undone and not dirty:
                # pattern complete but never submitted mid-way? cannot happen
                return None
            fixed = False
            if dirty:
                cell = next(bits(dirty))
                sched.add(self.clean_req(sched, cell))
                fixed = True
            elif undone:
                cell = next(bits(undone))
                sched.add(self.cross_req(sched, cell))
                fixed = True
            if not fixed:
                return None
        return None

    def solve(self):
        geo = self.geo
        best = None
        for primary in range(3):
            fills, uncovered = self.choose_fills(primary)
            # order: sweep each family's lines by line id; direction variants
            extra = list(bits(uncovered))
            # order extra cells along the primary family lines
            extra.sort(key=lambda i: (geo.line_of_cell[primary][i], i))
            for direction in (1, -1):
                order = sorted(fills, key=lambda fl: ((fl[0] - primary) % 3, direction * fl[1]))
                for others_off in (False, True):
                    prog = self.solve_variant(order, extra, others_off)
                    if prog is not None and (best is None or len(prog) < len(best)):
                        best = prog
        return best

    def solve_variant(self, order, extra, others_off):
        geo = self.geo
        state = State()
        prog = []
        res = self.build_pass(state, order, extra, others_off)
        if res is None:
            return None
        lines, state = res
        prog.extend(lines)
        cur_order = order
        for k in range(1, self.K):
            cands = []
            for o in (cur_order, list(reversed(cur_order))):
                r = self.build_pass(state, o, extra, others_off)
                if r is not None:
                    cands.append((len(r[0]), o, r))
            if not cands:
                return None
            cands.sort(key=lambda x: x[0])
            _, cur_order, (lines, state) = cands[0]
            prog.extend(lines)
        return prog


def solve_case(case):
    geo = Geo.get(case.board_radius)
    planner = Planner(case, geo)
    prog = planner.solve()
    if prog is None:
        return case.index, None
    # final verification
    st = simulate(geo, planner.pmask, planner.smask, prog, State())
    if st.submitted < case.count:
        return case.index, None
    return case.index, prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--outdir", default="output_files")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--only", type=int, help="solve only this case index (1-based)")
    args = ap.parse_args()

    cases = parse_input(args.input)
    if args.only:
        cases = [c for c in cases if c.index == args.only]
    t0 = time.time()
    if args.jobs > 1 and len(cases) > 1:
        with Pool(args.jobs) as pool:
            results = pool.map(solve_case, cases, chunksize=1)
    else:
        results = [solve_case(c) for c in cases]

    results.sort()
    total = 0
    failed = []
    out_lines = []
    for (idx, prog), case in zip(results, cases):
        if prog is None:
            failed.append(idx)
            prog = []
        total += len(prog)
        for step in prog:
            out_lines.append(" ".join(step))
        out_lines.append("")
    name = os.path.basename(args.input)
    out_name = "out_" + name[3:] if name.startswith("in_") else "out_" + name
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, out_name)
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines).rstrip("\n") + "\n")
    print(f"{name}: {len(cases)} cases, score {total}, failed {failed}, "
          f"{time.time() - t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
