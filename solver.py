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
from itertools import combinations, permutations, product
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
# planning

class Plan:
    """A partial program for one pass with the exactly simulated state at tick T.

    Every event constrains all currently active emitters (kept, moved or
    switched off), so the beams active at every tick are known when the event
    is added and the matter can be simulated incrementally.  Inactive emitters
    are unconstrained and move towards their next constraint as early as
    possible, which is harmless because they emit nothing."""

    __slots__ = ("geo", "T", "off", "act", "last", "matter", "submitted", "tracks", "filled", "events")

    def __init__(self, geo, state):
        self.geo = geo
        self.T = 0
        self.off = list(state.off)
        self.act = list(state.act)
        self.last = [0] * 6
        self.matter = state.matter
        self.submitted = 0
        self.tracks = [[] for _ in range(6)]
        self.filled = 0
        self.events = None   # persistent linked list of (T, label)

    def clone(self):
        p = Plan.__new__(Plan)
        p.geo = self.geo
        p.T = self.T
        p.off = list(self.off)
        p.act = list(self.act)
        p.last = list(self.last)
        p.matter = self.matter
        p.submitted = self.submitted
        p.tracks = list(self.tracks)
        p.filled = self.filled
        p.events = self.events
        return p

    def need(self, t, o, a):
        return abs(o - self.off[t]) + (1 if a != self.act[t] else 0)

    def earliest(self, req):
        T = self.T + 1
        for t, (o, a) in req.items():
            T = max(T, self.last[t] + self.need(t, o, a))
        return T

    def add(self, req, pmask, smask, label=None):
        geo = self.geo
        T0 = self.T
        T1 = self.earliest(req)
        n = T1 - T0
        p = self.clone()
        tick_masks = [[] for _ in range(n)]
        for t, (o1, a1) in req.items():
            o0, a0, lt = self.off[t], self.act[t], self.last[t]
            tr = list(self.tracks[t])
            tr.extend(["-"] * (T1 - len(tr)))
            d = o1 - o0
            step = "F" if d > 0 else "B"
            sgn = 1 if d > 0 else -1
            nd = abs(d)
            if a0 and a1:
                for i in range(nd):
                    tr[T0 + i] = step
                for k in range(n):
                    pos = o0 + sgn * min(k + 1, nd)
                    tick_masks[k].append(geo.ray[t][geo.pidx(t, pos)])
            elif a1:
                for i in range(nd):
                    tr[lt + i] = step
                tr[T1 - 1] = "T"
                tick_masks[n - 1].append(geo.ray[t][geo.pidx(t, o1)])
            elif a0:
                tr[T0] = "T"
                for i in range(nd):
                    tr[T0 + 1 + i] = step
            else:
                for i in range(nd):
                    tr[lt + i] = step
            p.tracks[t] = tr
            p.off[t] = o1
            p.act[t] = a1
            p.last[t] = T1
        matter = self.matter
        for k in range(n):
            c1 = c2 = c3 = 0
            for m in tick_masks[k]:
                c3 |= c2 & m
                c2 |= c1 & m
                c1 |= m
            matter = (matter | (c2 & ~c3)) & ~c3
            if (matter & pmask) == pmask and not (matter & smask):
                matter &= ~pmask
                p.submitted += 1
        p.matter = matter
        p.T = T1
        p.events = (self.events, (T1, label))
        return p

    def event_list(self):
        out = []
        e = self.events
        while e is not None:
            out.append(e[1])
            e = e[0]
        return out[::-1]

    def lines(self):
        out = []
        for i in range(self.T):
            out.append([tr[i] if i < len(tr) else "-" for tr in self.tracks])
        return out

    def state(self):
        return State(self.off, self.act, self.matter, 0)


class Planner:
    BEAM = 8
    FINAL = 3
    FINAL_ORDERED = 2
    NLINES = 4
    HMAX = 5
    ORDERED = True
    POINT = True

    def __init__(self, case, geo):
        self.case = case
        self.geo = geo
        self.K = case.count
        self.pmask = sum(1 << geo.idx[c] for c in case.matter)
        self.smask = sum(1 << geo.idx[c] for c in case.skirt)
        self.lines = [{} for _ in range(3)]
        for f in range(3):
            for L, lm in geo.line_mask[f].items():
                p = lm & self.pmask
                s = lm & self.smask
                if p or s:
                    self.lines[f][L] = (p, s)

    # -- cover selection ---------------------------------------------------
    def choose_fills(self, primary):
        uncovered = self.pmask
        fills = []
        for f in [primary] + [g for g in range(3) if g != primary]:
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

    # -- event options -----------------------------------------------------
    def off_reqs(self, plan, group):
        """Requirements switching `group` off and keeping other active ones."""
        req = {}
        for t in range(6):
            if plan.act[t]:
                req[t] = (plan.off[t], t not in group)
        return req

    def fill_options(self, plan, f, L, limit=16):
        """Requirement dicts for filling line L of family f: the pair on the
        line, blockers on (up to 4 of) its skirt cells, and every other active
        emitter either switched off or parked where it does not hit a pattern
        cell of the line.  Returns the `limit` earliest options."""
        geo = self.geo
        a, b = PAIR[f]
        base = {a: (geo.off_for_line[a][L], True), b: (geo.off_for_line[b][L], True)}
        p_on, s_on = self.lines[f].get(L, (0, 0))
        holes = list(bits(s_on))
        blockers = [t for t in range(6) if FAMILY[t] != f]
        Lmask = geo.line_mask[f][L]
        opts = []
        for hs in combinations(holes, min(len(holes), 4)):
            for perm in permutations(blockers, len(hs)):
                req = dict(base)
                for h, t in zip(hs, perm):
                    req[t] = (geo.off_for_line[t][geo.line_of_cell[FAMILY[t]][h]], True)
                choice_lists = []
                for t in blockers:
                    if t in req or not plan.act[t]:
                        continue
                    cur = plan.off[t]
                    cs = [(cur, False)]
                    for o in (cur, cur - 1, cur + 1, cur - 2, cur + 2):
                        if -geo.maxoff <= o <= geo.maxoff and not (geo.ray[t][geo.pidx(t, o)] & Lmask & self.pmask):
                            cs.append((o, True))
                            if o != cur:
                                break
                    choice_lists.append((t, cs))
                for combo in product(*[cs for _, cs in choice_lists]):
                    r = dict(req)
                    for (t, _), c in zip(choice_lists, combo):
                        r[t] = c
                    opts.append((plan.earliest(r), r))
        opts.sort(key=lambda x: x[0])
        return [r for _, r in opts[:limit]]

    def point_options(self, plan, f, L, limit=8):
        """One emitter of family f on line L and up to four secondaries crossing
        it on undone pattern cells (creating them directly). Other active
        emitters are switched off, or kept when they hit no skirt cell of L."""
        geo = self.geo
        a, b = PAIR[f]
        Lmask = geo.line_mask[f][L]
        cells = sorted(bits(self.lines[f][L][0] & ~plan.matter),
                       key=lambda i: geo.line_of_cell[(f + 1) % 3][i])
        secondaries = [t for t in range(6) if FAMILY[t] != f]
        opts = []
        for A, Ap in ((a, b), (b, a)):
            base = {A: (geo.off_for_line[A][L], True)}
            if plan.act[Ap]:
                base[Ap] = (plan.off[Ap], False)
            k = min(4, len(cells))
            for lo in range(len(cells) - k + 1):
                subset = cells[lo:lo + k]
                for perm in permutations(secondaries, k):
                    req = dict(base)
                    for cell, t in zip(subset, perm):
                        req[t] = (geo.off_for_line[t][geo.line_of_cell[FAMILY[t]][cell]], True)
                    rest = [t for t in secondaries if t not in req and plan.act[t]]
                    choice_lists = []
                    for t in rest:
                        cur = plan.off[t]
                        cs = [(cur, False)]
                        if not (geo.ray[t][geo.pidx(t, cur)] & Lmask & self.smask):
                            cs.append((cur, True))
                        choice_lists.append((t, cs))
                    for combo in product(*[cs for _, cs in choice_lists]):
                        r = dict(req)
                        for (t, _), c in zip(choice_lists, combo):
                            r[t] = c
                        opts.append((plan.earliest(r), r))
        opts.sort(key=lambda x: x[0])
        return [r for _, r in opts[:limit]]

    def est_point_time(self, plan, f, L, cells):
        geo = self.geo
        a, b = PAIR[f]
        est = plan.T + 1
        est = max(est, min(plan.last[t] + plan.need(t, geo.off_for_line[t][L], True) for t in (a, b)))
        secondaries = [t for t in range(6) if FAMILY[t] != f]
        ds = sorted(min(plan.last[t] + plan.need(t, geo.off_for_line[t][geo.line_of_cell[FAMILY[t]][c]], True)
                        for t in secondaries) for c in cells)
        return max(est, ds[0]) if ds else est

    def est_fill_time(self, plan, f, L):
        """Optimistic earliest tick at which line L of family f could be filled."""
        geo = self.geo
        est = plan.T + 1
        for t in PAIR[f]:
            est = max(est, plan.last[t] + plan.need(t, geo.off_for_line[t][L], True))
        p, s = self.lines[f][L]
        blockers = [t for t in range(6) if FAMILY[t] != f]
        for h in bits(s):
            best = min(plan.last[t] + plan.need(t, geo.off_for_line[t][geo.line_of_cell[FAMILY[t]][h]], True)
                       for t in blockers)
            est = max(est, best)
        return est

    def remaining_estimate(self, undone):
        """Greedy number of fills still needed to cover the undone cells."""
        n = 0
        while undone:
            best = 0
            for f in range(3):
                for L, (p, s) in self.lines[f].items():
                    if popcount(s) > self.HMAX:
                        continue
                    g = popcount(p & undone)
                    if g > best:
                        best, bl = g, p
            if best == 0:
                return n + popcount(undone) / 3.0
            undone &= ~bl
            n += 1
        return n

    def key(self, plan):
        if plan.submitted:
            return plan.T
        undone = self.pmask & ~plan.matter
        dirty = plan.matter & self.smask
        return plan.T + self.remaining_estimate(undone) + 1.5 * popcount(dirty) \
            + 1.5 * popcount(plan.filled & undone)

    def ordered_key(self, plan, remaining_mask, nxt):
        if plan.submitted:
            return plan.T
        dirty = plan.matter & self.smask & ~remaining_mask
        undone = plan.filled & self.pmask & ~plan.matter & ~remaining_mask
        look = 0 if nxt is None else self.est_fill_time(plan, *nxt) - (plan.T + 1)
        return plan.T + 2.0 * (popcount(dirty) + popcount(undone)) + look

    def run_fills_ordered(self, state, order):
        """Beam search over blocker choices for a fixed sequence of fills."""
        geo = self.geo
        beam = [Plan(geo, state)]
        rem = [0] * (len(order) + 1)
        for i in range(len(order) - 1, -1, -1):
            f, L = order[i]
            rem[i] = rem[i + 1] | geo.line_mask[f][L]
        for i, (f, L) in enumerate(order):
            nxt = order[i + 1] if i + 1 < len(order) else None
            a, b = PAIR[f]
            cands = {}
            for plan in beam:
                starts = [plan]
                if plan.act[a] or plan.act[b]:
                    for group in ((a,), (b,), (a, b), tuple(range(6))):
                        req = self.off_reqs(plan, group)
                        if any(not v[1] for v in req.values()):
                            starts.append(plan.add(req, self.pmask, self.smask, ("off", group)))
                for sp in starts:
                    for req in self.fill_options(sp, f, L):
                        np_ = sp.add(req, self.pmask, self.smask, ("fill", f, L))
                        np_.filled |= geo.line_mask[f][L]
                        k = self.ordered_key(np_, rem[i + 1], nxt)
                        sk = (tuple(np_.off), tuple(np_.act), np_.matter)
                        if sk not in cands or cands[sk][0] > k:
                            cands[sk] = (k, np_)
            beam = [v[1] for v in sorted(cands.values(), key=lambda x: x[0])[: self.BEAM]]
        return beam

    def ordered_variants(self):
        if not hasattr(self, "_variants"):
            vs = []
            for primary in range(3):
                fills, _ = self.choose_fills(primary)
                for direction in (1, -1):
                    vs.append(sorted(fills, key=lambda fl: ((fl[0] - primary) % 3, direction * fl[1])))
            self._variants = vs
        return self._variants

    def run_fills(self, state):
        """Beam search over sequences of fills; returns finished plans."""
        geo = self.geo
        beam = [Plan(geo, state)]
        results = []
        for _ in range(60):
            cands = {}
            for plan in beam:
                undone = self.pmask & ~plan.matter
                if plan.submitted or (not undone and not (plan.matter & self.smask)):
                    results.append(plan)
                    continue
                lines = []
                dirty_now = plan.matter & self.smask
                for f in range(3):
                    for L, (p, s) in self.lines[f].items():
                        g = popcount(p & undone)
                        if g == 0 and not (s & dirty_now):
                            continue
                        if popcount(s) <= self.HMAX:
                            est = self.est_fill_time(plan, f, L)
                            lines.append((est - 0.5 * g + 0.5 * max(0, popcount(s) - 4), "fill", f, L))
                        if self.POINT and g:
                            cells = list(bits(p & undone))
                            est = self.est_point_time(plan, f, L, cells)
                            lines.append((est - 0.5 * min(4, g) + 0.5, "point", f, L))
                if not lines:
                    results.append(plan)
                    continue
                dirty = plan.matter & self.smask
                if dirty:
                    cl = []
                    for cell in bits(dirty):
                        for T, r in self.fix_reqs(plan, cell, "clean")[:2]:
                            cl.append((T, cell, r))
                    cl.sort(key=lambda x: x[0])
                    for T, cell, r in cl[:4]:
                        np_ = plan.add(r, self.pmask, self.smask, ("clean", cell))
                        k = self.key(np_)
                        sk = (tuple(np_.off), tuple(np_.act), np_.matter)
                        if sk not in cands or cands[sk][0] > k:
                            cands[sk] = (k, np_)
                lines.sort()
                for _, kind, f, L in lines[: self.NLINES]:
                    a, b = PAIR[f]
                    if kind == "point":
                        for req in self.point_options(plan, f, L):
                            np_ = plan.add(req, self.pmask, self.smask, ("point", f, L))
                            k = self.key(np_)
                            sk = (tuple(np_.off), tuple(np_.act), np_.matter)
                            if sk not in cands or cands[sk][0] > k:
                                cands[sk] = (k, np_)
                        continue
                    starts = [plan]
                    if plan.act[a] or plan.act[b]:
                        for group in ((a,), (b,), (a, b), tuple(range(6))):
                            req = self.off_reqs(plan, group)
                            if any(not v[1] for v in req.values()):
                                starts.append(plan.add(req, self.pmask, self.smask, ("off", group)))
                    for sp in starts:
                        for req in self.fill_options(sp, f, L):
                            np_ = sp.add(req, self.pmask, self.smask, ("fill", f, L))
                            np_.filled |= geo.line_mask[f][L]
                            k = self.key(np_)
                            sk = (tuple(np_.off), tuple(np_.act), np_.matter)
                            if sk not in cands or cands[sk][0] > k:
                                cands[sk] = (k, np_)
            if not cands:
                break
            beam = [v[1] for v in sorted(cands.values(), key=lambda x: x[0])[: self.BEAM]]
        results.extend(beam)
        results.sort(key=self.key)
        return results

    def fix_reqs(self, plan, cell, kind):
        """Requirements creating (kind='cross', two beams) or destroying
        ('clean', three beams) one cell, everything else switched off.
        Returns a list of (earliest tick, req) sorted by tick."""
        geo = self.geo
        out = []
        fams = [(0, 1), (0, 2), (1, 2)] if kind == "cross" else [(0, 1, 2)]
        for fs in fams:
            for ts in product(*[PAIR[f] for f in fs]):
                req = {}
                for t in ts:
                    req[t] = (geo.off_for_line[t][geo.line_of_cell[FAMILY[t]][cell]], True)
                rest = [t for t in range(6) if t not in req and plan.act[t]]
                for keep in product((False, True), repeat=len(rest)):
                    r = dict(req)
                    for t, k in zip(rest, keep):
                        r[t] = (plan.off[t], k)
                    out.append((plan.earliest(r), r))
        out.sort(key=lambda x: x[0])
        return out

    def point_key(self, plan):
        if plan.submitted:
            return plan.T
        undone = self.pmask & ~plan.matter
        dirty = self.smask & plan.matter
        return plan.T + 1.0 * popcount(undone) + 1.5 * popcount(dirty)

    def extend_cross(self, plan, req, T):
        """Tries to add beams of unused emitters so that more undone pattern
        cells are created in the same tick. Returns the best (key, plan)."""
        geo = self.geo
        best_plan = plan.add(req, self.pmask, self.smask)
        best = (self.point_key(best_plan), best_plan, req)
        improved = True
        while improved:
            improved = False
            cur_req = best[2]
            for t in range(6):
                if cur_req.get(t, (0, False))[1]:
                    continue
                for o in range(-geo.maxoff, geo.maxoff + 1):
                    r = dict(cur_req)
                    r[t] = (o, True)
                    if plan.earliest(r) > T:
                        continue
                    np_ = plan.add(r, self.pmask, self.smask)
                    k = self.point_key(np_)
                    if k < best[0]:
                        best = (k, np_, r)
                        improved = True
        return best[0], best[1]

    def finish(self, plan, submitted_before):
        """Adds crossings/cleanups until the pattern is submitted."""
        for _ in range(len(self.case.matter) + len(self.case.skirt) + 10):
            if plan.submitted > 0:
                return plan
            undone = self.pmask & ~plan.matter
            dirty = self.smask & plan.matter
            cands = []
            for cell in bits(undone):
                for T, r in self.fix_reqs(plan, cell, "cross")[:2]:
                    cands.append((T, 0, r))
            for cell in bits(dirty):
                for T, r in self.fix_reqs(plan, cell, "clean")[:1]:
                    cands.append((T, 1, r))
                for f in range(3):
                    L = self.geo.line_of_cell[f][cell]
                    if popcount(self.lines[f][L][1]) <= 4:
                        for r in self.fill_options(plan, f, L, limit=2):
                            cands.append((plan.earliest(r), 1, r))
            if not cands:
                return None
            cands.sort(key=lambda x: x[0])
            Tmin = cands[0][0]
            best = None
            for T, kind, r in cands:
                if T > Tmin + 1:
                    break
                if kind == 0:
                    k, np_ = self.extend_cross(plan, r, T)
                else:
                    np_ = plan.add(r, self.pmask, self.smask)
                    k = self.point_key(np_)
                if best is None or k < best[0]:
                    best = (k, np_)
            plan = best[1]
        return None

    def run_pass(self, state):
        best = None
        pools = [self.run_fills(state)[: self.FINAL]]
        if self.ORDERED:
            for order in self.ordered_variants():
                pools.append(self.run_fills_ordered(state, order)[: self.FINAL_ORDERED])
        for pool in pools:
            for plan in pool:
                done = self.finish(plan, 0)
                if done is not None and (best is None or done.T < best.T):
                    best = done
        return best

    def solve(self):
        state = State()
        prog = []
        cache = {}
        submitted = 0
        while submitted < self.K:
            key = (tuple(state.off), tuple(state.act), state.matter)
            if key not in cache:
                cache[key] = self.run_pass(state)
            best = cache[key]
            if best is None:
                return None
            prog.extend(best.lines())
            submitted += best.submitted
            state = best.state()
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
