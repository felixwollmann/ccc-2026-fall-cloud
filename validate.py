#!/usr/bin/env python3
"""Validator and scorer for level 6 (Nanofabricator).

Simulates every program in an output file against the patterns in the matching
input file, checks that each pattern is submitted at least the required number
of times, and reports the score (total number of instruction lines, lower is
better).

The simulation is a port of the official sandbox (sandbox-level-6.html):
  - board radius R from the input -> internal radius R+1, grid of 2R+1 rows
    and 4R+1 columns (same character layout as the pattern)
  - 6 emitters ride the border rail, starting in their corners, inactive
  - each tick: run one instruction line, then create matter where exactly two
    active beams cross and destroy it where three or more cross, then submit
    the pattern if it is present and no other matter touches it

Usage:
    python3 validate.py <input_file> [output_file] [-v]
    python3 validate.py input_files/          # all in_*/out_* pairs in a dir

If output_file is omitted, it is derived from the input name (in_X -> out_X).
"""

import argparse
import os
import sys

VALID_INSTRUCTIONS = {"-", "T", "F", "B"}

# Neighbour offsets in the (col, row) character grid.
DIR_TO = {
    "E": (2, 0),
    "SE": (1, 1),
    "SW": (-1, 1),
    "W": (-2, 0),
    "NW": (-1, -1),
    "NE": (1, -1),
}
# Firing direction of emitter t (0-based).
FIRE_OV = [(2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)]


class ValidationError(Exception):
    pass


class Board:
    """Board geometry for a given input radius (border included)."""

    def __init__(self, board_radius):
        radius = board_radius + 1
        self.radius = radius
        self.rows = 2 * radius - 1
        self.cols = 4 * radius - 3

        # Interior cells (the only cells matter can exist on).
        self.interior = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self._base_cell(c, r) == "O":
                    self.interior.add((c, r))

        # Rail positions, clockwise from the left corner.
        self.perimeter = []
        if radius > 1:
            c, r = 0, radius - 1
            for d in ("NE", "E", "SE", "SW", "W", "NW"):
                dc, dr = DIR_TO[d]
                for _ in range(radius - 1):
                    self.perimeter.append((c, r))
                    c, r = c + dc, r + dr

        # ray_cells[p][t]: interior cells hit by emitter t standing at rail position p.
        self.ray_cells = []
        for pc, pr in self.perimeter:
            per_track = []
            for dc, dr in FIRE_OV:
                cells = []
                c, r = pc + dc, pr + dr
                while self._in_grid(c, r):
                    if (c, r) in self.interior:
                        cells.append((c, r))
                    c, r = c + dc, r + dr
                per_track.append(cells)
            self.ray_cells.append(per_track)

        self.start_indices = [i * max(radius - 1, 1) for i in range(6)]

    def _in_grid(self, c, r):
        radius, cols = self.radius, self.cols
        if r < 0 or r >= 2 * radius - 1:
            return False
        if (r + c + radius + 1) % 2 != 0:
            return False
        if r < radius:
            return c + r >= radius - 1 and c - r <= cols - radius + 1
        return c + (2 * radius - 1 - r) >= radius - 1 and c + r - 2 * radius + 1 < cols - radius + 1

    def _base_cell(self, c, r):
        """'-' gap, ':' outside, '#' border rail, 'O' interior."""
        radius, rows, cols = self.radius, self.rows, self.cols
        if (r + c + radius + 1) % 2 != 0:
            return "-"
        r_rail = r == 0 or r == rows - 1
        if r < radius:
            c_rail = c == radius - 1 - r or c == cols - 1 - radius + 1 + r
            if c + r >= radius - 1 and c - r <= cols - radius + 1:
                return "#" if r_rail or c_rail else "O"
            return ":"
        c_rail = c == radius - 1 + r - 2 * radius + 2 or c == cols - 1 + radius - 1 - r
        if c + 2 * radius - 1 - r >= radius - 1 and c + r - 2 * radius + 1 < cols - radius + 1:
            return "#" if r_rail or c_rail else "O"
        return ":"

    def render(self, matter, indices=None):
        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                base = self._base_cell(c, r)
                if base == "O":
                    row.append("M" if (c, r) in matter else "O")
                elif base == "#":
                    row.append("*")
                else:
                    row.append("-")
            grid.append(row)
        if indices is not None:
            for t, idx in enumerate(indices):
                c, r = self.perimeter[idx]
                grid[r][c] = str(t + 1)
        return "\n".join("".join(row) for row in grid)


class Case:
    def __init__(self, index, board_radius, rows, cols, count, pattern_lines):
        self.index = index
        self.board_radius = board_radius
        self.rows = rows
        self.cols = cols
        self.count = count
        self.pattern_lines = pattern_lines
        self.board = Board(board_radius)

        if rows != self.board.rows or cols != self.board.cols:
            raise ValidationError(
                f"case {index}: pattern is {rows}x{cols}, board of radius {board_radius} "
                f"is {self.board.rows}x{self.board.cols}"
            )
        for i, ln in enumerate(pattern_lines):
            if len(ln) != cols:
                raise ValidationError(f"case {index}: pattern line {i + 1} has {len(ln)} chars, expected {cols}")

        # Pattern matter cells, and the 'O' cells bordering them that must stay empty.
        self.matter = set()
        for r, ln in enumerate(pattern_lines):
            for c, ch in enumerate(ln):
                if ch == "M":
                    self.matter.add((c, r))
        self.skirt = set()
        for r, ln in enumerate(pattern_lines):
            for c, ch in enumerate(ln):
                if ch == "O" and any((c + dc, r + dr) in self.matter for dc, dr in DIR_TO.values()):
                    self.skirt.add((c, r))


def parse_input(path):
    with open(path) as f:
        lines = [ln.rstrip("\r\n") for ln in f]
    pos = 0

    def next_nonblank():
        nonlocal pos
        while pos < len(lines) and not lines[pos].strip():
            pos += 1
        if pos >= len(lines):
            raise ValidationError(f"{path}: unexpected end of input")
        ln = lines[pos]
        pos += 1
        return ln

    n = int(next_nonblank().strip())
    cases = []
    for i in range(n):
        header = next_nonblank().split()
        if len(header) != 4:
            raise ValidationError(f"{path}: case {i + 1}: bad header {header}")
        board_radius, rows, cols, count = map(int, header)
        pattern = [next_nonblank().strip() for _ in range(rows)]
        cases.append(Case(i + 1, board_radius, rows, cols, count, pattern))
    return cases


def parse_output(path):
    """Returns a list of programs; each program is a list of (line_no, [6 tokens])."""
    programs = []
    current = []
    errors = []
    with open(path) as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                if current:
                    programs.append(current)
                    current = []
                continue
            tokens = line.split()
            if len(tokens) != 6:
                errors.append(f"line {line_no}: expected 6 instructions, got {len(tokens)}: {line!r}")
            bad = [t for t in tokens if t not in VALID_INSTRUCTIONS]
            if bad:
                errors.append(f"line {line_no}: unknown instruction(s) {bad}: {line!r}")
            tokens = (tokens + ["-"] * 6)[:6]
            current.append((line_no, [t if t in VALID_INSTRUCTIONS else "-" for t in tokens]))
    if current:
        programs.append(current)
    return programs, errors


def simulate(case, program, trace=False):
    """Runs the program and returns (submitted, final matter, final emitter indices)."""
    board = case.board
    perim_len = len(board.perimeter)
    indices = list(board.start_indices)
    active = [False] * 6
    matter = set()
    submitted = 0

    for tick, (line_no, step) in enumerate(program, 1):
        # 1. execute instruction line
        for t, ins in enumerate(step):
            if ins == "F":
                indices[t] = (indices[t] + 1) % perim_len
            elif ins == "B":
                indices[t] = (indices[t] - 1) % perim_len
            elif ins == "T":
                active[t] = not active[t]

        # 2. create / destroy matter
        hits = {}
        for t in range(6):
            if active[t]:
                for cell in board.ray_cells[indices[t]][t]:
                    hits[cell] = hits.get(cell, 0) + 1
        for cell, n in hits.items():
            if n == 2:
                matter.add(cell)
            elif n >= 3:
                matter.discard(cell)

        # 3. submit pattern
        accepted = case.matter <= matter and not (case.skirt & matter)
        if accepted:
            matter -= case.matter
            submitted += 1

        if trace:
            print(f"--- tick {tick} (line {line_no}): {' '.join(step)}"
                  f"  active={''.join('1' if a else '0' for a in active)}"
                  f"{'  SUBMITTED #' + str(submitted) if accepted else ''}")
            print(board.render(matter, indices))

    return submitted, matter, indices


def validate(input_path, output_path, verbose=False, trace=None):
    cases = parse_input(input_path)
    programs, errors = parse_output(output_path)

    if len(programs) != len(cases):
        errors.append(f"expected {len(cases)} programs, got {len(programs)}")

    score = 0
    active_lines = 0
    for case, program in zip(cases, programs):
        submitted, matter, indices = simulate(case, program, trace=(trace == case.index))
        lines = len(program)
        busy = sum(1 for _, step in program if any(t != "-" for t in step))
        score += lines
        active_lines += busy
        ok = submitted >= case.count
        status = "OK  " if ok else "FAIL"
        if verbose or not ok:
            print(f"  [{status}] case {case.index:3d}: r={case.board_radius} "
                  f"submitted {submitted}/{case.count}, {lines} lines ({busy} non-noop)")
        if not ok:
            errors.append(f"case {case.index}: submitted {submitted} of {case.count} required patterns")
            if verbose:
                print("    final board:")
                print("    " + case.board.render(matter, indices).replace("\n", "\n    "))

    return score, active_lines, errors


def default_output_path(input_path):
    d, name = os.path.split(input_path)
    if name.startswith("in_"):
        return os.path.join(d, "out_" + name[3:])
    raise ValidationError(f"cannot derive output path from {input_path!r}; pass it explicitly")


def main():
    ap = argparse.ArgumentParser(description="Validate and score level 6 outputs.")
    ap.add_argument("input", help="input file, or a directory containing in_*/out_* pairs")
    ap.add_argument("output", nargs="?", help="output file (default: in_X -> out_X)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every case")
    ap.add_argument("--trace", type=int, metavar="CASE", help="print the board after every tick of one case")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        pairs = []
        for name in sorted(os.listdir(args.input)):
            if name.startswith("in_") and name.endswith(".txt"):
                inp = os.path.join(args.input, name)
                out = default_output_path(inp)
                if os.path.exists(out):
                    pairs.append((inp, out))
                else:
                    print(f"skip {name}: no {os.path.basename(out)}")
    else:
        pairs = [(args.input, args.output or default_output_path(args.input))]

    all_ok = True
    total = 0
    for inp, out in pairs:
        print(f"{os.path.basename(inp)} <- {os.path.basename(out)}")
        try:
            score, active_lines, errors = validate(inp, out, args.verbose, args.trace)
        except (ValidationError, OSError, ValueError) as e:
            print(f"  INVALID: {e}")
            all_ok = False
            continue
        if errors:
            all_ok = False
            for e in errors:
                print(f"  ERROR: {e}")
            print(f"  INVALID (score would be {score})")
        else:
            total += score
            print(f"  VALID  score = {score} lines ({active_lines} non-noop)")

    if len(pairs) > 1:
        print(f"total score of valid files: {total}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
