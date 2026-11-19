package io.github.zzuegg.brainiac.experimental.logic;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Min-conflicts local search for binary {@link Csp}s (Minton, Johnston, Philips &amp; Laird,
 * "Minimizing conflicts: a heuristic repair method for constraint satisfaction and scheduling
 * problems", AIJ 1992): start from a complete assignment (random, or greedy: each variable in
 * turn takes its least-conflicting value) and repeatedly move a random conflicted variable to its
 * least-conflicting value. Two standard escapes from plateaus and local minima: a random walk (with
 * probability {@code walk} a random value instead) and a tabu list forbidding a variable to return
 * to a value it held within the last {@code tenure} steps. Conflict counts per variable and value
 * are maintained incrementally, so a step costs {@code O(degree * domain)}.
 */
public final class MinConflicts {

    /**
     * A run.
     *
     * @param solved whether a solution was reached
     * @param steps repair steps taken
     * @param bestViolations the fewest violated constraints seen
     * @param assignment the best assignment
     * @param millis wall time
     */
    public record Result(boolean solved, long steps, int bestViolations, int[] assignment, double millis) {
    }

    private MinConflicts() {
    }

    /**
     * Runs min-conflicts.
     *
     * @param csp the problem
     * @param maxSteps the step budget
     * @param greedyInit whether to start from a greedy assignment
     * @param walk the random-walk probability
     * @param tenure the tabu tenure (0 for none)
     * @param seed the seed
     * @return the result
     */
    public static Result solve(Csp csp, long maxSteps, boolean greedyInit, double walk, int tenure, long seed) {
        long t0 = System.nanoTime();
        SplittableRandom r = new SplittableRandom(seed);
        int n = csp.variables();
        int[] cur = new int[n];
        java.util.Arrays.fill(cur, -1);
        int[][] conf = new int[n][];
        for (int x = 0; x < n; x++) {
            conf[x] = new int[csp.domain(x)];
        }
        for (int x = 0; x < n; x++) {
            int v;
            if (greedyInit) {
                int best = Integer.MAX_VALUE;
                List<Integer> ties = new ArrayList<>();
                for (int a = 0; a < conf[x].length; a++) {
                    if (conf[x][a] < best) {
                        best = conf[x][a];
                        ties.clear();
                    }
                    if (conf[x][a] == best) {
                        ties.add(a);
                    }
                }
                v = ties.get(r.nextInt(ties.size()));
            } else {
                v = r.nextInt(csp.domain(x));
            }
            set(csp, conf, cur, x, v);
        }
        long[][] tabuUntil = new long[n][];
        for (int x = 0; x < n; x++) {
            tabuUntil[x] = new long[csp.domain(x)];
        }
        int violations = csp.violations(cur);
        int best = violations;
        int[] bestAssign = cur.clone();
        long step = 0;
        while (violations > 0 && step < maxSteps) {
            step++;
            List<Integer> conflicted = new ArrayList<>();
            for (int x = 0; x < n; x++) {
                if (conf[x][cur[x]] > 0) {
                    conflicted.add(x);
                }
            }
            int x = conflicted.get(r.nextInt(conflicted.size()));
            int v;
            if (r.nextDouble() < walk) {
                v = r.nextInt(csp.domain(x));
            } else {
                int bc = Integer.MAX_VALUE;
                List<Integer> ties = new ArrayList<>();
                for (int a = 0; a < conf[x].length; a++) {
                    if (a == cur[x]) {
                        continue;
                    }
                    boolean tabu = tabuUntil[x][a] > step;
                    if (tabu && violations - conf[x][cur[x]] + conf[x][a] >= best) {
                        continue; // aspiration: a tabu move is allowed only if it beats the best
                    }
                    if (conf[x][a] < bc) {
                        bc = conf[x][a];
                        ties.clear();
                    }
                    if (conf[x][a] == bc) {
                        ties.add(a);
                    }
                }
                // plain min-conflicts stays put when every other value is worse; with a tabu list
                // the best non-tabu move is always taken (as in tabu search)
                if (ties.isEmpty() || (bc > conf[x][cur[x]] && tenure == 0)) {
                    continue;
                }
                v = ties.get(r.nextInt(ties.size()));
            }
            if (v == cur[x]) {
                continue;
            }
            violations += conf[x][v] - conf[x][cur[x]];
            if (tenure > 0) {
                tabuUntil[x][cur[x]] = step + tenure;
            }
            set(csp, conf, cur, x, v);
            if (violations < best) {
                best = violations;
                bestAssign = cur.clone();
            }
        }
        return new Result(violations == 0, step, best, bestAssign, (System.nanoTime() - t0) / 1e6);
    }

    /** Assigns {@code x = v}, updating the conflict counts of its neighbours. */
    private static void set(Csp csp, int[][] conf, int[] cur, int x, int v) {
        int old = cur[x];
        for (int c : csp.constraintsOf(x)) {
            int y = csp.other(c, x);
            for (int b = 0; b < conf[y].length; b++) {
                if (old >= 0 && !csp.allows(c, x, old, b)) {
                    conf[y][b]--;
                }
                if (!csp.allows(c, x, v, b)) {
                    conf[y][b]++;
                }
            }
        }
        cur[x] = v;
    }
}
