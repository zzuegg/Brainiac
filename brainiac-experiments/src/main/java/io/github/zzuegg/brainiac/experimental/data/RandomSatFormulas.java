package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Random CNF formulas from the SR(n) distribution of NeuroSAT (Selsam et al., 2019,
 * <a href="https://arxiv.org/abs/1802.03685">arXiv:1802.03685</a>), with a small DPLL
 * solver to label them. SR(n) adds random clauses (size {@code 1 + Bernoulli(0.3) +
 * Geometric(0.4)} over {@code n} variables, random signs) until the formula turns
 * unsatisfiable; the last clause with one literal negated gives a satisfiable twin. The
 * pair differs in a single literal, so no surface statistic tells them apart.
 *
 * <p>A clause is an {@code int[]} of literals {@code +v} or {@code -v}, {@code v} in
 * {@code 1..n}.
 */
public final class RandomSatFormulas {

    private RandomSatFormulas() {
    }

    /**
     * A labelled formula.
     *
     * @param variables the variable count {@code n}
     * @param clauses the clauses
     * @param satisfiable the label
     */
    public record Formula(int variables, List<int[]> clauses, boolean satisfiable) {
    }

    /**
     * One SR(n) pair: the unsatisfiable formula and its satisfiable twin.
     *
     * @param n the variable count
     * @param random the source
     * @return {satisfiable, unsatisfiable}
     */
    public static Formula[] srPair(int n, RandomGenerator random) {
        List<int[]> clauses = new ArrayList<>();
        while (true) {
            int k = 1 + (random.nextDouble() < 0.3 ? 1 : 0);
            while (random.nextDouble() >= 0.4) {
                k++;
            }
            k = Math.min(k, n);
            int[] clause = new int[k];
            List<Integer> vars = new ArrayList<>();
            for (int v = 1; v <= n; v++) {
                vars.add(v);
            }
            for (int i = 0; i < k; i++) {
                int v = vars.remove(random.nextInt(vars.size()));
                clause[i] = random.nextBoolean() ? v : -v;
            }
            clauses.add(clause);
            if (solve(n, clauses) == null) {
                List<int[]> sat = new ArrayList<>(clauses.subList(0, clauses.size() - 1));
                int[] flipped = clause.clone();
                int i = random.nextInt(k);
                flipped[i] = -flipped[i];
                sat.add(flipped);
                return new Formula[] {new Formula(n, sat, true), new Formula(n, List.copyOf(clauses), false)};
            }
        }
    }

    /**
     * Finds a satisfying assignment by DPLL with unit propagation.
     *
     * @param n the variable count
     * @param clauses the clauses
     * @return {@code assignment[v]} for {@code v = 1..n} (index 0 unused), or {@code null} if
     *     unsatisfiable
     */
    public static boolean[] solve(int n, List<int[]> clauses) {
        int[] value = new int[n + 1];
        return dpll(clauses, value) ? toBooleans(value) : null;
    }

    private static boolean[] toBooleans(int[] value) {
        boolean[] b = new boolean[value.length];
        for (int v = 1; v < value.length; v++) {
            b[v] = value[v] > 0;
        }
        return b;
    }

    private static boolean dpll(List<int[]> clauses, int[] value) {
        List<Integer> assigned = new ArrayList<>();
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int[] c : clauses) {
                int unassigned = 0;
                int last = 0;
                boolean sat = false;
                for (int lit : c) {
                    int v = value[Math.abs(lit)];
                    if (v == 0) {
                        unassigned++;
                        last = lit;
                    } else if (v > 0 == lit > 0) {
                        sat = true;
                        break;
                    }
                }
                if (sat) {
                    continue;
                }
                if (unassigned == 0) {
                    undo(assigned, value);
                    return false;
                }
                if (unassigned == 1) {
                    value[Math.abs(last)] = last > 0 ? 1 : -1;
                    assigned.add(Math.abs(last));
                    changed = true;
                }
            }
        }
        int branch = 0;
        for (int[] c : clauses) {
            for (int lit : c) {
                if (value[Math.abs(lit)] == 0) {
                    branch = Math.abs(lit);
                    break;
                }
            }
            if (branch != 0) {
                break;
            }
        }
        if (branch == 0) {
            return true;
        }
        for (int s : new int[] {1, -1}) {
            value[branch] = s;
            if (dpll(clauses, value)) {
                return true;
            }
            value[branch] = 0;
        }
        undo(assigned, value);
        return false;
    }

    private static void undo(List<Integer> assigned, int[] value) {
        for (int v : assigned) {
            value[v] = 0;
        }
    }

    /**
     * @param clauses the clauses
     * @param assignment {@code assignment[v]}, {@code v = 1..n}
     * @return whether every clause has a true literal
     */
    public static boolean satisfies(List<int[]> clauses, boolean[] assignment) {
        for (int[] c : clauses) {
            boolean ok = false;
            for (int lit : c) {
                if (assignment[Math.abs(lit)] == lit > 0) {
                    ok = true;
                    break;
                }
            }
            if (!ok) {
                return false;
            }
        }
        return true;
    }
}
