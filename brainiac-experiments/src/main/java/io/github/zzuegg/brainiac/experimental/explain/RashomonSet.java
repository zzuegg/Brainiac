package io.github.zzuegg.brainiac.experimental.explain;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Rashomon sets of sparse decision trees and the predictive multiplicity they reveal.
 *
 * <p>The Rashomon set is every tree of at most the given depth whose objective is within
 * {@code epsilon} of the optimum. {@link #enumerate} lists it sorted by objective, best first, in
 * the spirit of SORTD (Arslan et al., NeurIPS 2025, <a href="https://arxiv.org/abs/2511.03344">arXiv:2511.03344</a>):
 * every dynamic-programming subproblem keeps a sorted list of its own good subtrees, and a split's
 * list is formed by combining its children's lists under the budget left by the other child's
 * optimal cost (taken from the {@link OptimalTrees} cache), so no tree outside the set is ever
 * built and the per-subproblem lists are shared between paths. A brute-force enumeration
 * ({@link #bruteForce}) is the reference for small depths.
 *
 * <p>Multiplicity (Marx, Calmon and Ustun, ICML 2020, <a href="https://arxiv.org/abs/1909.06677">arXiv:1909.06677</a>):
 * {@link #ambiguity} is the share of rows whose prediction some set member flips relative to a
 * reference model, {@link #discrepancy} the largest share flipped by a single member;
 * {@link #importance} gives each feature's model reliance across the set (the Rashomon
 * importance distribution of Donnelly et al., NeurIPS 2023).
 */
public final class RashomonSet {

    /**
     * A member of the set.
     *
     * @param tree the tree
     * @param objective its objective
     */
    public record Member(OptimalTrees.Tree tree, double objective) {
    }

    private record Key(long[] bits, int depth) {
        @Override
        public boolean equals(Object o) {
            return o instanceof Key k && k.depth == depth && Arrays.equals(k.bits, bits);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(bits) * 31 + depth;
        }
    }

    private record Cached(double bound, List<Member> members) {
    }

    private final OptimalTrees problem;
    private final Map<Key, Cached> lists = new HashMap<>();
    private final int limit;

    /**
     * Creates the enumerator.
     *
     * @param problem the solved (or unsolved) optimal-tree problem whose cache is reused
     * @param limit the most members kept per subproblem (a safety cap)
     */
    public RashomonSet(OptimalTrees problem, int limit) {
        this.problem = problem;
        this.limit = limit;
    }

    /**
     * Every tree within {@code epsilon} of the optimum, best first.
     *
     * @param depth the maximum depth
     * @param epsilon the additive objective tolerance
     * @return the members, sorted by objective
     */
    public List<Member> enumerate(int depth, double epsilon) {
        long[] all = problem.all();
        double best = problem.minimumCost(all, depth);
        return list(all, depth, best + epsilon + 1e-12);
    }

    private List<Member> list(long[] s, int depth, double bound) {
        Key key = new Key(s, depth);
        Cached c = lists.get(key);
        if (c != null && c.bound >= bound) {
            List<Member> out = new ArrayList<>();
            for (Member m : c.members) {
                if (m.objective() <= bound) {
                    out.add(m);
                }
            }
            return out;
        }
        List<Member> out = new ArrayList<>();
        double leaf = problem.leafCost(s);
        if (leaf <= bound) {
            out.add(new Member(OptimalTrees.Tree.leaf(problem.majority(s)), leaf));
        }
        if (depth > 0) {
            for (int f = 0; f < problem.features(); f++) {
                long[][] parts = problem.split(s, f);
                if (OptimalTrees.count(parts[0]) == 0 || OptimalTrees.count(parts[1]) == 0) {
                    continue;
                }
                double minLeft = problem.minimumCost(parts[0], depth - 1);
                double minRight = problem.minimumCost(parts[1], depth - 1);
                if (minLeft + minRight > bound) {
                    continue;
                }
                List<Member> left = list(parts[0], depth - 1, bound - minRight);
                List<Member> right = list(parts[1], depth - 1, bound - minLeft);
                for (Member l : left) {
                    for (Member r : right) {
                        if (l.objective() + r.objective() > bound) {
                            break;
                        }
                        out.add(new Member(new OptimalTrees.Tree(f, l.tree(), r.tree(), problem.majority(s)),
                                l.objective() + r.objective()));
                    }
                    if (out.size() > limit) {
                        break;
                    }
                }
            }
        }
        out.sort(Comparator.comparingDouble(Member::objective));
        if (out.size() > limit) {
            out = new ArrayList<>(out.subList(0, limit));
        }
        lists.put(key, new Cached(bound, out));
        return out;
    }

    /**
     * Brute-force enumeration of every tree of at most the given depth (no pruning, no
     * sharing), the reference.
     *
     * @param depth the maximum depth (2 or 3 on few features)
     * @param epsilon the tolerance
     * @return the members within tolerance, sorted
     */
    public List<Member> bruteForce(int depth, double epsilon) {
        List<Member> all = new ArrayList<>();
        for (OptimalTrees.Tree t : allTrees(problem.all(), depth)) {
            all.add(new Member(t, problem.objective(t)));
        }
        double best = all.stream().mapToDouble(Member::objective).min().orElse(0);
        List<Member> out = new ArrayList<>();
        for (Member m : all) {
            if (m.objective() <= best + epsilon + 1e-12) {
                out.add(m);
            }
        }
        out.sort(Comparator.comparingDouble(Member::objective));
        return out;
    }

    private List<OptimalTrees.Tree> allTrees(long[] s, int depth) {
        List<OptimalTrees.Tree> out = new ArrayList<>();
        out.add(OptimalTrees.Tree.leaf(problem.majority(s)));
        if (depth == 0) {
            return out;
        }
        for (int f = 0; f < problem.features(); f++) {
            long[][] parts = problem.split(s, f);
            if (OptimalTrees.count(parts[0]) == 0 || OptimalTrees.count(parts[1]) == 0) {
                continue;
            }
            List<OptimalTrees.Tree> left = allTrees(parts[0], depth - 1);
            List<OptimalTrees.Tree> right = allTrees(parts[1], depth - 1);
            for (OptimalTrees.Tree l : left) {
                for (OptimalTrees.Tree r : right) {
                    out.add(new OptimalTrees.Tree(f, l, r, problem.majority(s)));
                }
            }
        }
        return out;
    }

    /**
     * Ambiguity: the share of rows on which at least one model disagrees with the reference.
     *
     * @param models the set
     * @param reference the reference predictions
     * @param x the binary rows
     * @return the share
     */
    public static double ambiguity(List<OptimalTrees.Tree> models, int[] reference, double[][] x) {
        int flipped = 0;
        for (int i = 0; i < x.length; i++) {
            for (OptimalTrees.Tree t : models) {
                if (t.predict(x[i]) != reference[i]) {
                    flipped++;
                    break;
                }
            }
        }
        return flipped / (double) x.length;
    }

    /**
     * Discrepancy: the largest share of rows one model flips relative to the reference.
     *
     * @param models the set
     * @param reference the reference predictions
     * @param x the binary rows
     * @return the share
     */
    public static double discrepancy(List<OptimalTrees.Tree> models, int[] reference, double[][] x) {
        double worst = 0;
        for (OptimalTrees.Tree t : models) {
            int flipped = 0;
            for (int i = 0; i < x.length; i++) {
                if (t.predict(x[i]) != reference[i]) {
                    flipped++;
                }
            }
            worst = Math.max(worst, flipped / (double) x.length);
        }
        return worst;
    }

    /**
     * The Rashomon ratio proxy: the set size over the number of trees of the depth (counted by
     * brute force only for tiny problems, so this takes the count as a parameter).
     *
     * @param setSize the Rashomon set size
     * @param hypothesisCount the number of candidate trees
     * @return the ratio
     */
    public static double ratio(int setSize, double hypothesisCount) {
        return setSize / hypothesisCount;
    }

    /**
     * Model reliance of every feature for every member: the increase of the error rate when the
     * feature's column is shuffled.
     *
     * @param models the set
     * @param x binary rows
     * @param y labels
     * @param seed the shuffle seed
     * @return {@code [model][feature]}
     */
    public static double[][] importance(List<OptimalTrees.Tree> models, double[][] x, double[] y, long seed) {
        int p = x[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        int[] perm = new int[x.length];
        for (int i = 0; i < perm.length; i++) {
            perm[i] = i;
        }
        for (int i = perm.length - 1; i > 0; i--) {
            int k = r.nextInt(i + 1);
            int t = perm[i];
            perm[i] = perm[k];
            perm[k] = t;
        }
        double[][] out = new double[models.size()][p];
        for (int m = 0; m < models.size(); m++) {
            OptimalTrees.Tree t = models.get(m);
            double base = error(t, x, y, -1, perm);
            for (int f : t.features()) {
                out[m][f] = error(t, x, y, f, perm) - base;
            }
        }
        return out;
    }

    private static double error(OptimalTrees.Tree t, double[][] x, double[] y, int feature, int[] perm) {
        int wrong = 0;
        for (int i = 0; i < x.length; i++) {
            double[] row = x[i];
            if (feature >= 0) {
                row = row.clone();
                row[feature] = x[perm[i]][feature];
            }
            if (t.predict(row) != (int) y[i]) {
                wrong++;
            }
        }
        return wrong / (double) x.length;
    }
}
