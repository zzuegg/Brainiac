package io.github.zzuegg.brainiac.experimental.logic;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * A learned clause scorer for {@link ResolutionProver} in the style of ENIGMA (Jakubův &amp; Urban,
 * "ENIGMA: efficient learning-based inference guiding machine", CICM 2017): clauses are mapped to
 * hashed counts of vertical symbol walks (sign, predicate, then up to two function or constant
 * symbols below each argument, variables collapsed to {@code *}), plus length, weight, depth,
 * variable count, conjecture ancestry and overlap with the conjecture's symbols. Constants named
 * like {@code c17} (instance-specific individuals) are anonymised so that a model trained on one
 * problem applies to others; constants of the conjecture get their own token. A logistic
 * regression trained on the given clauses of earlier proofs (positive: used in the proof;
 * negative: processed but unused) scores new clauses.
 */
public final class ClauseSelector {

    private final int dim;
    private final double[] w;
    private double bias;
    private ResolutionProver.Problem cached;
    private String[] names;
    private BitSet conjecture;

    /**
     * A training example.
     *
     * @param features hashed feature indices (repeats count)
     * @param positive whether the clause was used in the proof
     */
    public record Example(int[] features, boolean positive) {
    }

    /**
     * Creates an untrained selector.
     *
     * @param bits log2 of the hashed feature dimension
     */
    public ClauseSelector(int bits) {
        this.dim = 1 << bits;
        this.w = new double[dim];
    }

    private void prepare(ResolutionProver.Problem p) {
        if (cached == p) {
            return;
        }
        cached = p;
        conjecture = p.conjectureSymbols();
        names = new String[p.symbols().size()];
        for (int s = 0; s < names.length; s++) {
            String n = p.symbols().get(s);
            if (p.arities().get(s) == 0 && conjecture.get(s)) {
                names[s] = "@g";
            } else if (p.arities().get(s) == 0 && n.matches("[a-z]+[0-9]+(h[0-9]+)?")) {
                names[s] = "@c";
            } else {
                names[s] = n;
            }
        }
    }

    private int hash(String s) {
        return Math.floorMod(s.hashCode() * 0x9E3779B1, dim);
    }

    /**
     * The hashed features of a clause.
     *
     * @param p the problem the clause belongs to
     * @param c the clause
     * @return feature indices
     */
    public int[] features(ResolutionProver.Problem p, ResolutionProver.Clause c) {
        prepare(p);
        List<Integer> f = new ArrayList<>();
        Set<Integer> shared = new HashSet<>();
        int depth = 0;
        int vars = 0;
        for (ResolutionProver.Literal l : c.literals()) {
            String head = (l.positive() ? "+" : "-") + names[l.predicate()];
            f.add(hash("L" + head));
            if (conjecture.get(l.predicate())) {
                shared.add(l.predicate());
            }
            for (ResolutionProver.Term t : l.args()) {
                String a = token(t);
                f.add(hash(head + "/" + a));
                for (ResolutionProver.Term u : t.args()) {
                    f.add(hash(head + "/" + a + "/" + token(u)));
                    f.add(hash(a + "/" + token(u)));
                }
                depth = Math.max(depth, t.depth());
                vars += countVars(t, shared);
            }
        }
        f.add(hash("len=" + Math.min(c.literals().size(), 6)));
        f.add(hash("w=" + Math.min(c.weight() / 3, 12)));
        f.add(hash("depth=" + depth));
        f.add(hash("vars=" + Math.min(vars, 6)));
        f.add(hash("goal=" + c.goal()));
        f.add(hash("shared=" + Math.min(shared.size(), 4)));
        f.add(hash("unit=" + (c.literals().size() == 1) + c.goal()));
        f.add(hash("ground=" + (vars == 0) + "/" + c.goal()));
        return f.stream().mapToInt(Integer::intValue).toArray();
    }

    private String token(ResolutionProver.Term t) {
        return t.symbol() < 0 ? "*" : names[t.symbol()];
    }

    private int countVars(ResolutionProver.Term t, Set<Integer> shared) {
        if (t.symbol() < 0) {
            return 1;
        }
        if (conjecture.get(t.symbol())) {
            shared.add(t.symbol());
        }
        int n = 0;
        for (ResolutionProver.Term a : t.args()) {
            n += countVars(a, shared);
        }
        return n;
    }

    /**
     * Labelled examples from a proof: every given clause, positive when it is in the proof.
     *
     * @param p the problem
     * @param r a successful proof attempt
     * @return the examples (empty if no proof)
     */
    public List<Example> examples(ResolutionProver.Problem p, ResolutionProver.Result r) {
        List<Example> out = new ArrayList<>();
        if (!r.proved()) {
            return out;
        }
        Set<Integer> inProof = new HashSet<>();
        for (ResolutionProver.Clause c : r.proof()) {
            inProof.add(c.id());
        }
        for (ResolutionProver.Clause c : r.givenClauses()) {
            out.add(new Example(features(p, c), inProof.contains(c.id())));
        }
        return out;
    }

    /**
     * Trains the logistic regression by SGD with class-balanced weights.
     *
     * @param data the examples
     * @param epochs passes over the data
     * @param lr the learning rate
     * @param l2 the L2 penalty
     * @param seed the shuffling seed
     */
    public void train(List<Example> data, int epochs, double lr, double l2, long seed) {
        long pos = data.stream().filter(Example::positive).count();
        long neg = data.size() - pos;
        if (pos == 0 || neg == 0) {
            return;
        }
        double wPos = data.size() / (2.0 * pos);
        double wNeg = data.size() / (2.0 * neg);
        SplittableRandom r = new SplittableRandom(seed);
        int[] order = new int[data.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        for (int e = 0; e < epochs; e++) {
            for (int i = order.length - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            for (int i : order) {
                Example ex = data.get(i);
                double z = logit(ex.features());
                double p = 1 / (1 + Math.exp(-z));
                double g = (p - (ex.positive() ? 1 : 0)) * (ex.positive() ? wPos : wNeg);
                for (int f : ex.features()) {
                    w[f] -= lr * (g + l2 * w[f]);
                }
                bias -= lr * g;
            }
        }
    }

    private double logit(int[] features) {
        double z = bias;
        for (int f : features) {
            z += w[f];
        }
        return z;
    }

    /**
     * Scores a clause (higher is more promising).
     *
     * @param p the problem
     * @param c the clause
     * @return the logit of "used in a proof"
     */
    public double score(ResolutionProver.Problem p, ResolutionProver.Clause c) {
        return logit(features(p, c));
    }

    /**
     * The queue priority used by {@link ResolutionProver.Selection#LEARNED} (lower is selected
     * first): ENIGMA's combination of the clause weight with the classifier, here the weight
     * times {@code 1 + 9 (1 - p)} so that clauses predicted useless count up to ten times heavier.
     *
     * @param p the problem
     * @param c the clause
     * @return the priority
     */
    public double priority(ResolutionProver.Problem p, ResolutionProver.Clause c) {
        double prob = 1 / (1 + Math.exp(-score(p, c)));
        return c.weight() * (1 + 9 * (1 - prob));
    }
}
