package io.github.zzuegg.brainiac.experimental.symbolic;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

/**
 * A planning heuristic learned by ranking rather than regression (Garrett, Kaelbling &amp;
 * Lozano-Pérez, "Learning to rank for synthesizing planning heuristics", IJCAI 2016; the ranking
 * objective of Wang &amp; Trevizan's action-set heuristics, AAAI 2025,
 * <a href="https://arxiv.org/abs/2504.20318">arXiv:2504.20318</a>).
 *
 * <p>Greedy best-first search only needs states in the right order, so a linear model on
 * {@link WlFeatures} is fitted to rank, at every step of an optimal training plan, the plan's
 * successor below each of its siblings (and below its parent), with the pairwise logistic loss
 * {@code log(1 + exp(h(better) - h(worse)))} and an L2 penalty, by full-batch Adam. The same
 * ranker scores action sets for {@link PartialSpaceSearch} ({@link #trainActionSets}).
 */
public final class RankingHeuristic {

    private final WlFeatures wl;
    private final boolean actionSets;
    private double[] w;
    private double trainMillis;

    private RankingHeuristic(WlFeatures wl, boolean actionSets) {
        this.wl = wl;
        this.actionSets = actionSets;
    }

    /** @return the training time in milliseconds */
    public double trainMillis() {
        return trainMillis;
    }

    /** @return whether this ranker scores action sets */
    public boolean scoresActionSets() {
        return actionSets;
    }

    /**
     * Trains a state ranker on optimal plans.
     *
     * @param samples states of optimal plans
     * @param iterations WL rounds
     * @param steps Adam steps
     * @return the ranker
     */
    public static RankingHeuristic trainStates(List<LearnedPlanningHeuristic.Sample> samples, int iterations,
            int steps) {
        long t0 = System.nanoTime();
        RankingHeuristic h = new RankingHeuristic(new WlFeatures(iterations), false);
        List<double[]> better = new ArrayList<>();
        List<double[]> worse = new ArrayList<>();
        for (LearnedPlanningHeuristic.Sample s : samples) {
            if (s.next() == null) {
                continue;
            }
            double[] good = h.wl.features(s.task(), s.next(), true);
            double[] parent = h.wl.features(s.task(), s.state(), true);
            better.add(good);
            worse.add(parent);
            for (StripsTask.Action a : s.task().applicable(s.state())) {
                BitSet t = a.apply(s.state());
                if (!t.equals(s.next())) {
                    better.add(good);
                    worse.add(h.wl.features(s.task(), t, true));
                }
            }
        }
        h.wl.freeze();
        h.fit(better, worse, steps);
        h.trainMillis = (System.nanoTime() - t0) / 1e6;
        return h;
    }

    /**
     * Trains an action-set ranker on the partial-space decisions of optimal plans: at every
     * level (schema, then each argument) the set consistent with the plan's action must rank
     * below its siblings, and the next state's full set below the current state's.
     *
     * @param samples states of optimal plans
     * @param iterations WL rounds
     * @param steps Adam steps
     * @return the ranker
     */
    public static RankingHeuristic trainActionSets(List<LearnedPlanningHeuristic.Sample> samples, int iterations,
            int steps) {
        long t0 = System.nanoTime();
        RankingHeuristic h = new RankingHeuristic(new WlFeatures(iterations), true);
        List<double[]> better = new ArrayList<>();
        List<double[]> worse = new ArrayList<>();
        for (LearnedPlanningHeuristic.Sample s : samples) {
            if (s.next() == null) {
                continue;
            }
            StripsTask t = s.task();
            List<StripsTask.Action> all = t.applicable(s.state());
            better.add(h.wl.features(t, s.next(), t.applicable(s.next()), true));
            worse.add(h.wl.features(t, s.state(), all, true));
            StripsTask.Action star = s.nextAction();
            int arity = star.args().length;
            for (int level = -1; level < arity; level++) {
                // the sets one decision deeper than the plan's prefix of length level
                List<List<StripsTask.Action>> siblings = PartialSpaceSearch.split(all, star, level);
                List<StripsTask.Action> chosen = siblings.get(0);
                if (siblings.size() < 2) {
                    continue;
                }
                double[] good = h.wl.features(t, s.state(), chosen, true);
                for (int k = 1; k < siblings.size(); k++) {
                    better.add(good);
                    worse.add(h.wl.features(t, s.state(), siblings.get(k), true));
                }
            }
        }
        h.wl.freeze();
        h.fit(better, worse, steps);
        h.trainMillis = (System.nanoTime() - t0) / 1e6;
        return h;
    }

    private void fit(List<double[]> better, List<double[]> worse, int steps) {
        int d = wl.dimension();
        double[][] diff = new double[better.size()][d];
        for (int i = 0; i < diff.length; i++) {
            double[] b = better.get(i);
            double[] c = worse.get(i);
            for (int j = 0; j < d; j++) {
                diff[i][j] = (j < b.length ? b[j] : 0) - (j < c.length ? c[j] : 0);
            }
        }
        w = new double[d];
        double[] m = new double[d];
        double[] v = new double[d];
        double lr = 0.05;
        double l2 = 1e-4;
        for (int t = 1; t <= steps; t++) {
            double[] g = new double[d];
            for (double[] x : diff) {
                double z = 0;
                for (int j = 0; j < d; j++) {
                    z += w[j] * x[j];
                }
                double p = 1 / (1 + Math.exp(-z));
                for (int j = 0; j < d; j++) {
                    g[j] += p * x[j] / diff.length;
                }
            }
            for (int j = 0; j < d; j++) {
                g[j] += l2 * w[j];
                m[j] = 0.9 * m[j] + 0.1 * g[j];
                v[j] = 0.999 * v[j] + 0.001 * g[j] * g[j];
                w[j] -= lr * (m[j] / (1 - Math.pow(0.9, t))) / (Math.sqrt(v[j] / (1 - Math.pow(0.999, t))) + 1e-8);
            }
        }
    }

    private double score(double[] x) {
        double s = 0;
        for (int j = 0; j < Math.min(x.length, w.length); j++) {
            s += w[j] * x[j];
        }
        return s;
    }

    /**
     * Scores a state (lower is better; a ranking score, not a cost estimate).
     *
     * @param task the task
     * @param state the state
     * @return the score
     */
    public double score(StripsTask task, BitSet state) {
        return score(actionSets ? wl.features(task, state, task.applicable(state), false)
            : wl.features(task, state, false));
    }

    /**
     * Scores a state with an action set.
     *
     * @param task the task
     * @param state the state
     * @param actions the action set
     * @return the score
     */
    public double score(StripsTask task, BitSet state, List<StripsTask.Action> actions) {
        return score(wl.features(task, state, actions, false));
    }

    /**
     * @param task the task to search
     * @return the ranker as a search evaluator (goal states get minus infinity is avoided: the
     *     search's goal test handles them)
     */
    public StateSpaceSearch.Evaluator evaluator(StripsTask task) {
        return (s, d) -> StateSpaceSearch.Evaluation.of(score(task, s));
    }
}
