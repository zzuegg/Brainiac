package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.StationaryKernel;
import io.github.zzuegg.brainiac.train.HyperparameterSearch;
import io.github.zzuegg.brainiac.train.HyperparameterSearch.Choice;
import io.github.zzuegg.brainiac.train.HyperparameterSearch.IntRange;
import io.github.zzuegg.brainiac.train.HyperparameterSearch.LogUniform;
import io.github.zzuegg.brainiac.train.HyperparameterSearch.Objective;
import io.github.zzuegg.brainiac.train.HyperparameterSearch.Range;
import io.github.zzuegg.brainiac.train.HyperparameterSearch.Result;
import io.github.zzuegg.brainiac.train.HyperparameterSearch.Trial;
import io.github.zzuegg.brainiac.train.HyperparameterSearch.Uniform;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Model-based hyperparameter search over the same spaces, objectives and results as core
 * {@link HyperparameterSearch}, written as the {@code bayesOpt} and {@code bohb} methods that
 * class would gain.
 * <ul>
 *   <li>{@link #bayesOpt}: Gaussian-process Bayesian optimisation (Snoek, Larochelle and Adams
 *       2012, <a href="https://arxiv.org/abs/1206.2944">arXiv:1206.2944</a>). Configurations are
 *       mapped to the unit cube; a Matérn 5/2 {@link GaussianProcess} with marginal-likelihood
 *       hyperparameters models the standardised scores, and each new trial maximises expected
 *       improvement (or a lower confidence bound) over random and locally perturbed
 *       candidates.</li>
 *   <li>{@link #bohb}: BOHB (Falkner, Klein and Hutter 2018,
 *       <a href="https://arxiv.org/abs/1807.01774">arXiv:1807.01774</a>): Hyperband's
 *       brackets of successive halving, but each new configuration is drawn from a
 *       tree-Parzen model (a product kernel density of the good versus the bad trials) fitted
 *       on the largest budget with enough observations, with a share of random draws kept for
 *       exploration.</li>
 * </ul>
 */
public final class ModelBasedSearch {

    /** How Bayesian optimisation scores a candidate. */
    public enum Acquisition {
        /** Expected improvement over the best score so far. */
        EXPECTED_IMPROVEMENT,
        /** The lower confidence bound {@code mu - 2 sigma} (UCB for minimisation). */
        LOWER_CONFIDENCE_BOUND
    }

    private final Map<String, Range> space;
    private final List<String> names;
    private final SplittableRandom random;

    /**
     * Creates a search.
     *
     * @param space each hyperparameter's range, by name
     * @param seed seeds the draws
     */
    public ModelBasedSearch(Map<String, Range> space, long seed) {
        this.space = new LinkedHashMap<>(Objects.requireNonNull(space, "space"));
        this.names = List.copyOf(this.space.keySet());
        this.random = new SplittableRandom(seed);
    }

    /** A random point of the unit cube. */
    private double[] randomPoint() {
        double[] u = new double[names.size()];
        for (int i = 0; i < u.length; i++) {
            u[i] = random.nextDouble();
        }
        return u;
    }

    /**
     * Maps a unit-cube point to a configuration: uniform ranges linearly, log-uniform ranges in
     * the logarithm, integers and choices by equal-width bins.
     *
     * @param u a point of {@code [0, 1]^p}
     * @return the configuration
     */
    public Map<String, Object> decode(double[] u) {
        Map<String, Object> c = new LinkedHashMap<>();
        for (int i = 0; i < names.size(); i++) {
            double v = Math.clamp(u[i], 0, 1);
            Object value = switch (space.get(names.get(i))) {
                case Uniform r -> r.low() + v * (r.high() - r.low());
                case LogUniform r -> Math.exp(Math.log(r.low()) + v * (Math.log(r.high()) - Math.log(r.low())));
                case IntRange r -> Math.min(r.high(), r.low() + (int) Math.floor(v * (r.high() - r.low() + 1)));
                case Choice r -> r.options().get(Math.min(r.options().size() - 1, (int) Math.floor(v * r.options()
                        .size())));
            };
            c.put(names.get(i), value);
        }
        return c;
    }

    /**
     * Maps a configuration to the centre of its unit-cube cell.
     *
     * @param config the configuration
     * @return the point
     */
    public double[] encode(Map<String, Object> config) {
        double[] u = new double[names.size()];
        for (int i = 0; i < u.length; i++) {
            Object v = config.get(names.get(i));
            u[i] = switch (space.get(names.get(i))) {
                case Uniform r -> (((Number) v).doubleValue() - r.low()) / Math.max(r.high() - r.low(), 1e-12);
                case LogUniform r -> (Math.log(((Number) v).doubleValue()) - Math.log(r.low()))
                        / Math.max(Math.log(r.high()) - Math.log(r.low()), 1e-12);
                case IntRange r -> (((Number) v).intValue() - r.low() + 0.5) / (r.high() - r.low() + 1);
                case Choice r -> (r.options().indexOf(v) + 0.5) / r.options().size();
            };
        }
        return u;
    }

    /**
     * Gaussian-process Bayesian optimisation at a fixed budget.
     *
     * @param objective trains and scores, lower is better
     * @param trials how many configurations in all
     * @param budget the budget of each
     * @param startup random trials first, at least 2
     * @param acquisition the acquisition function
     * @return the trials and the best
     */
    public Result bayesOpt(Objective objective, int trials, int budget, int startup, Acquisition acquisition) {
        List<Trial> all = new ArrayList<>();
        List<double[]> points = new ArrayList<>();
        for (int t = 0; t < trials; t++) {
            double[] u;
            if (t < Math.max(2, startup)) {
                u = randomPoint();
            } else {
                u = proposeBo(points, all, acquisition);
            }
            Map<String, Object> c = decode(u);
            points.add(encode(c));
            all.add(new Trial(c, budget, objective.evaluate(c, budget)));
        }
        return result(all);
    }

    private double[] proposeBo(List<double[]> points, List<Trial> trials, Acquisition acquisition) {
        int n = points.size();
        double[] y = new double[n];
        double mean = 0;
        for (int i = 0; i < n; i++) {
            y[i] = trials.get(i).score();
            mean += y[i] / n;
        }
        double var = 0;
        for (double v : y) {
            var += (v - mean) * (v - mean) / n;
        }
        double sd = Math.sqrt(Math.max(var, 1e-12));
        double best = Double.POSITIVE_INFINITY;
        int bestIdx = 0;
        for (int i = 0; i < n; i++) {
            y[i] = (y[i] - mean) / sd;
            if (y[i] < best) {
                best = y[i];
                bestIdx = i;
            }
        }
        double[][] x = points.toArray(new double[0][]);
        GaussianProcess gp = GaussianProcess.optimize(x, y, StationaryKernel.matern52(0.3, 1), 1e-3, 30);
        int candidates = 512;
        double[][] cand = new double[candidates][];
        for (int i = 0; i < candidates; i++) {
            if (i % 2 == 0) {
                cand[i] = randomPoint();
            } else {
                double[] base = x[i % 4 == 1 ? bestIdx : random.nextInt(n)];
                cand[i] = new double[base.length];
                for (int d = 0; d < base.length; d++) {
                    cand[i][d] = Math.clamp(base[d] + 0.08 * random.nextGaussian(), 0, 1);
                }
            }
        }
        GaussianProcess.Prediction p = gp.predict(cand);
        double bestScore = Double.NEGATIVE_INFINITY;
        double[] pick = cand[0];
        for (int i = 0; i < candidates; i++) {
            double s = Math.sqrt(p.latentVariance()[i]);
            double m = p.mean()[i];
            double a = switch (acquisition) {
                case EXPECTED_IMPROVEMENT -> {
                    double imp = best - m - 0.01;
                    double z = imp / s;
                    yield imp * MlMetrics.normalCdf(z) + s * Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
                }
                case LOWER_CONFIDENCE_BOUND -> -(m - 2 * s);
            };
            if (a > bestScore) {
                bestScore = a;
                pick = cand[i];
            }
        }
        return pick;
    }

    /**
     * BOHB: Hyperband with a tree-Parzen model proposing the configurations of every bracket.
     *
     * @param objective trains and scores, lower is better
     * @param maxBudget the largest budget one configuration gets
     * @param eta the reduction factor, at least 2
     * @param iterations how many times to run all brackets
     * @param randomFraction the share of configurations drawn at random, for example {@code 1/3}
     * @return the trials and the best
     */
    public Result bohb(Objective objective, int maxBudget, int eta, int iterations, double randomFraction) {
        if (eta < 2 || maxBudget < 1) {
            throw new IllegalArgumentException("bohb needs eta >= 2 and a budget >= 1");
        }
        int sMax = (int) Math.floor(Math.log(maxBudget) / Math.log(eta) + 1e-9);
        List<Trial> all = new ArrayList<>();
        for (int it = 0; it < iterations; it++) {
            for (int s = sMax; s >= 0; s--) {
                int n = (int) Math.ceil((double) (sMax + 1) / (s + 1) * Math.pow(eta, s));
                List<Map<String, Object>> configs = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    double[] u = random.nextDouble() < randomFraction ? null : proposeKde(all);
                    configs.add(decode(u == null ? randomPoint() : u));
                }
                for (int round = 0; round <= s && !configs.isEmpty(); round++) {
                    int budget = Math.max(1, (int) Math.round(maxBudget * Math.pow(eta, round - s)));
                    List<Trial> scored = new ArrayList<>();
                    for (Map<String, Object> c : configs) {
                        Trial t = new Trial(c, budget, objective.evaluate(c, budget));
                        scored.add(t);
                        all.add(t);
                    }
                    scored.sort(Comparator.comparingDouble(Trial::score));
                    int keep = Math.max(1, scored.size() / eta);
                    configs = new ArrayList<>();
                    for (int i = 0; i < keep; i++) {
                        configs.add(scored.get(i).config());
                    }
                }
            }
        }
        return result(all);
    }

    /**
     * BOHB's proposal: on the largest budget with at least {@code p + 2} trials, split the trials
     * into the best 15% and the rest, sample candidates around good trials with a Gaussian
     * kernel, and keep the one maximising {@code l(x) / g(x)}. Returns {@code null} (draw at
     * random) when no budget has enough trials yet.
     */
    private double[] proposeKde(List<Trial> all) {
        int p = names.size();
        Map<Integer, List<Trial>> byBudget = new LinkedHashMap<>();
        for (Trial t : all) {
            byBudget.computeIfAbsent(t.budget(), k -> new ArrayList<>()).add(t);
        }
        List<Trial> use = null;
        int bestBudget = -1;
        for (Map.Entry<Integer, List<Trial>> e : byBudget.entrySet()) {
            if (e.getValue().size() >= p + 2 && e.getKey() > bestBudget) {
                bestBudget = e.getKey();
                use = e.getValue();
            }
        }
        if (use == null) {
            return null;
        }
        List<Trial> sorted = new ArrayList<>(use);
        sorted.sort(Comparator.comparingDouble(Trial::score));
        int good = Math.max(p + 1, (int) Math.ceil(0.15 * sorted.size()));
        good = Math.min(good, sorted.size() - 1);
        List<double[]> l = new ArrayList<>();
        List<double[]> g = new ArrayList<>();
        for (int i = 0; i < sorted.size(); i++) {
            (i < good ? l : g).add(encode(sorted.get(i).config()));
        }
        double[] bwL = bandwidth(l);
        double[] bwG = bandwidth(g);
        double[] pick = null;
        double bestRatio = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < 64; c++) {
            double[] base = l.get(random.nextInt(l.size()));
            double[] u = new double[p];
            for (int d = 0; d < p; d++) {
                u[d] = Math.clamp(base[d] + 2 * bwL[d] * random.nextGaussian(), 0, 1);
            }
            double ratio = Math.log(kde(l, u, bwL) + 1e-300) - Math.log(kde(g, u, bwG) + 1e-300);
            if (ratio > bestRatio) {
                bestRatio = ratio;
                pick = u;
            }
        }
        return pick;
    }

    /** Scott's rule per dimension, floored so that a tight group still explores. */
    private static double[] bandwidth(List<double[]> pts) {
        int p = pts.getFirst().length;
        double[] bw = new double[p];
        for (int d = 0; d < p; d++) {
            double mean = 0;
            for (double[] q : pts) {
                mean += q[d] / pts.size();
            }
            double var = 0;
            for (double[] q : pts) {
                var += (q[d] - mean) * (q[d] - mean) / pts.size();
            }
            bw[d] = Math.max(0.03, 1.06 * Math.sqrt(var) * Math.pow(pts.size(), -0.2));
        }
        return bw;
    }

    private static double kde(List<double[]> pts, double[] u, double[] bw) {
        double s = 0;
        for (double[] q : pts) {
            double d2 = 0;
            double norm = 1;
            for (int i = 0; i < u.length; i++) {
                double d = (u[i] - q[i]) / bw[i];
                d2 += d * d;
                norm *= bw[i];
            }
            s += Math.exp(-0.5 * d2) / norm;
        }
        return s / pts.size();
    }

    private static Result result(List<Trial> all) {
        int top = all.stream().mapToInt(Trial::budget).max().orElse(0);
        Trial best = all.stream().filter(t -> t.budget() == top).min(Comparator.comparingDouble(Trial::score))
                .orElse(null);
        return new Result(List.copyOf(all), best);
    }
}
