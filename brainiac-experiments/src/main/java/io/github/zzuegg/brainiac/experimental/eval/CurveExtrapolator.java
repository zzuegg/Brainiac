package io.github.zzuegg.brainiac.experimental.eval;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.LogDensity;
import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learning-curve extrapolation: predict where a training curve ends from its first few
 * percent, so a ranking (or an early-stopping decision) can be made without finishing every run.
 * Curves are given as budgets {@code x} in (0, 1] (a fraction of the target budget) and scores
 * {@code y} where <b>lower is better</b> (negate accuracies).
 * <ul>
 *   <li>{@link #powerLawLeastSquares}: the baseline, {@code y = c + a x^-α} fitted by least
 *       squares;</li>
 *   <li>{@link #ensemble}: Domhan et al.'s parametric ensemble (IJCAI 2015,
 *       <a href="https://www.ijcai.org/Proceedings/15/Papers/487.pdf">paper</a>): a weighted
 *       sum of curve families (pow3, exp3, log-power, Janoschek), each first fitted by least
 *       squares, then sampled jointly with the weights and the noise level by MCMC
 *       ({@link HamiltonianMonteCarlo#randomWalkMetropolis}), under a prior that the curve keeps
 *       improving after the last observation. The predictive samples give a mean, a spread and
 *       the probability of ending below a threshold (the early-stopping rule);</li>
 *   <li>{@link Pfn}: LC-PFN (Adriaensen et al., NeurIPS 2023,
 *       <a href="https://arxiv.org/abs/2310.20447">arXiv:2310.20447</a>): a transformer trained
 *       only on curves sampled from that parametric prior, which reads the observed points in
 *       context and outputs a discretised ("Riemann") distribution over the value at a query
 *       budget in one forward pass. It reuses the masking layout of the tabular
 *       {@code PriorFittedNetwork}: every token attends to the context points only.</li>
 * </ul>
 */
public final class CurveExtrapolator {

    private CurveExtrapolator() {
    }

    /** A parametric curve family, in unconstrained parameters. */
    public enum Family {
        /** {@code c + a x^-α}. */
        POW3(3),
        /** {@code c + a e^(-b x)}. */
        EXP3(3),
        /** {@code c + a / (1 + (x / e^b)^α)}. */
        LOG_POWER(4),
        /** Janoschek: {@code α - (α - β) e^(-κ x^δ)}. */
        JANOSCHEK(4);

        private final int parameters;

        Family(int parameters) {
            this.parameters = parameters;
        }

        /** @return the number of parameters */
        public int parameters() {
            return parameters;
        }

        /**
         * Evaluates the family.
         *
         * @param p the parameters (positive ones in log space)
         * @param offset where this family's parameters start in {@code p}
         * @param x the budget
         * @return the curve value
         */
        public double value(double[] p, int offset, double x) {
            return switch (this) {
                case POW3 -> p[offset] + Math.exp(p[offset + 1]) * Math.pow(x, -Math.exp(p[offset + 2]));
                case EXP3 -> p[offset] + Math.exp(p[offset + 1]) * Math.exp(-Math.exp(p[offset + 2]) * x);
                case LOG_POWER -> p[offset] + Math.exp(p[offset + 1])
                        / (1 + Math.pow(x / Math.exp(p[offset + 2]), Math.exp(p[offset + 3])));
                case JANOSCHEK -> p[offset] + Math.exp(p[offset + 1])
                        * Math.exp(-Math.exp(p[offset + 2]) * Math.pow(x, Math.exp(p[offset + 3])));
            };
        }
    }

    /**
     * A fitted curve.
     *
     * @param family the family
     * @param parameters its parameters
     * @param sse the sum of squared residuals on the observed points
     */
    public record Fit(Family family, double[] parameters, double sse) {
        /**
         * The curve at a budget.
         *
         * @param x the budget
         * @return the value
         */
        public double predict(double x) {
            return family.value(parameters, 0, x);
        }
    }

    /**
     * A predictive distribution of the curve value at a query budget.
     *
     * @param mean the predictive mean
     * @param sd the predictive standard deviation
     * @param samples predictive samples (or Riemann bin centres repeated by mass)
     */
    public record Prediction(double mean, double sd, double[] samples) {
        /**
         * The probability of ending below a threshold.
         *
         * @param threshold the score to beat
         * @return the share of predictive mass below it
         */
        public double probabilityBelow(double threshold) {
            int n = 0;
            for (double s : samples) {
                n += s < threshold ? 1 : 0;
            }
            return samples.length == 0 ? 0 : n / (double) samples.length;
        }
    }

    /**
     * Least-squares fit of one family, by Nelder–Mead from several starts, on y scaled to the
     * observed range.
     *
     * @param family the family
     * @param x budgets
     * @param y scores (lower is better)
     * @return the best fit, in the original scale of y
     */
    public static Fit fit(Family family, double[] x, double[] y) {
        double lo = Arrays.stream(y).min().orElse(0);
        double range = Math.max(1e-12, Arrays.stream(y).max().orElse(1) - lo);
        double[] z = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            z[i] = (y[i] - lo) / range;
        }
        double[] best = null;
        double bestSse = Double.MAX_VALUE;
        SplittableRandom r = new SplittableRandom(family.ordinal() + 17L);
        for (int start = 0; start < 6; start++) {
            double[] p0 = initial(family, r, start);
            double[] p = nelderMead(q -> sse(family, q, x, z), p0, 600);
            double s = sse(family, p, x, z);
            if (s < bestSse) {
                bestSse = s;
                best = p;
            }
        }
        double[] scaled = best.clone();
        scaled[0] = lo + range * best[0];
        scaled[1] = best[1] + Math.log(range);
        return new Fit(family, scaled, bestSse * range * range);
    }

    /**
     * The baseline: a least-squares power law {@code c + a x^-α}.
     *
     * @param x budgets
     * @param y scores (lower is better)
     * @return the fit
     */
    public static Fit powerLawLeastSquares(double[] x, double[] y) {
        return fit(Family.POW3, x, y);
    }

    private static double[] initial(Family f, SplittableRandom r, int start) {
        double[] p = new double[f.parameters()];
        p[0] = start == 0 ? -0.2 : -0.6 * r.nextDouble();
        p[1] = Math.log(0.2 + r.nextDouble());
        for (int i = 2; i < p.length; i++) {
            p[i] = start == 0 ? -0.5 : r.nextDouble() * 3 - 2;
        }
        return p;
    }

    private static double sse(Family f, double[] p, double[] x, double[] y) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            double e = f.value(p, 0, x[i]) - y[i];
            s += e * e;
        }
        return Double.isFinite(s) ? s : Double.MAX_VALUE;
    }

    /** A function of a parameter vector. */
    @FunctionalInterface
    private interface Objective {
        double value(double[] p);
    }

    private static double[] nelderMead(Objective f, double[] start, int iterations) {
        int d = start.length;
        double[][] s = new double[d + 1][];
        double[] v = new double[d + 1];
        s[0] = start.clone();
        for (int i = 0; i < d; i++) {
            s[i + 1] = start.clone();
            s[i + 1][i] += 0.5;
        }
        for (int i = 0; i <= d; i++) {
            v[i] = f.value(s[i]);
        }
        for (int it = 0; it < iterations; it++) {
            Integer[] o = new Integer[d + 1];
            for (int i = 0; i <= d; i++) {
                o[i] = i;
            }
            final double[] vv = v;
            Arrays.sort(o, (a, b) -> Double.compare(vv[a], vv[b]));
            double[][] ss = new double[d + 1][];
            double[] sv = new double[d + 1];
            for (int i = 0; i <= d; i++) {
                ss[i] = s[o[i]];
                sv[i] = v[o[i]];
            }
            s = ss;
            v = sv;
            double[] c = new double[d];
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    c[j] += s[i][j] / d;
                }
            }
            double[] xr = lerp(c, s[d], -1);
            double fr = f.value(xr);
            if (fr < v[0]) {
                double[] xe = lerp(c, s[d], -2);
                double fe = f.value(xe);
                if (fe < fr) {
                    s[d] = xe;
                    v[d] = fe;
                } else {
                    s[d] = xr;
                    v[d] = fr;
                }
            } else if (fr < v[d - 1]) {
                s[d] = xr;
                v[d] = fr;
            } else {
                double[] xc = lerp(c, s[d], 0.5);
                double fc = f.value(xc);
                if (fc < v[d]) {
                    s[d] = xc;
                    v[d] = fc;
                } else {
                    for (int i = 1; i <= d; i++) {
                        s[i] = lerp(s[0], s[i], 0.5);
                        v[i] = f.value(s[i]);
                    }
                }
            }
        }
        int best = 0;
        for (int i = 1; i <= d; i++) {
            best = v[i] < v[best] ? i : best;
        }
        return s[best];
    }

    private static double[] lerp(double[] c, double[] x, double t) {
        double[] out = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            out[i] = c[i] + t * (x[i] - c[i]);
        }
        return out;
    }

    /**
     * Domhan et al.'s ensemble: least-squares fits of every family as the start of a joint
     * random-walk Metropolis chain over all family parameters, softmax weights and the log
     * noise level; the likelihood is Gaussian around the weighted sum, and the prior rejects
     * combined curves that get worse after the last observation.
     *
     * @param x budgets, increasing
     * @param y scores (lower is better)
     * @param query the budget to predict
     * @param samples kept MCMC samples (as many again are warm-up)
     * @param seed the chain seed
     * @return the predictive distribution at {@code query}
     */
    public static Prediction ensemble(double[] x, double[] y, double query, int samples, long seed) {
        double lo = Arrays.stream(y).min().orElse(0);
        double range = Math.max(1e-12, Arrays.stream(y).max().orElse(1) - lo);
        double[] z = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            z[i] = (y[i] - lo) / range;
        }
        Family[] families = Family.values();
        int dim = 0;
        int[] offsets = new int[families.length];
        for (int k = 0; k < families.length; k++) {
            offsets[k] = dim;
            dim += families[k].parameters();
        }
        int weights = dim;
        dim += families.length + 1;
        double[] start = new double[dim];
        double[] sses = new double[families.length];
        for (int k = 0; k < families.length; k++) {
            Fit f = fit(families[k], x, z);
            System.arraycopy(f.parameters(), 0, start, offsets[k], families[k].parameters());
            sses[k] = f.sse();
        }
        double minSse = Arrays.stream(sses).min().orElse(0);
        for (int k = 0; k < families.length; k++) {
            start[weights + k] = -(sses[k] - minSse) / Math.max(1e-6, 2 * minSse / x.length + 1e-4);
        }
        start[dim - 1] = Math.log(Math.sqrt(minSse / x.length) + 1e-3);
        double xLast = x[x.length - 1];
        final int nw = weights;
        final int d = dim;
        LogDensity target = (p, gradient) -> {
            double sigma = Math.exp(p[d - 1]);
            if (sigma < 1e-4 || sigma > 1) {
                return Double.NEGATIVE_INFINITY;
            }
            double ll = 0;
            for (int i = 0; i < x.length; i++) {
                double e = combined(families, offsets, nw, p, x[i]) - z[i];
                ll -= e * e / (2 * sigma * sigma) + Math.log(sigma);
            }
            double prev = combined(families, offsets, nw, p, xLast);
            for (int k = 1; k <= 4; k++) {
                double next = combined(families, offsets, nw, p, xLast + (query - xLast) * k / 4);
                if (!(next <= prev + 1e-9) || next < -3) {
                    return Double.NEGATIVE_INFINITY;
                }
                prev = next;
            }
            double prior = 0;
            for (int i = 0; i < d - 1; i++) {
                prior -= p[i] * p[i] / (2 * 100);
            }
            return Double.isFinite(ll) ? ll + prior : Double.NEGATIVE_INFINITY;
        };
        if (!Double.isFinite(target.value(start))) {
            Fit pow = powerLawLeastSquares(x, y);
            double v = Math.min(pow.predict(query), y[y.length - 1]);
            return new Prediction(v, 0, new double[] {v});
        }
        HamiltonianMonteCarlo.Result chain = HamiltonianMonteCarlo.randomWalkMetropolis(target, start, 0.02,
                samples, samples, new SplittableRandom(seed));
        double[] pred = new double[chain.samples().length];
        double mean = 0;
        for (int i = 0; i < pred.length; i++) {
            pred[i] = lo + range * combined(families, offsets, nw, chain.samples()[i], query);
            mean += pred[i] / pred.length;
        }
        double var = 0;
        for (double v : pred) {
            var += (v - mean) * (v - mean) / pred.length;
        }
        return new Prediction(mean, Math.sqrt(var), pred);
    }

    private static double combined(Family[] families, int[] offsets, int weights, double[] p, double x) {
        double max = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < families.length; k++) {
            max = Math.max(max, p[weights + k]);
        }
        double norm = 0;
        double sum = 0;
        for (int k = 0; k < families.length; k++) {
            double w = Math.exp(p[weights + k] - max);
            norm += w;
            sum += w * families[k].value(p, offsets[k], x);
        }
        return sum / norm;
    }

    /**
     * Samples a curve from the parametric prior that LC-PFN is trained on: a random family with
     * random parameters, on scores in roughly [0, 1], plus Gaussian noise.
     *
     * @param x the budgets to evaluate
     * @param random the random source
     * @return {@code [x.length + 1]}: noisy values at {@code x}, then the noise-free value at 1
     */
    public static double[] samplePriorCurve(double[] x, SplittableRandom random) {
        Family f = Family.values()[random.nextInt(Family.values().length)];
        double[] p = new double[f.parameters()];
        p[0] = random.nextDouble() * 0.5;
        p[1] = Math.log(0.05 + random.nextDouble());
        switch (f) {
            case POW3 -> p[2] = Math.log(0.1 + random.nextDouble());
            case EXP3 -> p[2] = Math.log(0.5 + 8 * random.nextDouble());
            case LOG_POWER -> {
                p[2] = Math.log(0.02 + 0.5 * random.nextDouble());
                p[3] = Math.log(0.5 + 2 * random.nextDouble());
            }
            case JANOSCHEK -> {
                p[2] = Math.log(0.5 + 6 * random.nextDouble());
                p[3] = Math.log(0.3 + 1.5 * random.nextDouble());
            }
        }
        double noise = 0.03 * random.nextDouble() * Math.exp(p[1]);
        double[] out = new double[x.length + 1];
        for (int i = 0; i < x.length; i++) {
            out[i] = f.value(p, 0, x[i]) + noise * random.nextGaussian();
        }
        out[x.length] = f.value(p, 0, 1.0);
        return out;
    }

    /**
     * LC-PFN: a prior-fitted network for curve extrapolation. Observed curves are resampled to
     * a fixed number of context points; y is normalised by the first and last observed values
     * ({@code (y - y_last) / |y_first - y_last|}); every context token embeds {@code (x, y)},
     * every query token embeds {@code (x, ·)} plus a learned query vector; core
     * {@link TransformerBlock}s run with a mask that lets every token attend to context tokens
     * only; a head gives logits over equal-width bins of normalised y (a Riemann distribution).
     */
    public static final class Pfn {

        private static final float MASKED = -1e9f;
        private static final double LOW = -2.5;
        private static final double HIGH = 1.5;

        private final int context;
        private final int bins;
        private final int dim;
        private final Var embed;
        private final Var embedBias;
        private final Var queryToken;
        private final List<TransformerBlock> blocks = new ArrayList<>();
        private final LayerNorm norm;
        private final Var head;
        private final Var headBias;
        private final Optimizer optimizer;

        /**
         * Creates an untrained LC-PFN.
         *
         * @param context context points per curve
         * @param bins output bins
         * @param dim the token width
         * @param blocks transformer blocks
         * @param lr the Adam learning rate
         * @param random the initialisation source
         */
        public Pfn(int context, int bins, int dim, int blocks, float lr, SplittableRandom random) {
            this.context = context;
            this.bins = bins;
            this.dim = dim;
            this.embed = Var.param(Tensor.normal(Shape.of(2, dim), 0, 1f, random));
            this.embedBias = Var.param(Tensor.zeros(Shape.of(dim)));
            this.queryToken = Var.param(Tensor.normal(Shape.of(dim), 0, 1f, random));
            for (int b = 0; b < blocks; b++) {
                this.blocks.add((TransformerBlock) TransformerBlock.of(2).ffnSize(2 * dim)
                        .create(Shape.of(8, dim), random));
            }
            this.norm = LayerNorm.of(dim);
            this.head = Var.param(Tensor.normal(Shape.of(dim, bins), 0, (float) (1 / Math.sqrt(dim)), random));
            this.headBias = Var.param(Tensor.zeros(Shape.of(bins)));
            this.optimizer = Optimizer.adam(lr);
        }

        /** @return the trainable parameters */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(embed, embedBias, queryToken));
            for (TransformerBlock b : blocks) {
                p.addAll(b.parameters());
            }
            p.addAll(norm.parameters());
            p.add(head);
            p.add(headBias);
            return p;
        }

        private double[][] resample(double[] x, double[] y) {
            double[][] out = new double[2][context];
            for (int i = 0; i < context; i++) {
                double t = x[0] + (x[x.length - 1] - x[0]) * i / (context - 1.0);
                int j = 0;
                while (j + 1 < x.length - 1 && x[j + 1] < t) {
                    j++;
                }
                int k = Math.min(x.length - 1, j + 1);
                double w = x[k] == x[j] ? 0 : (t - x[j]) / (x[k] - x[j]);
                out[0][i] = t;
                out[1][i] = y[j] + w * (y[k] - y[j]);
            }
            return out;
        }

        private static double scale(double[] y) {
            return Math.max(1e-9, Math.abs(y[0] - y[y.length - 1]));
        }

        private Var logits(List<double[][]> curves, double[] queries, boolean training) {
            int b = curves.size();
            int q = queries.length;
            int rows = context + q;
            float[] in = new float[b * rows * 2];
            float[] isQuery = new float[b * rows];
            for (int t = 0; t < b; t++) {
                double[][] c = curves.get(t);
                for (int i = 0; i < context; i++) {
                    in[(t * rows + i) * 2] = (float) c[0][i];
                    in[(t * rows + i) * 2 + 1] = (float) c[1][i];
                }
                for (int i = 0; i < q; i++) {
                    in[(t * rows + context + i) * 2] = (float) queries[i];
                    isQuery[t * rows + context + i] = 1;
                }
            }
            Var h = Var.of(Tensor.of(Shape.of(b, rows, 2), in)).matmul(embed).add(embedBias)
                    .add(Var.of(Tensor.of(Shape.of(b, rows, 1), isQuery)).mul(queryToken.reshape(1, 1, dim)));
            float[] mask = new float[rows * rows];
            for (int i = 0; i < rows; i++) {
                for (int j = context; j < rows; j++) {
                    mask[i * rows + j] = MASKED;
                }
            }
            Tensor bias = Tensor.of(Shape.of(1, 1, rows, rows), mask);
            for (TransformerBlock block : blocks) {
                h = block.forward(List.of(h), bias, training);
            }
            Var out = h.slice(1, context, rows).reshape(b * q, dim);
            return norm.forward(out).matmul(head).add(headBias);
        }

        private int bin(double z) {
            int k = (int) Math.floor((z - LOW) / (HIGH - LOW) * bins);
            return Math.max(0, Math.min(bins - 1, k));
        }

        private double centre(int k) {
            return LOW + (k + 0.5) * (HIGH - LOW) / bins;
        }

        /**
         * Pre-trains on curves from {@link #samplePriorCurve}, each cut at a random 5-30% of the
         * target budget and observed at {@code 4 * context} points, with the query at 1.
         *
         * @param steps Adam steps
         * @param batch curves per step
         * @param seed the prior's seed
         * @return the mean cross-entropy over the last 10% of steps
         */
        public double pretrain(int steps, int batch, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            double tail = 0;
            int tailCount = 0;
            for (int s = 0; s < steps; s++) {
                List<double[][]> curves = new ArrayList<>();
                float[] targets = new float[batch];
                for (int t = 0; t < batch; t++) {
                    double cut = 0.05 + 0.25 * r.nextDouble();
                    int n = 4 * context;
                    double[] x = new double[n];
                    for (int i = 0; i < n; i++) {
                        x[i] = cut * (i + 1) / n;
                    }
                    double[] full = samplePriorCurve(x, r);
                    double[] y = Arrays.copyOf(full, n);
                    double[][] c = resample(x, y);
                    double sc = scale(c[1]);
                    double last = c[1][context - 1];
                    for (int i = 0; i < context; i++) {
                        c[1][i] = (c[1][i] - last) / sc;
                    }
                    curves.add(c);
                    targets[t] = bin((full[n] - last) / sc);
                }
                Var loss = Functional.crossEntropy(logits(curves, new double[] {1.0}, true),
                        Tensor.of(Shape.of(batch), targets));
                loss.backward();
                optimizer.step(parameters());
                if (s >= steps - Math.max(1, steps / 10)) {
                    tail += loss.value().toArray()[0];
                    tailCount++;
                }
            }
            return tail / tailCount;
        }

        /**
         * Extrapolates one curve in a single forward pass.
         *
         * @param x observed budgets, increasing
         * @param y observed scores (lower is better)
         * @param query the budget to predict
         * @return the Riemann predictive distribution, as 200 quantile samples
         */
        public Prediction predict(double[] x, double[] y, double query) {
            double[][] c = resample(x, y);
            double sc = scale(c[1]);
            double last = c[1][context - 1];
            for (int i = 0; i < context; i++) {
                c[1][i] = (c[1][i] - last) / sc;
            }
            float[] l = Var.noGrad(() -> logits(List.<double[][]>of(c), new double[] {query}, false)).value().toArray();
            double max = Double.NEGATIVE_INFINITY;
            for (float v : l) {
                max = Math.max(max, v);
            }
            double[] p = new double[bins];
            double norm = 0;
            for (int k = 0; k < bins; k++) {
                p[k] = Math.exp(l[k] - max);
                norm += p[k];
            }
            double mean = 0;
            double sq = 0;
            for (int k = 0; k < bins; k++) {
                p[k] /= norm;
                double v = last + sc * centre(k);
                mean += p[k] * v;
                sq += p[k] * v * v;
            }
            double[] samples = new double[200];
            double cum = 0;
            int k = 0;
            for (int i = 0; i < samples.length; i++) {
                double u = (i + 0.5) / samples.length;
                while (k < bins - 1 && cum + p[k] < u) {
                    cum += p[k];
                    k++;
                }
                samples[i] = last + sc * centre(k);
            }
            return new Prediction(mean, Math.sqrt(Math.max(0, sq - mean * mean)), samples);
        }

        @Override
        public String toString() {
            return "LC-PFN(" + context + " context points, " + bins + " bins, " + dim + " wide, " + blocks.size()
                    + " blocks)";
        }
    }
}
