package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.circuits.InceptionCircuit;
import io.github.zzuegg.brainiac.experimental.circuits.LearnSpn;
import io.github.zzuegg.brainiac.experimental.circuits.SquaredCircuit;
import io.github.zzuegg.brainiac.experimental.circuits.TensorizedCircuit;
import io.github.zzuegg.brainiac.experimental.ml.DirichletProcessMixture;
import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.VariationalGaussianMixture;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 11 items 11.3.11 to 11.3.13: probabilistic circuits and variational mixtures.
 * <ul>
 *   <li>{@link #digits} (11.3.11): binarised 3x5 digit glyphs on a 4x6 canvas (random shift, 5%
 *       pixel flips) with the label as a 25th variable. {@link TensorizedCircuit} (EM and Adam)
 *       and {@link LearnSpn} against a Bernoulli mixture fitted by EM and independent pixels:
 *       test log-likelihood of the pixels (label marginalised), and classification by conditional
 *       inference with 50% of the pixels missing, against mean imputation plus logistic
 *       regression trained on complete data.</li>
 *   <li>{@link #marginals} (11.3.11): a random discrete Bayesian network over 8 three-state
 *       variables; the circuit's one-pass marginals with half the variables missing against
 *       brute-force summation of its own joint (exactness), and both against the true
 *       marginals, with the time per query.</li>
 *   <li>{@link #squared} (11.3.12): at matched parameter counts, the monotone circuit (one tree,
 *       EM) against real and complex {@link SquaredCircuit}s and an {@link InceptionCircuit}, on a
 *       planted distribution with cancellation, {@code p(s) proportional to (c + sum_i w_i s_i)^2}
 *       over {@code s in {-1, 1}^10}, and on the digits.</li>
 *   <li>{@link #clustering} (11.3.13): blobs with 3 to 10 clusters, unequal sizes and a diffuse
 *       outlier group: one {@link VariationalGaussianMixture} fit against
 *       {@link GaussianMixture#selectByBic}, collapsed Gibbs for a Dirichlet-process mixture and
 *       k-means with the elbow rule, by ARI, the error in the number of clusters, held-out
 *       log-likelihood (plug-in for VB) and runtime.</li>
 * </ul>
 * {@code RealNvpFlow} and {@code PixelCnn} likelihoods and {@code FactorGraph} belief
 * propagation were not run at this budget.
 */
public final class CircuitShootout {

    private CircuitShootout() {
    }

    private static final String[] FONT = {
        "111101101101111", "010110010010111", "111001111100111", "111001111001111", "101101111001001",
        "111100111001111", "111100111101111", "111001001001001", "111101111101111", "111101111001111"};

    /**
     * Binarised digits with the label as the last column.
     *
     * @param n samples
     * @param flip the pixel flip probability
     * @param seed the seed
     * @return {@code [n][25]} category indices
     */
    public static int[][] digitData(int n, double flip, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[][] out = new int[n][25];
        for (int i = 0; i < n; i++) {
            int digit = r.nextInt(10);
            int dx = r.nextInt(2);
            int dy = r.nextInt(2);
            for (int y = 0; y < 5; y++) {
                for (int x = 0; x < 3; x++) {
                    if (FONT[digit].charAt(y * 3 + x) == '1') {
                        out[i][(y + dy) * 4 + x + dx] = 1;
                    }
                }
            }
            for (int p = 0; p < 24; p++) {
                if (r.nextDouble() < flip) {
                    out[i][p] = 1 - out[i][p];
                }
            }
            out[i][24] = digit;
        }
        return out;
    }

    private static double[][] asDouble(int[][] x) {
        double[][] o = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                o[i][j] = x[i][j] < 0 ? Double.NaN : x[i][j];
            }
        }
        return o;
    }

    /** A mixture of independent categoricals fitted by EM, with exact marginals. */
    static final class CategoricalMixture {
        final double[] pi;
        final double[][][] p;

        CategoricalMixture(int[][] x, int[] cats, int k, int iterations, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            int d = cats.length;
            pi = new double[k];
            Arrays.fill(pi, 1.0 / k);
            p = new double[k][d][];
            for (int j = 0; j < k; j++) {
                for (int v = 0; v < d; v++) {
                    p[j][v] = new double[cats[v]];
                    double s = 0;
                    for (int c = 0; c < cats[v]; c++) {
                        p[j][v][c] = 0.5 + r.nextDouble();
                        s += p[j][v][c];
                    }
                    for (int c = 0; c < cats[v]; c++) {
                        p[j][v][c] /= s;
                    }
                }
            }
            for (int it = 0; it < iterations; it++) {
                double[] nk = new double[k];
                double[][][] cnt = new double[k][d][];
                for (int j = 0; j < k; j++) {
                    for (int v = 0; v < d; v++) {
                        cnt[j][v] = new double[cats[v]];
                    }
                }
                for (int[] row : x) {
                    double[] post = posterior(row);
                    for (int j = 0; j < k; j++) {
                        nk[j] += post[j];
                        for (int v = 0; v < d; v++) {
                            if (row[v] >= 0) {
                                cnt[j][v][row[v]] += post[j];
                            }
                        }
                    }
                }
                for (int j = 0; j < k; j++) {
                    pi[j] = (nk[j] + 0.01) / (x.length + 0.01 * k);
                    for (int v = 0; v < d; v++) {
                        double s = 0;
                        for (double c : cnt[j][v]) {
                            s += c + 0.1;
                        }
                        for (int c = 0; c < cats[v]; c++) {
                            p[j][v][c] = (cnt[j][v][c] + 0.1) / s;
                        }
                    }
                }
            }
        }

        double[] logJoint(int[] row) {
            double[] lp = new double[pi.length];
            for (int j = 0; j < pi.length; j++) {
                lp[j] = Math.log(pi[j]);
                for (int v = 0; v < row.length; v++) {
                    if (row[v] >= 0) {
                        lp[j] += Math.log(p[j][v][row[v]]);
                    }
                }
            }
            return lp;
        }

        double[] posterior(int[] row) {
            double[] lp = logJoint(row);
            double max = Arrays.stream(lp).max().orElse(0);
            double s = 0;
            for (int j = 0; j < lp.length; j++) {
                lp[j] = Math.exp(lp[j] - max);
                s += lp[j];
            }
            for (int j = 0; j < lp.length; j++) {
                lp[j] /= s;
            }
            return lp;
        }

        double logLikelihood(int[] row) {
            double[] lp = logJoint(row);
            double max = Arrays.stream(lp).max().orElse(0);
            double s = 0;
            for (double v : lp) {
                s += Math.exp(v - max);
            }
            return max + Math.log(s);
        }

        int classify(int[] row, int label, int classes) {
            int[] q = row.clone();
            int best = 0;
            double bl = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < classes; c++) {
                q[label] = c;
                double ll = logLikelihood(q);
                if (ll > bl) {
                    bl = ll;
                    best = c;
                }
            }
            return best;
        }
    }

    private static int[][] hide(int[][] x, int columns, double share, boolean hideLabel, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[][] out = new int[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i].clone();
            for (int v = 0; v < columns; v++) {
                if (r.nextDouble() < share) {
                    out[i][v] = -1;
                }
            }
            if (hideLabel) {
                out[i][x[i].length - 1] = -1;
            }
        }
        return out;
    }

    /**
     * Digits: likelihood and classification with missing pixels.
     *
     * @param train training samples
     * @param test test samples
     * @param units circuit units per region
     * @param replicas circuit trees
     * @param emIterations EM iterations
     * @param seed the seed
     * @return method to {@code {test pixel log-likelihood, accuracy with 50% pixels missing,
     *     complete-input accuracy}}
     */
    public static Map<String, double[]> digits(int train, int test, int units, int replicas, int emIterations,
            long seed) {
        int[][] tr = digitData(train, 0.05, seed);
        int[][] te = digitData(test, 0.05, seed + 1);
        int[] cats = new int[25];
        Arrays.fill(cats, 2);
        cats[24] = 10;
        int[][] pixelsOnly = hide(te, 0, 0, true, seed);
        int[][] half = hide(te, 24, 0.5, true, seed + 2);
        Map<String, double[]> out = new LinkedHashMap<>();
        TensorizedCircuit em = new TensorizedCircuit(cats, units, replicas, seed).initialise(asDouble(tr), seed).fitEm(asDouble(tr), emIterations,
                0.01);
        out.put("tensorised circuit (EM)", circuitScores(em, pixelsOnly, half, te));
        TensorizedCircuit adam = new TensorizedCircuit(cats, units, replicas, seed + 5)
                .initialise(asDouble(tr), seed + 5).fitAdam(asDouble(tr), 100, 64, 0.05, seed);
        out.put("tensorised circuit (Adam)", circuitScores(adam, pixelsOnly, half, te));
        LearnSpn spn = LearnSpn.learn(tr, cats, 1e-3, 30, seed);
        double ll = 0;
        int ok = 0;
        int okFull = 0;
        for (int i = 0; i < te.length; i++) {
            ll += spn.logLikelihood(pixelsOnly[i]) / te.length;
            ok += spn.classify(half[i], 24) == te[i][24] ? 1 : 0;
            okFull += spn.classify(pixelsOnly[i], 24) == te[i][24] ? 1 : 0;
        }
        out.put("LearnSPN", new double[] {ll, ok / (double) te.length, okFull / (double) te.length});
        CategoricalMixture mix = new CategoricalMixture(tr, cats, 20, 40, seed);
        ll = 0;
        ok = 0;
        okFull = 0;
        for (int i = 0; i < te.length; i++) {
            ll += mix.logLikelihood(pixelsOnly[i]) / te.length;
            ok += mix.classify(half[i], 24, 10) == te[i][24] ? 1 : 0;
            okFull += mix.classify(pixelsOnly[i], 24, 10) == te[i][24] ? 1 : 0;
        }
        out.put("Bernoulli mixture (EM, 20)", new double[] {ll, ok / (double) te.length, okFull / (double) te.length});
        CategoricalMixture indep = new CategoricalMixture(tr, cats, 1, 2, seed);
        ll = 0;
        for (int[] row : pixelsOnly) {
            ll += indep.logLikelihood(row) / te.length;
        }
        out.put("independent pixels", new double[] {ll, Double.NaN, Double.NaN});
        out.put("mean imputation + logistic regression", imputationBaseline(tr, te, half));
        return out;
    }

    private static double[] circuitScores(TensorizedCircuit c, int[][] pixelsOnly, int[][] half, int[][] te) {
        double ll = 0;
        int ok = 0;
        int okFull = 0;
        for (int i = 0; i < te.length; i++) {
            ll += c.logLikelihood(asDouble(new int[][] {pixelsOnly[i]})[0]) / te.length;
            ok += c.classify(asDouble(new int[][] {half[i]})[0], 24) == te[i][24] ? 1 : 0;
            okFull += c.classify(asDouble(new int[][] {pixelsOnly[i]})[0], 24) == te[i][24] ? 1 : 0;
        }
        return new double[] {ll, ok / (double) te.length, okFull / (double) te.length};
    }

    /** Multinomial logistic regression on complete pixels; missing pixels are mean-imputed. */
    private static double[] imputationBaseline(int[][] tr, int[][] te, int[][] half) {
        int d = 24;
        int k = 10;
        double[] mean = new double[d];
        for (int[] row : tr) {
            for (int v = 0; v < d; v++) {
                mean[v] += row[v] / (double) tr.length;
            }
        }
        double[][] w = new double[k][d + 1];
        for (int it = 0; it < 300; it++) {
            double[][] g = new double[k][d + 1];
            for (int[] row : tr) {
                double[] p = softmax(w, row, mean);
                for (int c = 0; c < k; c++) {
                    double e = p[c] - (row[24] == c ? 1 : 0);
                    g[c][d] += e;
                    for (int v = 0; v < d; v++) {
                        g[c][v] += e * row[v];
                    }
                }
            }
            for (int c = 0; c < k; c++) {
                for (int v = 0; v <= d; v++) {
                    w[c][v] -= 0.5 * g[c][v] / tr.length + 1e-4 * w[c][v];
                }
            }
        }
        int ok = 0;
        int okFull = 0;
        for (int i = 0; i < te.length; i++) {
            ok += argmax(softmax(w, half[i], mean)) == te[i][24] ? 1 : 0;
            okFull += argmax(softmax(w, te[i], mean)) == te[i][24] ? 1 : 0;
        }
        return new double[] {Double.NaN, ok / (double) te.length, okFull / (double) te.length};
    }

    private static double[] softmax(double[][] w, int[] row, double[] mean) {
        int k = w.length;
        int d = mean.length;
        double[] z = new double[k];
        for (int c = 0; c < k; c++) {
            z[c] = w[c][d];
            for (int v = 0; v < d; v++) {
                z[c] += w[c][v] * (row[v] < 0 ? mean[v] : row[v]);
            }
        }
        double max = Arrays.stream(z).max().orElse(0);
        double s = 0;
        for (int c = 0; c < k; c++) {
            z[c] = Math.exp(z[c] - max);
            s += z[c];
        }
        for (int c = 0; c < k; c++) {
            z[c] /= s;
        }
        return z;
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) {
            b = v[i] > v[b] ? i : b;
        }
        return b;
    }

    /**
     * Exact marginals on a small discrete Bayesian network.
     *
     * @param queries marginal queries with half the variables missing
     * @param seed the seed
     * @return {@code {max |circuit marginal - brute-force sum of the circuit|, mean |circuit
     *     marginal - true marginal| (log scale), mean |mixture marginal - true| (log scale), ms per
     *     circuit query, ms per brute-force query}}
     */
    public static double[] marginals(int queries, long seed) {
        int d = 8;
        int c = 3;
        SplittableRandom r = new SplittableRandom(seed);
        // chain-plus-skip network: x_v depends on x_{v-1} and x_{v-2}
        double[][][][] cpt = new double[d][c][c][c];
        for (int v = 0; v < d; v++) {
            for (int a = 0; a < c; a++) {
                for (int b = 0; b < c; b++) {
                    double s = 0;
                    for (int k = 0; k < c; k++) {
                        cpt[v][a][b][k] = Math.pow(r.nextDouble(), 3) + 0.02;
                        s += cpt[v][a][b][k];
                    }
                    for (int k = 0; k < c; k++) {
                        cpt[v][a][b][k] /= s;
                    }
                }
            }
        }
        int states = (int) Math.pow(c, d);
        double[] joint = new double[states];
        for (int s = 0; s < states; s++) {
            int[] x = decode(s, d, c);
            double p = 1;
            for (int v = 0; v < d; v++) {
                p *= cpt[v][v > 0 ? x[v - 1] : 0][v > 1 ? x[v - 2] : 0][x[v]];
            }
            joint[s] = p;
        }
        int n = 3000;
        int[][] data = new int[n][];
        double[] cdf = new double[states];
        double acc = 0;
        for (int s = 0; s < states; s++) {
            acc += joint[s];
            cdf[s] = acc;
        }
        for (int i = 0; i < n; i++) {
            int s = Arrays.binarySearch(cdf, r.nextDouble() * acc);
            data[i] = decode(s < 0 ? -s - 1 : s, d, c);
        }
        int[] cats = new int[d];
        Arrays.fill(cats, c);
        TensorizedCircuit circuit = new TensorizedCircuit(cats, 6, 2, seed).initialise(asDouble(data), seed).fitEm(asDouble(data), 30, 0.01);
        CategoricalMixture mix = new CategoricalMixture(data, cats, 12, 40, seed);
        double exactErr = 0;
        double truthErr = 0;
        double mixErr = 0;
        long circuitNs = 0;
        long bruteNs = 0;
        double[] circuitJoint = null;
        long t0 = System.nanoTime();
        circuitJoint = new double[states];
        for (int s = 0; s < states; s++) {
            circuitJoint[s] = Math.exp(circuit.logLikelihood(asDouble(new int[][] {decode(s, d, c)})[0]));
        }
        bruteNs += System.nanoTime() - t0;
        for (int q = 0; q < queries; q++) {
            int[] x = data[r.nextInt(n)].clone();
            for (int v = 0; v < d; v++) {
                if (r.nextBoolean()) {
                    x[v] = -1;
                }
            }
            t0 = System.nanoTime();
            double lc = circuit.logLikelihood(asDouble(new int[][] {x})[0]);
            circuitNs += System.nanoTime() - t0;
            t0 = System.nanoTime();
            double brute = 0;
            double truth = 0;
            for (int s = 0; s < states; s++) {
                if (matches(decode(s, d, c), x)) {
                    brute += circuitJoint[s];
                    truth += joint[s];
                }
            }
            bruteNs += System.nanoTime() - t0;
            exactErr = Math.max(exactErr, Math.abs(lc - Math.log(brute)));
            truthErr += Math.abs(lc - Math.log(truth)) / queries;
            mixErr += Math.abs(mix.logLikelihood(x) - Math.log(truth)) / queries;
        }
        return new double[] {exactErr, truthErr, mixErr, circuitNs / 1e6 / queries, bruteNs / 1e6 / queries};
    }

    private static int[] decode(int s, int d, int c) {
        int[] x = new int[d];
        for (int v = 0; v < d; v++) {
            x[v] = s % c;
            s /= c;
        }
        return x;
    }

    private static boolean matches(int[] full, int[] partial) {
        for (int v = 0; v < full.length; v++) {
            if (partial[v] >= 0 && partial[v] != full[v]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Samples the planted cancellation distribution {@code p(s) proportional to
     * (c + sum_i w_i s_i)^2}.
     *
     * @param d binary variables
     * @param n samples
     * @param seed the seed
     * @return {@code {int[][] samples, Double entropy in nats}}
     */
    static Object[] planted(int d, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] w = new double[d];
        for (int i = 0; i < d; i++) {
            w[i] = (r.nextBoolean() ? 1 : -1) * (0.5 + r.nextDouble());
        }
        int states = 1 << d;
        double[] p = new double[states];
        double z = 0;
        for (int s = 0; s < states; s++) {
            double f = 0.3;
            for (int i = 0; i < d; i++) {
                f += w[i] * (((s >> i) & 1) == 1 ? 1 : -1);
            }
            p[s] = f * f;
            z += p[s];
        }
        double entropy = 0;
        double[] cdf = new double[states];
        double acc = 0;
        for (int s = 0; s < states; s++) {
            p[s] /= z;
            if (p[s] > 0) {
                entropy -= p[s] * Math.log(p[s]);
            }
            acc += p[s];
            cdf[s] = acc;
        }
        int[][] x = new int[n][d];
        for (int k = 0; k < n; k++) {
            int s = Arrays.binarySearch(cdf, r.nextDouble() * acc);
            s = s < 0 ? -s - 1 : s;
            for (int i = 0; i < d; i++) {
                x[k][i] = (s >> i) & 1;
            }
        }
        return new Object[] {x, entropy};
    }

    /**
     * Monotone against squared and Inception circuits at matched sizes.
     *
     * @param planted the planted cancellation task (else the digit pixels)
     * @param units units per region
     * @param steps Adam steps for the squared models
     * @param seed the seed
     * @return method to {@code {test log-likelihood, parameters}}; the planted task adds a
     *     {@code "true distribution"} row with the negative entropy
     */
    public static Map<String, double[]> squared(boolean planted, int units, int steps, long seed) {
        int[][] tr;
        int[][] te;
        Map<String, double[]> out = new LinkedHashMap<>();
        if (planted) {
            Object[] c = planted(10, 2500, seed);
            int[][] all = (int[][]) c[0];
            tr = Arrays.copyOfRange(all, 0, 1500);
            te = Arrays.copyOfRange(all, 1500, 2500);
            out.put("true distribution", new double[] {-(double) c[1], 0});
        } else {
            int[][] all = digitData(1500, 0.05, seed);
            tr = new int[1000][];
            te = new int[500][];
            for (int i = 0; i < 1500; i++) {
                int[] row = Arrays.copyOf(all[i], 24);
                if (i < 1000) {
                    tr[i] = row;
                } else {
                    te[i - 1000] = row;
                }
            }
        }
        int d = tr[0].length;
        int[] cats = new int[d];
        Arrays.fill(cats, 2);
        TensorizedCircuit mono = new TensorizedCircuit(cats, units, 1, seed).initialise(asDouble(tr), seed).fitEm(asDouble(tr), 60, 0.01);
        out.put("monotone circuit (EM)", new double[] {mono.meanLogLikelihood(asDouble(te)), mono.parameterCount()});
        TensorizedCircuit monoBig = new TensorizedCircuit(cats, 2 * units, 1, seed).initialise(asDouble(tr), seed).fitEm(asDouble(tr), 60, 0.01);
        out.put("monotone circuit, 2K units (EM)", new double[] {monoBig.meanLogLikelihood(asDouble(te)),
            monoBig.parameterCount()});
        SquaredCircuit real = new SquaredCircuit(cats, units, false, 1, seed).fit(tr, steps, 128, 0.03f, seed);
        out.put("squared circuit (real)", new double[] {real.meanLogLikelihood(te), real.parameterCount()});
        SquaredCircuit cplx = new SquaredCircuit(cats, units, true, 1, seed).fit(tr, steps, 128, 0.03f, seed);
        out.put("squared circuit (complex)", new double[] {cplx.meanLogLikelihood(te), cplx.parameterCount()});
        InceptionCircuit inc = new InceptionCircuit(cats, units, false, 2, seed).fit(tr, steps, 128, 0.03f, seed);
        out.put("Inception circuit (U=2, real)", new double[] {inc.meanLogLikelihood(te), inc.parameterCount()});
        return out;
    }

    /**
     * Unknown-K clustering with a variational Bayesian mixture.
     *
     * @param ks true cluster counts
     * @param n points per data set
     * @param seed the seed
     * @return method to mean {@code {ARI, |K error|, held-out log-likelihood, ms}}
     */
    public static Map<String, double[]> clustering(int[] ks, int n, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        double sigma = 0.6;
        for (int k : ks) {
            double[][] centres = new double[k][];
            for (int i = 0; i < k; i++) {
                boolean ok;
                do {
                    centres[i] = new double[] {-10 + 20 * r.nextDouble(), -10 + 20 * r.nextDouble()};
                    ok = true;
                    for (int j = 0; j < i; j++) {
                        ok &= MlMatrices.squaredDistance(centres[i], centres[j]) > 16;
                    }
                } while (!ok);
            }
            double[] weight = new double[k];
            for (int i = 0; i < k; i++) {
                weight[i] = i == 0 ? 3 : 0.4 + r.nextDouble();
            }
            int[] y = new int[n];
            double[][] x = blobs(centres, weight, sigma, n, y, r);
            double[][] test = blobs(centres, weight, sigma, n, new int[n], r);
            long t0 = System.nanoTime();
            VariationalGaussianMixture vb = VariationalGaussianMixture.fit(x, 16,
                    VariationalGaussianMixture.Prior.FINITE_DIRICHLET, 1e-2, sigma * sigma, 300, seed);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            double vbLl = 0;
            for (double[] p : test) {
                vbLl += vb.logDensity(p) / test.length;
            }
            add(out, "VB-GMM (Dirichlet 0.01, K=16)", score(y, vb.labels(), k, vbLl, ms), ks.length);
            t0 = System.nanoTime();
            VariationalGaussianMixture sb = VariationalGaussianMixture.fit(x, 16,
                    VariationalGaussianMixture.Prior.STICK_BREAKING, 1, sigma * sigma, 300, seed);
            ms = (System.nanoTime() - t0) / 1_000_000;
            double sbLl = 0;
            for (double[] p : test) {
                sbLl += sb.logDensity(p) / test.length;
            }
            add(out, "VB-GMM (stick-breaking, alpha 1)", score(y, sb.labels(), k, sbLl, ms), ks.length);
            t0 = System.nanoTime();
            GaussianMixture gmm = GaussianMixture.selectByBic(x, 12, 60, seed);
            ms = (System.nanoTime() - t0) / 1_000_000;
            add(out, "GMM + BIC (K = 1..12)", score(y, gmm.predict(x), k, gmm.meanLogLikelihood(test), ms),
                    ks.length);
            t0 = System.nanoTime();
            DirichletProcessMixture.Result g = DirichletProcessMixture.gibbs(x, 1,
                    DirichletProcessMixture.Prior.of(x, sigma * sigma), 30, null, seed);
            ms = (System.nanoTime() - t0) / 1_000_000;
            add(out, "DP collapsed Gibbs", score(y, g.labels(), k, g.meanLogLikelihood(test), ms), ks.length);
            t0 = System.nanoTime();
            int ke = elbow(x, 12, seed);
            int[] km = KMeans.fit(x, ke, KMeans.Init.PLUS_PLUS, 3, 50, seed).labels();
            ms = (System.nanoTime() - t0) / 1_000_000;
            add(out, "k-means + elbow", score(y, km, k, Double.NaN, ms), ks.length);
        }
        return out;
    }

    private static double[][] blobs(double[][] centres, double[] weight, double sigma, int n, int[] labels,
            SplittableRandom r) {
        int k = centres.length;
        double total = Arrays.stream(weight).sum();
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            if (r.nextDouble() < 0.03) {
                // diffuse outlier group, labelled as its own cluster k
                labels[i] = k;
                x[i] = new double[] {-12 + 24 * r.nextDouble(), -12 + 24 * r.nextDouble()};
                continue;
            }
            double u = r.nextDouble() * total;
            int c = 0;
            while (c < k - 1 && (u -= weight[c]) > 0) {
                c++;
            }
            labels[i] = c;
            x[i] = new double[] {centres[c][0] + sigma * r.nextGaussian(), centres[c][1] + sigma * r.nextGaussian()};
        }
        return x;
    }

    private static int elbow(double[][] x, int maxK, long seed) {
        double[] inertia = new double[maxK];
        for (int k = 1; k <= maxK; k++) {
            inertia[k - 1] = KMeans.fit(x, k, KMeans.Init.PLUS_PLUS, 2, 50, seed).inertia();
        }
        int best = 1;
        double bestGap = Double.NEGATIVE_INFINITY;
        for (int k = 1; k <= maxK; k++) {
            double xn = (k - 1.0) / (maxK - 1);
            double yn = (inertia[k - 1] - inertia[maxK - 1]) / (inertia[0] - inertia[maxK - 1]);
            if ((1 - xn) - yn > bestGap) {
                bestGap = (1 - xn) - yn;
                best = k;
            }
        }
        return best;
    }

    private static double[] score(int[] truth, int[] found, int k, double heldOut, long ms) {
        int[] core = java.util.stream.IntStream.range(0, truth.length).filter(i -> truth[i] < k).toArray();
        int[] t = new int[core.length];
        int[] f = new int[core.length];
        for (int i = 0; i < core.length; i++) {
            t[i] = truth[core[i]];
            f[i] = found[core[i]];
        }
        // clusters holding at least 2% of the points count towards K
        int[] sizes = new int[Arrays.stream(found).max().orElse(0) + 1];
        for (int v : found) {
            if (v >= 0) {
                sizes[v]++;
            }
        }
        int kHat = (int) Arrays.stream(sizes).filter(s -> s >= 0.02 * found.length).count();
        return new double[] {MlMetrics.adjustedRandIndex(t, f), Math.abs(kHat - k), heldOut, ms};
    }

    private static void add(Map<String, double[]> out, String key, double[] v, int runs) {
        double[] acc = out.computeIfAbsent(key, s -> new double[v.length]);
        for (int i = 0; i < v.length; i++) {
            acc[i] += v[i] / runs;
        }
    }
}
