package io.github.zzuegg.brainiac.experimental.sbi;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Summary statistics for approximate Bayesian computation, working on a fixed pool of prior
 * simulations so every observation reuses the same budget.
 * <ul>
 *   <li>{@link #rejection}: keep the parameters of the nearest simulations in standardised
 *       summary space;</li>
 *   <li>{@link #regressionAdjusted}: Beaumont, Zhang and Balding (Genetics 2002): weight the
 *       accepted pairs with an Epanechnikov kernel of the distance, fit the local-linear
 *       regression {@code theta = a + B (s - s_o)} by weighted least squares and replace each
 *       accepted parameter by {@code theta - B (s - s_o)}, which removes the first-order effect
 *       of the tolerance;</li>
 *   <li>{@link #semiAutomatic}: Fearnhead and Prangle (JRSS-B 2012): use fitted values of a
 *       pilot regression {@code E[theta | x]} (linear ridge, or an MLP) as the summaries, which
 *       are optimal for posterior-mean estimation;</li>
 *   <li>{@link #infomax}: a learned summary network trained to maximise the InfoNCE bound on the
 *       mutual information between {@code s(x)} and {@code theta} (Chen et al., "Neural
 *       approximate sufficient statistics", ICLR 2021), with a separable critic.</li>
 * </ul>
 */
public final class SummaryStatistics {

    private SummaryStatistics() {
    }

    /** How semi-automatic ABC regresses parameters on data. */
    public enum Regression {
        /** Ridge regression on the given features. */
        LINEAR,
        /** A small tanh MLP trained by mean squared error. */
        MLP
    }

    private static double[] scale(double[][] s) {
        return SbiArrays.sd(s, SbiArrays.mean(s));
    }

    private static Integer[] nearest(double[][] summaries, double[] observed, double[] sc, double[] dist) {
        Integer[] order = new Integer[summaries.length];
        for (int i = 0; i < summaries.length; i++) {
            double d = 0;
            for (int j = 0; j < observed.length; j++) {
                double z = (summaries[i][j] - observed[j]) / sc[j];
                d += z * z;
            }
            dist[i] = Math.sqrt(d);
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(dist[a], dist[b]));
        return order;
    }

    /**
     * Rejection ABC on a simulation pool.
     *
     * @param thetas pool parameters
     * @param summaries pool summaries
     * @param observed observed summaries
     * @param keep accepted count
     * @return accepted parameters
     */
    public static double[][] rejection(double[][] thetas, double[][] summaries, double[] observed, int keep) {
        double[] dist = new double[thetas.length];
        Integer[] order = nearest(summaries, observed, scale(summaries), dist);
        double[][] out = new double[keep][];
        for (int i = 0; i < keep; i++) {
            out[i] = thetas[order[i]].clone();
        }
        return out;
    }

    /**
     * Rejection ABC with local-linear regression adjustment, resampled by the kernel weights.
     *
     * @param thetas pool parameters
     * @param summaries pool summaries
     * @param observed observed summaries
     * @param keep accepted count (the tolerance is the distance of the last accepted)
     * @param random the random source for resampling
     * @return adjusted parameters, {@code keep} of them
     */
    public static double[][] regressionAdjusted(double[][] thetas, double[][] summaries, double[] observed, int keep,
            SplittableRandom random) {
        double[] sc = scale(summaries);
        double[] dist = new double[thetas.length];
        Integer[] order = nearest(summaries, observed, sc, dist);
        double eps = dist[order[keep - 1]] * 1.0001 + 1e-12;
        int q = observed.length;
        int d = thetas[0].length;
        double[] w = new double[keep];
        double[][] z = new double[keep][q + 1];
        for (int i = 0; i < keep; i++) {
            int k = order[i];
            double u = dist[k] / eps;
            w[i] = Math.max(1e-12, 1 - u * u);
            z[i][0] = 1;
            for (int j = 0; j < q; j++) {
                z[i][j + 1] = (summaries[k][j] - observed[j]) / sc[j];
            }
        }
        double[][] ztwz = new double[q + 1][q + 1];
        for (int i = 0; i < keep; i++) {
            for (int a = 0; a <= q; a++) {
                for (int b = 0; b <= q; b++) {
                    ztwz[a][b] += w[i] * z[i][a] * z[i][b];
                }
            }
        }
        for (int a = 1; a <= q; a++) {
            ztwz[a][a] += 1e-6 * keep;
        }
        double[][] coef = new double[d][];
        for (int c = 0; c < d; c++) {
            double[] rhs = new double[q + 1];
            for (int i = 0; i < keep; i++) {
                for (int a = 0; a <= q; a++) {
                    rhs[a] += w[i] * z[i][a] * thetas[order[i]][c];
                }
            }
            coef[c] = Matrices.solve(ztwz, rhs);
        }
        double[][] adj = new double[keep][d];
        double[] cdf = new double[keep];
        double tot = 0;
        for (int i = 0; i < keep; i++) {
            for (int c = 0; c < d; c++) {
                double v = thetas[order[i]][c];
                for (int a = 1; a <= q; a++) {
                    v -= coef[c][a] * z[i][a];
                }
                adj[i][c] = v;
            }
            tot += w[i];
            cdf[i] = tot;
        }
        double[][] out = new double[keep][];
        for (int i = 0; i < keep; i++) {
            double u = random.nextDouble() * tot;
            int k = Arrays.binarySearch(cdf, u);
            k = k < 0 ? -k - 1 : k;
            out[i] = adj[Math.min(k, keep - 1)].clone();
        }
        return out;
    }

    /**
     * Semi-automatic summaries: fitted posterior means from a regression of parameters on
     * features of the data.
     *
     * @param kind linear or MLP
     * @param thetas pilot parameters
     * @param features pilot data features
     * @param seed the seed (MLP)
     * @return the summary function {@code features -> E[theta | x]}
     */
    public static UnaryOperator<double[]> semiAutomatic(Regression kind, double[][] thetas, double[][] features,
            long seed) {
        double[] fm = SbiArrays.mean(features);
        double[] fs = SbiArrays.sd(features, fm);
        double[] tm = SbiArrays.mean(thetas);
        double[] ts = SbiArrays.sd(thetas, tm);
        int p = fm.length;
        int d = tm.length;
        if (kind == Regression.LINEAR) {
            double[][] xtx = new double[p + 1][p + 1];
            double[][] xty = new double[d][p + 1];
            for (int i = 0; i < thetas.length; i++) {
                double[] z = new double[p + 1];
                z[0] = 1;
                for (int j = 0; j < p; j++) {
                    z[j + 1] = (features[i][j] - fm[j]) / fs[j];
                }
                for (int a = 0; a <= p; a++) {
                    for (int b = 0; b <= p; b++) {
                        xtx[a][b] += z[a] * z[b];
                    }
                    for (int c = 0; c < d; c++) {
                        xty[c][a] += z[a] * thetas[i][c];
                    }
                }
            }
            for (int a = 1; a <= p; a++) {
                xtx[a][a] += 1e-3 * thetas.length;
            }
            double[][] coef = new double[d][];
            for (int c = 0; c < d; c++) {
                coef[c] = Matrices.solve(xtx, xty[c]);
            }
            return f -> {
                double[] out = new double[d];
                for (int c = 0; c < d; c++) {
                    double v = coef[c][0];
                    for (int j = 0; j < p; j++) {
                        v += coef[c][j + 1] * (f[j] - fm[j]) / fs[j];
                    }
                    out[c] = v;
                }
                return out;
            };
        }
        SimpleMlp net = SimpleMlp.tanh(seed, p, 64, 64, d);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(128, thetas.length);
        for (int it = 0; it < 600; it++) {
            int[] idx = SbiArrays.batch(thetas.length, b, r);
            Var pred = net.forward(Var.of(SbiArrays.standardisedRows(features, idx, fm, fs)));
            Var loss = pred.sub(Var.of(SbiArrays.standardisedRows(thetas, idx, tm, ts))).square().mean();
            loss.backward();
            opt.step(net.parameters());
        }
        return f -> {
            double[] z = Var.noGrad(() -> net.forward(Var.of(SbiArrays.standardised(new double[][] {f}, fm, fs))))
                    .value().toDoubleArray();
            double[] out = new double[d];
            for (int c = 0; c < d; c++) {
                out[c] = tm[c] + ts[c] * z[c];
            }
            return out;
        };
    }

    /**
     * An InfoNCE-trained summary network {@code s(x)} of dimension {@code dim}: the critic is
     * {@code s(x) . h(theta)} with an MLP {@code h}, and the loss is the contrastive
     * cross-entropy over the batch.
     *
     * @param thetas training parameters
     * @param features training data features
     * @param dim summary dimension
     * @param steps Adam steps on batches of 128
     * @param seed the seed
     * @return the summary function
     */
    public static UnaryOperator<double[]> infomax(double[][] thetas, double[][] features, int dim, int steps,
            long seed) {
        double[] fm = SbiArrays.mean(features);
        double[] fs = SbiArrays.sd(features, fm);
        double[] tm = SbiArrays.mean(thetas);
        double[] ts = SbiArrays.sd(thetas, tm);
        SimpleMlp s = SimpleMlp.tanh(seed, fm.length, 64, 64, dim);
        SimpleMlp h = SimpleMlp.tanh(seed + 1, tm.length, 64, dim);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(128, thetas.length);
        java.util.List<Var> params = new java.util.ArrayList<>(s.parameters());
        params.addAll(h.parameters());
        float[] id = new float[b * b];
        for (int i = 0; i < b; i++) {
            id[i * b + i] = 1;
        }
        Tensor eye = Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(b, b), id);
        for (int it = 0; it < steps; it++) {
            int[] idx = SbiArrays.batch(thetas.length, b, r);
            Var sx = s.forward(Var.of(SbiArrays.standardisedRows(features, idx, fm, fs)));
            Var ht = h.forward(Var.of(SbiArrays.standardisedRows(thetas, idx, tm, ts)));
            Var logits = sx.matmul(ht.transpose());
            Var diag = logits.mul(Var.of(eye)).sum(1, false);
            Var loss = diag.sub(logits.logsumexp(1, false)).mean().neg();
            loss.backward();
            opt.step(params);
        }
        return f -> Var.noGrad(() -> s.forward(Var.of(SbiArrays.standardised(new double[][] {f}, fm, fs)))).value()
                .toDoubleArray();
    }
}
