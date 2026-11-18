package io.github.zzuegg.brainiac.experimental.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Model extraction: an attacker with query access to a prediction {@link Api} trains a copy.
 *
 * <ul>
 *   <li>{@link #equationSolving}: a softmax (multinomial logistic) victim satisfies
 *       {@code log p_k - log p_0 = (w_k - w_0)·x + (b_k - b_0)}, so {@code d + 1} confidence
 *       queries determine every class's parameters up to the softmax's shift; least squares
 *       on a few more is exact up to rounding (Tramèr et al., USENIX Security 2016,
 *       <a href="https://arxiv.org/abs/1609.02943">arXiv:1609.02943</a>).</li>
 *   <li>{@link #jacobianAugmentation}: train a substitute on the labels of a small seed set,
 *       grow the set by {@code x + λ sign(∇_x z_{ŷ}(x))} along the substitute's Jacobian, query
 *       the new points, repeat (Papernot et al., AsiaCCS 2017,
 *       <a href="https://arxiv.org/abs/1602.02697">arXiv:1602.02697</a>).</li>
 *   <li>{@link #knockoff}: query natural images from another distribution and train on the
 *       returned probabilities, choosing queries at random or adaptively (highest entropy of
 *       the current copy, in rounds) (Orekondy, Schiele and Fritz, CVPR 2019,
 *       <a href="https://arxiv.org/abs/1812.02766">arXiv:1812.02766</a>).</li>
 * </ul>
 */
public final class ModelExtraction {

    /** A prediction service: probability vectors for a batch of inputs. */
    @FunctionalInterface
    public interface Api {

        /**
         * Answers queries.
         *
         * @param rows the inputs
         * @return one probability vector per row
         */
        float[][] query(float[][] rows);
    }

    private ModelExtraction() {
    }

    /**
     * The service of a logits model returning softmax probabilities.
     *
     * @param model the victim
     * @return the API
     */
    public static Api of(UnaryOperator<Var> model) {
        Objects.requireNonNull(model);
        return rows -> rowsOf(model.apply(Var.of(Tensor.of(rows))).softmax(1).value(), rows.length);
    }

    /**
     * Label-only answers: the one-hot vector of the argmax.
     *
     * @param api the full service
     * @return the label-only service
     */
    public static Api labelOnly(Api api) {
        return rows -> {
            float[][] p = api.query(rows);
            float[][] out = new float[p.length][p[0].length];
            for (int i = 0; i < p.length; i++) {
                out[i][QueryOracle.argmax(p[i])] = 1;
            }
            return out;
        };
    }

    /**
     * Equation solving against a softmax-linear victim.
     *
     * @param api the service
     * @param features the input dimension {@code d}
     * @param queries random Gaussian queries, at least {@code d + 1}
     * @param seed the seed
     * @return weights {@code [d + 1][classes]} (last row the biases), relative to class 0
     */
    public static double[][] equationSolving(Api api, int features, int queries, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] q = new float[queries][features];
        for (float[] row : q) {
            for (int j = 0; j < features; j++) {
                row[j] = (float) (r.nextGaussian() * 0.5 + 0.5);
            }
        }
        float[][] p = api.query(q);
        int k = p[0].length;
        double[] a = new double[queries * (features + 1)];
        double[] b = new double[queries * k];
        for (int i = 0; i < queries; i++) {
            for (int j = 0; j < features; j++) {
                a[i * (features + 1) + j] = q[i][j];
            }
            a[i * (features + 1) + features] = 1;
            double base = Math.log(Math.max(p[i][0], 1e-30));
            for (int c = 0; c < k; c++) {
                b[i * k + c] = Math.log(Math.max(p[i][c], 1e-30)) - base;
            }
        }
        Tensor at = Tensor.of(Shape.of(queries, features + 1), a, DType.FLOAT64, Device.cpu());
        Tensor bt = Tensor.of(Shape.of(queries, k), b, DType.FLOAT64, Device.cpu());
        double[] w = Linalg.lstsq(at, bt, 0).toDoubleArray();
        double[][] out = new double[features + 1][k];
        for (int j = 0; j <= features; j++) {
            System.arraycopy(w, j * k, out[j], 0, k);
        }
        return out;
    }

    /**
     * The softmax-linear model of extracted weights.
     *
     * @param w weights {@code [d + 1][classes]}, last row the biases
     * @return its logits function
     */
    public static UnaryOperator<Var> linear(double[][] w) {
        int d = w.length - 1;
        int k = w[0].length;
        float[] wv = new float[d * k];
        float[] bv = new float[k];
        for (int j = 0; j < d; j++) {
            for (int c = 0; c < k; c++) {
                wv[j * k + c] = (float) w[j][c];
            }
        }
        for (int c = 0; c < k; c++) {
            bv[c] = (float) w[d][c];
        }
        Tensor wt = Tensor.of(Shape.of(d, k), wv);
        Tensor bt = Tensor.of(Shape.of(k), bv);
        return x -> x.matmul(wt).add(bt);
    }

    /**
     * Trains a copy on (soft) targets with Adam.
     *
     * @param x the inputs
     * @param targets probability targets, one row each
     * @param sizes the copy's layer sizes, input first
     * @param steps training steps
     * @param seed the seed
     * @return the copy
     */
    public static SimpleMlp train(float[][] x, float[][] targets, int[] sizes, int steps, long seed) {
        SimpleMlp copy = SimpleMlp.relu(seed, sizes);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int b = Math.min(64, x.length);
        for (int s = 0; s < steps; s++) {
            float[][] bx = new float[b][];
            float[][] by = new float[b][];
            for (int i = 0; i < b; i++) {
                int j = r.nextInt(x.length);
                bx[i] = x[j];
                by[i] = targets[j];
            }
            Var logp = copy.forward(Var.of(Tensor.of(bx))).logSoftmax(1);
            logp.mul(Tensor.of(by)).sum().mul(-1f / b).backward();
            opt.step(copy.parameters());
        }
        return copy;
    }

    /**
     * Knockoff Nets.
     *
     * @param api the service
     * @param pool the attacker's natural images
     * @param budget queries
     * @param adaptive choose queries by the copy's entropy in five rounds, else at random
     * @param sizes the copy's layer sizes
     * @param steps training steps of the final copy
     * @param seed the seed
     * @return the copy
     */
    public static SimpleMlp knockoff(Api api, float[][] pool, int budget, boolean adaptive, int[] sizes, int steps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] order = BlackBoxAttacks.permutation(pool.length, r);
        int n = Math.min(budget, pool.length);
        if (!adaptive) {
            float[][] q = new float[n][];
            for (int i = 0; i < n; i++) {
                q[i] = pool[order[i]];
            }
            return train(q, api.query(q), sizes, steps, seed + 1);
        }
        boolean[] used = new boolean[pool.length];
        List<float[]> xs = new ArrayList<>();
        List<float[]> ps = new ArrayList<>();
        int round = Math.max(1, n / 5);
        SimpleMlp copy = null;
        while (xs.size() < n) {
            int take = Math.min(round, n - xs.size());
            int[] pick = new int[take];
            if (copy == null) {
                int c = 0;
                for (int i = 0; i < pool.length && c < take; i++) {
                    if (!used[order[i]]) {
                        pick[c++] = order[i];
                    }
                }
            } else {
                float[] z = copy.forward(Var.of(Tensor.of(pool))).softmax(1).value().toArray();
                int k = z.length / pool.length;
                double[] entropy = new double[pool.length];
                for (int i = 0; i < pool.length; i++) {
                    double h = 0;
                    for (int c = 0; c < k; c++) {
                        double p = z[i * k + c];
                        h -= p > 0 ? p * Math.log(p) : 0;
                    }
                    entropy[i] = used[i] ? Double.NEGATIVE_INFINITY : h + 1e-6 * r.nextDouble();
                }
                boolean[] top = PoisonDetectors.top(entropy, take);
                int c = 0;
                for (int i = 0; i < pool.length && c < take; i++) {
                    if (top[i]) {
                        pick[c++] = i;
                    }
                }
            }
            float[][] q = new float[take][];
            for (int i = 0; i < take; i++) {
                used[pick[i]] = true;
                q[i] = pool[pick[i]];
            }
            float[][] p = api.query(q);
            for (int i = 0; i < take; i++) {
                xs.add(q[i]);
                ps.add(p[i]);
            }
            copy = train(xs.toArray(new float[0][]), ps.toArray(new float[0][]), sizes,
                    xs.size() < n ? steps / 3 : steps, seed + xs.size());
        }
        return copy;
    }

    /**
     * Jacobian-based dataset augmentation.
     *
     * @param api the service (its argmax labels are used)
     * @param seedSet the attacker's initial inputs
     * @param rounds augmentation rounds (the set doubles each round)
     * @param lambda the step along the Jacobian sign
     * @param budget the query cap
     * @param sizes the substitute's layer sizes
     * @param steps training steps per round
     * @param seed the seed
     * @return the substitute
     */
    public static SimpleMlp jacobianAugmentation(Api api, float[][] seedSet, int rounds, float lambda, int budget,
            int[] sizes, int steps, long seed) {
        List<float[]> xs = new ArrayList<>(List.of(seedSet));
        List<float[]> ys = new ArrayList<>(List.of(labelOnly(api).query(seedSet)));
        SimpleMlp sub = train(xs.toArray(new float[0][]), ys.toArray(new float[0][]), sizes, steps, seed);
        for (int t = 0; t < rounds && xs.size() < budget; t++) {
            int m = Math.min(xs.size(), budget - xs.size());
            float[][] base = new float[m][];
            float[] ids = new float[m];
            for (int i = 0; i < m; i++) {
                base[i] = xs.get(i);
                ids[i] = QueryOracle.argmax(ys.get(i));
            }
            Var in = Var.param(Tensor.of(base));
            Var picked = sub.forward(in).takeAlong(Tensor.of(Shape.of(m, 1), ids)).sum();
            float[] g = Var.gradients(picked, List.of(in), false).get(0).value().sign().toArray();
            int d = base[0].length;
            float sign = t % 2 == 0 ? lambda : -lambda;
            float[][] fresh = new float[m][d];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < d; j++) {
                    fresh[i][j] = base[i][j] + sign * g[i * d + j];
                }
            }
            float[][] labels = labelOnly(api).query(fresh);
            for (int i = 0; i < m; i++) {
                xs.add(fresh[i]);
                ys.add(labels[i]);
            }
            sub = train(xs.toArray(new float[0][]), ys.toArray(new float[0][]), sizes, steps, seed + t + 1);
        }
        return sub;
    }

    /**
     * Top-1 agreement of two models.
     *
     * @param a one model
     * @param b another
     * @param x inputs
     * @return the share of rows with the same argmax
     */
    public static double agreement(UnaryOperator<Var> a, UnaryOperator<Var> b, float[][] x) {
        int[] pa = TransferAttacks.predict(a, x);
        int[] pb = TransferAttacks.predict(b, x);
        int same = 0;
        for (int i = 0; i < pa.length; i++) {
            same += pa[i] == pb[i] ? 1 : 0;
        }
        return same / (double) pa.length;
    }

    static float[][] rowsOf(Tensor t, int n) {
        int k = Math.toIntExact(t.size() / n);
        float[] v = t.toArray();
        float[][] out = new float[n][k];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, i * k, out[i], 0, k);
        }
        return out;
    }
}
