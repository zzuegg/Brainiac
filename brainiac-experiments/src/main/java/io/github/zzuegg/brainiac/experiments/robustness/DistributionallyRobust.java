package io.github.zzuegg.brainiac.experiments.robustness;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.robust.AdversarialAttacks;
import io.github.zzuegg.brainiac.experimental.robust.AdversarialTraining;
import io.github.zzuegg.brainiac.experimental.robust.DistributionalRobustness;
import io.github.zzuegg.brainiac.experimental.train.GroupRobustTraining;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Distributionally robust optimisation beyond groups (13.6.10).
 * <ul>
 *   <li>{@link #subpopulation}: on {@link ShiftSuites#backgrounds} (a Waterbirds-like task where
 *       the background agrees with the class in most training images), CVaR, chi-squared and KL
 *       DRO ({@link DistributionalRobustness}) without group labels against ERM, JTT (no group
 *       labels) and GroupDRO (with group labels, the oracle) from {@link GroupRobustTraining}, by
 *       worst-group and average test accuracy on a balanced test set, optionally with flipped
 *       training labels (which DRO upweights);</li>
 *   <li>{@link #perturbation}: Wasserstein-DRO logistic regression against plain and
 *       L2-regularised logistic regression and PGD adversarial training of the same linear model
 *       ({@link AdversarialTraining}), by accuracy under L2 perturbations of growing radius and by
 *       training time, on a {@link TabularStressSuite} table.</li>
 * </ul>
 */
public final class DistributionallyRobust {

    private DistributionallyRobust() {
    }

    private static Model mlp(long seed) {
        return Model.sequential().input(ShiftSuites.PIXELS).dense(32, Activation.RELU).dense(2).seed(seed).build();
    }

    private static int[] predict(Model m, Tensor x) {
        float[] z = Var.noGrad(() -> m.forward(Var.of(x), false)).value().toArray();
        int[] out = new int[z.length / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = z[2 * i + 1] > z[2 * i] ? 1 : 0;
        }
        return out;
    }

    /** {average accuracy, worst-group accuracy}. */
    private static double[] groupAccuracy(int[] p, int[] y, int[] group) {
        int[] right = new int[4];
        int[] count = new int[4];
        int all = 0;
        for (int i = 0; i < y.length; i++) {
            count[group[i]]++;
            if (p[i] == y[i]) {
                right[group[i]]++;
                all++;
            }
        }
        double worst = 1;
        for (int g = 0; g < 4; g++) {
            if (count[g] > 0) {
                worst = Math.min(worst, right[g] / (double) count[g]);
            }
        }
        return new double[] {all / (double) y.length, worst};
    }

    private static Tensor rows(Tensor x, int[] idx) {
        float[] f = new float[idx.length];
        for (int i = 0; i < idx.length; i++) {
            f[i] = idx[i];
        }
        return x.gather(0, Tensor.of(Shape.of(idx.length), f));
    }

    private static Tensor ids(int[] y, int[] idx) {
        float[] f = new float[idx.length];
        for (int i = 0; i < idx.length; i++) {
            f[i] = y[idx[i]];
        }
        return Tensor.of(Shape.of(idx.length), f);
    }

    /**
     * Worst-group and average accuracy of each method.
     *
     * @param n training images
     * @param agreement how often background and class agree in training, such as 0.95
     * @param labelNoise the share of training labels flipped
     * @param steps training steps
     * @param seed the seed
     * @return method to {average accuracy, worst-group accuracy}
     */
    public static Map<String, double[]> subpopulation(int n, double agreement, double labelNoise, int steps,
            long seed) {
        ShiftSuites.Grouped train = ShiftSuites.backgrounds(n, agreement, seed);
        ShiftSuites.Grouped test = ShiftSuites.backgrounds(800, 0.5, seed + 1);
        int[] y = train.split().y().clone();
        SplittableRandom noise = new SplittableRandom(seed + 2);
        for (int i = 0; i < y.length; i++) {
            if (noise.nextDouble() < labelNoise) {
                y[i] = 1 - y[i];
            }
        }
        Tensor x = train.split().x();
        int b = 64;
        Map<String, double[]> out = new LinkedHashMap<>();
        Map<String, Object[]> dro = new LinkedHashMap<>();
        dro.put("ERM", new Object[] {DistributionalRobustness.Kind.ERM, 0.0});
        dro.put("CVaR (alpha 0.1)", new Object[] {DistributionalRobustness.Kind.CVAR, 0.1});
        dro.put("chi-square (lambda 0.3)", new Object[] {DistributionalRobustness.Kind.CHI_SQUARE, 0.3});
        dro.put("KL (tau 0.3)", new Object[] {DistributionalRobustness.Kind.KL, 0.3});
        for (Map.Entry<String, Object[]> e : dro.entrySet()) {
            DistributionalRobustness.Kind kind = (DistributionalRobustness.Kind) e.getValue()[0];
            double param = (Double) e.getValue()[1];
            Model m = mlp(seed);
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed);
            for (int s = 0; s < steps; s++) {
                int[] idx = new int[b];
                for (int i = 0; i < b; i++) {
                    idx[i] = r.nextInt(n);
                }
                Tensor xb = rows(x, idx);
                Tensor yb = ids(y, idx);
                opt.minimize(m.parameters(), () -> DistributionalRobustness.loss(kind,
                        Functional.crossEntropyPerRow(m.forward(Var.of(xb), true), yb), param));
            }
            out.put(e.getKey(), groupAccuracy(predict(m, test.split().x()), test.split().y(), test.group()));
        }
        // JTT: a short ERM run marks errors, a second run upsamples them
        Model id = mlp(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps / 4; s++) {
            int[] idx = new int[b];
            for (int i = 0; i < b; i++) {
                idx[i] = r.nextInt(n);
            }
            Tensor xb = rows(x, idx);
            Tensor yb = ids(y, idx);
            opt.minimize(id.parameters(), () -> Functional.crossEntropy(id.forward(Var.of(xb), true), yb));
        }
        GroupRobustTraining.JustTrainTwice jtt = new GroupRobustTraining.JustTrainTwice(
                GroupRobustTraining.JustTrainTwice.errorSet(predict(id, x), y), 10);
        Model second = mlp(seed);
        Optimizer opt2 = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            int[] idx = jtt.sample(b, r);
            Tensor xb = rows(x, idx);
            Tensor yb = ids(y, idx);
            opt2.minimize(second.parameters(), () -> Functional.crossEntropy(second.forward(Var.of(xb), true), yb));
        }
        out.put("JTT", groupAccuracy(predict(second, test.split().x()), test.split().y(), test.group()));
        // GroupDRO with the group labels (the oracle)
        Model gm = mlp(seed);
        Optimizer opt3 = Optimizer.adam(3e-3f);
        GroupRobustTraining.GroupDro gdro = new GroupRobustTraining.GroupDro(4, 0.05);
        int[] group = train.group();
        for (int s = 0; s < steps; s++) {
            int[] idx = new int[b];
            for (int i = 0; i < b; i++) {
                idx[i] = r.nextInt(n);
            }
            Tensor xb = rows(x, idx);
            Tensor yb = ids(y, idx);
            opt3.minimize(gm.parameters(), () -> {
                Var per = Functional.crossEntropyPerRow(gm.forward(Var.of(xb), true), yb).reshape(b);
                List<Var> losses = new ArrayList<>();
                for (int g = 0; g < 4; g++) {
                    float[] mask = new float[b];
                    int c = 0;
                    for (int i = 0; i < b; i++) {
                        if (group[idx[i]] == g) {
                            mask[i] = 1;
                            c++;
                        }
                    }
                    losses.add(c == 0 ? null : per.mul(Tensor.of(Shape.of(b), mask)).sum().mul(1f / c));
                }
                return gdro.loss(losses);
            });
        }
        out.put("GroupDRO (group labels)", groupAccuracy(predict(gm, test.split().x()), test.split().y(),
                test.group()));
        return out;
    }

    /**
     * Robust accuracy of linear classifiers under L2 perturbations.
     *
     * @param n training rows
     * @param epsilons perturbation radii to score
     * @param trainEpsilon the radius used for training (Wasserstein DRO and PGD)
     * @param seed the seed
     * @return method to {robust accuracy per radius..., training ms}
     */
    public static Map<String, double[]> perturbation(int n, double[] epsilons, double trainEpsilon, long seed) {
        TabularStressSuite suite = new TabularStressSuite(4, 0.5, false, seed);
        double[][][] xs = new double[2][][];
        int[][] ys = new int[2][];
        for (int part = 0; part < 2; part++) {
            SyntheticRecsys.Table t = suite.table(part == 0 ? n : 1000, seed + part);
            int d = suite.columns();
            float[] v = t.features().toArray();
            float[] l = t.labels().toArray();
            int rows = l.length;
            xs[part] = new double[rows][d];
            ys[part] = new int[rows];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < d; j++) {
                    xs[part][i][j] = v[i * d + j];
                }
                ys[part][i] = l[i] > 0.5 ? 1 : 0;
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        Map<String, DistributionalRobustness.Linear> models = new LinkedHashMap<>();
        Map<String, Double> times = new LinkedHashMap<>();
        long t0 = System.nanoTime();
        models.put("logistic", DistributionalRobustness.wassersteinLogistic(xs[0], ys[0], 0, 0, 100));
        times.put("logistic", (System.nanoTime() - t0) / 1e6);
        t0 = System.nanoTime();
        models.put("L2 logistic", DistributionalRobustness.wassersteinLogistic(xs[0], ys[0], 0, 0.05, 100));
        times.put("L2 logistic", (System.nanoTime() - t0) / 1e6);
        t0 = System.nanoTime();
        models.put("Wasserstein DRO", DistributionalRobustness.wassersteinLogistic(xs[0], ys[0], trainEpsilon, 0,
                100));
        times.put("Wasserstein DRO", (System.nanoTime() - t0) / 1e6);
        t0 = System.nanoTime();
        models.put("PGD training", pgdLinear(xs[0], ys[0], trainEpsilon, 300, seed));
        times.put("PGD training", (System.nanoTime() - t0) / 1e6);
        for (Map.Entry<String, DistributionalRobustness.Linear> e : models.entrySet()) {
            double[] row = new double[epsilons.length + 1];
            for (int k = 0; k < epsilons.length; k++) {
                row[k] = e.getValue().robustAccuracy(xs[1], ys[1], epsilons[k]);
            }
            row[epsilons.length] = times.get(e.getKey());
            out.put(e.getKey(), row);
        }
        return out;
    }

    /** A linear model trained with L2 PGD adversarial training. */
    private static DistributionalRobustness.Linear pgdLinear(double[][] x, int[] y, double epsilon, int steps,
            long seed) {
        int n = x.length;
        int d = x[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        Var w = Var.param(Tensor.normal(Shape.of(d, 2), 0, 0.01f, r));
        Var b = Var.param(Tensor.zeros(Shape.of(2)));
        AdversarialTraining at = AdversarialTraining.of(v -> v.matmul(w).add(b), List.of(w, b),
                Optimizer.adam(0.03f)).method(AdversarialTraining.Method.PGD_AT).norm(AdversarialAttacks.Norm.L2)
                .epsilon((float) epsilon).steps(5).seed(seed);
        int bs = 64;
        for (int s = 0; s < steps; s++) {
            float[] xb = new float[bs * d];
            float[] yb = new float[bs];
            for (int i = 0; i < bs; i++) {
                int k = r.nextInt(n);
                for (int j = 0; j < d; j++) {
                    xb[i * d + j] = (float) x[k][j];
                }
                yb[i] = y[k];
            }
            at.step(Tensor.of(Shape.of(bs, d), xb), Tensor.of(Shape.of(bs), yb));
        }
        float[] wv = w.value().toArray();
        float[] bv = b.value().toArray();
        double[] diff = new double[d];
        for (int j = 0; j < d; j++) {
            diff[j] = wv[j * 2 + 1] - wv[j * 2];
        }
        return new DistributionalRobustness.Linear(diff, bv[1] - bv[0]);
    }
}
