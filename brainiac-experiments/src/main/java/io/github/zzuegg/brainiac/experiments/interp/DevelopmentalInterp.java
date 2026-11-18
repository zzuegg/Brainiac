package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.interp.LocalLearningCoefficient;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.experiments.optim.GrokkingExperiment;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 9 item 9.4.8: the local learning coefficient as a developmental probe.
 *
 * <p>An MLP (46-128-23) learns modular addition mod 23 from 80% of the table
 * ({@link GrokkingExperiment#modularAddition}) full-batch with AdamW and strong weight decay,
 * long enough to grok (its test accuracy jumps after it has memorised the training set; the
 * GrokkingExperiment test itself stops before that). At checkpoints the
 * {@link LocalLearningCoefficient} is estimated by localised SGLD, next to the baselines
 * that do not need sampling: the Hessian trace (Hutchinson, Rademacher probes) and its top
 * eigenvalue (power iteration), both from exact Hessian-vector products of
 * {@link CurvatureProducts}, and the weight norm.
 *
 * <p>Two further checks: estimator stability (the final checkpoint's LLC at three SGLD step
 * sizes), and ranking: models of equal size trained with different weight decays, ranked by
 * LLC and by weight norm, against their test accuracy (Spearman correlation; a lower LLC
 * should mean better generalisation).
 */
public final class DevelopmentalInterp {

    private static final int P = 23;

    private DevelopmentalInterp() {
    }

    /**
     * One checkpoint.
     *
     * @param step the training step
     * @param trainAccuracy training accuracy
     * @param testAccuracy test accuracy
     * @param llc the local learning coefficient estimate
     * @param hessianTrace the Hessian trace of the training loss
     * @param topEigenvalue its largest eigenvalue
     * @param weightNorm the parameters' L2 norm
     */
    public record Checkpoint(int step, double trainAccuracy, double testAccuracy, double llc, double hessianTrace,
            double topEigenvalue, double weightNorm) {
    }

    /**
     * The study's results.
     *
     * @param curve the checkpoints
     * @param stability the final LLC at SGLD step sizes 3e-4, 1e-3 and 3e-3
     * @param rankLlc the Spearman correlation between LLC and test accuracy across weight decays
     * @param rankNorm the same for the weight norm
     * @param diverged whether any LLC chain diverged
     */
    public record Study(List<Checkpoint> curve, double[] stability, double rankLlc, double rankNorm,
            boolean diverged) {
    }

    private static Model mlp(long seed) {
        return Model.sequential().input(2 * P).dense(128, Activation.RELU).dense(P, Activation.IDENTITY).seed(seed)
                .build();
    }

    private static double accuracy(Model m, Tensor x, Tensor y) {
        float[] p = Var.noGrad(() -> m.forward(Var.of(x))).value().argmax(1).toArray();
        float[] t = y.toArray();
        int hit = 0;
        for (int i = 0; i < t.length; i++) {
            hit += p[i] == t[i] ? 1 : 0;
        }
        return (double) hit / t.length;
    }

    private static double norm(Model m) {
        double s = 0;
        for (Var v : m.parameters()) {
            s += v.value().square().sum().item();
        }
        return Math.sqrt(s);
    }

    private static LocalLearningCoefficient.Estimate llc(Model m, Tensor x, Tensor y, float stepSize, long seed) {
        int n = x.shape().dim(0);
        SplittableRandom r = new SplittableRandom(seed);
        double full = Var.noGrad(() -> Functional.crossEntropy(m.forward(Var.of(x)), y)).value().item();
        int batch = 64;
        LocalLearningCoefficient.Estimate e = LocalLearningCoefficient.estimate(m.parameters(), step -> {
            int from = r.nextInt(n - batch + 1);
            return Functional.crossEntropy(m.forward(Var.of(x.slice(0, from, from + batch))), y.slice(0, from,
                    from + batch));
        }, full, n, new LocalLearningCoefficient.Settings(stepSize, 100f, 2, 80, 30), seed);
        return e;
    }

    private static double[] curvature(Model m, Tensor x, Tensor y, long seed) {
        List<Var> params = m.parameters();
        int size = CurvatureProducts.size(params);
        SplittableRandom r = new SplittableRandom(seed);
        double trace = 0;
        int probes = 4;
        for (int k = 0; k < probes; k++) {
            double[] v = new double[size];
            for (int i = 0; i < size; i++) {
                v[i] = r.nextBoolean() ? 1 : -1;
            }
            double[] hv = CurvatureProducts.hessianVector(() -> Functional.crossEntropy(m.forward(Var.of(x)), y),
                    params, v);
            trace += CurvatureProducts.dot(v, hv) / probes;
        }
        double[] v = new double[size];
        for (int i = 0; i < size; i++) {
            v[i] = r.nextGaussian();
        }
        double eig = 0;
        for (int it = 0; it < 8; it++) {
            double nv = Math.sqrt(CurvatureProducts.dot(v, v));
            for (int i = 0; i < size; i++) {
                v[i] /= nv;
            }
            double[] hv = CurvatureProducts.hessianVector(() -> Functional.crossEntropy(m.forward(Var.of(x)), y),
                    params, v);
            eig = CurvatureProducts.dot(v, hv);
            v = hv;
        }
        for (Var p : params) {
            p.zeroGrad();
        }
        return new double[] {trace, eig};
    }

    /**
     * Runs the study.
     *
     * @param steps training steps
     * @param every steps between checkpoints
     * @param rankSteps training steps of each weight-decay model in the ranking check
     * @param seed the seed
     * @return the results
     */
    public static Study run(int steps, int every, int rankSteps, long seed) {
        Tensor[] d = GrokkingExperiment.modularAddition(P, 0.8, seed);
        Model m = mlp(seed);
        Optimizer opt = Optimizer.adamW(1e-2f, 0.9f, 0.98f, 1e-8f, 1f);
        List<Checkpoint> curve = new ArrayList<>();
        boolean diverged = false;
        for (int s = 1; s <= steps; s++) {
            Functional.crossEntropy(m.forward(Var.of(d[0]), true), d[1]).backward();
            opt.step(m.parameters());
            if (s % every == 0) {
                LocalLearningCoefficient.Estimate e = llc(m, d[0], d[1], 1e-3f, seed + s);
                diverged |= e.diverged();
                double[] c = curvature(m, d[0], d[1], seed + s);
                curve.add(new Checkpoint(s, accuracy(m, d[0], d[1]), accuracy(m, d[2], d[3]), e.llc(), c[0], c[1],
                        norm(m)));
            }
        }
        float[] sizes = {3e-4f, 1e-3f, 3e-3f};
        double[] stability = new double[sizes.length];
        for (int i = 0; i < sizes.length; i++) {
            LocalLearningCoefficient.Estimate e = llc(m, d[0], d[1], sizes[i], seed + 900 + i);
            diverged |= e.diverged();
            stability[i] = e.llc();
        }
        float[] decays = {0f, 0.1f, 0.3f, 1f};
        double[] acc = new double[decays.length];
        double[] llcs = new double[decays.length];
        double[] norms = new double[decays.length];
        for (int k = 0; k < decays.length; k++) {
            Model mk = mlp(seed + 1);
            Optimizer ok = Optimizer.adamW(1e-2f, 0.9f, 0.98f, 1e-8f, decays[k]);
            for (int s = 0; s < rankSteps; s++) {
                Functional.crossEntropy(mk.forward(Var.of(d[0]), true), d[1]).backward();
                ok.step(mk.parameters());
            }
            acc[k] = accuracy(mk, d[2], d[3]);
            LocalLearningCoefficient.Estimate e = llc(mk, d[0], d[1], 1e-3f, seed + 500 + k);
            diverged |= e.diverged();
            llcs[k] = e.llc();
            norms[k] = norm(mk);
        }
        return new Study(curve, stability, spearman(llcs, acc), spearman(norms, acc), diverged);
    }

    private static double spearman(double[] a, double[] b) {
        double[] ra = ranks(a);
        double[] rb = ranks(b);
        int n = a.length;
        double ma = (n - 1) / 2.0;
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < n; i++) {
            sab += (ra[i] - ma) * (rb[i] - ma);
            saa += (ra[i] - ma) * (ra[i] - ma);
            sbb += (rb[i] - ma) * (rb[i] - ma);
        }
        return saa == 0 || sbb == 0 ? 0 : sab / Math.sqrt(saa * sbb);
    }

    private static double[] ranks(double[] v) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            int less = 0;
            int equal = 0;
            for (double w : v) {
                less += w < v[i] ? 1 : 0;
                equal += w == v[i] ? 1 : 0;
            }
            r[i] = less + (equal - 1) / 2.0;
        }
        return r;
    }
}
