package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.experimental.train.ExactNaturalGradient;
import io.github.zzuegg.brainiac.experimental.train.Kfac;
import io.github.zzuegg.brainiac.experimental.train.KroneckerCapture;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The diagnostics split of the second-order shootout (wave 7, item 7.3.7, after Kunstner et
 * al. 2019): on a tiny spiral classifier, exact natural-gradient steps with the true Fisher,
 * the empirical Fisher and the (saddle-free) Hessian against SGD and Adam, and the cosine
 * similarity of each cheap preconditioner's step (K-FAC, the AdaFisher diagonal, Shampoo,
 * Adam's square-root empirical-Fisher diagonal, the empirical Fisher itself) to the exact
 * natural-gradient step at the same point.
 */
public final class NaturalGradientDiagnostics {

    /** The damping of every exact step, relative to the loss' curvature scale. */
    public static final double DAMPING = 0.05;

    private NaturalGradientDiagnostics() {
    }

    /**
     * The losses reached by each exact or first-order method after the same number of steps.
     *
     * @param sizes the network's layer sizes, for example {@code {2, 8, 3}}
     * @param steps steps per method
     * @param seed the seed
     * @return the final full-data loss per method
     */
    public static Map<String, Double> losses(int[] sizes, int steps, long seed) {
        ToyTasks.Data data = ToyTasks.spiral(15, 3, 0.1, seed);
        Map<String, Double> out = new LinkedHashMap<>();
        for (Map.Entry<String, ExactNaturalGradient> e : Map.of(
                "natural gradient (Fisher)", new ExactNaturalGradient(ExactNaturalGradient.Curvature.FISHER, 1f, DAMPING),
                "empirical Fisher", new ExactNaturalGradient(ExactNaturalGradient.Curvature.EMPIRICAL_FISHER, 1f, DAMPING),
                "Newton (|Hessian|)", new ExactNaturalGradient(ExactNaturalGradient.Curvature.HESSIAN, 1f, DAMPING),
                "gradient descent", new ExactNaturalGradient(ExactNaturalGradient.Curvature.IDENTITY, 1f, 0))
                .entrySet().stream().sorted(Map.Entry.comparingByKey()).toList()) {
            SimpleMlp net = SimpleMlp.tanh(seed, sizes);
            for (int t = 0; t < steps; t++) {
                e.getValue().step(net.parameters(), p -> net.forward(Var.of(data.inputs())), data.ids());
            }
            out.put(e.getKey(), NumericsTasks.loss(net, data));
        }
        SimpleMlp adam = SimpleMlp.tanh(seed, sizes);
        Optimizer opt = Optimizer.adam(0.05f);
        for (int t = 0; t < steps; t++) {
            Functional.crossEntropy(adam.forward(Var.of(data.inputs())), data.ids()).backward();
            opt.step(adam.parameters());
        }
        out.put("Adam", NumericsTasks.loss(adam, data));
        return out;
    }

    /**
     * The cosine of each approximate preconditioned step to the exact natural-gradient step,
     * at the network's initialization and after a few gradient steps, averaged.
     *
     * @param sizes the layer sizes
     * @param seed the seed
     * @return the mean cosine per approximation
     */
    public static Map<String, Double> alignment(int[] sizes, long seed) {
        ToyTasks.Data data = ToyTasks.spiral(15, 3, 0.1, seed);
        SimpleMlp net = SimpleMlp.tanh(seed, sizes);
        Map<String, Double> sum = new LinkedHashMap<>();
        int points = 3;
        Optimizer warm = Optimizer.adam(0.05f);
        for (int k = 0; k < points; k++) {
            alignOnce(net, data, sum);
            for (int t = 0; t < 10; t++) {
                Functional.crossEntropy(net.forward(Var.of(data.inputs())), data.ids()).backward();
                warm.step(net.parameters());
            }
        }
        sum.replaceAll((key, v) -> v / points);
        return sum;
    }

    private static void alignOnce(SimpleMlp net, ToyTasks.Data data, Map<String, Double> sum) {
        double damping = 1e-3;
        List<Var> params = net.parameters();
        ExactNaturalGradient.Matrices m = ExactNaturalGradient.compute(params, p -> net.forward(Var.of(data.inputs())),
                data.ids(), false);
        double[] g = m.gradient();
        double[] exact = ExactNaturalGradient.direction(m.fisher(), g, damping);
        Map<String, double[]> dirs = new LinkedHashMap<>();
        dirs.put("gradient", ExactNaturalGradient.direction(null, g, 0));
        dirs.put("empirical Fisher", ExactNaturalGradient.direction(m.empiricalFisher(), g, damping));

        // K-FAC and the AdaFisher diagonal from the exact (expected) factors: the loss summed over
        // classes weighted by p is approximated by many sampled-label passes
        Kfac kfac = new Kfac(1f, (float) damping, 0f, 1, 0f, 0f, false);
        KroneckerCapture cap = new KroneckerCapture();
        Var logits = NumericsTasks.forward(net, Var.of(data.inputs()), cap);
        List<List<KroneckerCapture.Factors>> draws = new ArrayList<>();
        java.util.SplittableRandom r = new java.util.SplittableRandom(11);
        for (int d = 0; d < 16; d++) {
            draws.add(cap.factors(Functional.crossEntropy(logits,
                    io.github.zzuegg.brainiac.experimental.train.Sophia.sampleLabels(logits.value(), r))));
        }
        List<KroneckerCapture.Factors> pooled = pool(draws);
        kfac.updateCurvature(pooled);
        List<double[][]> grads = layerGradients(net, g);
        double[] kf = new double[g.length];
        double[] adaFisher = new double[g.length];
        double[] shampoo = new double[g.length];
        int o = 0;
        for (int l = 0; l < net.depth(); l++) {
            double[][] gl = grads.get(l);
            double[][] dk = kfac.precondition(net.weight(l), gl);
            double[][] aCov = pooled.get(l).inputCovariance();
            double[][] gCov = pooled.get(l).gradientCovariance();
            double[][] ds = shampooDirection(gl);
            int in = gl.length - 1;
            int out = gl[0].length;
            for (int part = 0; part < 2; part++) {
                int rows = part == 0 ? in : 1;
                for (int i = 0; i < rows; i++) {
                    int row = part == 0 ? i : in;
                    for (int j = 0; j < out; j++) {
                        kf[o] = -dk[row][j];
                        adaFisher[o] = -gl[row][j] / (aCov[row][row] * gCov[j][j] + damping);
                        shampoo[o] = -ds[row][j];
                        o++;
                    }
                }
            }
        }
        dirs.put("K-FAC", kf);
        dirs.put("AdaFisher diagonal", adaFisher);
        dirs.put("Shampoo", shampoo);
        double[] adam = new double[g.length];
        double[][] ef = m.empiricalFisher();
        for (int i = 0; i < g.length; i++) {
            adam[i] = -g[i] / (Math.sqrt(Math.max(0, ef[i][i])) + 1e-8);
        }
        dirs.put("Adam (sqrt EF diagonal)", adam);
        for (Map.Entry<String, double[]> e : dirs.entrySet()) {
            sum.merge(e.getKey(), CurvatureProducts.cosine(e.getValue(), exact), Double::sum);
        }
    }

    private static List<KroneckerCapture.Factors> pool(List<List<KroneckerCapture.Factors>> draws) {
        List<KroneckerCapture.Factors> out = new ArrayList<>();
        for (int l = 0; l < draws.get(0).size(); l++) {
            List<double[]> a = new ArrayList<>();
            List<double[]> d = new ArrayList<>();
            for (List<KroneckerCapture.Factors> draw : draws) {
                a.addAll(List.of(draw.get(l).a()));
                d.addAll(List.of(draw.get(l).delta()));
            }
            out.add(new KroneckerCapture.Factors(draws.get(0).get(l).site(), a.toArray(double[][]::new),
                    d.toArray(double[][]::new)));
        }
        return out;
    }

    /** Splits a flat gradient over {@code [W0, b0, W1, b1, ...]} into combined {@code [W; b]} blocks. */
    private static List<double[][]> layerGradients(SimpleMlp net, double[] g) {
        List<double[][]> out = new ArrayList<>();
        int o = 0;
        for (int l = 0; l < net.depth(); l++) {
            int in = net.weight(l).shape().dim(0);
            int outDim = net.weight(l).shape().dim(1);
            double[][] m = new double[in + 1][outDim];
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < outDim; j++) {
                    m[i][j] = g[o++];
                }
            }
            for (int j = 0; j < outDim; j++) {
                m[in][j] = g[o++];
            }
            out.add(m);
        }
        return out;
    }

    /** One Shampoo step direction {@code L^-1/4 G R^-1/4} from this gradient alone. */
    private static double[][] shampooDirection(double[][] g) {
        Tensor gt = io.github.zzuegg.brainiac.experimental.ml.MlMatrices.tensor(g).to(
                io.github.zzuegg.brainiac.tensor.DType.FLOAT32);
        Tensor l = gt.matmul(gt.transpose());
        Tensor r = gt.transpose().matmul(gt);
        float eps = 1e-6f;
        Tensor li = Linalg.symmetricPower(l.add(Linalg.identity(l.shape().dim(0), l.dtype(), l.device()).mul(eps)),
                -0.25, eps);
        Tensor ri = Linalg.symmetricPower(r.add(Linalg.identity(r.shape().dim(0), r.dtype(), r.device()).mul(eps)),
                -0.25, eps);
        return io.github.zzuegg.brainiac.experimental.ml.MlMatrices.matrix(li.matmul(gt).matmul(ri));
    }
}
