package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ChaoticSeries;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.experimental.train.HessianFree;
import io.github.zzuegg.brainiac.experimental.train.Kfac;
import io.github.zzuegg.brainiac.experimental.train.KroneckerCapture;
import io.github.zzuegg.brainiac.experimental.train.Lbfgs;
import io.github.zzuegg.brainiac.experimental.train.LevenbergMarquardt;
import io.github.zzuegg.brainiac.experiments.generative.NeuralFields;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * The full-batch and small-network splits of the second-order shootout (wave 7, items 7.3.3,
 * 7.3.4 and 7.3.6), each method at a fixed iteration budget with its wall time:
 *
 * <ul>
 *   <li>{@link #neuralField}: a Fourier-feature coordinate MLP fitting a {@link NeuralFields}
 *       picture with the full batch, by Adam, SGD with momentum, {@link Lbfgs#fullBatch} and
 *       {@link HessianFree};</li>
 *   <li>{@link #autoencoder}: the classic deep-autoencoder curvature benchmark on
 *       {@link ToyTasks#digits} (64-32-8-32-64 tanh), by Adam, K-FAC, L-BFGS and
 *       Hessian-free;</li>
 *   <li>{@link #smallNetwork}: a 4-8-1 tanh network predicting the next Mackey-Glass value
 *       ({@link ChaoticSeries}), by Adam, L-BFGS and {@link LevenbergMarquardt}.</li>
 * </ul>
 */
public final class FullBatchSecondOrder {

    private FullBatchSecondOrder() {
    }

    /**
     * One method's outcome.
     *
     * @param loss the final mean squared error
     * @param millis the wall time
     * @param iterations the iterations run
     * @param millisToTarget wall time until the loss first fell below the target, -1 if never
     */
    public record Row(double loss, long millis, int iterations, long millisToTarget) {
    }

    private interface Method {
        double iterate(int t);
    }

    private static Row measure(Method m, int iterations, double target, Function<Integer, Double> lossAt) {
        long t0 = System.nanoTime();
        long hit = -1;
        double loss = Double.NaN;
        for (int t = 0; t < iterations; t++) {
            loss = m.iterate(t);
            if (hit < 0 && loss < target) {
                hit = (System.nanoTime() - t0) / 1_000_000;
            }
        }
        long ms = (System.nanoTime() - t0) / 1_000_000;
        return new Row(lossAt.apply(0), ms, iterations, hit);
    }

    private static Var mse(Var out, Tensor target) {
        return out.sub(target).square().mean();
    }

    private static Method firstOrder(Optimizer opt, SimpleMlp net, Tensor x, Tensor y) {
        return t -> {
            Var l = mse(net.forward(Var.of(x)), y);
            l.backward();
            opt.step(net.parameters());
            return l.value().item();
        };
    }

    private static Method lbfgs(SimpleMlp net, Tensor x, Tensor y) {
        Lbfgs opt = Lbfgs.fullBatch(1f, 20);
        return t -> {
            mse(net.forward(Var.of(x)), y).backward();
            opt.step(net.parameters(), () -> {
                for (Var p : net.parameters()) {
                    p.zeroGrad();
                }
                Var l = mse(net.forward(Var.of(x)), y);
                l.backward();
                return l.value().item();
            });
            return lossOf(net, x, y);
        };
    }

    private static Method hessianFree(SimpleMlp net, Tensor x, Tensor y, int maxCg) {
        HessianFree hf = new HessianFree(1.0, maxCg, 0.95);
        float scale = 1f / y.size();
        return t -> hf.step(net.parameters(), p -> net.forward(Var.of(x)), out -> mse(out, y),
                CurvatureProducts.OutputCurvature.squaredError(scale));
    }

    private static double lossOf(SimpleMlp net, Tensor x, Tensor y) {
        return Var.noGrad(() -> mse(net.forward(Var.of(x)), y)).value().item();
    }

    /**
     * Random Fourier features {@code [sin 2 pi B c, cos 2 pi B c, c]} of the pixel coordinates.
     *
     * @param side the picture side
     * @param features frequencies
     * @param scale the frequencies' standard deviation
     * @param seed the seed
     * @return {@code [side * side, 2 features + 2]}
     */
    public static Tensor fourierCoordinates(int side, int features, double scale, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] b = new double[features][2];
        for (double[] row : b) {
            row[0] = r.nextGaussian() * scale;
            row[1] = r.nextGaussian() * scale;
        }
        int width = 2 * features + 2;
        float[] out = new float[side * side * width];
        for (int py = 0; py < side; py++) {
            for (int px = 0; px < side; px++) {
                double cx = px / (side - 1.0) * 2 - 1;
                double cy = py / (side - 1.0) * 2 - 1;
                int o = (py * side + px) * width;
                for (int k = 0; k < features; k++) {
                    double a = 2 * Math.PI * (b[k][0] * cx + b[k][1] * cy);
                    out[o + k] = (float) Math.sin(a);
                    out[o + features + k] = (float) Math.cos(a);
                }
                out[o + 2 * features] = (float) cx;
                out[o + 2 * features + 1] = (float) cy;
            }
        }
        return Tensor.of(Shape.of(side * side, width), out);
    }

    /**
     * The neural-field split.
     *
     * @param iterations the iteration budget of Hessian-free; L-BFGS gets five times as many
     *     (about the same wall time as the first-order methods), first-order methods ten times
     * @param target the target mean squared error
     * @param seed the seed
     * @return one row per method
     */
    public static Map<String, Row> neuralField(int iterations, double target, long seed) {
        int side = 20;
        Tensor x = fourierCoordinates(side, 10, 1.5, seed);
        Tensor y = Tensor.of(Shape.of(side * side, 1), NeuralFields.picture(side, 4));
        Map<String, Row> out = new LinkedHashMap<>();
        int width = x.shape().dim(1);
        SimpleMlp adam = SimpleMlp.tanh(seed, width, 16, 1);
        out.put("Adam", measure(firstOrder(Optimizer.adam(0.01f), adam, x, y), iterations * 10, target,
                k -> lossOf(adam, x, y)));
        SimpleMlp sgd = SimpleMlp.tanh(seed, width, 16, 1);
        out.put("SGD momentum", measure(firstOrder(Optimizer.momentum(0.1f, 0.9f), sgd, x, y), iterations * 10, target,
                k -> lossOf(sgd, x, y)));
        SimpleMlp lb = SimpleMlp.tanh(seed, width, 16, 1);
        out.put("L-BFGS", measure(lbfgs(lb, x, y), iterations * 5, target, k -> lossOf(lb, x, y)));
        SimpleMlp hf = SimpleMlp.tanh(seed, width, 16, 1);
        out.put("Hessian-free", measure(hessianFree(hf, x, y, 30), iterations, target, k -> lossOf(hf, x, y)));
        return out;
    }

    /**
     * The deep-autoencoder split.
     *
     * @param iterations the iteration budget of Hessian-free; L-BFGS gets five times as many,
     *     Adam and K-FAC ten times
     * @param target the target reconstruction error
     * @param seed the seed
     * @return one row per method
     */
    public static Map<String, Row> autoencoder(int iterations, double target, long seed) {
        Tensor x = ToyTasks.digits(10, 20, 8, 0.05, seed).inputs();
        Map<String, Row> out = new LinkedHashMap<>();
        int[] sizes = {64, 32, 8, 32, 64};
        SimpleMlp adam = SimpleMlp.tanh(seed, sizes);
        out.put("Adam", measure(firstOrder(Optimizer.adam(0.01f), adam, x, x), iterations * 10, target,
                k -> lossOf(adam, x, x)));
        SimpleMlp kf = SimpleMlp.tanh(seed, sizes);
        Kfac kfac = Kfac.kfac(0.3f, 1e-2f);
        SplittableRandom noise = new SplittableRandom(seed + 5);
        out.put("K-FAC", measure(t -> {
            KroneckerCapture cap = new KroneckerCapture();
            Var y = NumericsTasks.forward(kf, Var.of(x), cap);
            // Gaussian likelihood: targets sampled around the model's own prediction
            Tensor sampled = y.value().add(Tensor.normal(y.shape(), 0, 1, noise));
            kfac.updateCurvature(cap.factors(y.sub(Var.of(sampled)).square().sum(1, false).mean().mul(0.5f)));
            Var l = mse(y, x);
            l.backward();
            kfac.step(kf.parameters());
            return l.value().item();
        }, iterations * 10, target, k -> lossOf(kf, x, x)));
        SimpleMlp lb = SimpleMlp.tanh(seed, sizes);
        out.put("L-BFGS", measure(lbfgs(lb, x, x), iterations * 5, target, k -> lossOf(lb, x, x)));
        SimpleMlp hf = SimpleMlp.tanh(seed, sizes);
        out.put("Hessian-free", measure(hessianFree(hf, x, x, 25), iterations, target, k -> lossOf(hf, x, x)));
        return out;
    }

    /**
     * The small-network split: one-step prediction of the Mackey-Glass series from the last
     * four values.
     *
     * @param iterations the iteration budget of Levenberg-Marquardt and L-BFGS; Adam gets 50
     *     times as many steps
     * @param seed the seed
     * @return one row per method
     */
    public static Map<String, Row> smallNetwork(int iterations, long seed) {
        double[] s = ChaoticSeries.mackeyGlass(124, 17, 1.0, 300);
        int window = 4;
        int n = s.length - window;
        float[] xs = new float[n * window];
        float[] ys = new float[n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < window; k++) {
                xs[i * window + k] = (float) (s[i + k] - 0.9);
            }
            ys[i] = (float) (s[i + window] - 0.9);
        }
        Tensor x = Tensor.of(Shape.of(n, window), xs);
        Tensor y = Tensor.of(Shape.of(n, 1), ys);
        Map<String, Row> out = new LinkedHashMap<>();
        SimpleMlp adam = SimpleMlp.tanh(seed, window, 8, 1);
        out.put("Adam", measure(firstOrder(Optimizer.adam(0.01f), adam, x, y), iterations * 50, 0,
                k -> lossOf(adam, x, y)));
        SimpleMlp lb = SimpleMlp.tanh(seed, window, 8, 1);
        out.put("L-BFGS", measure(lbfgs(lb, x, y), iterations, 0, k -> lossOf(lb, x, y)));
        SimpleMlp lm = SimpleMlp.tanh(seed, window, 8, 1);
        LevenbergMarquardt solver = new LevenbergMarquardt(1e-2);
        List<Var> params = lm.parameters();
        out.put("Levenberg-Marquardt", measure(t -> solver.step(params, p -> lm.forward(Var.of(x)).sub(Var.of(y))),
                iterations, 0, k -> lossOf(lm, x, y)));
        return out;
    }
}
