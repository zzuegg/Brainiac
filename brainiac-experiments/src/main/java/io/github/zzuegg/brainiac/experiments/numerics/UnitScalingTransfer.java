package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.Fp8Format;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.UnitScaledOps;
import io.github.zzuegg.brainiac.experimental.train.UnitMup;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Unit scaling and u-muP (wave 7, item 7.3.16) on a 64-w-w-10 tanh MLP over
 * {@link ToyTasks#digits}:
 *
 * <ul>
 *   <li>{@link #sweep} and {@link #optimum}: the optimal Adam learning rate at each width, for the standard
 *       parameterization (Glorot init, one rate) and for u-muP ({@link UnitScaledOps},
 *       {@link UnitMup}); the hypothesis is that u-muP's optimum does not move with width;</li>
 *   <li>{@link #fp8}: training with every matmul output rounded to FP8 E4M3 and every gradient
 *       flowing back through it rounded to E5M2, with no loss scaling and no per-tensor scales;
 *       reports the share of nonzero gradient values flushed to zero and the loss against fp32.</li>
 * </ul>
 */
public final class UnitScalingTransfer {

    private UnitScalingTransfer() {
    }

    /** The two parameterizations. */
    public enum Scheme {
        /** Standard parameterization. */
        STANDARD,
        /** Unit-scaled maximal update parameterization. */
        UNIT_MUP
    }

    private static final class Net {
        final List<Var> params;
        final Optimizer optimizer;
        final java.util.function.BiFunction<Tensor, UnaryOperator<Var>, Var> logits;
        final Scheme scheme;

        Net(Scheme scheme, int width, float lr, long seed) {
            this.scheme = scheme;
            SplittableRandom r = new SplittableRandom(seed);
            if (scheme == Scheme.STANDARD) {
                SimpleMlp mlp = SimpleMlp.tanh(seed, 64, width, width, 10);
                params = mlp.parameters();
                optimizer = Optimizer.adam(lr);
                logits = (x, cast) -> {
                    Var h = Var.of(x);
                    for (int l = 0; l < mlp.depth(); l++) {
                        h = cast.apply(mlp.linear(l, h));
                        h = l < mlp.depth() - 1 ? h.tanh() : h;
                    }
                    return h;
                };
            } else {
                UnitMup mup = new UnitMup();
                Var w0 = mup.weight(64, width, r);
                Var w1 = mup.weight(width, width, r);
                Var w2 = mup.weight(width, 10, r);
                Var b0 = Var.param(Tensor.zeros(Shape.of(1, width)));
                Var b1 = Var.param(Tensor.zeros(Shape.of(1, width)));
                Var b2 = Var.param(Tensor.zeros(Shape.of(1, 10)));
                params = List.of(w0, b0, w1, b1, w2, b2);
                optimizer = mup.adam(lr, List.of(w0, b0, b1, b2), List.of(w1), List.of(w2));
                logits = (x, cast) -> {
                    Var h = UnitScaledOps.tanh(cast.apply(UnitScaledOps.matmul(Var.of(x), w0).add(b0)));
                    h = UnitScaledOps.tanh(cast.apply(UnitScaledOps.matmul(h, w1).add(b1)));
                    return cast.apply(UnitScaledOps.readout(h, w2).add(b2));
                };
            }
        }

        Var loss(Tensor x, Tensor y, UnaryOperator<Var> cast) {
            Var z = logits.apply(x, cast);
            return scheme == Scheme.STANDARD ? Functional.crossEntropy(z, y) : UnitScaledOps.crossEntropy(z, y);
        }

        double evalLoss(ToyTasks.Data data) {
            Tensor z = Var.noGrad(() -> logits.apply(data.inputs(), UnaryOperator.identity())).value();
            return Functional.crossEntropy(Var.of(z), data.ids()).value().item();
        }
    }

    private static double train(Scheme scheme, int width, float lr, int steps, long seed, UnaryOperator<Var> cast) {
        ToyTasks.Data data = ToyTasks.digits(10, 30, 8, 0.1, seed);
        Net net = new Net(scheme, width, lr, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int t = 0; t < steps; t++) {
            Tensor[] b = NumericsTasks.batch(data, 64, r);
            net.loss(b[0], b[1], cast).backward();
            net.optimizer.step(net.params);
        }
        double l = net.evalLoss(data);
        return Double.isFinite(l) ? l : 10;
    }

    /**
     * The learning-rate sweep.
     *
     * @param scheme the parameterization
     * @param widths the hidden widths
     * @param log2Rates the base learning rates, as powers of two
     * @param steps steps per run
     * @param seed the seed
     * @return per width, the final training loss at each rate
     */
    public static Map<Integer, double[]> sweep(Scheme scheme, int[] widths, int[] log2Rates, int steps, long seed) {
        Map<Integer, double[]> out = new LinkedHashMap<>();
        for (int w : widths) {
            double[] losses = new double[log2Rates.length];
            for (int k = 0; k < log2Rates.length; k++) {
                losses[k] = train(scheme, w, (float) Math.pow(2, log2Rates[k]), steps, seed, UnaryOperator.identity());
            }
            out.put(w, losses);
        }
        return out;
    }

    /**
     * The best rate's exponent per width.
     *
     * @param sweep from {@link #sweep}
     * @param log2Rates the rates' exponents
     * @return per width, the exponent with the lowest loss
     */
    public static Map<Integer, Integer> optimum(Map<Integer, double[]> sweep, int[] log2Rates) {
        Map<Integer, Integer> out = new LinkedHashMap<>();
        sweep.forEach((w, losses) -> {
            int best = 0;
            for (int k = 1; k < losses.length; k++) {
                best = losses[k] < losses[best] ? k : best;
            }
            out.put(w, log2Rates[best]);
        });
        return out;
    }

    /** The outcome of low-precision training. */
    public record Fp8Row(double fp32Loss, double fp8Loss, double underflow) {
    }

    /**
     * Trains in fp32 and in simulated FP8 without scaling.
     *
     * @param scheme the parameterization
     * @param width the width
     * @param lr the base learning rate
     * @param steps steps
     * @param seed the seed
     * @return both losses and the share of nonzero gradient values that FP8 flushed to zero
     */
    public static Fp8Row fp8(Scheme scheme, int width, float lr, int steps, long seed) {
        double fp32 = train(scheme, width, lr, steps, seed, UnaryOperator.identity());
        long[] counts = new long[2];
        UnaryOperator<Var> cast = v -> Var.custom(t -> Fp8Format.E4M3.roundAll(t[0]), (t, o, g) -> {
            Tensor q = Fp8Format.E5M2.roundAll(g);
            float[] before = g.toArray();
            float[] after = q.toArray();
            for (int i = 0; i < before.length; i++) {
                if (before[i] != 0) {
                    counts[0]++;
                    if (after[i] == 0) {
                        counts[1]++;
                    }
                }
            }
            return new Tensor[] {q};
        }, v);
        double fp8 = train(scheme, width, lr, steps, seed, cast);
        return new Fp8Row(fp32, fp8, counts[0] == 0 ? 0 : (double) counts[1] / counts[0]);
    }
}
