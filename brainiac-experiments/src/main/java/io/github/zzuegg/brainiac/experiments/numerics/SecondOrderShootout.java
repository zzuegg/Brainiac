package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.AdaFisher;
import io.github.zzuegg.brainiac.experimental.train.AdaHessian;
import io.github.zzuegg.brainiac.experimental.train.Kfac;
import io.github.zzuegg.brainiac.experimental.train.KroneckerCapture;
import io.github.zzuegg.brainiac.experimental.train.Lbfgs;
import io.github.zzuegg.brainiac.experimental.train.PsgdKron;
import io.github.zzuegg.brainiac.experimental.train.Shampoo;
import io.github.zzuegg.brainiac.experimental.train.Soap;
import io.github.zzuegg.brainiac.experimental.train.Sophia;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * The minibatch split of the second-order shootout (wave 7, items 7.3.1, 7.3.2, 7.3.3 online
 * and 7.3.5): a tanh MLP on three-arm spirals trained for the same number of minibatch steps
 * by K-FAC, EKFAC, AdaFisher, AdaHessian and online L-BFGS against AdamW, Shampoo, SOAP, PSGD
 * Kron and Sophia-H. Reports the final full-data loss (mean over seeds), the steps to reach a
 * target loss, wall time per step, and how many seeds diverged.
 */
public final class SecondOrderShootout {

    private SecondOrderShootout() {
    }

    /**
     * One method's outcome over the seeds.
     *
     * @param loss the mean final cross-entropy over seeds that did not diverge
     * @param accuracy the mean final accuracy
     * @param stepsToTarget the mean steps until the full-data loss first fell below the target
     *     ({@code steps} when it never did)
     * @param millisPerStep the mean wall time per step
     * @param diverged seeds whose loss became non-finite or exceeded 10
     */
    public record Row(double loss, double accuracy, double stepsToTarget, double millisPerStep, int diverged) {
    }

    private interface Trainer {
        void step(SimpleMlp net, Tensor x, Tensor y, int t);
    }

    /** The methods compared, in report order. */
    public static final List<String> METHODS = List.of("AdamW", "Shampoo", "SOAP", "PSGD Kron", "Sophia-H",
            "K-FAC", "EKFAC", "AdaFisher", "AdaHessian", "oLBFGS");

    /**
     * Runs every method.
     *
     * @param steps minibatch steps per run
     * @param batch the minibatch size
     * @param target the target full-data loss
     * @param seeds the seeds
     * @return one row per method
     */
    public static Map<String, Row> run(int steps, int batch, double target, long... seeds) {
        Map<String, Row> out = new LinkedHashMap<>();
        for (String m : METHODS) {
            double loss = 0;
            double acc = 0;
            double hit = 0;
            double millis = 0;
            int ok = 0;
            int diverged = 0;
            for (long seed : seeds) {
                double[] r = runOne(m, steps, batch, target, seed);
                hit += r[2];
                millis += r[3];
                if (!Double.isFinite(r[0]) || r[0] > 10) {
                    diverged++;
                } else {
                    loss += r[0];
                    acc += r[1];
                    ok++;
                }
            }
            out.put(m, new Row(ok == 0 ? Double.NaN : loss / ok, ok == 0 ? 0 : acc / ok, hit / seeds.length,
                    millis / seeds.length, diverged));
        }
        return out;
    }

    private static double[] runOne(String method, int steps, int batch, double target, long seed) {
        ToyTasks.Data data = ToyTasks.spiral(100, 3, 0.1, seed);
        SimpleMlp net = SimpleMlp.tanh(seed, 2, 32, 32, 3);
        SplittableRandom r = new SplittableRandom(seed * 31 + 7);
        Trainer trainer = trainer(method, seed);
        int hit = steps;
        long t0 = System.nanoTime();
        for (int t = 0; t < steps; t++) {
            Tensor[] b = NumericsTasks.batch(data, batch, r);
            trainer.step(net, b[0], b[1], t);
            if (hit == steps && (t + 1) % 10 == 0 && NumericsTasks.loss(net, data) < target) {
                hit = t + 1;
            }
        }
        double ms = (System.nanoTime() - t0) / 1e6 / steps;
        return new double[] {NumericsTasks.loss(net, data), NumericsTasks.accuracy(net, data), hit, ms};
    }

    private static Var ce(SimpleMlp net, Tensor x, Tensor y) {
        return Functional.crossEntropy(net.forward(Var.of(x)), y);
    }

    private static Trainer plain(Optimizer opt) {
        return (net, x, y, t) -> {
            ce(net, x, y).backward();
            opt.step(net.parameters());
        };
    }

    private static Trainer trainer(String method, long seed) {
        SplittableRandom labels = new SplittableRandom(seed + 99);
        return switch (method) {
            case "AdamW" -> plain(Optimizer.adamW(0.02f, 1e-4f));
            case "Shampoo" -> plain(new Shampoo(0.05f, 0.9f, 10, 64));
            case "SOAP" -> plain(Soap.of(0.01f));
            case "PSGD Kron" -> plain(PsgdKron.of(0.01f));
            case "Sophia-H" -> {
                Sophia sophia = Sophia.of(0.01f);
                yield (net, x, y, t) -> {
                    if (t % 10 == 0) {
                        sophia.updateHessianHutchinson(net.parameters(), () -> ce(net, x, y), seed + t);
                    }
                    ce(net, x, y).backward();
                    sophia.step(net.parameters());
                };
            }
            case "K-FAC", "EKFAC" -> {
                Kfac kfac = method.equals("K-FAC") ? Kfac.kfac(0.05f, 3e-3f) : Kfac.ekfac(0.05f, 3e-3f);
                yield (net, x, y, t) -> {
                    KroneckerCapture cap = new KroneckerCapture();
                    Var logits = NumericsTasks.forward(net, Var.of(x), cap);
                    Tensor sampled = Sophia.sampleLabels(logits.value(), labels);
                    kfac.updateCurvature(cap.factors(Functional.crossEntropy(logits, sampled)));
                    Functional.crossEntropy(logits, y).backward();
                    kfac.step(net.parameters());
                };
            }
            case "AdaFisher" -> {
                AdaFisher ada = AdaFisher.of(0.02f);
                yield (net, x, y, t) -> {
                    KroneckerCapture cap = new KroneckerCapture();
                    Var logits = NumericsTasks.forward(net, Var.of(x), cap);
                    Tensor sampled = Sophia.sampleLabels(logits.value(), labels);
                    ada.updateCurvature(cap.factors(Functional.crossEntropy(logits, sampled)));
                    Functional.crossEntropy(logits, y).backward();
                    ada.step(net.parameters());
                };
            }
            case "AdaHessian" -> {
                AdaHessian ada = AdaHessian.of(0.15f, seed);
                yield (net, x, y, t) -> {
                    ada.updateHessian(net.parameters(), () -> ce(net, x, y));
                    ce(net, x, y).backward();
                    ada.step(net.parameters());
                };
            }
            case "oLBFGS" -> {
                Lbfgs lbfgs = Lbfgs.online(0.2f, 10, 1e-2);
                yield (net, x, y, t) -> {
                    ce(net, x, y).backward();
                    lbfgs.step(net.parameters(), () -> {
                        for (Var p : net.parameters()) {
                            p.zeroGrad();
                        }
                        Var l = ce(net, x, y);
                        l.backward();
                        return l.value().item();
                    });
                };
            }
            default -> throw new IllegalArgumentException(method);
        };
    }
}
