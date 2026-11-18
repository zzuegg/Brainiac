package io.github.zzuegg.brainiac.experimental.robust;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Input-space adversarial attacks on a classifier given as a logits function: the fast
 * gradient sign method (FGSM; Goodfellow, Shlens and Szegedy 2015,
 * <a href="https://arxiv.org/abs/1412.6572">arXiv:1412.6572</a>) and projected gradient
 * descent with random starts and restarts (PGD; Madry et al., ICLR 2018,
 * <a href="https://arxiv.org/abs/1706.06083">arXiv:1706.06083</a>), in the {@code ℓ∞} and
 * {@code ℓ2} norms. Each attack maximises the cross-entropy of the true class within the
 * budget, optionally inside a valid input box. PGD with restarts keeps, per example, the
 * restart with the highest loss (or the first that flips the prediction).
 */
public final class AdversarialAttacks {

    /** The threat model's norm. */
    public enum Norm {
        /** {@code ||δ||∞ <= ε} */
        LINF,
        /** {@code ||δ||2 <= ε}, per example */
        L2
    }

    private AdversarialAttacks() {
    }

    /**
     * The gradient of the summed cross-entropy with respect to the inputs.
     *
     * @param logits the model
     * @param x the inputs
     * @param ids the true classes
     * @return {@code dL/dx}
     */
    public static Tensor inputGradient(UnaryOperator<Var> logits, Tensor x, Tensor ids) {
        Var in = Var.param(x);
        Var loss = Functional.crossEntropy(logits.apply(in), ids).mul((float) ids.shape().dim(0));
        return Var.gradients(loss, List.of(in), false).get(0).value();
    }

    /**
     * FGSM: one signed gradient step of size {@code ε} ({@code ℓ∞}), or one normalised gradient
     * step ({@code ℓ2}).
     *
     * @param logits the model
     * @param x the clean inputs
     * @param ids the true classes
     * @param norm the norm
     * @param epsilon the budget
     * @return the adversarial inputs
     */
    public static Tensor fgsm(UnaryOperator<Var> logits, Tensor x, Tensor ids, Norm norm, float epsilon) {
        Tensor g = inputGradient(logits, x, ids);
        return x.add(direction(g, norm).mul(epsilon));
    }

    /**
     * PGD from a uniform random start inside the ball.
     *
     * @param logits the model
     * @param x the clean inputs
     * @param ids the true classes
     * @param norm the norm
     * @param epsilon the budget
     * @param stepSize the step size, typically {@code 2.5 ε / steps}
     * @param steps the ascent steps
     * @param restarts independent random starts; the strongest per example is kept
     * @param random the random source of the starts
     * @return the adversarial inputs
     */
    public static Tensor pgd(UnaryOperator<Var> logits, Tensor x, Tensor ids, Norm norm, float epsilon,
            float stepSize, int steps, int restarts, SplittableRandom random) {
        int n = x.shape().dim(0);
        int d = Math.toIntExact(x.size() / n);
        float[] best = null;
        double[] bestLoss = new double[n];
        java.util.Arrays.fill(bestLoss, Double.NEGATIVE_INFINITY);
        for (int r = 0; r < Math.max(1, restarts); r++) {
            Tensor adv = x.add(randomStart(x.shape(), n, d, norm, epsilon, random));
            for (int s = 0; s < steps; s++) {
                Tensor g = inputGradient(logits, adv, ids);
                adv = project(adv.add(direction(g, norm).mul(stepSize)), x, norm, epsilon);
            }
            double[] loss = perExampleLoss(logits, adv, ids);
            float[] a = adv.toArray();
            if (best == null) {
                best = a;
                bestLoss = loss;
            } else {
                for (int i = 0; i < n; i++) {
                    if (loss[i] > bestLoss[i]) {
                        bestLoss[i] = loss[i];
                        System.arraycopy(a, i * d, best, i * d, d);
                    }
                }
            }
        }
        return Tensor.of(x.shape(), best);
    }

    /**
     * Clean or adversarial accuracy.
     *
     * @param logits the model
     * @param x the inputs
     * @param ids the true classes
     * @return the share classified correctly
     */
    public static double accuracy(UnaryOperator<Var> logits, Tensor x, Tensor ids) {
        return logits.apply(Var.of(x)).value().argmax(1).equalTo(ids).mean().item();
    }

    /**
     * The per-example cross-entropy.
     *
     * @param logits the model
     * @param x the inputs
     * @param ids the true classes
     * @return one loss per example
     */
    public static double[] perExampleLoss(UnaryOperator<Var> logits, Tensor x, Tensor ids) {
        float[] l = Functional.crossEntropyPerRow(logits.apply(Var.of(x)), ids).value().toArray();
        double[] out = new double[l.length];
        for (int i = 0; i < l.length; i++) {
            out[i] = l[i];
        }
        return out;
    }

    /**
     * The steepest-ascent direction of unit norm: the sign for {@code ℓ∞}, the per-example
     * normalised gradient for {@code ℓ2}.
     *
     * @param g the gradient {@code [n, ...]}
     * @param norm the norm
     * @return the direction
     */
    public static Tensor direction(Tensor g, Norm norm) {
        if (norm == Norm.LINF) {
            return g.sign();
        }
        int n = g.shape().dim(0);
        int d = Math.toIntExact(g.size() / n);
        float[] v = g.toArray();
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < d; j++) {
                s += v[i * d + j] * (double) v[i * d + j];
            }
            float inv = (float) (1 / Math.max(Math.sqrt(s), 1e-12));
            for (int j = 0; j < d; j++) {
                v[i * d + j] *= inv;
            }
        }
        return Tensor.of(g.shape(), v);
    }

    /**
     * Projects inputs back into the ball around the clean ones.
     *
     * @param adv the perturbed inputs
     * @param x the clean inputs
     * @param norm the norm
     * @param epsilon the radius
     * @return the projection
     */
    public static Tensor project(Tensor adv, Tensor x, Norm norm, float epsilon) {
        Tensor delta = adv.sub(x);
        float[] v = delta.toArray();
        if (norm == Norm.LINF) {
            for (int i = 0; i < v.length; i++) {
                v[i] = Math.max(-epsilon, Math.min(epsilon, v[i]));
            }
            return x.add(Tensor.of(x.shape(), v));
        }
        int n = x.shape().dim(0);
        int d = Math.toIntExact(x.size() / n);
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < d; j++) {
                s += v[i * d + j] * (double) v[i * d + j];
            }
            double norm2 = Math.sqrt(s);
            if (norm2 > epsilon) {
                float f = (float) (epsilon / norm2);
                for (int j = 0; j < d; j++) {
                    v[i * d + j] *= f;
                }
            }
        }
        return x.add(Tensor.of(x.shape(), v));
    }

    private static Tensor randomStart(io.github.zzuegg.brainiac.tensor.Shape shape, int n, int d, Norm norm,
            float epsilon, SplittableRandom random) {
        float[] v = new float[n * d];
        if (norm == Norm.LINF) {
            for (int i = 0; i < v.length; i++) {
                v[i] = (float) random.nextDouble(-epsilon, epsilon);
            }
        } else {
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = 0; j < d; j++) {
                    v[i * d + j] = (float) random.nextGaussian();
                    s += v[i * d + j] * (double) v[i * d + j];
                }
                double radius = epsilon * Math.pow(random.nextDouble(), 1.0 / d) / Math.sqrt(s);
                for (int j = 0; j < d; j++) {
                    v[i * d + j] *= (float) radius;
                }
            }
        }
        return Tensor.of(shape, v);
    }
}
