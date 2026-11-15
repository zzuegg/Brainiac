package io.github.zzuegg.brainiac.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Turns a policy network's output into an action distribution: a categorical over
 * discrete actions (the output is the logits), or a diagonal Gaussian over continuous
 * ones (the output is the mean, and a learned, state-independent log standard deviation
 * gives the spread).
 */
final class PolicyHead {

    private static final float HALF_LOG_2PI = (float) (0.5 * Math.log(2 * Math.PI));

    final boolean discrete;
    final int size;
    final Space.Box box;
    final Var logStd;

    PolicyHead(Space actions, float initialLogStd, Device device) {
        if (actions instanceof Space.Discrete d) {
            discrete = true;
            size = d.n();
            box = null;
            logStd = null;
        } else {
            box = (Space.Box) actions;
            discrete = false;
            size = box.size();
            float[] s = new float[size];
            java.util.Arrays.fill(s, initialLogStd);
            logStd = Var.param(Tensor.of(Shape.of(1, size), s).to(device));
        }
    }

    /** Values per action as stored: 1 for discrete, the vector length for continuous. */
    int actionSize() {
        return discrete ? 1 : size;
    }

    /** Samples (or picks greedily) from one output row; continuous samples are not clipped. */
    float[] sample(float[] out, SplittableRandom random, boolean explore) {
        if (discrete) {
            if (!explore) {
                return new float[] {Nets.argmax(out)};
            }
            double max = Double.NEGATIVE_INFINITY;
            for (float v : out) {
                max = Math.max(max, v);
            }
            double[] p = new double[size];
            double sum = 0;
            for (int i = 0; i < size; i++) {
                p[i] = Math.exp(out[i] - max);
                sum += p[i];
            }
            double u = random.nextDouble() * sum;
            for (int i = 0; i < size; i++) {
                u -= p[i];
                if (u <= 0) {
                    return new float[] {i};
                }
            }
            return new float[] {size - 1};
        }
        float[] a = new float[size];
        float[] s = logStd.value().toArray();
        for (int i = 0; i < size; i++) {
            a[i] = explore ? (float) (out[i] + Math.exp(s[i]) * random.nextGaussian()) : out[i];
        }
        return a;
    }

    /** What the environment receives: continuous actions clipped into the box. */
    float[] toEnvironment(float[] action) {
        return discrete ? action : box.clip(action);
    }

    /** Log-probability of the actions, {@code [n, 1]}. */
    Var logProb(Var out, Tensor actions) {
        if (discrete) {
            Var logp = Activation.LOG_SOFTMAX.forward(out);
            return logp.mul(Nets.oneHot(actions.toArray(), size, out.value().device())).sum(1, true);
        }
        Var z = out.neg().add(actions).mul(logStd.neg().exp());
        return z.square().mul(-0.5f).sub(logStd).sub(HALF_LOG_2PI).sum(1, true);
    }

    /** Entropy of the distribution, {@code [n, 1]} (or {@code [1, 1]} for a Gaussian). */
    Var entropy(Var out) {
        if (discrete) {
            Var logp = Activation.LOG_SOFTMAX.forward(out);
            return logp.exp().mul(logp).sum(1, true).neg();
        }
        return logStd.add(HALF_LOG_2PI + 0.5f).sum(1, true);
    }
}
