package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.Denoiser;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Flow matching with straight (rectified-flow) paths: Lipman et al.,
 * <a href="https://arxiv.org/abs/2210.02747">2210.02747</a>, and Liu et al.,
 * <a href="https://arxiv.org/abs/2209.03003">2209.03003</a>. A sample on the straight line
 * from noise {@code z} (at {@code t = 0}) to data {@code x} (at {@code t = 1}) is
 * {@code x_t = (1 - t) z + t x}; the {@link Denoiser} regresses the constant velocity
 * {@code x - z}. Generation integrates {@code dx/dt = v(x, t)} with the
 * {@linkplain #sampleEuler Euler} or the second-order {@linkplain #sampleHeun Heun} method.
 *
 * <p>{@link #reflowStep Reflow} retrains on (noise, generated sample) pairs produced by the
 * model itself. Those couplings no longer cross, so the learned paths straighten and a
 * single Euler step gets close to the many-step result. Labels with dropout give
 * classifier-free guidance on the velocity.
 */
@SuppressWarnings("exports")   // experimental.* packages are exported by the lead at integration
public final class FlowMatching {
    private final Denoiser net;
    private final Optimizer optimizer;
    private final float labelDropout;
    private final SplittableRandom random;

    private FlowMatching(Denoiser net, float learningRate, float labelDropout, long seed) {
        this.net = net;
        this.optimizer = Optimizer.adam(learningRate);
        this.labelDropout = labelDropout;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Creates a model.
     *
     * @param net the velocity network
     * @param learningRate the Adam learning rate
     * @param labelDropout the share of labels dropped while training, for guidance
     * @param seed the seed for noise and times
     * @return the model
     */
    public static FlowMatching of(Denoiser net, float learningRate, float labelDropout, long seed) {
        return new FlowMatching(net, learningRate, labelDropout, seed);
    }

    /**
     * Changes the learning rate, for example to anneal it at the end of training.
     *
     * @param learningRate the new rate
     */
    public void learningRate(float learningRate) {
        optimizer.learningRate(learningRate);
    }

    /**
     * One step of conditional flow matching with fresh noise.
     *
     * @param data data samples
     * @param labels their labels, or -1
     * @return the loss before the step
     */
    public float trainStep(float[][] data, int[] labels) {
        return reflowStep(Nets.rows(Nets.gaussian(data.length, net.dimension(), random)), data, labels);
    }

    /**
     * One step on given couplings: regresses {@code target - source} along their straight
     * line. With independent noise as the source this is plain flow matching; with
     * (noise, model sample) pairs it is reflow.
     *
     * @param source the start points (noise)
     * @param target the end points (data or samples)
     * @param labels the labels, or -1
     * @return the loss before the step
     */
    public float reflowStep(float[][] source, float[][] target, int[] labels) {
        int n = source.length;
        float[] t = new float[n];
        float[] oneMinus = new float[n];
        int[] l = labels.clone();
        for (int i = 0; i < n; i++) {
            t[i] = (float) random.nextDouble();
            oneMinus[i] = 1 - t[i];
            if (random.nextDouble() < labelDropout) {
                l[i] = -1;
            }
        }
        Tensor z = Nets.matrix(source);
        Tensor x = Nets.matrix(target);
        Tensor xt = z.mul(Nets.column(oneMinus)).add(x.mul(Nets.column(t)));
        Var loss = net.forward(Var.of(xt), t, l).sub(x.sub(z)).square().mean();
        float value = loss.value().item();
        loss.backward();
        Nets.clip(net.parameters(), 1f);
        optimizer.step(net.parameters());
        return value;
    }

    /**
     * Evaluates the velocity field, with guidance.
     *
     * @param x points
     * @param t the time, the same for all
     * @param labels labels, or -1
     * @param guidance 1 for none, larger to push towards the label
     * @return the velocities
     */
    public float[][] velocity(float[][] x, float t, int[] labels, float guidance) {
        float[] ts = new float[x.length];
        Arrays.fill(ts, t);
        Tensor xt = Nets.matrix(x);
        float[][] cond = Nets.rows(net.forward(Var.of(xt), ts, labels).value());
        if (guidance == 1f) {
            return cond;
        }
        int[] none = new int[x.length];
        Arrays.fill(none, -1);
        float[][] unc = Nets.rows(net.forward(Var.of(xt), ts, none).value());
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < cond[i].length; j++) {
                cond[i][j] = unc[i][j] + guidance * (cond[i][j] - unc[i][j]);
            }
        }
        return cond;
    }

    /**
     * Draws the starting noise for a sampler.
     *
     * @param n the number of samples
     * @param seed the seed
     * @return standard normal points
     */
    public float[][] noise(int n, long seed) {
        return Nets.rows(Nets.gaussian(n, net.dimension(), new SplittableRandom(seed)));
    }

    /**
     * Integrates from noise to data with Euler steps.
     *
     * @param start the noise (not modified)
     * @param labels labels, or -1
     * @param steps the number of steps
     * @param guidance the guidance weight, 1 for none
     * @param trajectory receives the state after every step, or {@code null}
     * @return the samples
     */
    public float[][] sampleEuler(float[][] start, int[] labels, int steps, float guidance, List<float[][]> trajectory) {
        float[][] x = copy(start);
        float h = 1f / steps;
        for (int k = 0; k < steps; k++) {
            axpy(x, h, velocity(x, k * h, labels, guidance));
            if (trajectory != null) {
                trajectory.add(copy(x));
            }
        }
        return x;
    }

    /**
     * Integrates from noise to data with Heun's method (explicit trapezoid): two
     * evaluations per step, second-order accurate.
     *
     * @param start the noise (not modified)
     * @param labels labels, or -1
     * @param steps the number of steps
     * @param guidance the guidance weight, 1 for none
     * @param trajectory receives the state after every step, or {@code null}
     * @return the samples
     */
    public float[][] sampleHeun(float[][] start, int[] labels, int steps, float guidance, List<float[][]> trajectory) {
        float[][] x = copy(start);
        float h = 1f / steps;
        for (int k = 0; k < steps; k++) {
            float t = k * h;
            float[][] v1 = velocity(x, t, labels, guidance);
            float[][] pred = copy(x);
            axpy(pred, h, v1);
            float[][] v2 = velocity(pred, t + h, labels, guidance);
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[i].length; j++) {
                    x[i][j] += h * 0.5f * (v1[i][j] + v2[i][j]);
                }
            }
            if (trajectory != null) {
                trajectory.add(copy(x));
            }
        }
        return x;
    }

    private static void axpy(float[][] x, float a, float[][] y) {
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                x[i][j] += a * y[i][j];
            }
        }
    }

    static float[][] copy(float[][] a) {
        float[][] c = new float[a.length][];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i].clone();
        }
        return c;
    }
}
