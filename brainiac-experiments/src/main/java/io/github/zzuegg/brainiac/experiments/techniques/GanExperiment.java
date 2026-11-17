package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GradientPenalty;
import io.github.zzuegg.brainiac.experimental.nn.SpectralNormDense;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * A Wasserstein GAN on a 2-D toy distribution (two Gaussian blobs), with the critic's
 * 1-Lipschitz constraint enforced either by the WGAN-GP gradient penalty
 * ({@link GradientPenalty#wgan}, computed without second-order autograd) or by spectral
 * normalization of every critic layer ({@link SpectralNormDense}). The critic takes five
 * steps per generator step, as in the WGAN-GP paper, with Adam(1e-3, 0.5, 0.9).
 *
 * <p>Reports how far the generated distribution is from the data with a sliced Wasserstein
 * distance (the mean 1-D Wasserstein distance over random projections). In these short runs
 * the spectrally normalized critic converges; the gradient-penalty GAN oscillates around the
 * two blobs (its critic is accurate, see {@link #criticEstimate}, but WGAN-GP needs far more
 * generator steps on mixtures than a unit test can afford).
 */
public final class GanExperiment {

    /** How the critic is kept 1-Lipschitz. */
    public enum Constraint {
        /** WGAN-GP gradient penalty with weight 10. */
        GRADIENT_PENALTY,
        /** Spectral normalization of the critic's layers. */
        SPECTRAL_NORM
    }

    private GanExperiment() {
    }

    /**
     * Samples the data distribution: two blobs at {@code (-2, 0)} and {@code (2, 1)}.
     *
     * @param n the samples
     * @param r the random source
     * @return {@code [n, 2]}
     */
    public static Tensor data(int n, SplittableRandom r) {
        float[] x = new float[2 * n];
        for (int i = 0; i < n; i++) {
            boolean right = r.nextBoolean();
            x[2 * i] = (float) ((right ? 2 : -2) + 0.3 * r.nextGaussian());
            x[2 * i + 1] = (float) ((right ? 1 : 0) + 0.3 * r.nextGaussian());
        }
        return Tensor.of(Shape.of(n, 2), x);
    }

    /**
     * Trains a GAN.
     *
     * @param constraint the Lipschitz constraint
     * @param generatorSteps generator steps (the critic takes five per step)
     * @param seed the seed
     * @return {@code {distance before training, distance after}} to fresh data
     */
    public static float[] run(Constraint constraint, int generatorSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Model generator = Model.sequential().input(2).dense(64, Activation.RELU).dense(64, Activation.RELU)
                .dense(2, Activation.IDENTITY).seed(seed).build();
        Model.Builder cb = Model.sequential().input(2).seed(seed + 1);
        if (constraint == Constraint.SPECTRAL_NORM) {
            cb.add(SpectralNormDense.spec(64, Activation.leakyRelu(0.2f)))
                    .add(SpectralNormDense.spec(64, Activation.leakyRelu(0.2f)))
                    .add(SpectralNormDense.spec(1, Activation.IDENTITY));
        } else {
            cb.dense(64, Activation.leakyRelu(0.2f)).dense(64, Activation.leakyRelu(0.2f)).dense(1, Activation.IDENTITY);
        }
        Model critic = cb.build();
        // two time-scale updates: the critic learns faster than the generator
        Optimizer gOpt = Optimizer.adam(5e-4f, 0.5f, 0.9f, 1e-8f);
        Optimizer cOpt = Optimizer.adam(1e-3f, 0.5f, 0.9f, 1e-8f);
        float before = distance(generator, r);
        for (int s = 0; s < generatorSteps; s++) {
            for (int c = 0; c < 5; c++) {
                Tensor real = data(64, r);
                Tensor fake = generator.forward(Var.of(noise(64, r))).value();
                Var loss = critic.forward(Var.of(fake)).mean().sub(critic.forward(Var.of(real)).mean());
                if (constraint == Constraint.GRADIENT_PENALTY) {
                    loss = loss.add(GradientPenalty.wgan(critic::forward, critic.parameters(), real, fake, r).mul(10f));
                }
                loss.backward();
                cOpt.step(critic.parameters());
            }
            critic.forward(generator.forward(Var.of(noise(64, r)))).mean().neg().backward();
            gOpt.step(generator.parameters());
            for (Var p : critic.parameters()) {
                p.zeroGrad();
            }
        }
        return new float[] {before, distance(generator, r)};
    }

    /**
     * Trains only a WGAN-GP critic between two fixed distributions whose Wasserstein-1
     * distance is known exactly: a standard normal and the same normal shifted by
     * {@code shift} along the first axis (distance {@code shift}). The trained critic's
     * {@code E[D(real)] - E[D(fake)]} on fresh samples estimates that distance, and its
     * input gradients should have norm about 1.
     *
     * @param shift the shift, the true distance
     * @param steps critic steps
     * @param seed the seed
     * @return {@code {Wasserstein estimate, mean input-gradient norm at the data}}
     */
    public static float[] criticEstimate(float shift, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Model critic = Model.sequential().input(2).dense(64, Activation.leakyRelu(0.2f))
                .dense(64, Activation.leakyRelu(0.2f)).dense(1, Activation.IDENTITY).seed(seed).build();
        Optimizer opt = Optimizer.adam(1e-3f, 0.5f, 0.9f, 1e-8f);
        Tensor offset = Tensor.of(shift, 0f);
        for (int s = 0; s < steps; s++) {
            Tensor fake = noise(128, r);
            Tensor real = noise(128, r).add(offset);
            critic.forward(Var.of(fake)).mean().sub(critic.forward(Var.of(real)).mean())
                    .add(GradientPenalty.wgan(critic::forward, critic.parameters(), real, fake, r).mul(10f)).backward();
            opt.step(critic.parameters());
        }
        Tensor fake = noise(2048, r);
        Tensor real = noise(2048, r).add(offset);
        float estimate = critic.forward(Var.of(real)).mean().sub(critic.forward(Var.of(fake)).mean()).value().item();
        float norm = GradientPenalty.gradientNorm(critic::forward, critic.parameters(), real, 1e-2f).mean().value()
                .item();
        return new float[] {estimate, norm};
    }

    private static Tensor noise(int n, SplittableRandom r) {
        return Tensor.normal(Shape.of(n, 2), 0, 1, r);
    }

    /** The sliced Wasserstein-1 distance between 512 generated and 512 data samples. */
    private static float distance(Model generator, SplittableRandom r) {
        float[] g = generator.forward(Var.of(noise(512, r))).value().toArray();
        float[] d = data(512, r).toArray();
        double total = 0;
        for (int k = 0; k < 16; k++) {
            double angle = Math.PI * k / 16;
            double cx = Math.cos(angle);
            double cy = Math.sin(angle);
            double[] pg = new double[512];
            double[] pd = new double[512];
            for (int i = 0; i < 512; i++) {
                pg[i] = cx * g[2 * i] + cy * g[2 * i + 1];
                pd[i] = cx * d[2 * i] + cy * d[2 * i + 1];
            }
            java.util.Arrays.sort(pg);
            java.util.Arrays.sort(pd);
            for (int i = 0; i < 512; i++) {
                total += Math.abs(pg[i] - pd[i]) / 512 / 16;
            }
        }
        return (float) total;
    }
}
