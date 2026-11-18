package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.EdmDenoiser;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/**
 * Distribution matching distillation (DMD, Yin et al.,
 * <a href="https://arxiv.org/abs/2311.18828">arXiv:2311.18828</a>; with DMD2's two
 * time-scale updates, <a href="https://arxiv.org/abs/2405.14867">arXiv:2405.14867</a>): a
 * one-step generator {@code G(z)} is trained so that its output distribution matches the
 * teacher diffusion model's. The gradient of the KL divergence between the two, diffused to a
 * random noise level, is the difference of their scores: {@code s_fake - s_real}. The real
 * score is the frozen teacher; the fake score is a second denoiser trained online, by plain
 * denoising, on the generator's own samples. Generated points move against that difference
 * (normalised per sample by the teacher's denoising error), and an optional regression loss
 * on a few precomputed (noise, teacher ODE output) pairs anchors the mode structure.
 *
 * <p>Both denoisers are data-prediction models in the {@code x = x0 + sigma n} form
 * ({@link EdmDenoiser}), where {@code s = (D(x) - x) / sigma^2}, so
 * {@code s_fake - s_real = (D_fake - D_real) / sigma^2}.
 */
public final class DistributionMatchingDistiller {
    private final Denoiser teacher;
    private final EdmDenoiser fake;
    private final Function<Var, Var> generator;
    private final List<Var> generatorParameters;
    private final Optimizer generatorOptimizer;
    private final Optimizer fakeOptimizer;
    private float sigmaMin = 0.05f;
    private float sigmaMax = 3f;
    private int fakeSteps = 2;
    private Tensor pairedNoise;
    private Tensor pairedTargets;
    private float regressionWeight;

    private DistributionMatchingDistiller(Denoiser teacher, EdmDenoiser fake, Function<Var, Var> generator,
            List<Var> generatorParameters, Optimizer generatorOptimizer, Optimizer fakeOptimizer) {
        this.teacher = Objects.requireNonNull(teacher);
        this.fake = Objects.requireNonNull(fake);
        this.generator = Objects.requireNonNull(generator);
        this.generatorParameters = List.copyOf(generatorParameters);
        this.generatorOptimizer = Objects.requireNonNull(generatorOptimizer);
        this.fakeOptimizer = Objects.requireNonNull(fakeOptimizer);
    }

    /**
     * Creates the distiller.
     *
     * @param teacher the frozen data-prediction teacher {@code D_real(x; sigma)}
     * @param fake the fake-score denoiser, trained online on generator samples
     * @param generator maps noise {@code [n, dim]} to samples in one step
     * @param generatorParameters its parameters
     * @param generatorOptimizer updates them
     * @param fakeOptimizer updates the fake-score denoiser
     * @return the distiller
     */
    public static DistributionMatchingDistiller of(Denoiser teacher, EdmDenoiser fake, Function<Var, Var> generator,
            List<Var> generatorParameters, Optimizer generatorOptimizer, Optimizer fakeOptimizer) {
        return new DistributionMatchingDistiller(teacher, fake, generator, generatorParameters, generatorOptimizer,
                fakeOptimizer);
    }

    /**
     * Sets the range of noise levels at which the two scores are compared (log-uniform).
     *
     * @param min the smallest level
     * @param max the largest level
     * @return this distiller
     */
    public DistributionMatchingDistiller noiseRange(float min, float max) {
        this.sigmaMin = min;
        this.sigmaMax = max;
        return this;
    }

    /**
     * Sets how many fake-score updates follow each generator update (DMD2 uses 5; 1 is DMD).
     *
     * @param steps the updates
     * @return this distiller
     */
    public DistributionMatchingDistiller fakeSteps(int steps) {
        this.fakeSteps = steps;
        return this;
    }

    /**
     * Adds DMD's regression loss: the generator must map these noises to the teacher's ODE
     * outputs for them.
     *
     * @param noise the noises {@code [m, dim]}
     * @param targets the teacher's deterministic samples from them
     * @param weight the loss weight
     * @return this distiller
     */
    public DistributionMatchingDistiller regression(Tensor noise, Tensor targets, float weight) {
        this.pairedNoise = noise;
        this.pairedTargets = targets;
        this.regressionWeight = weight;
        return this;
    }

    /**
     * One round: a generator update on the distribution-matching (and regression) loss, then
     * the fake-score updates on fresh generator samples.
     *
     * @param noise generator inputs {@code [n, dim]}
     * @param random the source of noise levels and noise
     * @return {@code {mean |grad| of the matching direction, regression loss, fake-score loss}}
     */
    public float[] step(Tensor noise, RandomGenerator random) {
        int n = noise.shape().dim(0);
        Var x = generator.apply(Var.of(noise));
        Tensor xv = x.value();
        float[] sigma = new float[n];
        for (int i = 0; i < n; i++) {
            sigma[i] = (float) Math.exp(Math.log(sigmaMin) + random.nextDouble() * Math.log(sigmaMax / sigmaMin));
        }
        Tensor sc = Tensor.of(Shape.of(n, 1), sigma.clone());
        Tensor noisy = xv.add(Tensor.normal(xv.shape(), 0, 1, random).mul(sc));
        Tensor sigmaRow = Tensor.of(Shape.of(n), sigma);
        Tensor real = teacher.apply(Var.of(noisy), sigmaRow).value();
        Tensor fakeD = fake.apply(Var.of(noisy), sigmaRow).value();
        // DMD's normaliser: the teacher's mean absolute denoising error per sample
        Tensor weight = xv.sub(real).abs().mean(1, true).add(1e-4f);
        Tensor grad = fakeD.sub(real).div(weight);
        Tensor target = xv.sub(grad);
        Var loss = x.sub(Var.of(target)).square().sum(1, false).mean().mul(0.5f);
        float regression = 0;
        if (pairedNoise != null && regressionWeight > 0) {
            Var reg = generator.apply(Var.of(pairedNoise)).sub(Var.of(pairedTargets)).square().sum(1, false).mean();
            regression = reg.value().item();
            loss = loss.add(reg.mul(regressionWeight));
        }
        loss.backward();
        generatorOptimizer.step(generatorParameters);
        for (Var p : fake.parameters()) {
            p.zeroGrad();
        }
        float fakeLoss = 0;
        for (int k = 0; k < fakeSteps; k++) {
            Tensor samples = generator.apply(Var.of(noise)).value();
            Var fl = fake.loss(samples, random);
            fakeLoss = fl.value().item();
            fl.backward();
            fakeOptimizer.step(fake.parameters());
            for (Var p : generatorParameters) {
                p.zeroGrad();
            }
        }
        return new float[] {grad.abs().mean().item(), regression, fakeLoss};
    }

    /**
     * Generates in one step.
     *
     * @param noise standard normal noise
     * @return the samples
     */
    public Tensor generate(Tensor noise) {
        return generator.apply(Var.of(noise)).value();
    }
}
