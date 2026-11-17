package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * The training loop of a generative adversarial network: alternating steps of a
 * discriminator, which learns to tell real samples from generated ones, and a generator,
 * which learns to fool it, each with its own optimizer. An optional R1 penalty (Mescheder et
 * al., 2018) keeps the discriminator's gradient on real data small; spectral normalization is
 * {@link Functional#spectralNorm}.
 */
public final class Adversarial {

    /** The adversarial loss. */
    public enum Loss {
        /** The non-saturating logistic loss of the original GAN, as StyleGAN uses. */
        NON_SATURATING,
        /** The hinge loss, as BigGAN and SAGAN use. */
        HINGE,
        /** The Wasserstein critic loss; pair it with a penalty or spectral normalization. */
        WASSERSTEIN
    }

    /**
     * The losses of one step.
     *
     * @param discriminator the discriminator's loss, penalty included, at its last step
     * @param generator the generator's loss
     */
    public record Step(float discriminator, float generator) {
    }

    private final Function<Var, Var> generator;
    private final List<Var> generatorParameters;
    private final Optimizer generatorOptimizer;
    private final Function<Var, Var> discriminator;
    private final List<Var> discriminatorParameters;
    private final Optimizer discriminatorOptimizer;
    private final Loss loss;
    private float r1Gamma;
    private int discriminatorSteps = 1;

    /**
     * @param generator maps noise to samples
     * @param generatorParameters its parameters
     * @param generatorOptimizer updates them
     * @param discriminator maps samples to one score each, higher meaning real
     * @param discriminatorParameters its parameters
     * @param discriminatorOptimizer updates them
     * @param loss the adversarial loss
     */
    public Adversarial(Function<Var, Var> generator, List<Var> generatorParameters, Optimizer generatorOptimizer,
            Function<Var, Var> discriminator, List<Var> discriminatorParameters, Optimizer discriminatorOptimizer,
            Loss loss) {
        this.generator = Objects.requireNonNull(generator);
        this.generatorParameters = List.copyOf(generatorParameters);
        this.generatorOptimizer = Objects.requireNonNull(generatorOptimizer);
        this.discriminator = Objects.requireNonNull(discriminator);
        this.discriminatorParameters = List.copyOf(discriminatorParameters);
        this.discriminatorOptimizer = Objects.requireNonNull(discriminatorOptimizer);
        this.loss = Objects.requireNonNull(loss);
    }

    /**
     * @param gamma the R1 penalty's weight on real data, 0 for none; 10 is common
     * @return this
     */
    public Adversarial withR1(float gamma) {
        this.r1Gamma = gamma;
        return this;
    }

    /**
     * @param steps discriminator steps per generator step, as WGAN uses 5
     * @return this
     */
    public Adversarial withDiscriminatorSteps(int steps) {
        if (steps < 1) {
            throw new IllegalArgumentException("discriminator steps must be at least 1, got " + steps);
        }
        this.discriminatorSteps = steps;
        return this;
    }

    /**
     * One round: {@code discriminatorSteps} discriminator updates, then one generator update.
     *
     * @param real a batch of real samples; the same batch serves every discriminator step
     * @param noise draws a batch of generator inputs
     * @return the losses
     */
    public Step step(Tensor real, Supplier<Tensor> noise) {
        float dLoss = 0;
        for (int k = 0; k < discriminatorSteps; k++) {
            clear();
            Var fake = generator.apply(Var.of(noise.get())).detach();
            Var r = discriminator.apply(Var.of(real));
            Var f = discriminator.apply(fake);
            Var d = switch (loss) {
                case NON_SATURATING -> r.neg().softplus().mean().add(f.softplus().mean());
                case HINGE -> r.neg().add(1).relu().mean().add(f.add(1).relu().mean());
                case WASSERSTEIN -> f.mean().sub(r.mean());
            };
            if (r1Gamma > 0) {
                d = d.add(Functional.r1Penalty(discriminator, real, r1Gamma));
            }
            d.backward();
            dLoss = d.value().item();
            for (Var p : generatorParameters) {
                p.zeroGrad();
            }
            discriminatorOptimizer.step(discriminatorParameters);
        }
        clear();
        Var s = discriminator.apply(generator.apply(Var.of(noise.get())));
        Var g = loss == Loss.NON_SATURATING ? s.neg().softplus().mean() : s.mean().neg();
        g.backward();
        float gLoss = g.value().item();
        for (Var p : discriminatorParameters) {
            p.zeroGrad();
        }
        generatorOptimizer.step(generatorParameters);
        return new Step(dLoss, gLoss);
    }

    private void clear() {
        for (Var p : generatorParameters) {
            p.zeroGrad();
        }
        for (Var p : discriminatorParameters) {
            p.zeroGrad();
        }
    }
}
