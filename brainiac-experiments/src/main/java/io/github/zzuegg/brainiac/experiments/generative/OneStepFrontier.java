package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ContinuousConsistency;
import io.github.zzuegg.brainiac.experimental.nn.DpmSolver;
import io.github.zzuegg.brainiac.experimental.nn.EdmDenoiser;
import io.github.zzuegg.brainiac.experimental.nn.MomentMatchingGenerator;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MeanFlow;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.experimental.train.DistributionMatchingDistiller;
import io.github.zzuegg.brainiac.experimental.train.DriftingGenerator;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Adversarial;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * The 2025-26 one-step generators of wave 6 on the {@link OneStepGenerators} ring, scored the
 * same way (mean distance to the nearest blob centre, share within 0.3 of one, blobs covered)
 * so they line up against that harness's MeanFlow, shortcut, reflow and consistency baselines:
 * <ul>
 *   <li>DMD ({@link DistributionMatchingDistiller}, item 4.11): a one-step generator distilled
 *       from an {@link EdmDenoiser} teacher by matching real and fake scores, with a small
 *       regression set of teacher DPM-Solver samples;</li>
 *   <li>IMM ({@link MomentMatchingGenerator}, 4.16) and continuous-time consistency training
 *       (sCM, {@link ContinuousConsistency}, 4.16), at 1, 2 and 4 steps;</li>
 *   <li>drifting models ({@link DriftingGenerator}, 4.18);</li>
 *   <li>a GAN baseline (core {@link Adversarial}, non-saturating loss with R1) with the same
 *       generator network.</li>
 * </ul>
 */
@SuppressWarnings("exports")
public final class OneStepFrontier {
    private static final int BATCH = 128;

    private OneStepFrontier() {
    }

    private static Model generatorNet(long seed) {
        return Model.sequential().input(2).dense(64, Activation.SILU).dense(64, Activation.SILU).dense(2).seed(seed)
                .build();
    }

    private static Tensor batch(ToyPoints ring, SplittableRandom r) {
        return Nets.matrix(PointClouds.blobs(ring, BATCH, r));
    }

    private static Tensor probe(int samples, long seed) {
        return Nets.gaussian(samples, 2, new SplittableRandom(seed + 99));
    }

    /**
     * Trains an EDM teacher, then distils it into a one-step generator with DMD.
     *
     * @param ring the data
     * @param teacherIterations teacher training steps
     * @param iterations generator steps
     * @param samples how many points to score
     * @param seed the seed
     * @return {@code {DMD score, teacher score at 10 DPM-Solver++ steps}}
     */
    public static OneStepGenerators.Score[] dmd(ToyPoints ring, int teacherIterations, int iterations, int samples,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        EdmDenoiser teacher = edm(seed);
        Optimizer tOpt = Optimizer.adam(3e-3f);
        for (int i = 0; i < teacherIterations; i++) {
            teacher.loss(batch(ring, r), r).backward();
            Nets.clip(teacher.parameters(), 1f);
            tOpt.step(teacher.parameters());
        }
        double[] sigmas = EdmDenoiser.karrasSigmas(10, 0.002, 20, 7);
        OneStepGenerators.Score teacherScore = OneStepGenerators.score(ring, "EDM teacher (10 NFE)",
                Nets.rows(DpmSolver.sample(teacher, probe(samples, seed), sigmas)));
        Tensor pairedNoise = Nets.gaussian(64, 2, new SplittableRandom(seed + 5));
        Tensor pairedTargets = DpmSolver.sample(teacher, pairedNoise, sigmas);
        Model gen = generatorNet(seed + 1);
        EdmDenoiser fake = edm(seed + 2);
        DistributionMatchingDistiller dmd = DistributionMatchingDistiller.of(teacher, fake, gen::forward,
                gen.parameters(), Optimizer.adam(1e-3f), Optimizer.adam(3e-3f)).regression(pairedNoise,
                pairedTargets, 1f);
        // the fake score starts as a copy of the teacher, as in DMD
        for (int k = 0; k < teacher.parameters().size(); k++) {
            fake.parameters().get(k).assign(teacher.parameters().get(k).value());
        }
        for (int i = 0; i < iterations; i++) {
            dmd.step(Nets.gaussian(BATCH, 2, r), r);
        }
        return new OneStepGenerators.Score[] {OneStepGenerators.score(ring, "DMD",
                Nets.rows(dmd.generate(probe(samples, seed)))), teacherScore};
    }

    private static EdmDenoiser edm(long seed) {
        MlpDenoiser net = MlpDenoiser.of(2, 0, 64, 2, seed);
        return EdmDenoiser.of(Nets.unconditional(net), net.parameters(), 1f);
    }

    /**
     * Inductive moment matching, sampled with 1, 2 and 4 jumps.
     *
     * @param ring the data
     * @param iterations training steps
     * @param samples how many points to score
     * @param seed the seed
     * @return scores at 1, 2 and 4 steps
     */
    public static OneStepGenerators.Score[] imm(ToyPoints ring, int iterations, int samples, long seed) {
        MomentMatchingGenerator imm = MomentMatchingGenerator.of(MeanFlow.mlp(2, 64, 2, seed), 4, 0.25f, 0.25f);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < iterations; i++) {
            if (i == iterations * 3 / 4) {
                opt.learningRate(7e-4f);
            }
            imm.loss(batch(ring, r), r).backward();
            Nets.clip(imm.network().parameters(), 10f);
            opt.step(imm.network().parameters());
        }
        OneStepGenerators.Score[] out = new OneStepGenerators.Score[3];
        int[] steps = {1, 2, 4};
        for (int k = 0; k < 3; k++) {
            out[k] = OneStepGenerators.score(ring, "IMM " + steps[k],
                    Nets.rows(imm.sample(probe(samples, seed), steps[k])));
        }
        return out;
    }

    /**
     * Continuous-time consistency training (sCM), sampled with 1, 2 and 4 steps.
     *
     * @param ring the data
     * @param iterations training steps
     * @param samples how many points to score
     * @param seed the seed
     * @return scores at 1, 2 and 4 steps
     */
    public static OneStepGenerators.Score[] scm(ToyPoints ring, int iterations, int samples, long seed) {
        ContinuousConsistency scm = ContinuousConsistency.of(MeanFlow.mlp(2, 64, 2, seed), -1f, 1.4f,
                iterations / 4);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < iterations; i++) {
            if (i == iterations * 3 / 4) {
                opt.learningRate(7e-4f);
            }
            scm.loss(batch(ring, r), r).backward();
            Nets.clip(scm.network().parameters(), 10f);
            opt.step(scm.network().parameters());
        }
        float[][] times = {{}, {1.1f}, {1.3f, 0.8f, 0.4f}};
        OneStepGenerators.Score[] out = new OneStepGenerators.Score[3];
        int[] steps = {1, 2, 4};
        for (int k = 0; k < 3; k++) {
            out[k] = OneStepGenerators.score(ring, "sCM " + steps[k], Nets.rows(scm.sample(probe(samples, seed),
                    times[k], new SplittableRandom(seed + 3))));
        }
        return out;
    }

    /**
     * A drifting model.
     *
     * @param ring the data
     * @param iterations training steps
     * @param samples how many points to score
     * @param seed the seed
     * @return the score
     */
    public static OneStepGenerators.Score drifting(ToyPoints ring, int iterations, int samples, long seed) {
        Model gen = generatorNet(seed + 1);
        DriftingGenerator drift = DriftingGenerator.of(gen::forward, gen.parameters(), Optimizer.adam(2e-3f), 0.05f,
                0.2f, 1f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < iterations; i++) {
            drift.step(Nets.gaussian(BATCH, 2, r), batch(ring, r));
        }
        return OneStepGenerators.score(ring, "drifting", Nets.rows(drift.generate(probe(samples, seed))));
    }

    /**
     * The GAN baseline: the same generator, a 64-wide discriminator, non-saturating loss with
     * R1 (gamma 1), Adam(2e-3, 0.5, 0.9).
     *
     * @param ring the data
     * @param iterations generator steps
     * @param samples how many points to score
     * @param seed the seed
     * @return the score
     */
    public static OneStepGenerators.Score gan(ToyPoints ring, int iterations, int samples, long seed) {
        Model gen = generatorNet(seed + 1);
        Model disc = Model.sequential().input(2).dense(64, Activation.leakyRelu(0.2f))
                .dense(64, Activation.leakyRelu(0.2f)).dense(1).seed(seed + 2).build();
        Adversarial gan = new Adversarial(gen::forward, gen.parameters(), Optimizer.adam(2e-3f, 0.5f, 0.9f, 1e-8f),
                disc::forward, disc.parameters(), Optimizer.adam(2e-3f, 0.5f, 0.9f, 1e-8f),
                Adversarial.Loss.NON_SATURATING).withR1(1f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < iterations; i++) {
            gan.step(batch(ring, r), () -> Nets.gaussian(BATCH, 2, r));
        }
        return OneStepGenerators.score(ring, "GAN", Nets.rows(gen.forward(Var.of(probe(samples, seed))).value()));
    }
}
