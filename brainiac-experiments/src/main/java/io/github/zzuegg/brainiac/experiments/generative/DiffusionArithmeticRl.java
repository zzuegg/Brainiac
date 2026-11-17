package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.DiffusionGrpo;
import io.github.zzuegg.brainiac.experimental.llm.MaskedDiffusion;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Reinforcement learning for a masked diffusion LM on arithmetic, the text-diffusion side of
 * {@link TextDiffusionVsAutoregressive}: sequences {@code a+b=XY} (two-digit, zero-padded
 * sums of digits) are modelled by a small {@link MaskedDiffusion} denoiser. It is first
 * fine-tuned with the masked-diffusion loss on half of the problems, then trained with GRPO on
 * all of them with a verifiable reward, using either estimator of {@link DiffusionGrpo}
 * (d1's one-step diffu-GRPO or JustGRPO's left-to-right rollouts). Accuracy is always measured
 * with fully parallel decoding: both answer digits are unmasked in a single step.
 */
public final class DiffusionArithmeticRl {

    private static final int PLUS = 10;
    private static final int EQUALS = 11;
    private static final int MASK = 12;
    private static final int VOCAB = 13;
    private static final int LENGTH = 6;

    /**
     * Accuracies with parallel decoding (one step) and, in brackets of the same order, with
     * greedy left-to-right decoding.
     *
     * @param sft after supervised fine-tuning only
     * @param diffuGrpo after diffu-GRPO
     * @param justGrpo after JustGRPO
     * @param sftLeftToRight SFT only, decoded left to right
     * @param diffuGrpoLeftToRight diffu-GRPO, decoded left to right
     * @param justGrpoLeftToRight JustGRPO, decoded left to right
     */
    public record Result(double sft, double diffuGrpo, double justGrpo, double sftLeftToRight,
            double diffuGrpoLeftToRight, double justGrpoLeftToRight) {
    }

    private DiffusionArithmeticRl() {
    }

    static Model denoiser(long seed) {
        return Model.sequential().input(LENGTH).embedding(VOCAB, 16).flatten()
                .dense(128, Activation.TANH).dense(128, Activation.TANH).dense(LENGTH * VOCAB).seed(seed).build();
    }

    static int[] sequence(int a, int b) {
        int s = a + b;
        return new int[] {a, PLUS, b, EQUALS, s / 10, s % 10};
    }

    static int[] prompt(int a, int b) {
        return new int[] {a, PLUS, b, EQUALS, -1, -1};
    }

    static boolean correct(int[] seq) {
        return seq[4] * 10 + seq[5] == seq[0] + seq[2];
    }

    /**
     * @param diffusion the model
     * @return the share of all 100 problems answered right with one parallel greedy step
     */
    public static double accuracy(MaskedDiffusion diffusion) {
        SplittableRandom r = new SplittableRandom(0);
        int right = 0;
        for (int a = 0; a < 10; a++) {
            for (int b = 0; b < 10; b++) {
                int[] out = diffusion.sample(1, LENGTH, 1, prompt(a, b), Sampling.GREEDY, r, null)[0];
                right += correct(out) ? 1 : 0;
            }
        }
        return right / 100.0;
    }

    /**
     * @param diffusion the model
     * @return the share of all 100 problems answered right decoding greedily left to right
     */
    public static double leftToRightAccuracy(MaskedDiffusion diffusion) {
        DiffusionGrpo g = DiffusionGrpo.of(diffusion, VOCAB, MASK, DiffusionGrpo.Estimator.JUST_GRPO);
        SplittableRandom r = new SplittableRandom(0);
        int right = 0;
        for (int a = 0; a < 10; a++) {
            for (int b = 0; b < 10; b++) {
                right += correct(g.leftToRight(prompt(a, b), 1, Sampling.GREEDY, r)[0]) ? 1 : 0;
            }
        }
        return right / 100.0;
    }

    private static MaskedDiffusion copy(MaskedDiffusion from, long seed) {
        MaskedDiffusion to = MaskedDiffusion.of(denoiser(seed), VOCAB, MASK);
        List<Var> src = from.model().parameters();
        List<Var> dst = to.model().parameters();
        for (int i = 0; i < src.size(); i++) {
            dst.get(i).assign(src.get(i).value());
        }
        return to;
    }

    /**
     * @param sftSteps supervised masked-diffusion steps (batches of 32 from half the problems)
     * @param rlSteps GRPO steps for each estimator
     * @param seed the seed
     * @return the accuracies
     */
    public static Result run(int sftSteps, int rlSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> seen = new ArrayList<>();
        for (int a = 0; a < 10; a++) {
            for (int b = 0; b < 10; b++) {
                if (r.nextBoolean()) {
                    seen.add(sequence(a, b));
                }
            }
        }
        MaskedDiffusion sft = MaskedDiffusion.of(denoiser(seed), VOCAB, MASK);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < sftSteps; s++) {
            int[][] batch = new int[32][];
            for (int i = 0; i < batch.length; i++) {
                batch[i] = seen.get(r.nextInt(seen.size()));
            }
            Var loss = sft.loss(batch, r);
            loss.backward();
            opt.step(sft.model().parameters());
        }
        double[] after = new double[2];
        double[] afterL2r = new double[2];
        for (DiffusionGrpo.Estimator e : DiffusionGrpo.Estimator.values()) {
            MaskedDiffusion m = copy(sft, seed);
            DiffusionGrpo grpo = DiffusionGrpo.of(m, VOCAB, MASK, e).settings(Optimizer.adam(1e-3f), 8, 16, 2);
            SplittableRandom rr = new SplittableRandom(seed + 1);
            for (int s = 0; s < rlSteps; s++) {
                grpo.step(rand -> prompt(rand.nextInt(10), rand.nextInt(10)), seq -> correct(seq) ? 1 : 0, rr);
            }
            after[e.ordinal()] = accuracy(m);
            afterL2r[e.ordinal()] = leftToRightAccuracy(m);
        }
        return new Result(accuracy(sft), after[0], after[1], leftToRightAccuracy(sft), afterL2r[0], afterL2r[1]);
    }
}
