package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.PassAtK;
import io.github.zzuegg.brainiac.experimental.llm.ProlongedRl;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Does prolonged RL expand the reasoning boundary, or only sharpen it? (ProRL,
 * <a href="https://arxiv.org/abs/2505.24864">arXiv:2505.24864</a>, against the "RL only
 * sharpens" result that {@link ReasoningCoverage} reproduces.) A base model, supervised on part
 * of a mix of addition and multiplication, is trained by GRPO for the usual number of steps, by
 * GRPO for five times as long, and by {@link ProlongedRl} for five times as long (DAPO clipping,
 * KL penalty and a reference reset at every usual-length interval). Each is scored by pass@k on
 * both tasks up to large k, and by the forward KL from the base (RL's Razor: how far the
 * policy moved on the base's own samples, as in {@link ForgettingVsKl}).
 */
public final class ProlongedRlCoverage {

    /**
     * One trained model.
     *
     * @param passAtK pass@k at {@link Result#ks}, averaged over the two tasks
     * @param klFromBase the forward KL from the base on its own samples
     * @param resets reference resets
     */
    public record Arm(double[] passAtK, double klFromBase, int resets) {
    }

    /**
     * The comparison.
     *
     * @param ks the k values
     * @param arms the models by name
     */
    public record Result(int[] ks, Map<String, Arm> arms) {
    }

    private ProlongedRlCoverage() {
    }

    /**
     * Runs the comparison.
     *
     * @param maxOperand operand range of both tasks
     * @param usualSteps the usual GRPO length; the prolonged runs take five times as many
     * @param samples samples per prompt for pass@k
     * @param seed the seed
     * @return pass@k and KL per model
     */
    public static Result run(int maxOperand, int usualSteps, int samples, long seed) {
        ArithmeticTask add = ArithmeticTask.addition(maxOperand);
        ArithmeticTask mul = ArithmeticTask.multiplication(maxOperand);
        int times = add.tokenizer().encode("*")[0];
        WindowLm base = base(add, mul, seed);
        int[] ks = PassAtK.powersOfTwo(samples);
        Map<String, Arm> arms = new LinkedHashMap<>();
        arms.put("base", new Arm(passAtK(base, add, mul, samples, ks, seed), 0, 0));
        GrpoTrainer.Reward reward = (p, c) -> contains(p, times) ? mul.reward(p, c) : add.reward(p, c);
        int[][] lengths = {{1, 0}, {5, 0}, {5, usualSteps}};
        String[] names = {"GRPO x1", "GRPO x5", "ProRL x5"};
        for (int a = 0; a < names.length; a++) {
            WindowLm lm = base.copy();
            GrpoTrainer grpo = GrpoTrainer.of(lm)
                    .variant(a == 2 ? GrpoTrainer.Variant.DAPO : GrpoTrainer.Variant.GRPO)
                    .groupSize(8)
                    .promptsPerStep(16)
                    .innerEpochs(2)
                    .klCoefficient(0.05f)
                    .completions(add.maxAnswerTokens(), add.eos());
            ProlongedRl schedule = ProlongedRl.of(grpo, lm, lengths[a][1], () -> Optimizer.adam(3e-3f));
            SplittableRandom random = new SplittableRandom(seed + 11);
            for (int s = 0; s < lengths[a][0] * usualSteps; s++) {
                schedule.step(r -> (r.nextBoolean() ? add : mul).samplePrompt(r), reward, random);
            }
            double kl = (ForgettingVsKl.forwardKl(base, lm, add, 4, seed + 3)
                    + ForgettingVsKl.forwardKl(base, lm, mul, 4, seed + 3)) / 2;
            arms.put(names[a], new Arm(passAtK(lm, add, mul, samples, ks, seed), kl, schedule.resets()));
        }
        return new Result(ks, arms);
    }

    private static WindowLm base(ArithmeticTask add, ArithmeticTask mul, long seed) {
        WindowLm lm = WindowLm.of(add.vocabSize(), 6, 64, seed);
        SplittableRandom r = new SplittableRandom(seed + 17);
        List<int[]> ps = new ArrayList<>();
        List<int[]> gold = new ArrayList<>();
        for (ArithmeticTask task : List.of(add, mul)) {
            for (int[] p : task.allPrompts()) {
                if (r.nextDouble() < 0.5) {
                    ps.add(p);
                    gold.add(task.solution(p));
                }
            }
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < 60; s++) {
            lm.supervisedStep(ps, gold, opt);
        }
        return lm;
    }

    private static double[] passAtK(WindowLm lm, ArithmeticTask add, ArithmeticTask mul, int n, int[] ks,
            long seed) {
        double[] a = RlvrRecipes.passAtK(lm, add, n, seed + 99, ks);
        double[] m = RlvrRecipes.passAtK(lm, mul, n, seed + 99, ks);
        double[] out = new double[ks.length];
        for (int i = 0; i < ks.length; i++) {
            out[i] = (a[i] + m[i]) / 2;
        }
        return out;
    }

    private static boolean contains(int[] tokens, int token) {
        for (int t : tokens) {
            if (t == token) {
                return true;
            }
        }
        return false;
    }
}
