package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.PassAtK;
import io.github.zzuegg.brainiac.experimental.llm.PowerSampler;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Does sampling from the base model's power distribution recover what RL post-training buys?
 * (Karan and Du, <a href="https://arxiv.org/abs/2510.14901">arXiv:2510.14901</a>.) On
 * {@link ArithmeticTask} with the {@link RlvrRecipes#base} model, four ways of answering are
 * compared by pass@1, pass@k and generated tokens per answer: the base at temperature 1, the base
 * at the low temperature {@code 1/alpha}, {@link PowerSampler} on the base (training-free), and
 * the {@link GrpoTrainer}-trained model at temperature 1.
 */
public final class PowerSampling {

    /**
     * One way of answering.
     *
     * @param name the method
     * @param passAtK pass@k at {@link Result#ks}
     * @param tokensPerAnswer tokens generated per returned answer
     */
    public record Arm(String name, double[] passAtK, double tokensPerAnswer) {
    }

    /**
     * The comparison.
     *
     * @param ks the k values
     * @param arms the methods
     * @param acceptance the power sampler's acceptance rate
     */
    public record Result(int[] ks, List<Arm> arms, double acceptance) {

        /**
         * @param name a method
         * @return its arm
         */
        public Arm arm(String name) {
            return arms.stream().filter(a -> a.name().equals(name)).findFirst().orElseThrow();
        }
    }

    private interface Sampler {
        int[][] sample(int[][] prompts, SplittableRandom random);
    }

    private PowerSampling() {
    }

    /**
     * Runs the comparison.
     *
     * @param task the task
     * @param alpha the power (and the inverse low temperature)
     * @param mcmcSteps Metropolis-Hastings steps per answer
     * @param rlSteps GRPO steps for the RL arm
     * @param samples answers per prompt
     * @param seed the seed
     * @return pass@k per method
     */
    public static Result run(ArithmeticTask task, double alpha, int mcmcSteps, int rlSteps, int samples, long seed) {
        int[] ks = PassAtK.powersOfTwo(samples);
        WindowLm base = RlvrRecipes.base(task, 60, 0.5, seed);
        WindowLm rl = base.copy();
        RlvrRecipes.train("GRPO", task, rl, g -> g.variant(GrpoTrainer.Variant.GRPO), task::reward, rlSteps, seed,
                3e-3f, 0.05f, 0);
        int max = task.maxAnswerTokens();
        PowerSampler power = PowerSampler.of(base, alpha, mcmcSteps, max, task.eos());
        Arm baseArm = arm("base T=1", task, samples, ks, seed, (p, r) -> base.sample(p, max, task.eos(), 1.0, r));
        Arm cold = arm(String.format("base T=1/%.0f", alpha), task, samples, ks, seed,
                (p, r) -> base.sample(p, max, task.eos(), 1 / alpha, r));
        long before = power.generatedTokens();
        Arm powered = arm("power sampling", task, samples, ks, seed, power::sample);
        int answers = task.allPrompts().size() * samples;
        powered = new Arm(powered.name(), powered.passAtK(), (double) (power.generatedTokens() - before) / answers);
        Arm rlArm = arm("GRPO T=1", task, samples, ks, seed, (p, r) -> rl.sample(p, max, task.eos(), 1.0, r));
        return new Result(ks, List.of(baseArm, cold, powered, rlArm), power.acceptanceRate());
    }

    private static Arm arm(String name, ArithmeticTask task, int samples, int[] ks, long seed, Sampler sampler) {
        return arm(name, task.allPrompts(), task::correct, samples, ks, seed, sampler);
    }

    /**
     * The same comparison on two-step {@link ChainTask} solutions, where look-ahead matters more
     * because a wrong first step dooms the rest; the base is a partly supervised model and there
     * is no RL arm.
     *
     * @param task the task
     * @param sftSteps supervised steps for the base
     * @param alpha the power
     * @param mcmcSteps Metropolis-Hastings steps per answer
     * @param samples answers per prompt
     * @param seed the seed
     * @return pass@k per method
     */
    public static Result chain(ChainTask task, int sftSteps, double alpha, int mcmcSteps, int samples, long seed) {
        int[] ks = PassAtK.powersOfTwo(samples);
        List<int[]> prompts = task.allPrompts();
        WindowLm base = WindowLm.of(task.vocabSize(), 12, 64, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        List<int[]> sols = new ArrayList<>();
        for (int[] p : prompts) {
            sols.add(task.solution(p));
        }
        for (int s = 0; s < sftSteps; s++) {
            base.supervisedStep(prompts, sols, opt);
        }
        int max = task.maxSolutionTokens();
        PowerSampler power = PowerSampler.of(base, alpha, mcmcSteps, max, task.eos());
        Arm baseArm = arm("base T=1", prompts, task::correct, samples, ks, seed,
                (p, r) -> base.sample(p, max, task.eos(), 1.0, r));
        Arm cold = arm(String.format("base T=1/%.0f", alpha), prompts, task::correct, samples, ks, seed,
                (p, r) -> base.sample(p, max, task.eos(), 1 / alpha, r));
        Arm powered = arm("power sampling", prompts, task::correct, samples, ks, seed, power::sample);
        powered = new Arm(powered.name(), powered.passAtK(),
                (double) power.generatedTokens() / (prompts.size() * samples));
        return new Result(ks, List.of(baseArm, cold, powered), power.acceptanceRate());
    }

    private static Arm arm(String name, List<int[]> prompts, PassAtK.Verifier verifier, int samples, int[] ks,
            long seed, Sampler sampler) {
        int[][] batch = new int[prompts.size() * samples][];
        for (int i = 0; i < batch.length; i++) {
            batch[i] = prompts.get(i / samples);
        }
        int[][] out = sampler.sample(batch, new SplittableRandom(seed + 7));
        int[] counts = new int[prompts.size()];
        long tokens = 0;
        for (int i = 0; i < batch.length; i++) {
            tokens += out[i].length;
            if (verifier.correct(batch[i], out[i])) {
                counts[i / samples]++;
            }
        }
        return new Arm(name, PassAtK.curve(counts, samples, ks), (double) tokens / batch.length);
    }
}
