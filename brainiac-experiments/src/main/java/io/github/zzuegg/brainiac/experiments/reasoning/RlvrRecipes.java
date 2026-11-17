package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.PassAtK;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * The shared harness of the 2025 RLVR recipe comparisons on {@link ArithmeticTask} (an
 * extension of {@link ArithmeticRl}): it trains a {@link WindowLm} with a configured
 * {@link GrpoTrainer} and records what the papers plot, namely greedy accuracy, the pass@k
 * curve, the policy-entropy curve, the share of tokens that received a gradient, the clipped
 * share, the range of importance ratios and the sampler/learner mismatch.
 *
 * <p>Runs start either from scratch (as in {@link ArithmeticRl}) or from a weak supervised
 * base ({@link #base}), which is what the pass@k studies need: a model that already solves
 * some puzzles some of the time.
 */
public final class RlvrRecipes {

    /**
     * One training run.
     *
     * @param name the label
     * @param before greedy accuracy before training
     * @param after greedy accuracy after training
     * @param passAtK the pass@k curve after training at {@link #ks}
     * @param ks the k values
     * @param entropy the mean token entropy of the rollouts at each step
     * @param updatedFraction the mean share of tokens that received a policy gradient
     * @param clipFraction the mean share of clipped tokens
     * @param ratioMin the smallest token importance ratio seen
     * @param ratioMax the largest token importance ratio seen
     * @param ratioSpread the mean over steps of {@code ln(max ratio / min ratio)} in the step
     * @param accuracyCurve greedy accuracy after each fifth of training
     * @param behaviourKl the mean per-token {@code KL(sampler || learner)} estimate
     * @param finalReward the mean sampled reward of the last step
     */
    public record Run(String name, double before, double after, double[] passAtK, int[] ks, double[] entropy,
            double updatedFraction, double clipFraction, double ratioMin, double ratioMax, double ratioSpread,
            double[] accuracyCurve, double behaviourKl, double finalReward) {

        /** @return the mean entropy over the last tenth of training */
        public double finalEntropy() {
            int from = entropy.length - Math.max(1, entropy.length / 10);
            double s = 0;
            for (int i = from; i < entropy.length; i++) {
                s += entropy[i];
            }
            return s / (entropy.length - from);
        }

        /**
         * @param k one of {@link #ks}
         * @return pass@k
         */
        public double pass(int k) {
            for (int i = 0; i < ks.length; i++) {
                if (ks[i] == k) {
                    return passAtK[i];
                }
            }
            throw new IllegalArgumentException("k=" + k + " was not measured");
        }
    }

    private RlvrRecipes() {
    }

    /**
     * A weak supervised starting point: a fresh model fine-tuned for a few full-batch steps on
     * the gold answers of a random share of the puzzles, so it is right on some, near-right on
     * others, and has a spread of answers worth exploring.
     *
     * @param task the puzzles
     * @param steps supervised steps
     * @param share the share of puzzles in the supervised set
     * @param seed the seed
     * @return the base model
     */
    public static WindowLm base(ArithmeticTask task, int steps, double share, long seed) {
        WindowLm lm = WindowLm.of(task.vocabSize(), 6, 64, seed);
        SplittableRandom r = new SplittableRandom(seed + 17);
        List<int[]> ps = new ArrayList<>();
        List<int[]> gold = new ArrayList<>();
        for (int[] p : task.allPrompts()) {
            if (r.nextDouble() < share) {
                ps.add(p);
                gold.add(task.solution(p));
            }
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            lm.supervisedStep(ps, gold, opt);
        }
        return lm;
    }

    /**
     * The pass@k curve of a model on all puzzles.
     *
     * @param lm the model
     * @param task the puzzles
     * @param n samples per puzzle
     * @param seed the sampling seed
     * @param ks the k values
     * @return pass@k for each k
     */
    public static double[] passAtK(WindowLm lm, ArithmeticTask task, int n, long seed, int... ks) {
        return PassAtK.curve(lm, task.allPrompts(), task::correct, n, task.maxAnswerTokens(), task.eos(), 1.0,
                new SplittableRandom(seed), ks);
    }

    /**
     * Trains with a configured recipe.
     *
     * @param name the label
     * @param task the puzzles
     * @param start the starting model (trained in place), or {@code null} for a fresh one
     * @param configure sets the variant and options on a trainer that already has the
     *     harness defaults (Adam {@code learningRate}, group 8, 16 prompts, 2 epochs, KL
     *     {@code kl})
     * @param reward the reward
     * @param steps GRPO steps
     * @param seed the seed
     * @param learningRate the Adam learning rate
     * @param kl the KL coefficient
     * @param n samples per puzzle for the pass@k curve (0 to skip it)
     * @param ks the k values
     * @return the run
     */
    public static Run train(String name, ArithmeticTask task, WindowLm start, UnaryOperator<GrpoTrainer> configure,
            GrpoTrainer.Reward reward, int steps, long seed, float learningRate, float kl, int n, int... ks) {
        WindowLm lm = start != null ? start : WindowLm.of(task.vocabSize(), 6, 64, seed);
        double before = ArithmeticRl.accuracy(lm, task);
        GrpoTrainer grpo = configure.apply(GrpoTrainer.of(lm)
                .optimizer(Optimizer.adam(learningRate))
                .groupSize(8)
                .promptsPerStep(16)
                .innerEpochs(2)
                .klCoefficient(kl)
                .completions(task.maxAnswerTokens(), task.eos()));
        SplittableRandom random = new SplittableRandom(seed);
        double[] entropy = new double[steps];
        double updated = 0;
        double clipped = 0;
        double rmin = Double.POSITIVE_INFINITY;
        double rmax = Double.NEGATIVE_INFINITY;
        double mismatch = 0;
        double spread = 0;
        double last = 0;
        double[] curve = new double[5];
        for (int i = 0; i < steps; i++) {
            GrpoTrainer.Stats s = grpo.step(task::samplePrompt, reward, random);
            entropy[i] = s.entropy();
            updated += s.updatedFraction();
            clipped += s.clipFraction();
            rmin = Math.min(rmin, s.ratioMin());
            rmax = Math.max(rmax, s.ratioMax());
            mismatch += s.behaviourKl();
            spread += Math.log(s.ratioMax() / s.ratioMin());
            last = s.meanReward();
            if ((i + 1) * 5 / steps > i * 5 / steps) {
                curve[(i + 1) * 5 / steps - 1] = ArithmeticRl.accuracy(lm, task);
            }
        }
        double[] pass = n > 0 ? passAtK(lm, task, n, seed + 99, ks) : new double[0];
        return new Run(name, before, ArithmeticRl.accuracy(lm, task), pass, ks, entropy, updated / steps,
                clipped / steps, rmin, rmax, spread / steps, curve, mismatch / steps, last);
    }

    /**
     * Trains a fresh model with the true verifiable reward.
     *
     * @param name the label
     * @param task the puzzles
     * @param configure sets the recipe
     * @param steps GRPO steps
     * @param seed the seed
     * @param n samples per puzzle for pass@k (0 to skip)
     * @param ks the k values
     * @return the run
     */
    public static Run train(String name, ArithmeticTask task, UnaryOperator<GrpoTrainer> configure, int steps,
            long seed, int n, int... ks) {
        return train(name, task, null, configure, task::reward, steps, seed, 3e-3f, 0.05f, n, ks);
    }

    /**
     * Formats a pass@k curve.
     *
     * @param run the run
     * @return {@code k=value} pairs
     */
    public static String curve(Run run) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < run.ks().length; i++) {
            sb.append(i == 0 ? "" : " ").append(run.ks()[i]).append(':').append(String.format("%.2f", run.passAtK()[i]));
        }
        return sb.toString();
    }
}
