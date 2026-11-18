package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.align.SingleExampleSelector;
import io.github.zzuegg.brainiac.experimental.evolve.EvolutionStrategyFineTuner;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.PpoLmTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.train.MeZo;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Experiment X23, tiny reasoning RL: a randomly initialized {@link WindowLm} learns
 * arithmetic from nothing but a programmatic reward, trained with {@link GrpoTrainer} (or
 * its Dr. GRPO, DAPO, RLOO, ReMax, REINFORCE++ and GSPO variants), or with token-level PPO
 * and a learned critic ({@link #ppo}). The accuracy on all puzzles is measured greedily
 * before and after, showing the "accuracy climbs" curve of DeepSeek-R1-Zero in miniature.
 *
 * <p>{@link #fineTune} compares, from one weakly supervised base model and at an equal number
 * of reward evaluations, GRPO against two gradient-free fine-tuners: full-parameter evolution
 * strategies ({@link EvolutionStrategyFineTuner}, Qiu et al. 2025,
 * <a href="https://arxiv.org/abs/2509.24372">arXiv:2509.24372</a>) and MeZO's zeroth-order SGD
 * on the negated reward (Malladi et al. 2023,
 * <a href="https://arxiv.org/abs/2305.17333">arXiv:2305.17333</a>), with the KL divergence from
 * the base model.
 *
 * <p>{@link #oneShot} (wave 9, 9.2.13) asks what GRPO learns from a single training problem
 * (Wang et al. 2025, <a href="https://arxiv.org/abs/2504.20571">arXiv:2504.20571</a>): problems
 * are ranked by the historical variance of their accuracy in a pilot run
 * ({@link SingleExampleSelector}), and GRPO trains on the top 1, 2 or 16 of them, on one random
 * problem, or on the whole training half, with format-only and random-reward controls and the
 * entropy bonus on and off, while the accuracy on the held-out half is tracked.
 */
public final class ArithmeticRl {

    /**
     * The outcome.
     *
     * @param before greedy accuracy before training
     * @param after greedy accuracy after training
     * @param finalReward mean sampled reward in the last step
     */
    public record Result(double before, double after, double finalReward) {
    }

    private ArithmeticRl() {
    }

    /**
     * @param lm a model
     * @param task the puzzles
     * @return the fraction of all puzzles it answers right, greedily
     */
    public static double accuracy(WindowLm lm, ArithmeticTask task) {
        int right = 0;
        var prompts = task.allPrompts();
        int[][] outs = lm.sample(prompts.toArray(int[][]::new), task.maxAnswerTokens(), task.eos(), 0,
                new SplittableRandom(0));
        for (int i = 0; i < outs.length; i++) {
            if (task.correct(prompts.get(i), outs[i])) {
                right++;
            }
        }
        return (double) right / prompts.size();
    }

    /**
     * Trains a fresh model with GRPO.
     *
     * @param task the puzzles
     * @param variant the recipe
     * @param steps GRPO steps
     * @param seed the seed
     * @return accuracy before and after
     */
    public static Result run(ArithmeticTask task, GrpoTrainer.Variant variant, int steps, long seed) {
        return run(task, variant, steps, seed, 3e-3f, 0.2f);
    }

    /**
     * Trains a fresh model with GRPO.
     *
     * @param task the puzzles
     * @param variant the recipe
     * @param steps GRPO steps
     * @param seed the seed
     * @param learningRate the Adam learning rate
     * @param kl the KL coefficient
     * @return accuracy before and after
     */
    public static Result run(ArithmeticTask task, GrpoTrainer.Variant variant, int steps, long seed,
            float learningRate, float kl) {
        WindowLm lm = WindowLm.of(task.vocabSize(), 6, 64, seed);
        double before = accuracy(lm, task);
        GrpoTrainer grpo = GrpoTrainer.of(lm)
                .variant(variant)
                .optimizer(Optimizer.adam(learningRate))
                .groupSize(8)
                .promptsPerStep(16)
                .innerEpochs(2)
                .klCoefficient(kl)
                .completions(task.maxAnswerTokens(), task.eos());
        SplittableRandom random = new SplittableRandom(seed);
        GrpoTrainer.Stats stats = null;
        for (int i = 0; i < steps; i++) {
            stats = grpo.step(task::samplePrompt, task::reward, random);
        }
        return new Result(before, accuracy(lm, task), stats == null ? 0 : stats.meanReward());
    }

    /**
     * Trains a fresh model with token-level PPO and a learned critic ({@link PpoLmTrainer}),
     * at the same 128 sampled completions per step as the GRPO runs (16 prompts times 8).
     *
     * @param task the puzzles
     * @param steps PPO steps
     * @param seed the seed
     * @return accuracy before and after
     */
    public static Result ppo(ArithmeticTask task, int steps, long seed) {
        WindowLm lm = WindowLm.of(task.vocabSize(), 6, 64, seed);
        double before = accuracy(lm, task);
        PpoLmTrainer ppo = PpoLmTrainer.of(lm)
                .optimizers(Optimizer.adam(3e-3f), Optimizer.adam(3e-3f))
                .batch(128)
                .innerEpochs(2)
                .klCoefficient(0.05f)
                .completions(task.maxAnswerTokens(), task.eos());
        SplittableRandom random = new SplittableRandom(seed);
        PpoLmTrainer.Stats stats = null;
        for (int i = 0; i < steps; i++) {
            stats = ppo.step(task::samplePrompt, task::reward, random);
        }
        return new Result(before, accuracy(lm, task), stats == null ? 0 : stats.meanReward());
    }

    /** The fine-tuners of {@link #fineTune}. */
    public enum FineTuner {
        /** GRPO, 16 prompts x 8 sampled completions per step. */
        GRPO,
        /** Evolution strategies, 8 antithetic pairs per iteration, each scored greedily on 8 prompts. */
        ES,
        /** MeZO, a +- perturbation pair per step, each scored greedily on 64 prompts. */
        MEZO
    }

    /**
     * The outcome of a fine-tuning run.
     *
     * @param before greedy accuracy of the base model
     * @param after greedy accuracy after fine-tuning
     * @param rewardEvaluations completions scored by the reward
     * @param kl mean summed token KL(base || tuned) along the base model's greedy answers
     */
    public record FineTuneResult(double before, double after, long rewardEvaluations, double kl) {
    }

    /**
     * The shared starting point: a fresh model given a few supervised steps on 6 random puzzles,
     * so it knows the answer format but not the arithmetic.
     *
     * @param task the puzzles
     * @param sftSteps supervised steps
     * @param seed the seed
     * @return the base model
     */
    public static WindowLm base(ArithmeticTask task, int sftSteps, long seed) {
        WindowLm lm = WindowLm.of(task.vocabSize(), 6, 64, seed);
        SplittableRandom r = new SplittableRandom(seed + 17);
        List<int[]> prompts = new ArrayList<>();
        List<int[]> answers = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            int[] p = task.samplePrompt(r);
            prompts.add(p);
            answers.add(task.solution(p));
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int i = 0; i < sftSteps; i++) {
            lm.supervisedStep(prompts, answers, opt);
        }
        return lm;
    }

    private static double greedyReward(WindowLm lm, ArithmeticTask task, int[][] prompts) {
        int[][] outs = lm.sample(prompts, task.maxAnswerTokens(), task.eos(), 0, new SplittableRandom(0));
        double total = 0;
        for (int i = 0; i < outs.length; i++) {
            total += task.reward(prompts[i], outs[i]);
        }
        return total / outs.length;
    }

    private static int[][] batch(ArithmeticTask task, int size, SplittableRandom r) {
        int[][] out = new int[size][];
        for (int i = 0; i < size; i++) {
            out[i] = task.samplePrompt(r);
        }
        return out;
    }

    /**
     * Mean over all puzzles of the summed token KL(base || tuned) along the base model's greedy
     * completion.
     *
     * @param base the base model
     * @param tuned the fine-tuned model
     * @param task the puzzles
     * @return the KL in nats
     */
    public static double klFromBase(WindowLm base, WindowLm tuned, ArithmeticTask task) {
        double total = 0;
        List<int[]> prompts = task.allPrompts();
        for (int[] prompt : prompts) {
            int[] out = base.sample(new int[][] {prompt}, task.maxAnswerTokens(), task.eos(), 0,
                    new SplittableRandom(0))[0];
            int[] prefix = prompt.clone();
            for (int t = 0; t < out.length; t++) {
                total += kl(base.logits(prefix), tuned.logits(prefix)) / prompts.size();
                int[] next = java.util.Arrays.copyOf(prefix, prefix.length + 1);
                next[prefix.length] = out[t];
                prefix = next;
                if (out[t] == task.eos()) {
                    break;
                }
            }
        }
        return total;
    }

    private static double kl(float[] p, float[] q) {
        double mp = Double.NEGATIVE_INFINITY;
        double mq = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < p.length; i++) {
            mp = Math.max(mp, p[i]);
            mq = Math.max(mq, q[i]);
        }
        double zp = 0;
        double zq = 0;
        for (int i = 0; i < p.length; i++) {
            zp += Math.exp(p[i] - mp);
            zq += Math.exp(q[i] - mq);
        }
        double out = 0;
        for (int i = 0; i < p.length; i++) {
            double lp = p[i] - mp - Math.log(zp);
            double lq = q[i] - mq - Math.log(zq);
            out += Math.exp(lp) * (lp - lq);
        }
        return out;
    }

    /**
     * Fine-tunes the {@link #base} model with one method for a budget of reward evaluations.
     *
     * @param task the puzzles
     * @param method the fine-tuner
     * @param budget completions scored by the reward
     * @param seed the seed
     * @return accuracy before and after, and the KL from the base
     */
    public static FineTuneResult fineTune(ArithmeticTask task, FineTuner method, long budget, long seed) {
        return switch (method) {
            case GRPO -> fineTune(task, method, budget, seed, 0, 3e-3);
            case ES -> fineTune(task, method, budget, seed, ES_SIGMA, ES_RATE);
            case MEZO -> fineTune(task, method, budget, seed, MEZO_EPSILON, MEZO_RATE);
        };
    }

    private static final double ES_SIGMA = 0.05;
    private static final double ES_RATE = 3e-3;
    private static final double MEZO_EPSILON = 0.05;
    private static final double MEZO_RATE = 3e-2;

    /**
     * {@link #fineTune(ArithmeticTask, FineTuner, long, long)} with explicit hyperparameters.
     *
     * @param task the puzzles
     * @param method the fine-tuner
     * @param budget completions scored by the reward
     * @param seed the seed
     * @param noise the perturbation scale (ES sigma, MeZO epsilon; unused by GRPO)
     * @param rate the learning rate
     * @return accuracy before and after, and the KL from the base
     */
    public static FineTuneResult fineTune(ArithmeticTask task, FineTuner method, long budget, long seed, double noise,
            double rate) {
        WindowLm base = base(task, 40, seed);
        WindowLm lm = base.copy();
        double before = accuracy(lm, task);
        SplittableRandom r = new SplittableRandom(seed + 5);
        long used = 0;
        switch (method) {
            case GRPO -> {
                GrpoTrainer grpo = GrpoTrainer.of(lm).variant(GrpoTrainer.Variant.GRPO).optimizer(Optimizer.adam((float) rate))
                        .groupSize(8).promptsPerStep(16).innerEpochs(2).klCoefficient(0.2f)
                        .completions(task.maxAnswerTokens(), task.eos());
                while (used < budget) {
                    grpo.step(task::samplePrompt, task::reward, r);
                    used += 128;
                }
            }
            case ES -> {
                EvolutionStrategyFineTuner es = EvolutionStrategyFineTuner.of(lm.parameters(), noise, rate, 8, seed);
                while (used < budget) {
                    int[][] prompts = batch(task, 8, r);
                    es.step(() -> greedyReward(lm, task, prompts));
                    used += 16L * prompts.length;
                }
            }
            case MEZO -> {
                MeZo mezo = new MeZo((float) rate, (float) noise, 0, seed);
                while (used < budget) {
                    int[][] prompts = batch(task, 64, r);
                    // the cast picks MeZo's two-point step over Optimizer's step(parameters, Reevaluation)
                    Supplier<Float> loss = () -> (float) -greedyReward(lm, task, prompts);
                    mezo.step(lm.parameters(), loss);
                    used += 2L * prompts.length;
                }
            }
            default -> throw new IllegalArgumentException();
        }
        return new FineTuneResult(before, accuracy(lm, task), used, klFromBase(base, lm, task));
    }

    /**
     * One 9.2.13 arm.
     *
     * @param testAccuracy held-out greedy accuracy at each checkpoint, the start first
     * @param solvedAt the first checkpoint (1-based) at which every training problem is answered
     *     right greedily, or 0 if never
     * @param trainProblems training problems used
     */
    public record OneShotRow(double[] testAccuracy, int solvedAt, int trainProblems) {
    }

    private static double accuracy(WindowLm lm, ArithmeticTask task, List<int[]> prompts) {
        int[][] outs = lm.sample(prompts.toArray(int[][]::new), task.maxAnswerTokens(), task.eos(), 0,
                new SplittableRandom(0));
        double ok = 0;
        for (int i = 0; i < outs.length; i++) {
            ok += task.correct(prompts.get(i), outs[i]) ? 1 : 0;
        }
        return ok / outs.length;
    }

    /**
     * Runs 9.2.13 for one seed.
     *
     * @param task the puzzles (split in half: train and held-out)
     * @param steps GRPO steps per arm
     * @param checkpoints evaluations after the start
     * @param seed the seed
     * @return arm name to its row
     */
    public static Map<String, OneShotRow> oneShot(ArithmeticTask task, int steps, int checkpoints, long seed) {
        List<int[]> all = new ArrayList<>(task.allPrompts());
        SplittableRandom split = new SplittableRandom(seed + 1);
        java.util.Collections.shuffle(all, new java.util.Random(split.nextLong()));
        List<int[]> train = new ArrayList<>(all.subList(0, all.size() / 2));
        List<int[]> test = new ArrayList<>(all.subList(all.size() / 2, all.size()));
        WindowLm base = base(task, 40, seed);
        // the pilot run on the whole training half, recording per-problem accuracy
        WindowLm pilotLm = base.copy();
        GrpoTrainer pilot = GrpoTrainer.of(pilotLm).optimizer(Optimizer.adam(3e-3f)).groupSize(8).promptsPerStep(16)
                .klCoefficient(0.05f).completions(task.maxAnswerTokens(), task.eos());
        double[][] history = SingleExampleSelector.pilot(pilot, pilotLm, train, task::reward, task::correct, 5, 8, 8,
                task.maxAnswerTokens(), task.eos(), new SplittableRandom(seed + 2));
        int[] ranking = SingleExampleSelector.rank(history);
        int randomOne = new SplittableRandom(seed + 3).nextInt(train.size());
        Map<String, List<int[]>> sets = new LinkedHashMap<>();
        sets.put("full train half", train);
        sets.put("top-16 by variance", SingleExampleSelector.top(train, ranking, 16));
        sets.put("top-2 by variance", SingleExampleSelector.top(train, ranking, 2));
        sets.put("top-1 by variance", SingleExampleSelector.top(train, ranking, 1));
        sets.put("top-1, no entropy bonus", SingleExampleSelector.top(train, ranking, 1));
        sets.put("top-1, format-only reward", SingleExampleSelector.top(train, ranking, 1));
        sets.put("top-1, random reward", SingleExampleSelector.top(train, ranking, 1));
        sets.put("random 1", List.of(train.get(randomOne)));
        Map<String, OneShotRow> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<int[]>> e : sets.entrySet()) {
            String name = e.getKey();
            List<int[]> problems = e.getValue();
            WindowLm lm = base.copy();
            GrpoTrainer grpo = GrpoTrainer.of(lm).optimizer(Optimizer.adam(3e-3f)).groupSize(8).promptsPerStep(16)
                    .klCoefficient(0.05f).completions(task.maxAnswerTokens(), task.eos());
            SplittableRandom coin = new SplittableRandom(seed + 4);
            GrpoTrainer.Reward reward = name.contains("format-only") ? (p, c) -> task.parse(c) != null ? 1 : 0
                    : name.contains("random reward") ? (p, c) -> coin.nextBoolean() ? 1 : 0 : task::reward;
            if (!name.contains("no entropy")) {
                reward = SingleExampleSelector.entropyBonus(reward, lm, 0.02f);
            }
            double[] acc = new double[checkpoints + 1];
            acc[0] = accuracy(lm, task, test);
            int solvedAt = 0;
            SplittableRandom r = new SplittableRandom(seed + 5);
            int next = 1;
            for (int s = 1; s <= steps; s++) {
                grpo.step(x -> problems.get(x.nextInt(problems.size())), reward, r);
                if (next <= checkpoints && s == steps * next / checkpoints) {
                    acc[next] = accuracy(lm, task, test);
                    if (solvedAt == 0 && accuracy(lm, task, problems) == 1) {
                        solvedAt = next;
                    }
                    next++;
                }
            }
            out.put(name, new OneShotRow(acc, solvedAt, problems.size()));
        }
        return out;
    }
}
