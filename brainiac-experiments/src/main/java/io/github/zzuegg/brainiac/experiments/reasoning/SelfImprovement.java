package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.SelfRewardingTrainer;
import io.github.zzuegg.brainiac.experimental.llm.SpinTrainer;
import io.github.zzuegg.brainiac.experimental.llm.StarTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Preference and self-improvement experiments on arithmetic: DPO (and IPO, SimPO) from
 * labelled pairs, SPIN from the supervised data alone, and a self-rewarding loop where the
 * model judges its own answers, and STaR bootstrapping chain-of-thought from a few worked
 * examples (with and without rationalization). Each returns the greedy accuracy before and
 * after.
 */
public final class SelfImprovement {

    /**
     * Accuracy before and after.
     *
     * @param before greedy accuracy before
     * @param after greedy accuracy after
     */
    public record Result(double before, double after) {
    }

    private SelfImprovement() {
    }

    static int[] wrongAnswer(ArithmeticTask task, int[] prompt, SplittableRandom random) {
        int right = task.answer(prompt);
        int wrong;
        do {
            wrong = random.nextInt(10);
        } while (wrong == right);
        return task.tokenizer().encode(wrong + "#");
    }

    static int[] junk(ArithmeticTask task, SplittableRandom random) {
        int[] out = new int[1 + random.nextInt(task.maxAnswerTokens())];
        for (int i = 0; i < out.length; i++) {
            out[i] = random.nextInt(task.vocabSize() - 1);
        }
        return out;
    }

    /**
     * Briefly fine-tunes a model (as DPO assumes), then trains it on (right, wrong) answer
     * pairs.
     *
     * @param task the puzzles
     * @param objective the preference loss
     * @param steps optimizer steps
     * @param seed the seed
     * @return accuracy before and after
     */
    public static Result preference(ArithmeticTask task, DpoTrainer.Objective objective, int steps, long seed) {
        WindowLm lm = warmStart(task, 15, seed);
        SplittableRandom random = new SplittableRandom(seed);
        double before = ArithmeticRl.accuracy(lm, task);
        DpoTrainer dpo = DpoTrainer.of(lm).objective(objective).beta(0.5f).optimizer(Optimizer.adam(3e-3f));
        List<int[]> prompts = task.allPrompts();
        for (int s = 0; s < steps; s++) {
            if (s % 10 == 0) {
                dpo.reference(lm.copy());   // online, iterative DPO: refresh the reference
            }
            // on-policy negatives: the model's own sampled mistakes, else a random wrong answer
            int[][] own = lm.sample(prompts.toArray(int[][]::new), task.maxAnswerTokens(), task.eos(), 1.0, random);
            List<DpoTrainer.Preference> batch = new ArrayList<>();
            for (int i = 0; i < prompts.size(); i++) {
                int[] p = prompts.get(i);
                int[] rejected = own[i].length > 0 && !task.correct(p, own[i]) ? own[i] : wrongAnswer(task, p, random);
                batch.add(new DpoTrainer.Preference(p, task.solution(p), rejected));
            }
            dpo.step(batch);
        }
        return new Result(before, ArithmeticRl.accuracy(lm, task));
    }

    static WindowLm warmStart(ArithmeticTask task, int sftSteps, long seed) {
        WindowLm lm = WindowLm.of(task.vocabSize(), 6, 64, seed);
        List<int[]> prompts = task.allPrompts();
        List<int[]> answers = new ArrayList<>();
        for (int[] p : prompts) {
            answers.add(task.solution(p));
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < sftSteps; s++) {
            lm.supervisedStep(prompts, answers, opt);
        }
        return lm;
    }

    /**
     * Briefly fine-tunes a model, then runs SPIN on the same data.
     *
     * @param task the puzzles
     * @param sftSteps supervised steps before SPIN
     * @param iterations SPIN iterations
     * @param seed the seed
     * @return accuracy after fine-tuning and after SPIN
     */
    public static Result spin(ArithmeticTask task, int sftSteps, int iterations, long seed) {
        WindowLm lm = warmStart(task, sftSteps, seed);
        double before = ArithmeticRl.accuracy(lm, task);
        List<int[]> prompts = task.allPrompts();
        List<int[]> answers = new ArrayList<>();
        for (int[] p : prompts) {
            answers.add(task.solution(p));
        }
        SpinTrainer spin = SpinTrainer.of(lm).beta(0.5f).stepsPerIteration(15).optimizer(Optimizer.adam(3e-3f))
                .generation(task.maxAnswerTokens(), task.eos(), 1.0);
        SplittableRandom random = new SplittableRandom(seed);
        for (int i = 0; i < iterations; i++) {
            spin.iterate(prompts, answers, random);
        }
        return new Result(before, ArithmeticRl.accuracy(lm, task));
    }

    /**
     * The STaR hint for a chain problem: the right final answer in brackets before the
     * question, {@code [9]3+4+2=}.
     *
     * @param task the puzzles
     * @param prompt a problem
     * @return the hinted problem
     */
    static int[] hinted(ChainTask task, int[] prompt) {
        int[] h = task.tokenizer().encode("[" + task.answer(prompt) + "]");
        int[] out = Arrays.copyOf(h, h.length + prompt.length);
        System.arraycopy(prompt, 0, out, h.length, prompt.length);
        return out;
    }

    static double chainAccuracy(WindowLm lm, ChainTask task) {
        List<int[]> prompts = task.allPrompts();
        int[][] outs = lm.sample(prompts.toArray(int[][]::new), task.maxSolutionTokens(), task.eos(), 0,
                new SplittableRandom(0));
        int right = 0;
        for (int i = 0; i < outs.length; i++) {
            right += task.correct(prompts.get(i), outs[i]) ? 1 : 0;
        }
        return (double) right / outs.length;
    }

    /**
     * STaR on chain arithmetic: a model sees worked solutions (plain and hinted) for only
     * {@code labelled} problems, then bootstraps on all problems from final-answer checks
     * alone. With {@code rationalize} off it is rejection-sampling fine-tuning.
     *
     * @param task the puzzles
     * @param labelled how many worked examples seed the model
     * @param rounds STaR iterations
     * @param rationalize whether failures are retried with the answer as a hint
     * @param seed the seed
     * @return greedy accuracy after seeding and after the iterations
     */
    public static Result star(ChainTask task, int labelled, int rounds, boolean rationalize, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<int[]> prompts = task.allPrompts();
        List<int[]> seedPrompts = new ArrayList<>();
        List<int[]> seedSolutions = new ArrayList<>();
        List<int[]> shuffled = new ArrayList<>(prompts);
        java.util.Collections.shuffle(shuffled, new java.util.Random(seed));
        for (int[] p : shuffled.subList(0, labelled)) {
            seedPrompts.add(p);
            seedSolutions.add(task.solution(p));
            seedPrompts.add(hinted(task, p));
            seedSolutions.add(task.solution(p));
        }
        WindowLm lm = WindowLm.of(task.vocabSize(), 16, 64, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < 60; s++) {
            lm.supervisedStep(seedPrompts, seedSolutions, opt);
        }
        double before = chainAccuracy(lm, task);
        StarTrainer star = StarTrainer.of(lm, task::correct, p -> hinted(task, p)).rationalization(rationalize)
                .samples(4).generation(task.maxSolutionTokens(), task.eos(), 1.0)
                .finetune(60, () -> Optimizer.adam(3e-3f)).seed(seedPrompts, seedSolutions);
        for (int r = 0; r < rounds; r++) {
            star.iterate(prompts, random);
        }
        return new Result(before, chainAccuracy(lm, task));
    }

    /**
     * Seeds a model with a little answering and judging ability, then lets it improve by
     * judging its own samples.
     *
     * @param task the puzzles
     * @param sftSteps supervised steps on answers and judgments
     * @param iterations self-rewarding iterations
     * @param seed the seed
     * @return accuracy after seeding and after self-rewarding
     */
    public static Result selfRewarding(ArithmeticTask task, int sftSteps, int iterations, long seed) {
        var tok = task.tokenizer();
        WindowLm lm = WindowLm.of(task.vocabSize(), 8, 128, seed);
        SelfRewardingTrainer srt = SelfRewardingTrainer.of(lm, tok.encode("?")[0], tok.encode("1")[0],
                tok.encode("0")[0]).generation(task.maxAnswerTokens(), task.eos()).beta(0.5f).candidates(8)
                .stepsPerIteration(5).optimizer(Optimizer.adam(1e-3f)).chosenNll(1f);
        SplittableRandom random = new SplittableRandom(seed);
        List<int[]> prompts = task.allPrompts();
        Optimizer opt = Optimizer.adam(5e-3f);
        int answerSteps = 20;
        for (int s = 0; s < sftSteps; s++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            for (int[] p : prompts) {
                // judging is cheap to label: every candidate number, balanced by repeating the
                // right one; answering gets only a few demonstrations
                int right = task.answer(p);
                for (int v = 0; v < 10; v++) {
                    int[][] ex = srt.judgeExample(p, tok.encode(v + "#"), v == right);
                    for (int r = 0; r < (v == right ? 4 : 1); r++) {
                        ps.add(ex[0]);
                        cs.add(ex[1]);
                    }
                }
                int[][] bad = srt.judgeExample(p, junk(task, random), false);
                ps.add(bad[0]);
                cs.add(bad[1]);
                if (s < answerSteps) {
                    ps.add(p);
                    cs.add(task.solution(p));
                }
            }
            lm.supervisedStep(ps, cs, opt);
        }
        double before = ArithmeticRl.accuracy(lm, task);
        for (int i = 0; i < iterations; i++) {
            srt.iterate(prompts, random);
        }
        return new Result(before, ArithmeticRl.accuracy(lm, task));
    }
}
