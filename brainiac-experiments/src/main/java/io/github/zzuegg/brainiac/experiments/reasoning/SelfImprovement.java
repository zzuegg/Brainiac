package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.SelfRewardingTrainer;
import io.github.zzuegg.brainiac.experimental.llm.SpinTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Preference and self-improvement experiments on arithmetic: DPO (and IPO, SimPO) from
 * labelled pairs, SPIN from the supervised data alone, and a self-rewarding loop where the
 * model judges its own answers. Each returns the greedy accuracy before and after.
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
