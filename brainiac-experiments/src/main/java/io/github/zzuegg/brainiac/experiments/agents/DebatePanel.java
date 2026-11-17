package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.llm.AnswerDebate;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticTask;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A panel of independently trained tiny language models answers multiplication puzzles: alone,
 * by majority vote, as a product of experts, and after rounds of debate in answer space.
 */
public final class DebatePanel {

    /**
     * @param singleMean the mean accuracy of the agents alone (greedy)
     * @param singleBest the best agent's accuracy
     * @param vote majority vote over every agent's favourite
     * @param product the product of experts
     * @param debate debate, then a vote
     */
    public record Result(double singleMean, double singleBest, double vote, double product, double debate) {
    }

    private DebatePanel() {
    }

    /**
     * @param agents panel size
     * @param sftSteps supervised steps per agent
     * @param rounds debate rounds
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(int agents, int sftSteps, int rounds, long seed) {
        return run(agents, sftSteps, rounds, 0.5, seed);
    }

    static Result run(int agents, int sftSteps, int rounds, double agreement, long seed) {
        ArithmeticTask task = ArithmeticTask.multiplication(9);
        List<int[]> prompts = task.allPrompts();
        List<int[]> answers = new ArrayList<>();
        for (int[] p : prompts) {
            answers.add(task.solution(p));
        }
        List<WindowLm> models = new ArrayList<>();
        SplittableRandom random = new SplittableRandom(seed);
        for (int k = 0; k < agents; k++) {
            WindowLm lm = WindowLm.of(task.vocabSize(), 6, 64, seed + 17L * k);
            Optimizer opt = Optimizer.adam(3e-3f);
            for (int s = 0; s < sftSteps; s++) {
                List<int[]> ps = new ArrayList<>();
                List<int[]> as = new ArrayList<>();
                for (int i = 0; i < 48; i++) {
                    int j = random.nextInt(prompts.size());
                    ps.add(prompts.get(j));
                    as.add(answers.get(j));
                }
                lm.supervisedStep(ps, as, opt);
            }
            models.add(lm);
        }
        List<AnswerDebate.Debater> debaters = new ArrayList<>();
        for (WindowLm lm : models) {
            debaters.add(lm::sequenceLogProb);
        }
        AnswerDebate panel = AnswerDebate.of(debaters).agreement(agreement);
        double[] single = new double[agents];
        int vote = 0;
        int product = 0;
        int debate = 0;
        for (int[] prompt : prompts) {
            List<int[]> pool = new ArrayList<>();
            for (int k = 0; k < agents; k++) {
                int[] g = models.get(k).greedy(prompt, task.maxAnswerTokens(), task.eos());
                single[k] += task.correct(prompt, g) ? 1 : 0;
                pool.add(g);
                int[][] batch = new int[4][];
                for (int i = 0; i < batch.length; i++) {
                    batch[i] = prompt;
                }
                pool.addAll(List.of(models.get(k).sample(batch, task.maxAnswerTokens(), task.eos(), 1.0, random)));
            }
            vote += task.correct(prompt, panel.vote(prompt, pool)) ? 1 : 0;
            product += task.correct(prompt, panel.productOfExperts(prompt, pool)) ? 1 : 0;
            debate += task.correct(prompt, panel.debate(prompt, pool, rounds).answer()) ? 1 : 0;
        }
        double n = prompts.size();
        double mean = 0;
        double best = 0;
        for (double s : single) {
            mean += s / n / agents;
            best = Math.max(best, s / n);
        }
        return new Result(mean, best, vote / n, product / n, debate / n);
    }
}
