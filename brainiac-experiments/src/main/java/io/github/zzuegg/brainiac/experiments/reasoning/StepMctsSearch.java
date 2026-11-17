package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.ProcessRewardModel;
import io.github.zzuegg.brainiac.experimental.llm.ReasoningMcts;
import io.github.zzuegg.brainiac.experimental.llm.SequenceScorer;
import io.github.zzuegg.brainiac.experimental.llm.TestTimeCompute;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * MCTS over reasoning steps (item 5.4, rStar-style) against the test-time-compute baselines
 * of {@link TestTimeScaling}, at a similar number of process-reward-model calls. The setup is
 * the same: an under-trained {@link WindowLm} writes {@link ChainTask} solutions step by step,
 * and a {@link ProcessRewardModel} trained on its own labelled samples scores partial
 * solutions. Compared: one sample, best-of-N reranked by the PRM, Tree-of-Thoughts
 * depth-first search with the PRM, and {@link ReasoningMcts}.
 */
public final class StepMctsSearch {

    /**
     * Accuracy and cost of each method.
     *
     * @param single one sample
     * @param bestOfN best of N by the PRM
     * @param depthFirst ToT depth-first search
     * @param mcts MCTS over steps
     * @param bestOfNCalls PRM calls per question, best-of-N
     * @param depthFirstCalls the most PRM calls per question allowed to depth-first search
     * @param mctsCalls mean distinct PRM calls per question, MCTS
     */
    public record Result(double single, double bestOfN, double depthFirst, double mcts, int bestOfNCalls,
            int depthFirstCalls, double mctsCalls) {
    }

    private StepMctsSearch() {
    }

    /**
     * @param task the puzzles
     * @param sftSteps generator training steps (few, so it errs)
     * @param budget PRM calls per question for best-of-N and depth-first search
     * @param simulations MCTS simulations per committed step
     * @param seed the seed
     * @return the comparison
     */
    public static Result run(ChainTask task, int sftSteps, int budget, int simulations, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<int[]> prompts = task.allPrompts();
        int maxTokens = task.maxSolutionTokens();
        int eos = task.eos();
        WindowLm lm = WindowLm.of(task.vocabSize(), 12, 64, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        List<int[]> sols = new ArrayList<>();
        for (int[] p : prompts) {
            sols.add(task.solution(p));
        }
        for (int s = 0; s < sftSteps; s++) {
            lm.supervisedStep(prompts, sols, opt);
        }
        List<int[]> vp = new ArrayList<>();
        List<int[]> vc = new ArrayList<>();
        for (int k = 0; k < 4; k++) {
            int[][] outs = lm.sample(prompts.toArray(int[][]::new), maxTokens, eos, 1.0, random);
            for (int i = 0; i < outs.length; i++) {
                vp.add(prompts.get(i));
                vc.add(outs[i]);
            }
        }
        for (int i = 0; i < prompts.size(); i++) {
            vp.add(prompts.get(i));
            vc.add(sols.get(i));
        }
        List<float[]> labels = new ArrayList<>();
        for (int i = 0; i < vp.size(); i++) {
            labels.add(task.stepLabels(vp.get(i), vc.get(i)));
        }
        ProcessRewardModel prm = ProcessRewardModel.of(SequenceScorer.of(task.vocabSize(), 12, 64, seed + 2),
                task.stepEnd(), eos);
        prm.fit(vp, vc, labels, 300);
        TestTimeCompute.Scorer value = (p, c) -> prm.score(p, c, ProcessRewardModel.Aggregate.MIN);
        TestTimeCompute.Sampler sampler = TestTimeCompute.Sampler.of(lm, maxTokens, eos, 1.0);

        int single = 0;
        int best = 0;
        int dfs = 0;
        int mcts = 0;
        long mctsCalls = 0;
        int[][] singles = lm.sample(prompts.toArray(int[][]::new), maxTokens, eos, 1.0, random);
        for (int i = 0; i < prompts.size(); i++) {
            int[] p = prompts.get(i);
            single += task.correct(p, singles[i]) ? 1 : 0;
            best += task.correct(p, TestTimeCompute.bestOfN(p, sampler, budget, value, random)) ? 1 : 0;
            dfs += task.correct(p, TestTimeCompute.stepDepthFirst(lm, p, value, 0.3, 4, 3, 4, task.stepEnd(), eos,
                    budget, random)) ? 1 : 0;
            ReasoningMcts search = ReasoningMcts.of(lm, value, task.stepEnd(), eos).expansions(4)
                    .simulations(simulations).limits(3, 4);
            mcts += task.correct(p, search.solve(p, random)) ? 1 : 0;
            mctsCalls += search.valueCalls();
        }
        double m = prompts.size();
        return new Result(single / m, best / m, dfs / m, mcts / m, budget, budget, mctsCalls / m);
    }
}
