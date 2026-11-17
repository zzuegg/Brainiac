package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.ProcessRewardModel;
import io.github.zzuegg.brainiac.experimental.llm.SequenceScorer;
import io.github.zzuegg.brainiac.experimental.llm.TestTimeCompute;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Experiment X15, self-critique and test-time compute: a deliberately under-trained
 * {@link WindowLm} solves {@link ChainTask} puzzles; an outcome verifier and a
 * {@link ProcessRewardModel} are trained on its own labelled samples. The same generator is
 * then evaluated with one sample, best-of-N, self-consistency, verifier-weighted voting, a
 * critique-and-resample loop, and Tree-of-Thoughts beam and depth-first search guided by the
 * PRM.
 */
public final class TestTimeScaling {

    /**
     * Accuracy of each way of answering.
     *
     * @param singleSample one sample at temperature 1
     * @param greedy the most likely tokens
     * @param bestOfN best of N by the outcome verifier
     * @param selfConsistency majority vote of N
     * @param weightedVote votes weighted by the verifier
     * @param critique the critique-and-resample loop
     * @param critiqueSamples mean samples the critique loop used
     * @param prmBestOfN best of N by the PRM (min over steps)
     * @param beamSearch step-level beam search with the PRM as value
     * @param depthFirst step-level depth-first search with the PRM
     */
    public record Result(double singleSample, double greedy, double bestOfN, double selfConsistency,
            double weightedVote, double critique, double critiqueSamples, double prmBestOfN, double beamSearch,
            double depthFirst) {
    }

    private TestTimeScaling() {
    }

    /**
     * @param task the puzzles
     * @param sftSteps how long to fine-tune the generator (few, so it errs)
     * @param n samples per question for the sampling methods
     * @param seed the seed
     * @return the accuracies
     */
    public static Result run(ChainTask task, int sftSteps, int n, long seed) {
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

        // the verifiers learn from the generator's own labelled samples
        List<int[]> vp = new ArrayList<>();
        List<int[]> vc = new ArrayList<>();
        for (int k = 0; k < 4; k++) {
            int[][] outs = lm.sample(prompts.toArray(int[][]::new), maxTokens, eos, 1.0, random);
            for (int i = 0; i < outs.length; i++) {
                vp.add(prompts.get(i));
                vc.add(outs[i]);
            }
        }
        for (int i = 0; i < prompts.size(); i++) {   // and the reference solutions
            vp.add(prompts.get(i));
            vc.add(sols.get(i));
        }
        SequenceScorer orm = SequenceScorer.of(task.vocabSize(), 12, 64, seed + 1);
        List<int[]> full = new ArrayList<>();
        float[] outcome = new float[vp.size()];
        List<float[]> steps = new ArrayList<>();
        for (int i = 0; i < vp.size(); i++) {
            full.add(concat(vp.get(i), vc.get(i)));
            outcome[i] = task.correct(vp.get(i), vc.get(i)) ? 1 : 0;
            steps.add(task.stepLabels(vp.get(i), vc.get(i)));
        }
        orm.fit(full, outcome, 300);
        ProcessRewardModel prm = ProcessRewardModel.of(SequenceScorer.of(task.vocabSize(), 12, 64, seed + 2),
                task.stepEnd(), eos);
        prm.fit(vp, vc, steps, 300);

        TestTimeCompute.Sampler sampler = TestTimeCompute.Sampler.of(lm, maxTokens, eos, 1.0);
        TestTimeCompute.Scorer outcomeScore = (p, c) -> orm.score(concat(p, c));
        TestTimeCompute.Scorer processScore = (p, c) -> prm.score(p, c, ProcessRewardModel.Aggregate.MIN);

        int single = 0;
        int greedy = 0;
        int best = 0;
        int vote = 0;
        int weighted = 0;
        int critique = 0;
        int critiqueSamples = 0;
        int prmBest = 0;
        int beam = 0;
        int dfs = 0;
        int[][] singles = lm.sample(prompts.toArray(int[][]::new), maxTokens, eos, 1.0, random);
        int[][] greedies = lm.sample(prompts.toArray(int[][]::new), maxTokens, eos, 0, random);
        for (int i = 0; i < prompts.size(); i++) {
            int[] p = prompts.get(i);
            int answer = task.answer(p);
            single += task.correct(p, singles[i]) ? 1 : 0;
            greedy += task.correct(p, greedies[i]) ? 1 : 0;
            best += task.correct(p, TestTimeCompute.bestOfN(p, sampler, n, outcomeScore, random)) ? 1 : 0;
            Integer sc = TestTimeCompute.selfConsistency(p, sampler, n, task::finalAnswer, random);
            vote += sc != null && sc == answer ? 1 : 0;
            Integer wv = TestTimeCompute.weightedVote(p, sampler, n, task::finalAnswer, outcomeScore, random);
            weighted += wv != null && wv == answer ? 1 : 0;
            TestTimeCompute.Critique c = TestTimeCompute.critiqueLoop(p, sampler, outcomeScore, 0.8, n, random);
            critique += task.correct(p, c.completion()) ? 1 : 0;
            critiqueSamples += c.samples();
            prmBest += task.correct(p, TestTimeCompute.bestOfN(p, sampler, n, processScore, random)) ? 1 : 0;
            beam += task.correct(p, TestTimeCompute.stepBeamSearch(lm, p, processScore, 2, 4, 3, 4,
                    task.stepEnd(), eos, random)) ? 1 : 0;
            dfs += task.correct(p, TestTimeCompute.stepDepthFirst(lm, p, processScore, 0.3, 4, 3, 4,
                    task.stepEnd(), eos, 4 * n, random)) ? 1 : 0;
        }
        double m = prompts.size();
        return new Result(single / m, greedy / m, best / m, vote / m, weighted / m, critique / m,
                critiqueSamples / m, prmBest / m, beam / m, dfs / m);
    }

    static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
