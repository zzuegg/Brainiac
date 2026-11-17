package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.ProcessRewardModel;
import io.github.zzuegg.brainiac.experimental.llm.SequenceScorer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The shared setup of the test-time scaling experiments on {@link ChainTask} (the same as
 * {@link TestTimeScaling}'s): an under-trained generator, and verifier training data made of
 * its own labelled samples plus the reference solutions, with a discriminative outcome
 * verifier and a {@link ProcessRewardModel} trained on them.
 *
 * @param task the puzzles
 * @param generator the under-trained reasoner
 * @param prompts the verifier-training prompts
 * @param completions the verifier-training completions (generator samples and references)
 * @param orm the discriminative outcome verifier
 * @param prm the process reward model
 */
record ChainHarness(ChainTask task, WindowLm generator, List<int[]> prompts, List<int[]> completions,
        SequenceScorer orm, ProcessRewardModel prm) {

    static ChainHarness build(ChainTask task, int sftSteps, long seed) {
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
        SequenceScorer orm = SequenceScorer.of(task.vocabSize(), 12, 64, seed + 1);
        List<int[]> full = new ArrayList<>();
        float[] outcome = new float[vp.size()];
        List<float[]> steps = new ArrayList<>();
        for (int i = 0; i < vp.size(); i++) {
            full.add(TestTimeScaling.concat(vp.get(i), vc.get(i)));
            outcome[i] = task.correct(vp.get(i), vc.get(i)) ? 1 : 0;
            steps.add(task.stepLabels(vp.get(i), vc.get(i)));
        }
        orm.fit(full, outcome, 300);
        ProcessRewardModel prm = ProcessRewardModel.of(SequenceScorer.of(task.vocabSize(), 12, 64, seed + 2),
                task.stepEnd(), eos);
        prm.fit(vp, vc, steps, 300);
        return new ChainHarness(task, lm, vp, vc, orm, prm);
    }

    double ormScore(int[] prompt, int[] completion) {
        return orm.score(TestTimeScaling.concat(prompt, completion));
    }

    double prmScore(int[] prompt, int[] completion) {
        return prm.score(prompt, completion, ProcessRewardModel.Aggregate.MIN);
    }
}
