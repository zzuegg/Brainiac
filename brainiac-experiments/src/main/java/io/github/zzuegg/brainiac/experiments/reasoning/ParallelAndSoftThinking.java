package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.DeepConf;
import io.github.zzuegg.brainiac.experimental.llm.ParallelThinker;
import io.github.zzuegg.brainiac.experimental.llm.SoftThinkingDecoder;
import io.github.zzuegg.brainiac.experimental.llm.TestTimeCompute;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Soft thinking (<a href="https://arxiv.org/abs/2505.15778">arXiv:2505.15778</a>) and native
 * parallel thinking (<a href="https://arxiv.org/abs/2509.04475">arXiv:2509.04475</a>) on
 * {@link ChainTask}, against chain-of-thought (greedy and sampled), self-consistency over the same
 * parallel paths, and {@link DeepConf}. The generator is a partly supervised {@link WindowLm}, so
 * single samples are often wrong; every arm is scored on held-out questions by accuracy and
 * generated tokens per question. {@link CoconutCurriculum} runs on its own pointer-chasing task
 * and is not repeated here.
 */
public final class ParallelAndSoftThinking {
    /** The sampling temperature of every sampled path. */
    static final double PATH_TEMPERATURE = 0.5;

    /**
     * One arm.
     *
     * @param accuracy share of held-out questions answered correctly
     * @param tokens generated tokens per question
     */
    public record Arm(double accuracy, double tokens) {
    }

    private ParallelAndSoftThinking() {
    }

    /**
     * Runs the comparison.
     *
     * @param task the task
     * @param sftSteps supervised steps for the generator
     * @param paths parallel paths (and self-consistency votes)
     * @param seed the seed
     * @return accuracy and tokens per arm, in insertion order
     */
    public static Map<String, Arm> run(ChainTask task, int sftSteps, int paths, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<int[]> all = task.allPrompts();
        List<int[]> train = new ArrayList<>();
        List<int[]> test = new ArrayList<>();
        for (int[] p : all) {
            (random.nextDouble() < 0.5 ? train : test).add(p);
        }
        WindowLm lm = WindowLm.of(task.vocabSize(), 12, 64, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        List<int[]> sols = new ArrayList<>();
        for (int[] p : all) {
            sols.add(task.solution(p));
        }
        for (int s = 0; s < sftSteps; s++) {
            lm.supervisedStep(all, sols, opt);
        }
        int max = task.maxSolutionTokens();
        int eos = task.eos();
        Map<String, Arm> arms = new LinkedHashMap<>();

        int greedy = 0;
        int sampled = 0;
        long greedyTokens = 0;
        long sampledTokens = 0;
        int[][] once = lm.sample(test.toArray(int[][]::new), max, eos, PATH_TEMPERATURE, random);
        for (int i = 0; i < test.size(); i++) {
            int[] g = lm.greedy(test.get(i), max, eos);
            greedy += task.correct(test.get(i), g) ? 1 : 0;
            greedyTokens += g.length;
            sampled += task.correct(test.get(i), once[i]) ? 1 : 0;
            sampledTokens += once[i].length;
        }
        arms.put("CoT greedy", new Arm((double) greedy / test.size(), (double) greedyTokens / test.size()));
        arms.put("CoT sampled", new Arm((double) sampled / test.size(), (double) sampledTokens / test.size()));

        SoftThinkingDecoder soft = SoftThinkingDecoder.of(lm, task.stepEnd(), eos, max, max)
                .settings(3, 0.25, 0, 2);
        int softOk = 0;
        long softTokens = 0;
        for (int[] p : test) {
            SoftThinkingDecoder.Output o = soft.decode(p);
            softOk += task.correct(p, o.tokens()) ? 1 : 0;
            softTokens += o.tokens().length;
        }
        arms.put("soft thinking", new Arm((double) softOk / test.size(), (double) softTokens / test.size()));

        int pad = task.tokenizer().encode("?")[0];
        ParallelThinker thinker = ParallelThinker.of(lm, paths, max, 3, eos, pad, PATH_TEMPERATURE, 4, 0, 64, seed + 3);
        thinker.train(train, p -> task.tokenizer().encode(task.answer(p) + "#"), 16, 150, 3e-3f,
                new SplittableRandom(seed + 4));
        List<ParallelThinker.Output> outs = thinker.answer(test, new SplittableRandom(seed + 5));
        int merged = 0;
        int voted = 0;
        long parallelTokens = 0;
        long voteTokens = 0;
        for (int i = 0; i < test.size(); i++) {
            ParallelThinker.Output o = outs.get(i);
            merged += task.correct(test.get(i), o.answer()) ? 1 : 0;
            parallelTokens += o.tokens();
            List<Integer> answers = new ArrayList<>();
            for (int[] p : o.paths()) {
                answers.add(Objects.requireNonNullElse(task.finalAnswer(p), -1));
                voteTokens += p.length;
            }
            voted += TestTimeCompute.majorityVote(answers) == task.answer(test.get(i)) ? 1 : 0;
        }
        arms.put("self-consistency x" + paths, new Arm((double) voted / test.size(),
                (double) voteTokens / test.size()));
        arms.put("ParaThinker x" + paths, new Arm((double) merged / test.size(),
                (double) parallelTokens / test.size()));

        DeepConf deepConf = DeepConf.of(lm, max, eos).settings(PATH_TEMPERATURE, 5, 2);
        int dc = 0;
        long dcTokens = 0;
        SplittableRandom dr = new SplittableRandom(seed + 6);
        for (int[] p : test) {
            DeepConf.Vote<Integer> v = deepConf.offline(p, paths, 0.5,
                    c -> Objects.requireNonNullElse(task.finalAnswer(c), -1), dr);
            dc += v.answer() == task.answer(p) ? 1 : 0;
            dcTokens += v.tokens();
        }
        arms.put("DeepConf x" + paths, new Arm((double) dc / test.size(), (double) dcTokens / test.size()));
        return arms;
    }
}
