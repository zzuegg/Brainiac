package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.DeepConf;
import io.github.zzuegg.brainiac.experimental.llm.TestTimeCompute;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * DeepConf against plain self-consistency and PRM-ranked best-of-N, on the
 * {@link TestTimeScaling} setup (an under-trained {@link ChainTask} reasoner,
 * {@link ChainHarness}): accuracy against the tokens generated, at the same number of traces.
 */
public final class ConfidenceVoting {

    /**
     * Accuracy and mean tokens per question for each strategy.
     *
     * @param selfConsistency majority vote of all traces
     * @param selfConsistencyTokens its tokens
     * @param offline DeepConf-offline (top half by lowest group confidence, weighted)
     * @param offlineTokens its tokens
     * @param online DeepConf-online (warm-up threshold, early stopping)
     * @param onlineTokens its tokens
     * @param prmBestOfN the trace the process reward model scores best
     * @param prmTokens its tokens (generator only)
     */
    public record Result(double selfConsistency, double selfConsistencyTokens, double offline, double offlineTokens,
            double online, double onlineTokens, double prmBestOfN, double prmTokens) {
    }

    private ConfidenceVoting() {
    }

    /**
     * @param task the puzzles
     * @param sftSteps how long the reasoner is fine-tuned
     * @param n traces per question
     * @param warmup DeepConf-online's warm-up traces
     * @param keepTop the share of traces DeepConf keeps
     * @param topK the candidates per token confidence
     * @param window the group window in tokens
     * @param seed the seed
     * @return the measurements
     */
    public static Result run(ChainTask task, int sftSteps, int n, int warmup, double keepTop, int topK, int window,
            long seed) {
        ChainHarness h = ChainHarness.build(task, sftSteps, seed);
        DeepConf dc = DeepConf.of(h.generator(), task.maxSolutionTokens(), task.eos()).settings(1.0, topK, window);
        SplittableRandom random = new SplittableRandom(seed + 5);
        double sc = 0;
        double scTokens = 0;
        double off = 0;
        double offTokens = 0;
        double on = 0;
        double onTokens = 0;
        double prm = 0;
        List<int[]> prompts = task.allPrompts();
        for (int[] p : prompts) {
            int answer = task.answer(p);
            List<DeepConf.Trace> traces = dc.sample(p, n, Double.NEGATIVE_INFINITY, random);
            List<Integer> votes = new ArrayList<>();
            int[] best = null;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (DeepConf.Trace t : traces) {
                votes.add(task.finalAnswer(t.tokens()));
                scTokens += t.tokens().length;
                double s = h.prmScore(p, t.tokens());
                if (s > bestScore) {
                    bestScore = s;
                    best = t.tokens();
                }
            }
            Integer v = TestTimeCompute.majorityVote(votes);
            sc += v != null && v == answer ? 1 : 0;
            prm += best != null && task.correct(p, best) ? 1 : 0;
            Integer o = DeepConf.vote(traces, task::finalAnswer, keepTop);
            off += o != null && o == answer ? 1 : 0;
            offTokens += traces.stream().mapToInt(t -> t.tokens().length).sum();
            DeepConf.Vote<Integer> online = dc.online(p, n, warmup, keepTop, task::finalAnswer, random);
            on += online.answer() != null && online.answer() == answer ? 1 : 0;
            onTokens += online.tokens();
        }
        double m = prompts.size();
        return new Result(sc / m, scTokens / m, off / m, offTokens / m, on / m, onTokens / m, prm / m, scTokens / m);
    }
}
