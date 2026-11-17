package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GenerativeVerifier;
import io.github.zzuegg.brainiac.experimental.llm.TestTimeCompute;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Generative verifiers ({@link GenerativeVerifier}, GenRM) against the discriminative outcome
 * verifier and the process reward model of {@link TestTimeScaling}, as best-of-N rankers of
 * the same under-trained {@link ChainTask} reasoner ({@link ChainHarness}). All verifiers learn
 * from the same labelled samples. GenRM-CoT's rationale is the reference chain of thought
 * (the verifier re-solves the problem before judging, as in the paper's reference-guided
 * rationales), and at test time it averages {@code P(Yes)} over sampled rationales.
 */
public final class GenerativeVerification {

    /**
     * Best-of-N accuracy by ranker.
     *
     * @param single one sample
     * @param oracle any of the N is right (the ceiling)
     * @param discriminative the outcome verifier ({@code SequenceScorer}, classification loss)
     * @param prm the process reward model (minimum over steps)
     * @param genrmDirect GenRM, {@code P(Yes)} right after the query
     * @param genrmCot GenRM-CoT, mean {@code P(Yes)} over sampled rationales
     */
    public record Result(double single, double oracle, double discriminative, double prm, double genrmDirect,
            double genrmCot) {
    }

    private GenerativeVerification() {
    }

    private static int argmax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * @param task the puzzles
     * @param sftSteps how long the reasoner is fine-tuned
     * @param verifierSteps steps (batches of 64) for each generative verifier
     * @param n candidates per question
     * @param votes GenRM-CoT rationales per candidate
     * @param seed the seed
     * @return the accuracies
     */
    public static Result run(ChainTask task, int sftSteps, int verifierSteps, int n, int votes, long seed) {
        ChainHarness h = ChainHarness.build(task, sftSteps, seed);
        int vocab = task.vocabSize() + 2;
        int query = task.tokenizer().encode("?")[0];
        int yes = task.vocabSize();
        int no = yes + 1;
        GenerativeVerifier direct = GenerativeVerifier.of(WindowLm.of(vocab, 20, 64, seed + 3), query, yes, no);
        GenerativeVerifier cot = GenerativeVerifier.of(WindowLm.of(vocab, 20, 64, seed + 3), query, yes, no);
        boolean[] correct = new boolean[h.prompts().size()];
        List<int[]> rationales = new ArrayList<>();
        for (int i = 0; i < correct.length; i++) {
            correct[i] = task.correct(h.prompts().get(i), h.completions().get(i));
            rationales.add(task.solution(h.prompts().get(i)));
        }
        Optimizer o1 = Optimizer.adam(3e-3f);
        Optimizer o2 = Optimizer.adam(3e-3f);
        SplittableRandom random = new SplittableRandom(seed + 9);
        int batch = 64;
        for (int s = 0; s < verifierSteps; s++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            List<int[]> rs = new ArrayList<>();
            boolean[] ok = new boolean[batch];
            for (int b = 0; b < batch; b++) {
                int i = random.nextInt(correct.length);
                ps.add(h.prompts().get(i));
                cs.add(h.completions().get(i));
                rs.add(rationales.get(i));
                ok[b] = correct[i];
            }
            direct.step(ps, cs, ok, null, o1);
            cot.step(ps, cs, ok, rs, o2);
        }

        TestTimeCompute.Sampler sampler = TestTimeCompute.Sampler.of(h.generator(), task.maxSolutionTokens(), task.eos(),
                1.0);
        double single = 0;
        double oracle = 0;
        double disc = 0;
        double prm = 0;
        double gd = 0;
        double gc = 0;
        List<int[]> prompts = task.allPrompts();
        for (int[] p : prompts) {
            int[][] cands = sampler.sample(p, n, random);
            single += task.correct(p, cands[0]) ? 1 : 0;
            boolean any = false;
            for (int[] c : cands) {
                any |= task.correct(p, c);
            }
            oracle += any ? 1 : 0;
            TestTimeCompute.Sampler fixed = (q, k, r) -> cands;
            disc += task.correct(p, TestTimeCompute.bestOfN(p, fixed, n, h::ormScore, random)) ? 1 : 0;
            prm += task.correct(p, TestTimeCompute.bestOfN(p, fixed, n, h::prmScore, random)) ? 1 : 0;
            List<int[]> list = List.of(cands);
            gd += task.correct(p, cands[argmax(direct.scoreAll(p, list))]) ? 1 : 0;
            gc += task.correct(p, cands[argmax(cot.scoreAll(p, list, votes, task.maxSolutionTokens(), task.eos(), 1.0,
                    random))]) ? 1 : 0;
        }
        double m = prompts.size();
        return new Result(single / m, oracle / m, disc / m, prm / m, gd / m, gc / m);
    }
}
