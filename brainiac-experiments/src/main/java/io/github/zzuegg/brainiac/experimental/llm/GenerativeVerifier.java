package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A generative verifier, GenRM (Zhang et al., "Generative verifiers: reward modeling as
 * next-token prediction", <a href="https://arxiv.org/abs/2408.15240">2408.15240</a>): instead of
 * a scalar head trained with a classification loss (a discriminative reward model such as
 * {@link RewardModel} or {@link SequenceScorer}), the verifier is a language model asked
 * "is this solution correct?" and trained by next-token prediction to answer {@code Yes} or
 * {@code No}. The score is {@code P(Yes)}.
 * <ul>
 *   <li><b>Direct</b>: {@code problem solution QUERY -> Yes|No}.</li>
 *   <li><b>Chain of thought</b> (GenRM-CoT): the verifier first writes a verification
 *       rationale ({@code problem solution QUERY rationale -> Yes|No}), trained on reference
 *       rationales; at test time it samples several rationales and averages {@code P(Yes)}
 *       over them, a majority vote that spends verifier compute for accuracy.</li>
 * </ul>
 * Being an ordinary {@link WindowLm}, the same model can also be trained to solve problems,
 * which is how the paper unifies generation and verification.
 */
public final class GenerativeVerifier {

    private final WindowLm lm;
    private final int query;
    private final int yes;
    private final int no;

    private GenerativeVerifier(WindowLm lm, int query, int yes, int no) {
        this.lm = Objects.requireNonNull(lm);
        this.query = query;
        this.yes = yes;
        this.no = no;
    }

    /**
     * @param lm the verifier language model (its vocabulary holds the three special tokens)
     * @param query the token that asks "is it correct?"
     * @param yes the Yes token
     * @param no the No token
     * @return the verifier
     */
    public static GenerativeVerifier of(WindowLm lm, int query, int yes, int no) {
        return new GenerativeVerifier(lm, query, yes, no);
    }

    /** @return the verifier model */
    public WindowLm model() {
        return lm;
    }

    private int[] ask(int[] problem, int[] solution) {
        int[] c = Arrays.copyOf(problem, problem.length + solution.length + 1);
        System.arraycopy(solution, 0, c, problem.length, solution.length);
        c[c.length - 1] = query;
        return c;
    }

    /**
     * One next-token-prediction step on labelled solutions.
     *
     * @param problems the problems
     * @param solutions candidate solutions
     * @param correct whether each is correct
     * @param rationales a verification rationale for each (GenRM-CoT), or {@code null} for direct
     * @param optimizer the optimizer
     * @return the mean token loss
     */
    public float step(List<int[]> problems, List<int[]> solutions, boolean[] correct, List<int[]> rationales,
            Optimizer optimizer) {
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int i = 0; i < problems.size(); i++) {
            ps.add(ask(problems.get(i), solutions.get(i)));
            int[] r = rationales == null ? new int[0] : rationales.get(i);
            int[] target = Arrays.copyOf(r, r.length + 1);
            target[r.length] = correct[i] ? yes : no;
            cs.add(target);
        }
        return lm.supervisedStep(ps, cs, optimizer);
    }

    /**
     * Direct GenRM.
     *
     * @param problem the problem
     * @param solution a candidate
     * @return {@code P(Yes)}, renormalized over Yes and No
     */
    public double score(int[] problem, int[] solution) {
        return scoreAll(problem, List.of(solution))[0];
    }

    /**
     * GenRM-CoT with majority voting: samples rationales and averages {@code P(Yes)} after each.
     *
     * @param problem the problem
     * @param solution a candidate
     * @param votes rationales to sample
     * @param maxRationale the most rationale tokens
     * @param rationaleEnd the token that ends a rationale
     * @param temperature the rationale sampling temperature
     * @param random the random source
     * @return the mean {@code P(Yes)}
     */
    public double score(int[] problem, int[] solution, int votes, int maxRationale, int rationaleEnd,
            double temperature, SplittableRandom random) {
        return scoreAll(problem, List.of(solution), votes, maxRationale, rationaleEnd, temperature, random)[0];
    }

    /**
     * Direct GenRM for several candidates in one batch.
     *
     * @param problem the problem
     * @param solutions the candidates
     * @return {@code P(Yes)} for each
     */
    public double[] scoreAll(int[] problem, List<int[]> solutions) {
        List<int[]> contexts = new ArrayList<>();
        for (int[] s : solutions) {
            contexts.add(ask(problem, s));
        }
        return yesProbabilities(contexts, 1);
    }

    /**
     * GenRM-CoT for several candidates, all rationales sampled in one batch.
     *
     * @param problem the problem
     * @param solutions the candidates
     * @param votes rationales per candidate
     * @param maxRationale the most rationale tokens
     * @param rationaleEnd the token that ends a rationale
     * @param temperature the rationale sampling temperature
     * @param random the random source
     * @return the mean {@code P(Yes)} for each candidate
     */
    public double[] scoreAll(int[] problem, List<int[]> solutions, int votes, int maxRationale, int rationaleEnd,
            double temperature, SplittableRandom random) {
        int[][] ctx = new int[solutions.size() * votes][];
        for (int i = 0; i < solutions.size(); i++) {
            int[] context = ask(problem, solutions.get(i));
            for (int k = 0; k < votes; k++) {
                ctx[i * votes + k] = context;
            }
        }
        int[][] rationales = lm.sample(ctx, maxRationale, rationaleEnd, temperature, random);
        List<int[]> full = new ArrayList<>();
        for (int i = 0; i < ctx.length; i++) {
            int[] c = Arrays.copyOf(ctx[i], ctx[i].length + rationales[i].length);
            System.arraycopy(rationales[i], 0, c, ctx[i].length, rationales[i].length);
            full.add(c);
        }
        return yesProbabilities(full, votes);
    }

    /** {@code P(Yes)} renormalized over Yes and No after each context, averaged over consecutive groups. */
    private double[] yesProbabilities(List<int[]> contexts, int group) {
        float[] logits = LlmTensors.values(lm.logits(contexts));
        int v = lm.vocabSize();
        double[] out = new double[contexts.size() / group];
        for (int i = 0; i < contexts.size(); i++) {
            double[] p = LlmTensors.softmax(Arrays.copyOfRange(logits, i * v, (i + 1) * v), 1);
            out[i / group] += p[yes] / (p[yes] + p[no]) / group;
        }
        return out;
    }
}
