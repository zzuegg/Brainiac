package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Speculative cascades: cascades and speculative decoding in one loop. A cascade lets a small
 * model answer and calls the large one only when a deferral rule says so (cheap, but each
 * deferral is a sequential large-model call, and the small model's confident mistakes stay);
 * speculative decoding drafts with the small model and verifies every token with one parallel
 * pass of the large one (exactly the large model's output, but no cheaper in quality terms).
 * A speculative cascade drafts and verifies like speculative decoding, but at each drafted
 * position a <em>deferral rule</em> looking at both models' distributions decides whether the
 * small model's token may stand even where the large model would pick another; the first
 * position it rejects is replaced by the large model's choice. The rule trades quality for
 * accepted length, and the large model's passes stay parallel.
 *
 * <p>Greedy decoding with three rules ({@link Rule}); {@code alpha = 0} under
 * {@link Rule#TOKEN} is exactly greedy speculative decoding of the large model.
 * {@link #cascade} is the sequential confidence cascade (Chow's rule) baseline. Both models
 * must share the vocabulary and be able to {@linkplain LanguageModel.State#truncate go back}.
 *
 * <p>Narasimhan et al., "Faster Cascades via Speculative Decoding" (ICLR 2025),
 * <a href="https://arxiv.org/abs/2405.19261">arXiv:2405.19261</a>.
 */
public final class SpeculativeCascade {

    /** When the small model's drafted token may stand. */
    public enum Rule {
        /** Chow: when the small model's top probability is at least {@code 1 - alpha} */
        CHOW,
        /** Diff: when the small model's top probability is at least the large one's minus {@code alpha} */
        DIFF,
        /** token-specific: when the large model gives the token at least its top probability minus {@code alpha} */
        TOKEN
    }

    /**
     * A generation.
     *
     * @param tokens the new tokens
     * @param largePasses the large model's passes (each reads several tokens at once)
     * @param smallSteps the small model's single-token steps
     * @param deferred positions where the large model's token replaced the small one's
     */
    public record Result(int[] tokens, int largePasses, int smallSteps, int deferred) {

        /**
         * @param smallCost one small step's cost in large passes
         * @return the cost in large-pass units
         */
        public double cost(double smallCost) {
            return largePasses + smallCost * smallSteps;
        }
    }

    private SpeculativeCascade() {
    }

    /** Whether the small model's token stands under the rule. */
    static boolean accept(Rule rule, double alpha, double[] q, double[] p, int token) {
        double qMax = Arrays.stream(q).max().orElse(0);
        double pMax = Arrays.stream(p).max().orElse(0);
        return switch (rule) {
            case CHOW -> qMax >= 1 - alpha;
            case DIFF -> qMax >= pMax - alpha;
            case TOKEN -> p[token] >= pMax - alpha;
        };
    }

    /**
     * A speculative cascade.
     *
     * @param small the drafter
     * @param large the verifier
     * @param prompt the prompt
     * @param maxNew tokens to generate
     * @param lookahead tokens drafted per large pass
     * @param rule the deferral rule
     * @param alpha the rule's tolerance
     * @return the tokens and counts
     */
    public static Result speculative(LanguageModel small, LanguageModel large, int[] prompt, int maxNew,
            int lookahead, Rule rule, double alpha) {
        int vocab = large.vocabSize();
        LanguageModel.State ls = large.newState(1);
        LanguageModel.State ss = small.newState(1);
        double[] p = LogitMath.softmax(large.next(ls, new int[][] {prompt}, null).toArray());
        double[] q = LogitMath.softmax(small.next(ss, new int[][] {prompt}, null).toArray());
        int passes = 1;
        int steps = 1;
        int deferred = 0;
        List<Integer> out = new ArrayList<>();
        int first = argmax(q);
        if (!accept(rule, alpha, q, p, first)) {
            first = argmax(p);
            deferred++;
        }
        int pending = first;
        out.add(pending);
        while (out.size() < maxNew) {
            int base = ls.length();
            q = LogitMath.softmax(small.next(ss, new int[][] {{pending}}, null).toArray());
            steps++;
            int[] guess = new int[lookahead];
            double[][] qs = new double[lookahead][];
            for (int i = 0; i < lookahead; i++) {
                qs[i] = q;
                guess[i] = argmax(q);
                q = LogitMath.softmax(small.next(ss, new int[][] {{guess[i]}}, null).toArray());
                steps++;
            }
            int[] read = new int[lookahead + 1];
            read[0] = pending;
            System.arraycopy(guess, 0, read, 1, lookahead);
            float[] all = large.nextAll(ls, new int[][] {read}).toArray();
            passes++;
            int keep = 0;
            int next = -1;
            for (int i = 0; i < lookahead && out.size() < maxNew; i++) {
                double[] pi = LogitMath.softmax(Arrays.copyOfRange(all, i * vocab, (i + 1) * vocab));
                if (accept(rule, alpha, qs[i], pi, guess[i])) {
                    out.add(guess[i]);
                    keep++;
                } else {
                    next = argmax(pi);
                    deferred++;
                    break;
                }
            }
            if (out.size() >= maxNew) {
                break;
            }
            if (next < 0) {   // every guess stood: the large model's next token comes free
                next = LogitMath.argmax(Arrays.copyOfRange(all, lookahead * vocab, (lookahead + 1) * vocab));
            }
            out.add(next);
            ls = ls.truncate(base + 1 + keep);
            ss = ss.truncate(base + 1 + keep);
            pending = next;
        }
        int[] tokens = out.stream().mapToInt(Integer::intValue).limit(maxNew).toArray();
        return new Result(tokens, passes, steps, deferred);
    }

    /**
     * The sequential confidence cascade: the small model writes each token unless its top
     * probability is below {@code 1 - alpha}; then the large model reads everything it has not
     * read yet in one pass and writes that token.
     *
     * @param small the small model
     * @param large the large model
     * @param prompt the prompt
     * @param maxNew tokens to generate
     * @param alpha the tolerance (0 always defers, 1 never)
     * @return the tokens and counts
     */
    public static Result cascade(LanguageModel small, LanguageModel large, int[] prompt, int maxNew, double alpha) {
        LanguageModel.State ss = small.newState(1);
        LanguageModel.State ls = large.newState(1);
        double[] q = LogitMath.softmax(small.next(ss, new int[][] {prompt}, null).toArray());
        int steps = 1;
        int passes = 0;
        int deferred = 0;
        List<Integer> unread = new ArrayList<>();
        for (int t : prompt) {
            unread.add(t);
        }
        List<Integer> out = new ArrayList<>();
        while (out.size() < maxNew) {
            int token = argmax(q);
            if (Arrays.stream(q).max().orElse(0) < 1 - alpha) {
                int[] read = unread.stream().mapToInt(Integer::intValue).toArray();
                double[] p = LogitMath.softmax(large.next(ls, new int[][] {read}, null).toArray());
                passes++;
                unread.clear();
                token = argmax(p);
                deferred++;
            }
            out.add(token);
            unread.add(token);
            if (out.size() < maxNew) {
                q = LogitMath.softmax(small.next(ss, new int[][] {{token}}, null).toArray());
                steps++;
            }
        }
        return new Result(out.stream().mapToInt(Integer::intValue).toArray(), passes, steps, deferred);
    }

    private static int argmax(double[] x) {
        int best = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[i] > x[best]) {
                best = i;
            }
        }
        return best;
    }
}
