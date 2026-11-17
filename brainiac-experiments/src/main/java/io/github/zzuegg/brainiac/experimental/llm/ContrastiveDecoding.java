package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Contrastive decoding (Li et al. 2022, <a href="https://arxiv.org/abs/2210.15097">arXiv:2210.15097</a>):
 * the score of a token is {@code log p_expert - log p_amateur / tau}, restricted to the
 * tokens the expert finds plausible ({@code p_expert >= alpha * max p_expert}). A small
 * "amateur" model shares the large model's failure modes (repetition, generic text), so
 * subtracting it keeps what only the expert knows. O'Brien and Lewis 2023
 * (<a href="https://arxiv.org/abs/2309.09117">arXiv:2309.09117</a>) showed it also helps
 * reasoning; their variant {@code (1 + beta) log p_e - beta log p_a} is {@link #scores}
 * with {@code beta}.
 *
 * <p>It is a {@link LanguageModel} whose "logits" are the scores, so greedy decoding in the
 * core {@code Generator} is contrastive decoding: a one-line swap. Both models must share
 * the vocabulary.
 */
public final class ContrastiveDecoding implements LanguageModel {

    private static final float FLOOR = -1e30f;

    private final LanguageModel expert;
    private final LanguageModel amateur;
    private final double alpha;
    private final double beta;

    /**
     * @param expert the large model
     * @param amateur the small model
     * @param alpha the plausibility threshold, for example 0.1
     * @param beta the contrast strength; {@code beta = 1} with {@code alpha} is close to the
     *     original difference of log-probabilities, 0 is the expert alone
     */
    public ContrastiveDecoding(LanguageModel expert, LanguageModel amateur, double alpha, double beta) {
        if (expert.vocabSize() != amateur.vocabSize()) {
            throw new IllegalArgumentException("expert and amateur need the same vocabulary");
        }
        this.expert = expert;
        this.amateur = amateur;
        this.alpha = alpha;
        this.beta = beta;
    }

    /**
     * The contrastive scores of one row.
     *
     * @param expertLogits the expert's logits
     * @param amateurLogits the amateur's logits
     * @param alpha the plausibility threshold
     * @param beta the contrast strength
     * @return {@code (1 + beta) log p_e - beta log p_a} for plausible tokens, {@link LogitMath#MASKED} otherwise
     */
    public static float[] scores(float[] expertLogits, float[] amateurLogits, double alpha, double beta) {
        double[] le = LogitMath.logSoftmax(expertLogits);
        double[] la = LogitMath.logSoftmax(amateurLogits);
        double max = Double.NEGATIVE_INFINITY;
        for (double v : le) {
            max = Math.max(max, v);
        }
        double cut = max + Math.log(alpha);
        float[] out = new float[le.length];
        for (int i = 0; i < le.length; i++) {
            out[i] = le[i] >= cut ? (float) ((1 + beta) * le[i] - beta * la[i]) : LogitMath.MASKED;
        }
        return out;
    }

    private record Pair(State e, State a) implements State {
        @Override
        public int length() {
            return e.length();
        }

        @Override
        public State select(int[] rows) {
            return new Pair(e.select(rows), a.select(rows));
        }

        @Override
        public State truncate(int length) {
            return new Pair(e.truncate(length), a.truncate(length));
        }
    }

    @Override
    public int vocabSize() {
        return expert.vocabSize();
    }

    @Override
    public State newState(int batch) {
        return new Pair(expert.newState(batch), amateur.newState(batch));
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        Pair p = (Pair) state;
        return combine(expert.next(p.e(), tokens, padding), amateur.next(p.a(), tokens, padding));
    }

    @Override
    public Tensor nextAll(State state, int[][] tokens) {
        Pair p = (Pair) state;
        return combine(expert.nextAll(p.e(), tokens), amateur.nextAll(p.a(), tokens));
    }

    private Tensor combine(Tensor e, Tensor a) {
        Shape shape = e.shape();
        int vocab = vocabSize();
        float[] ev = e.toArray();
        float[] av = a.toArray();
        float[] out = new float[ev.length];
        for (int r = 0; r < ev.length / vocab; r++) {
            float[] s = scores(java.util.Arrays.copyOfRange(ev, r * vocab, (r + 1) * vocab),
                    java.util.Arrays.copyOfRange(av, r * vocab, (r + 1) * vocab), alpha, beta);
            for (int i = 0; i < vocab; i++) {
                out[r * vocab + i] = Float.isFinite(s[i]) ? s[i] : FLOOR;
            }
        }
        return Tensor.of(shape, out);
    }
}
