package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * DoLa, decoding by contrasting layers (Chuang et al. 2023,
 * <a href="https://arxiv.org/abs/2309.03883">arXiv:2309.03883</a>): factual knowledge tends to
 * appear in the later layers, so the next-token scores are the final layer's
 * log-probabilities minus those of an earlier ("premature") layer read through the same
 * output head, {@code log p_N - log p_M}, over the tokens the final layer finds plausible.
 * The premature layer is chosen per token among candidates as the one whose distribution
 * differs most from the final one (largest Jensen-Shannon divergence).
 *
 * <p>A {@link LanguageModel} over a {@link HookedDecoder} (which exposes every layer's
 * residual stream), so the core {@code Generator} decodes with DoLa unchanged.
 */
public final class DolaDecoding implements LanguageModel {

    private static final float FLOOR = -1e30f;

    private final HookedDecoder model;
    private final int[] candidates;
    private final double alpha;
    private int lastPremature = -1;

    /**
     * @param model the decoder
     * @param candidates the premature layers to choose from (0 is the embedding output,
     *     {@code i} the output of block {@code i - 1})
     * @param alpha the plausibility threshold, for example 0.1
     */
    public DolaDecoding(HookedDecoder model, int[] candidates, double alpha) {
        this.model = model;
        this.candidates = candidates.clone();
        this.alpha = alpha;
        for (int c : candidates) {
            if (c < 0 || c >= model.layers()) {
                throw new IllegalArgumentException("premature layers must be below the last, got " + c);
            }
        }
    }

    /** @return the premature layer chosen for the last row scored */
    public int lastPremature() {
        return lastPremature;
    }

    /**
     * The DoLa scores of one position from the logits of every candidate layer.
     *
     * @param finalLogits the last layer's logits
     * @param premature the candidates' logits
     * @param alpha the plausibility threshold
     * @param chosen receives the index of the chosen candidate at {@code [0]}; may be null
     * @return the scores, {@link LogitMath#MASKED} for implausible tokens
     */
    public static float[] scores(float[] finalLogits, float[][] premature, double alpha, int[] chosen) {
        double[] lf = LogitMath.logSoftmax(finalLogits);
        double[] pf = LogitMath.softmax(finalLogits);
        int best = 0;
        double bestJsd = -1;
        for (int c = 0; c < premature.length; c++) {
            double[] pm = LogitMath.softmax(premature[c]);
            double jsd = 0;
            for (int i = 0; i < pf.length; i++) {
                double m = 0.5 * (pf[i] + pm[i]);
                if (pf[i] > 0) {
                    jsd += 0.5 * pf[i] * Math.log(pf[i] / m);
                }
                if (pm[i] > 0) {
                    jsd += 0.5 * pm[i] * Math.log(pm[i] / m);
                }
            }
            if (jsd > bestJsd) {
                bestJsd = jsd;
                best = c;
            }
        }
        if (chosen != null) {
            chosen[0] = best;
        }
        double[] lm = LogitMath.logSoftmax(premature[best]);
        double max = Arrays.stream(lf).max().orElse(0);
        double cut = max + Math.log(alpha);
        float[] out = new float[lf.length];
        for (int i = 0; i < lf.length; i++) {
            out[i] = lf[i] >= cut ? (float) (lf[i] - lm[i]) : LogitMath.MASKED;
        }
        return out;
    }

    private final class Tracked implements State {
        int[][] tokens;

        Tracked(int batch) {
            tokens = new int[batch][0];
        }

        @Override
        public int length() {
            return tokens[0].length;
        }

        @Override
        public State select(int[] rows) {
            int[][] t = new int[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                t[i] = tokens[rows[i]].clone();
            }
            tokens = t;
            return this;
        }

        @Override
        public State truncate(int length) {
            for (int b = 0; b < tokens.length; b++) {
                tokens[b] = Arrays.copyOf(tokens[b], length);
            }
            return this;
        }
    }

    @Override
    public int vocabSize() {
        return model.vocabSize();
    }

    @Override
    public State newState(int batch) {
        return new Tracked(batch);
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        Tensor all = nextAll(state, tokens);
        int steps = all.shape().dim(1);
        return all.slice(1, steps - 1, steps).reshape(tokens.length, vocabSize());
    }

    @Override
    public Tensor nextAll(State state, int[][] tokens) {
        Tracked t = (Tracked) state;
        for (int b = 0; b < tokens.length; b++) {
            int[] n = Arrays.copyOf(t.tokens[b], t.tokens[b].length + tokens[b].length);
            System.arraycopy(tokens[b], 0, n, t.tokens[b].length, tokens[b].length);
            t.tokens[b] = n;
        }
        HookedDecoder.Run run = model.forward(t.tokens);
        List<Var> hidden = run.hidden();
        int batch = tokens.length;
        int steps = tokens[0].length;
        int total = t.tokens[0].length;
        int vocab = vocabSize();
        float[] fin = run.logits().value().toArray();
        float[][] layer = new float[candidates.length][];
        for (int c = 0; c < candidates.length; c++) {
            layer[c] = model.project(hidden.get(candidates[c])).value().toArray();
        }
        float[] out = new float[batch * steps * vocab];
        int[] chosen = new int[1];
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < steps; s++) {
                int off = (b * total + total - steps + s) * vocab;
                float[][] pm = new float[candidates.length][];
                for (int c = 0; c < candidates.length; c++) {
                    pm[c] = Arrays.copyOfRange(layer[c], off, off + vocab);
                }
                float[] sc = scores(Arrays.copyOfRange(fin, off, off + vocab), pm, alpha, chosen);
                lastPremature = candidates[chosen[0]];
                for (int i = 0; i < vocab; i++) {
                    out[(b * steps + s) * vocab + i] = Float.isFinite(sc[i]) ? sc[i] : FLOOR;
                }
            }
        }
        return Tensor.of(Shape.of(batch, steps, vocab), out);
    }
}
