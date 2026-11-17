package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * Classifier-free guidance for language models (Sanchez et al. 2023, "Stay on topic with
 * Classifier-Free Guidance", <a href="https://arxiv.org/abs/2306.17806">arXiv:2306.17806</a>):
 * run the model twice, once with the prompt (conditional) and once without it or with a
 * negative prompt (unconditional), and extrapolate the log-probabilities away from the
 * unconditional ones, {@code log p_u + gamma (log p_c - log p_u)}. {@code gamma > 1} makes
 * the continuation follow the prompt more closely.
 *
 * <p>A {@link LanguageModel} wrapper: the first tokens it reads per sequence are the prompt;
 * the unconditional branch instead reads the negative prompt followed by the prompt's last
 * token, and both read every token after. So the core {@code Generator} does CFG unchanged.
 */
public final class ClassifierFreeGuidance implements LanguageModel {

    private final LanguageModel model;
    private final int[] negative;
    private final double gamma;

    /**
     * @param model the model
     * @param negativePrompt what the unconditional branch reads before the prompt's last
     *     token; empty to drop the prompt entirely
     * @param gamma the guidance strength; 1 is the plain conditional model
     */
    public ClassifierFreeGuidance(LanguageModel model, int[] negativePrompt, double gamma) {
        this.model = model;
        this.negative = negativePrompt.clone();
        this.gamma = gamma;
    }

    /**
     * Guides one row.
     *
     * @param conditional the conditional logits
     * @param unconditional the unconditional logits
     * @param gamma the strength
     * @return the guided log-probabilities
     */
    public static float[] guide(float[] conditional, float[] unconditional, double gamma) {
        double[] c = LogitMath.logSoftmax(conditional);
        double[] u = LogitMath.logSoftmax(unconditional);
        float[] out = new float[c.length];
        for (int i = 0; i < c.length; i++) {
            out[i] = (float) (u[i] + gamma * (c[i] - u[i]));
        }
        return out;
    }

    private static final class Pair implements State {
        State cond;
        State uncond;
        boolean started;

        Pair(State cond, State uncond) {
            this.cond = cond;
            this.uncond = uncond;
        }

        @Override
        public int length() {
            return cond.length();
        }

        @Override
        public State select(int[] rows) {
            cond = cond.select(rows);
            uncond = uncond.select(rows);
            return this;
        }
    }

    @Override
    public int vocabSize() {
        return model.vocabSize();
    }

    @Override
    public State newState(int batch) {
        return new Pair(model.newState(batch), model.newState(batch));
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        Pair p = (Pair) state;
        Tensor c = model.next(p.cond, tokens, padding);
        int[][] u = tokens;
        if (!p.started) {
            p.started = true;
            u = new int[tokens.length][];
            for (int b = 0; b < tokens.length; b++) {
                u[b] = Arrays.copyOf(negative, negative.length + 1);
                u[b][negative.length] = tokens[b][tokens[b].length - 1];
            }
            padding = null;
        }
        Tensor un = model.next(p.uncond, u, padding);
        int batch = tokens.length;
        int vocab = vocabSize();
        float[] cv = c.toArray();
        float[] uv = un.toArray();
        float[] out = new float[cv.length];
        for (int b = 0; b < batch; b++) {
            float[] g = guide(Arrays.copyOfRange(cv, b * vocab, (b + 1) * vocab),
                    Arrays.copyOfRange(uv, b * vocab, (b + 1) * vocab), gamma);
            System.arraycopy(g, 0, out, b * vocab, vocab);
        }
        return Tensor.of(Shape.of(batch, vocab), out);
    }
}
