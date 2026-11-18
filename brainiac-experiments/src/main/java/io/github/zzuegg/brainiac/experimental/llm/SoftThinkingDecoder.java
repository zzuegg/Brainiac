package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Soft thinking (Zhang et al., "Soft thinking: unlocking the reasoning potential of LLMs in
 * continuous concept space", <a href="https://arxiv.org/abs/2505.15778">arXiv:2505.15778</a>):
 * during the reasoning part of a completion the model does not commit to a token. It feeds back
 * the probability-weighted mixture of the top token embeddings, a "concept token", so later steps
 * still see every reading the model was unsure between. It is training-free; for the one-hot
 * input of a {@link WindowLm} the mixture is simply the top-k probability vector in that
 * position's slot.
 *
 * <p>Thinking ends when the most likely token is the thinking-end token (it is then written as a
 * real token) or after {@code maxThink} positions, or early (the paper's <em>cold stop</em>) once
 * the next-token entropy stays below a threshold for a few positions. The answer after it is
 * decoded greedily with ordinary tokens. A single deterministic pass, like greedy decoding.
 */
public final class SoftThinkingDecoder {
    private final WindowLm lm;
    private final int thinkEnd;
    private final int eos;
    private final int maxThink;
    private final int maxAnswer;
    private int topK = 4;
    private double temperature = 1;
    private double coldEntropy = 0;
    private int coldSteps = 2;

    private SoftThinkingDecoder(WindowLm lm, int thinkEnd, int eos, int maxThink, int maxAnswer) {
        this.lm = lm;
        this.thinkEnd = thinkEnd;
        this.eos = eos;
        this.maxThink = maxThink;
        this.maxAnswer = maxAnswer;
    }

    /**
     * Creates a decoder.
     *
     * @param lm the model, left unchanged
     * @param thinkEnd the token that closes the reasoning part
     * @param eos the end token
     * @param maxThink the most concept tokens
     * @param maxAnswer the most answer tokens
     * @return the decoder
     */
    public static SoftThinkingDecoder of(WindowLm lm, int thinkEnd, int eos, int maxThink, int maxAnswer) {
        return new SoftThinkingDecoder(lm, thinkEnd, eos, maxThink, maxAnswer);
    }

    /**
     * Sets the concept-token mixture and the cold stop.
     *
     * @param topK how many tokens a concept token mixes
     * @param temperature softmax temperature of the mixture
     * @param coldEntropy entropy (nats) below which thinking may stop early; 0 disables
     * @param coldSteps consecutive confident positions that trigger the cold stop
     * @return this decoder
     */
    public SoftThinkingDecoder settings(int topK, double temperature, double coldEntropy, int coldSteps) {
        this.topK = topK;
        this.temperature = temperature;
        this.coldEntropy = coldEntropy;
        this.coldSteps = coldSteps;
        return this;
    }

    /**
     * A decoded completion.
     *
     * @param tokens the most likely token of every concept token, the thinking end, then the
     *     answer: a readable transcript
     * @param conceptTokens positions decoded as mixtures
     * @param mixtures the mixtures fed back, one per concept token
     */
    public record Output(int[] tokens, int conceptTokens, List<float[]> mixtures) {
    }

    /**
     * Decodes one prompt.
     *
     * @param prompt the prompt
     * @return the completion
     */
    public Output decode(int[] prompt) {
        int vocab = lm.vocabSize();
        List<float[]> rows = new ArrayList<>();
        for (int t : prompt) {
            float[] r = new float[vocab];
            r[t] = 1;
            rows.add(r);
        }
        List<Integer> transcript = new ArrayList<>();
        List<float[]> mixtures = new ArrayList<>();
        int confident = 0;
        for (int s = 0; s < maxThink; s++) {
            float[] logits = forward(rows);
            int top = LlmTensors.argmax(logits);
            double[] p = LlmTensors.softmax(logits, temperature);
            double entropy = LogitMath.entropy(p);
            confident = entropy < coldEntropy ? confident + 1 : 0;
            if (top == thinkEnd || top == eos || (confident >= coldSteps && s > 0)) {
                break;
            }
            float[] mix = topK(p);
            rows.add(mix);
            mixtures.add(mix);
            transcript.add(top);
        }
        float[] end = new float[vocab];
        end[thinkEnd] = 1;
        rows.add(end);
        transcript.add(thinkEnd);
        for (int s = 0; s < maxAnswer; s++) {
            int tok = LlmTensors.argmax(forward(rows));
            float[] r = new float[vocab];
            r[tok] = 1;
            rows.add(r);
            transcript.add(tok);
            if (tok == eos) {
                break;
            }
        }
        return new Output(transcript.stream().mapToInt(Integer::intValue).toArray(), mixtures.size(), mixtures);
    }

    private float[] topK(double[] p) {
        Integer[] idx = new Integer[p.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(p[b], p[a]));
        float[] mix = new float[p.length];
        double total = 0;
        for (int i = 0; i < Math.min(topK, p.length); i++) {
            total += p[idx[i]];
        }
        for (int i = 0; i < Math.min(topK, p.length); i++) {
            mix[idx[i]] = (float) (p[idx[i]] / total);
        }
        return mix;
    }

    /** The window input of {@link WindowLm#encode} with arbitrary (soft) rows per position. */
    private float[] forward(List<float[]> rows) {
        int vocab = lm.vocabSize();
        int window = lm.window();
        float[] x = new float[window * vocab + window + 1];
        int n = Math.min(window, rows.size());
        for (int i = 0; i < n; i++) {
            float[] r = rows.get(rows.size() - n + i);
            System.arraycopy(r, 0, x, (window - n + i) * vocab, vocab);
        }
        x[window * vocab + n] = 1;
        return LlmTensors.predict(lm.model(), x);
    }

    @Override
    public String toString() {
        return "SoftThinkingDecoder(topK=" + topK + ", T=" + temperature + ")";
    }
}
