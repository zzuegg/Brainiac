package io.github.zzuegg.brainiac.experimental.data;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Dynamic byte patching from the Byte Latent Transformer (BLT): a small byte language model
 * measures how surprising the next byte is, and a new patch starts where the next-byte
 * entropy is high. Predictable stretches (the rest of a word, repeated text) become long
 * patches that a large latent model processes as one step; hard places get short patches
 * and more compute. Unlike a fixed tokenizer there is no vocabulary; patches are just byte
 * spans.
 *
 * <p>Two boundary rules from the paper:
 * <ul>
 *   <li>{@link Rule#GLOBAL}: start a patch where {@code H(x_t) > threshold};</li>
 *   <li>{@link Rule#MONOTONIC}: start a patch where the entropy jumps,
 *       {@code H(x_t) - H(x_{t-1}) > threshold}.</li>
 * </ul>
 * A maximum patch length bounds the work per patch.
 *
 * <p>Paper: Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than Tokens",
 * <a href="https://arxiv.org/abs/2412.09871">arXiv:2412.09871</a>.
 */
public final class EntropyPatcher {

    /** Where a new patch begins. */
    public enum Rule {
        /** Entropy above a global threshold. */
        GLOBAL,
        /** Entropy rising by more than the threshold over the previous byte. */
        MONOTONIC
    }

    private final ByteNgramModel model;
    private final Rule rule;
    private final double threshold;
    private final int maxPatch;

    /**
     * Creates a patcher.
     *
     * @param model the small byte model that scores entropy
     * @param rule the boundary rule
     * @param threshold the entropy threshold in bits
     * @param maxPatch the longest patch
     */
    public EntropyPatcher(ByteNgramModel model, Rule rule, double threshold, int maxPatch) {
        this.model = Objects.requireNonNull(model, "model");
        this.rule = Objects.requireNonNull(rule, "rule");
        this.threshold = threshold;
        if (maxPatch < 1) {
            throw new IllegalArgumentException("maxPatch must be positive");
        }
        this.maxPatch = maxPatch;
    }

    /**
     * Returns the start index of every patch.
     *
     * @param bytes the bytes
     * @return the patch starts, beginning with 0
     */
    public int[] boundaries(byte[] bytes) {
        double[] h = model.entropies(bytes);
        List<Integer> starts = new ArrayList<>();
        int last = 0;
        for (int i = 0; i < bytes.length; i++) {
            boolean start = i == 0
                    || i - last >= maxPatch
                    || (rule == Rule.GLOBAL ? h[i] > threshold : h[i] - h[i - 1] > threshold);
            if (start) {
                starts.add(i);
                last = i;
            }
        }
        return starts.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Assigns every byte to its patch.
     *
     * @param bytes the bytes
     * @return the patch index of each byte
     */
    public int[] patchIds(byte[] bytes) {
        int[] starts = boundaries(bytes);
        int[] ids = new int[bytes.length];
        int p = 0;
        for (int i = 0; i < bytes.length; i++) {
            if (p + 1 < starts.length && starts[p + 1] == i) {
                p++;
            }
            ids[i] = p;
        }
        return ids;
    }

    /**
     * Splits text into patches.
     *
     * @param text the text, as UTF-8
     * @return the patches, decoded back to strings (a patch may split a multi-byte character)
     */
    public List<String> patches(String text) {
        byte[] b = text.getBytes(StandardCharsets.UTF_8);
        int[] starts = boundaries(b);
        List<String> out = new ArrayList<>();
        for (int i = 0; i < starts.length; i++) {
            int end = i + 1 < starts.length ? starts[i + 1] : b.length;
            out.add(new String(b, starts[i], end - starts[i], StandardCharsets.UTF_8));
        }
        return out;
    }
}
