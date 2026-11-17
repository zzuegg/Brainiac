package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.Tokenizer;

import java.util.HashMap;
import java.util.Map;

/**
 * Scores for comparing tokenizers by how well text can be <em>modelled</em> through them,
 * normalised per character so tokenizers with different vocabularies are comparable:
 *
 * <ul>
 *   <li>{@link #bigramBitsPerCharacter}: the held-out cross-entropy of a token bigram
 *       language model (Witten-Bell interpolated with an add-one unigram) trained on the
 *       training text, in bits per character. A larger vocabulary shortens sequences but
 *       spreads the counts thinner, so this has a real optimum: the small-model proxy for
 *       "which tokenizer makes a language model's job easiest".</li>
 *   <li>{@link #compressionBitsPerCharacter}: a minimum-description-length proxy: the
 *       unigram entropy of the token stream plus the cost of writing down the merges.</li>
 * </ul>
 */
public final class TokenizerScores {

    private TokenizerScores() {
    }

    /**
     * Held-out bits per character of a token bigram model.
     *
     * @param tokenizer the tokenizer
     * @param train text the bigram model is counted on
     * @param heldOut text it is scored on
     * @return bits per character of {@code heldOut}
     */
    public static double bigramBitsPerCharacter(Tokenizer tokenizer, String train, String heldOut) {
        int[] t = tokenizer.encode(train);
        int v = tokenizer.size();
        int[] uni = new int[v];
        Map<Long, Integer> bi = new HashMap<>();
        int[] followers = new int[v];
        int[] contexts = new int[v];
        for (int i = 0; i < t.length; i++) {
            uni[t[i]]++;
            if (i > 0) {
                long key = (long) t[i - 1] * v + t[i];
                if (bi.merge(key, 1, Integer::sum) == 1) {
                    followers[t[i - 1]]++;
                }
                contexts[t[i - 1]]++;
            }
        }
        int[] h = tokenizer.encode(heldOut);
        double bits = 0;
        for (int i = 0; i < h.length; i++) {
            double pu = (uni[h[i]] + 1.0) / (t.length + v);
            double p = pu;
            if (i > 0) {
                int prev = h[i - 1];
                int c = contexts[prev];
                if (c > 0) {
                    double lambda = c / (double) (c + followers[prev]);
                    int pair = bi.getOrDefault((long) prev * v + h[i], 0);
                    p = lambda * pair / c + (1 - lambda) * pu;
                }
            }
            bits -= Math.log(p) / Math.log(2);
        }
        return bits / Math.max(1, heldOut.length());
    }

    /**
     * Description length per character: unigram entropy of the token stream plus the
     * bits needed to list the merges ({@code 2 log2(size)} each).
     *
     * @param tokenizer the tokenizer
     * @param text the text
     * @return bits per character
     */
    public static double compressionBitsPerCharacter(Tokenizer tokenizer, String text) {
        int[] t = tokenizer.encode(text);
        Map<Integer, Integer> counts = new HashMap<>();
        for (int id : t) {
            counts.merge(id, 1, Integer::sum);
        }
        double data = 0;
        for (int c : counts.values()) {
            data -= c * Math.log(c / (double) t.length) / Math.log(2);
        }
        double merges = Math.max(0, tokenizer.size() - 256);
        double model = merges * 2 * Math.log(Math.max(2, tokenizer.size())) / Math.log(2);
        return (data + model) / Math.max(1, text.length());
    }
}
