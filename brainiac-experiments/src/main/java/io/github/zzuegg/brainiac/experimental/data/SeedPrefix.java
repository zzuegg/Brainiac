package io.github.zzuegg.brainiac.experimental.data;

import java.util.random.RandomGenerator;

/**
 * Seed-conditioning ("hash-conditioning", Nagarajan et al., "Roll the dice and look before you
 * leap: going beyond the creative limits of next-token prediction",
 * <a href="https://arxiv.org/abs/2504.15266">2504.15266</a>, ICML 2025 outstanding paper): each
 * training sequence is prefixed with a short random string of reserved seed tokens. At test
 * time a fresh random seed is drawn and the model decodes <em>greedily</em>: the randomness
 * that temperature sampling injects token by token, where it tends to break coherence, is
 * injected once, up front, as a latent "plan". The paper finds it matches or beats temperature
 * sampling at producing outputs that are valid, unique and novel, and the NeurIPS 2025 best
 * paper ("Artificial Hivemind") documents the output homogeneity it counters.
 *
 * <p>Seed tokens are the ids {@code [first, first + alphabet)}, appended after the ordinary
 * vocabulary.
 */
public final class SeedPrefix {

    private final int first;
    private final int alphabet;
    private final int length;

    /**
     * @param first the id of the first seed token
     * @param alphabet how many seed tokens there are
     * @param length the seed length
     */
    public SeedPrefix(int first, int alphabet, int length) {
        if (alphabet < 2 || length < 1) {
            throw new IllegalArgumentException("need at least two seed tokens and length >= 1");
        }
        this.first = first;
        this.alphabet = alphabet;
        this.length = length;
    }

    /** @return the vocabulary size including the seed tokens */
    public int vocabularySize() {
        return first + alphabet;
    }

    /** @return the seed length */
    public int length() {
        return length;
    }

    /**
     * @param random the random source
     * @return a fresh random seed
     */
    public int[] seed(RandomGenerator random) {
        int[] s = new int[length];
        for (int i = 0; i < length; i++) {
            s[i] = first + random.nextInt(alphabet);
        }
        return s;
    }

    /**
     * @param tokens a sequence
     * @param random the random source
     * @return the sequence prefixed with a fresh random seed
     */
    public int[] apply(int[] tokens, RandomGenerator random) {
        int[] s = seed(random);
        int[] out = new int[length + tokens.length];
        System.arraycopy(s, 0, out, 0, length);
        System.arraycopy(tokens, 0, out, length, tokens.length);
        return out;
    }
}
