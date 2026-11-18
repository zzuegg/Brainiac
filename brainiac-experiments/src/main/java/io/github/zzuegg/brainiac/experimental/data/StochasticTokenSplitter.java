package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.Tokenizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

/**
 * StochasTok: stochastic token splitting for subword understanding (Sims et al., ICLR 2026,
 * <a href="https://arxiv.org/abs/2506.01687">arXiv:2506.01687</a>). During training, some
 * tokens of an already tokenized sequence are replaced by a pair of shorter in-vocabulary tokens
 * that spell the same string, so the model sees how its tokens are composed while the
 * vocabulary and the tokenizer stay unchanged (it can be switched on for pretraining or only
 * for post-training). BPE-dropout ({@link BpeDropoutTokenizer}) perturbs the merge process
 * instead and needs the merges; StochasTok works on token ids of any tokenizer.
 *
 * <p>Each of {@code rounds} passes visits every token and, with probability {@code p}, replaces
 * it by one of its splits drawn uniformly, so tokens can be split repeatedly down to characters.
 */
public final class StochasticTokenSplitter {

    private final int[][][] splits;
    private final double probability;
    private final int rounds;

    private StochasticTokenSplitter(int[][][] splits, double probability, int rounds) {
        if (probability < 0 || probability > 1 || rounds < 1) {
            throw new IllegalArgumentException("probability must be in [0, 1] and rounds positive");
        }
        this.splits = splits;
        this.probability = probability;
        this.rounds = rounds;
    }

    /**
     * Finds every token's two-token splits from a tokenizer's token texts.
     *
     * @param tokenizer the tokenizer
     * @param probability the chance each visited token is split
     * @param rounds passes over the sequence
     * @return the splitter
     */
    public static StochasticTokenSplitter of(Tokenizer tokenizer, double probability, int rounds) {
        String[] texts = new String[tokenizer.size()];
        for (int t = 0; t < texts.length; t++) {
            texts[t] = tokenizer.decode(t);
        }
        return of(texts, probability, rounds);
    }

    /**
     * Finds every token's two-token splits from token texts.
     *
     * @param texts each token's text by id ({@code null} for none)
     * @param probability the chance each visited token is split
     * @param rounds passes over the sequence
     * @return the splitter
     */
    public static StochasticTokenSplitter of(String[] texts, double probability, int rounds) {
        Map<String, Integer> index = new HashMap<>();
        for (int t = 0; t < texts.length; t++) {
            if (texts[t] != null && !texts[t].isEmpty() && !texts[t].contains("�")) {
                index.putIfAbsent(texts[t], t);
            }
        }
        int[][][] splits = new int[texts.length][][];
        for (int t = 0; t < texts.length; t++) {
            List<int[]> ways = new ArrayList<>();
            String s = texts[t];
            for (int cut = 1; s != null && cut < s.length(); cut++) {
                Integer a = index.get(s.substring(0, cut));
                Integer b = index.get(s.substring(cut));
                if (a != null && b != null) {
                    ways.add(new int[] {a, b});
                }
            }
            splits[t] = ways.toArray(int[][]::new);
        }
        return new StochasticTokenSplitter(splits, probability, rounds);
    }

    /**
     * @param token a token
     * @return its two-token splits
     */
    public int[][] splits(int token) {
        return splits[token].clone();
    }

    /**
     * Splits a sequence at random.
     *
     * @param ids the tokens
     * @param random the random source
     * @return the split sequence, which decodes to the same text
     */
    public int[] split(int[] ids, RandomGenerator random) {
        int[] cur = ids;
        for (int r = 0; r < rounds; r++) {
            List<Integer> out = new ArrayList<>(cur.length + 4);
            for (int t : cur) {
                if (splits[t].length > 0 && random.nextDouble() < probability) {
                    int[] s = splits[t][random.nextInt(splits[t].length)];
                    out.add(s[0]);
                    out.add(s[1]);
                } else {
                    out.add(t);
                }
            }
            cur = out.stream().mapToInt(Integer::intValue).toArray();
        }
        return cur;
    }

    @Override
    public String toString() {
        return "StochasTok(p " + probability + ", rounds " + rounds + ")";
    }
}
