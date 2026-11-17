package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.BytePairTokenizer;
import io.github.zzuegg.brainiac.data.Tokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * BPE-dropout (Provilkov et al. 2020, <a href="https://arxiv.org/abs/1910.13267">arXiv:1910.13267</a>)
 * over a core {@link BytePairTokenizer}: while encoding, each applicable merge is skipped
 * with probability {@code p}, so the same word is split differently each time it is seen.
 * Training on these varied segmentations regularizes the model and makes it robust to
 * misspellings and rare words; at inference use {@code p = 0}, which is exactly the core
 * tokenizer's encoding. It is a drop-in {@link Tokenizer}: same ids, same decoding, and it
 * saves as the plain BPE tokenizer.
 */
public final class BpeDropoutTokenizer implements Tokenizer {

    /** The GPT-2 chunking the core tokenizers use: merges never cross a chunk. */
    static final Pattern CHUNKS = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    private final BytePairTokenizer base;
    private final double dropout;
    private final RandomGenerator random;
    private final int[][] merges;
    private final Map<Long, Integer> rank = new HashMap<>();

    /**
     * @param base the tokenizer whose merges are used
     * @param dropout the probability of skipping each merge, for example 0.1
     * @param seed the seed of the skips
     */
    public BpeDropoutTokenizer(BytePairTokenizer base, double dropout, long seed) {
        this.base = base;
        this.dropout = dropout;
        this.random = new SplittableRandom(seed);
        this.merges = base.merges();
        for (int i = 0; i < merges.length; i++) {
            rank.putIfAbsent(pair(merges[i][0], merges[i][1]), i);
        }
    }

    private static long pair(int a, int b) {
        return ((long) a << 32) | (b & 0xffffffffL);
    }

    /** @return the deterministic tokenizer */
    public BytePairTokenizer base() {
        return base;
    }

    /**
     * Encodes with a given dropout.
     *
     * @param text the text
     * @param p the probability of skipping each merge
     * @param rng the source of the skips
     * @return the ids
     */
    public int[] encode(String text, double p, RandomGenerator rng) {
        List<Integer> out = new ArrayList<>();
        Matcher m = CHUNKS.matcher(text);
        while (m.find()) {
            for (int id : encodeChunk(m.group(), p, rng)) {
                out.add(id);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private int[] encodeChunk(String chunk, double p, RandomGenerator rng) {
        byte[] b = chunk.getBytes(StandardCharsets.UTF_8);
        int[] word = new int[b.length];
        for (int i = 0; i < b.length; i++) {
            word[i] = b[i] & 0xff;
        }
        while (word.length > 1) {
            int best = Integer.MAX_VALUE;
            int at = -1;
            for (int i = 0; i + 1 < word.length; i++) {
                Integer r = rank.get(pair(word[i], word[i + 1]));
                if (r != null && r < best && (p == 0 || rng.nextDouble() >= p)) {
                    best = r;
                    at = i;
                }
            }
            if (at < 0) {
                break;
            }
            int[] next = new int[word.length - 1];
            System.arraycopy(word, 0, next, 0, at);
            next[at] = 256 + best;
            System.arraycopy(word, at + 2, next, at + 1, word.length - at - 2);
            word = next;
        }
        return word;
    }

    @Override
    public int[] encode(String text) {
        return encode(text, dropout, random);
    }

    @Override
    public String decode(int... ids) {
        return base.decode(ids);
    }

    @Override
    public int size() {
        return base.size();
    }

    @Override
    public String kind() {
        return base.kind();
    }

    @Override
    public void save(Path path) throws IOException {
        base.save(path);
    }

    @Override
    public String toString() {
        return "BpeDropoutTokenizer(" + base + ", p " + dropout + ")";
    }
}
