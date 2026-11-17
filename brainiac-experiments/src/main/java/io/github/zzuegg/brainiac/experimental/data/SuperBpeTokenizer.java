package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.BytePairTokenizer;
import io.github.zzuegg.brainiac.data.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * SuperBPE (Liu et al. 2025, "SuperBPE: Space Travel for Language Models",
 * <a href="https://arxiv.org/abs/2503.13423">arXiv:2503.13423</a>): byte-pair encoding in
 * two stages. Up to a transition size it learns ordinary subword merges inside words (the
 * core {@link BytePairTokenizer}); after that the word boundary is lifted and it learns
 * <em>superword</em> merges across spaces ("of the", "in order to"), so frequent multi-word
 * expressions become single tokens. The same vocabulary size encodes text in noticeably
 * fewer tokens, which the paper turns into better models at lower inference cost.
 *
 * <p>Superword units stop at line breaks and sentence punctuation. Ids, merges and
 * decoding follow the core tokenizer (merge {@code i} makes token {@code 256 + i}), so
 * {@link #save} writes a core BPE file that decodes every id (it encodes without the
 * superword stage, as the core has no hook for it).
 */
public final class SuperBpeTokenizer implements Tokenizer {

    private final BytePairTokenizer all;
    private final BytePairTokenizer subword;
    private final int transition;
    private final int[][] merges;
    private final Map<Long, Integer> rank = new HashMap<>();

    private SuperBpeTokenizer(int[][] merges, int transition) {
        this.merges = merges;
        this.transition = transition;
        this.all = BytePairTokenizer.of(merges);
        this.subword = BytePairTokenizer.of(Arrays.copyOf(merges, transition));
        for (int i = transition; i < merges.length; i++) {
            rank.put(pair(merges[i][0], merges[i][1]), i);
        }
    }

    private static long pair(int a, int b) {
        return ((long) a << 32) | (b & 0xffffffffL);
    }

    /**
     * Learns a SuperBPE vocabulary.
     *
     * @param text the training text
     * @param transitionSize the vocabulary size (bytes included) where superword merges start
     * @param vocabularySize the final vocabulary size
     * @return the tokenizer
     */
    public static SuperBpeTokenizer train(String text, int transitionSize, int vocabularySize) {
        BytePairTokenizer stage1 = BytePairTokenizer.train(text, transitionSize);
        int[][] base = stage1.merges();
        List<int[]> merges = new ArrayList<>(Arrays.asList(base));
        Map<String, Integer> unitCounts = new LinkedHashMap<>();
        for (String u : units(text)) {
            unitCounts.merge(u, 1, Integer::sum);
        }
        List<int[]> seqs = new ArrayList<>();
        List<Integer> freq = new ArrayList<>();
        for (Map.Entry<String, Integer> e : unitCounts.entrySet()) {
            seqs.add(stage1.encode(e.getKey()));
            freq.add(e.getValue());
        }
        int next = 256 + merges.size();
        while (next < vocabularySize) {
            Map<Long, Long> counts = new HashMap<>();
            for (int s = 0; s < seqs.size(); s++) {
                int[] q = seqs.get(s);
                for (int i = 0; i + 1 < q.length; i++) {
                    counts.merge(pair(q[i], q[i + 1]), (long) freq.get(s), Long::sum);
                }
            }
            long best = -1;
            long bestCount = 1;
            for (Map.Entry<Long, Long> e : counts.entrySet()) {
                if (e.getValue() > bestCount || (e.getValue() == bestCount && best >= 0 && e.getKey() < best)) {
                    best = e.getKey();
                    bestCount = e.getValue();
                }
            }
            if (best < 0) {
                break;
            }
            int a = (int) (best >>> 32);
            int b = (int) best;
            merges.add(new int[] {a, b});
            for (int s = 0; s < seqs.size(); s++) {
                seqs.set(s, merge(seqs.get(s), a, b, next));
            }
            next++;
        }
        return new SuperBpeTokenizer(merges.toArray(new int[0][]), base.length);
    }

    /** Spans that superword merges stay within: split after line breaks and sentence ends. */
    static List<String> units(String text) {
        List<String> out = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '\n' || c == '.' || c == '!' || c == '?') {
                out.add(text.substring(start, i + 1));
                start = i + 1;
            }
        }
        if (start < text.length()) {
            out.add(text.substring(start));
        }
        return out;
    }

    private static int[] merge(int[] q, int a, int b, int id) {
        int[] out = new int[q.length];
        int n = 0;
        for (int i = 0; i < q.length; i++) {
            if (i + 1 < q.length && q[i] == a && q[i + 1] == b) {
                out[n++] = id;
                i++;
            } else {
                out[n++] = q[i];
            }
        }
        return Arrays.copyOf(out, n);
    }

    /** @return the vocabulary size where superword merges begin */
    public int transitionSize() {
        return 256 + transition;
    }

    /** @return the plain BPE tokenizer of the first stage */
    public BytePairTokenizer subwordTokenizer() {
        return subword;
    }

    @Override
    public int[] encode(String text) {
        List<Integer> out = new ArrayList<>();
        for (String u : units(text)) {
            int[] q = subword.encode(u);
            while (q.length > 1) {
                int best = Integer.MAX_VALUE;
                for (int i = 0; i + 1 < q.length; i++) {
                    Integer r = rank.get(pair(q[i], q[i + 1]));
                    if (r != null && r < best) {
                        best = r;
                    }
                }
                if (best == Integer.MAX_VALUE) {
                    break;
                }
                q = merge(q, merges[best][0], merges[best][1], 256 + best);
            }
            for (int id : q) {
                out.add(id);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public String decode(int... ids) {
        return all.decode(ids);
    }

    @Override
    public int size() {
        return all.size();
    }

    @Override
    public String kind() {
        return "bpe";
    }

    @Override
    public void save(Path path) throws IOException {
        all.save(path);
    }

    /**
     * @param text a text
     * @return how many tokens span more than one word (contain an inner space)
     */
    public int superwordTokens(String text) {
        int n = 0;
        for (int id : encode(text)) {
            String s = decode(id);
            if (s.strip().contains(" ")) {
                n++;
            }
        }
        return n;
    }

    @Override
    public String toString() {
        return "SuperBpeTokenizer(" + size() + " tokens, superwords from " + transitionSize() + ")";
    }
}
