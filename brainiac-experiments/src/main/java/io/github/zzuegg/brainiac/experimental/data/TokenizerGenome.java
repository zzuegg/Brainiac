package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.BytePairTokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A byte-pair tokenizer as a genome: an ordered list of merges (each joins two byte
 * strings) and a vocabulary-size gene saying how many of them are used. It decodes into a
 * core {@link BytePairTokenizer}, so an evolved tokenizer is a drop-in
 * {@code Tokenizer}. Merges are stored as strings rather than ids, so removing or
 * reordering one never invalidates the others: a merge whose parts cannot be formed by the
 * merges before it is simply skipped.
 *
 * <p>Mutations: grow or shrink the vocabulary gene, add the currently most frequent pair
 * (guided), add a random frequent pair, drop a merge, swap two neighbouring merges.
 * Crossover keeps a prefix of one parent and appends the other's merges.
 *
 * @param merges the merges in priority order, each {@code {left, right}} as ISO-8859-1
 *     strings holding raw bytes
 * @param active how many of the merges are used
 */
public record TokenizerGenome(List<String[]> merges, int active) {

    /** GPT-2 style pre-tokenization, as the core byte-pair tokenizer uses. */
    private static final Pattern CHUNKS = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    /** Copies the merges and clamps the vocabulary gene. */
    public TokenizerGenome {
        List<String[]> copy = new ArrayList<>();
        for (String[] m : merges) {
            copy.add(new String[] {m[0], m[1]});
        }
        merges = List.copyOf(copy);
        active = Math.max(0, Math.min(active, merges.size()));
    }

    /**
     * The genome of a trained byte-pair tokenizer.
     *
     * @param bpe the tokenizer
     * @return its merges, all active
     */
    public static TokenizerGenome of(BytePairTokenizer bpe) {
        List<String[]> m = new ArrayList<>();
        for (int[] pair : bpe.merges()) {
            m.add(new String[] {latin(bpe.tokenBytes(pair[0])), latin(bpe.tokenBytes(pair[1]))});
        }
        return new TokenizerGenome(m, m.size());
    }

    private static String latin(byte[] b) {
        return new String(b, StandardCharsets.ISO_8859_1);
    }

    /** @return the vocabulary size of the decoded tokenizer, at most {@code 256 + active} */
    public int vocabularySize() {
        return toTokenizer().size();
    }

    /**
     * Decodes the genome: the first {@code active} merges that can be formed.
     *
     * @return the tokenizer
     */
    public BytePairTokenizer toTokenizer() {
        Map<String, Integer> ids = new HashMap<>();
        for (int b = 0; b < 256; b++) {
            ids.put(String.valueOf((char) b), b);
        }
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < active; i++) {
            String[] m = merges.get(i);
            Integer a = ids.get(m[0]);
            Integer b = ids.get(m[1]);
            String joined = m[0] + m[1];
            if (a == null || b == null || ids.containsKey(joined)) {
                continue;
            }
            ids.put(joined, 256 + out.size());
            out.add(new int[] {a, b});
        }
        return BytePairTokenizer.of(out.toArray(int[][]::new));
    }

    /**
     * Counts neighbouring token pairs of a text under this genome's tokenizer.
     *
     * @param text the text
     * @return pair {@code {left, right}} strings with their counts
     */
    public Map<List<String>, Integer> pairCounts(String text) {
        BytePairTokenizer tok = toTokenizer();
        Map<List<String>, Integer> counts = new HashMap<>();
        Map<String, Integer> chunks = new HashMap<>();
        Matcher m = CHUNKS.matcher(text);
        while (m.find()) {
            chunks.merge(m.group(), 1, Integer::sum);
        }
        for (Map.Entry<String, Integer> e : chunks.entrySet()) {
            int[] ids = tok.encode(e.getKey());
            for (int i = 0; i + 1 < ids.length; i++) {
                counts.merge(List.of(latin(tok.tokenBytes(ids[i])), latin(tok.tokenBytes(ids[i + 1]))),
                        e.getValue(), Integer::sum);
            }
        }
        return counts;
    }

    /**
     * A copy with one random change.
     *
     * @param sample text that guided mutations count pairs in
     * @param random the random source
     * @return the mutant
     */
    public TokenizerGenome mutate(String sample, SplittableRandom random) {
        List<String[]> m = new ArrayList<>(merges);
        int act = active;
        switch (random.nextInt(6)) {
            case 0 -> act = Math.min(m.size(), act + 1 + random.nextInt(Math.max(1, m.size() / 8 + 1)));
            case 1 -> act = Math.max(0, act - 1 - random.nextInt(Math.max(1, act / 8 + 1)));
            case 2, 3 -> {
                Map<List<String>, Integer> counts = pairCounts(sample);
                if (!counts.isEmpty()) {
                    List<Map.Entry<List<String>, Integer>> top = new ArrayList<>(counts.entrySet());
                    top.sort((x, y) -> y.getValue() - x.getValue());
                    int pick = random.nextInt(3) == 0 ? random.nextInt(Math.min(10, top.size())) : 0;
                    List<String> p = top.get(pick).getKey();
                    m.add(act, new String[] {p.get(0), p.get(1)});
                    act++;
                }
            }
            case 4 -> {
                if (act > 0) {
                    m.remove(random.nextInt(act));
                    act--;
                }
            }
            default -> {
                if (act > 1) {
                    int i = random.nextInt(act - 1);
                    String[] t = m.get(i);
                    m.set(i, m.get(i + 1));
                    m.set(i + 1, t);
                }
            }
        }
        return new TokenizerGenome(m, act);
    }

    /**
     * Crossover: a prefix of this genome's active merges, then the other's merges not
     * already present; the vocabulary gene from either parent.
     *
     * @param other the other parent
     * @param random the random source
     * @return the child
     */
    public TokenizerGenome cross(TokenizerGenome other, SplittableRandom random) {
        int cut = active == 0 ? 0 : random.nextInt(active + 1);
        List<String[]> m = new ArrayList<>(merges.subList(0, cut));
        Set<String> have = new HashSet<>();
        for (String[] x : m) {
            have.add(x[0] + "\u0000" + x[1]);
        }
        for (String[] x : other.merges) {
            if (have.add(x[0] + "\u0000" + x[1])) {
                m.add(x);
            }
        }
        return new TokenizerGenome(m, random.nextBoolean() ? active : other.active);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof TokenizerGenome g) || g.active != active) {
            return false;
        }
        for (int i = 0; i < active; i++) {
            if (!merges.get(i)[0].equals(g.merges.get(i)[0]) || !merges.get(i)[1].equals(g.merges.get(i)[1])) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        int h = active;
        for (int i = 0; i < active; i++) {
            h = 31 * h + merges.get(i)[0].hashCode() * 17 + merges.get(i)[1].hashCode();
        }
        return h;
    }

    @Override
    public String toString() {
        return "TokenizerGenome(" + active + " of " + merges.size() + " merges)";
    }
}
