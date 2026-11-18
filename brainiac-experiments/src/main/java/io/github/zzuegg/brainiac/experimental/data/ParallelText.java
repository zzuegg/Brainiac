package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A synthetic parallel corpus with gold word alignments: {@link ToyTreebank} sentences (the
 * source language, English-like SVO) rendered into a target language by its constituency
 * tree with
 *
 * <ul>
 *   <li>a lexicon: each source word has one or two target translations (synonyms drawn at
 *       random);
 *   <li>reordering: adjectives after nouns inside noun phrases (local), postpositions
 *       instead of prepositions, and verb-final verb phrases (long-distance: the verb moves
 *       past its object and prepositional phrases);
 *   <li>fertility: transitive, placing and saying verbs become a stem plus a tense marker,
 *       both aligned to the verb;
 *   <li>null words: the articles "the" and "a" have no translation, and the target inserts a
 *       subject particle after every subject and an object particle after every object, which
 *       have no source.
 * </ul>
 *
 * <p>Gold alignments are the sure links {@code {source position, target position}}.
 */
public final class ParallelText {

    /** Target tokens beyond the translated words: two tense markers and two particles. */
    public static final int EXTRA = 4;

    private ParallelText() {
    }

    /**
     * A sentence pair.
     *
     * @param source source word ids
     * @param target target word ids
     * @param alignment sure links {@code {i, j}}
     */
    public record Pair(int[] source, int[] target, List<int[]> alignment) {
    }

    /** @return the source vocabulary size */
    public static int sourceVocabulary() {
        return ToyTreebank.VOCABULARY.size();
    }

    /** @return the target vocabulary size */
    public static int targetVocabulary() {
        return 2 * ToyTreebank.VOCABULARY.size() + EXTRA;
    }

    /**
     * Generates sentence pairs.
     *
     * @param n how many
     * @param minLength the shortest source
     * @param maxLength the longest source
     * @param seed the seed (the lexicon is fixed by it as well)
     * @return the pairs
     */
    public static List<Pair> generate(int n, int minLength, int maxLength, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int v = sourceVocabulary();
        int[] perm = new int[2 * v];
        for (int i = 0; i < perm.length; i++) {
            perm[i] = i;
        }
        SplittableRandom lexR = new SplittableRandom(12345);
        for (int i = perm.length - 1; i > 0; i--) {
            int j = lexR.nextInt(i + 1);
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        boolean[] synonyms = new boolean[v];
        for (int i = 0; i < v; i++) {
            synonyms[i] = lexR.nextDouble() < 0.3;
        }
        List<Pair> out = new ArrayList<>();
        for (ToyTreebank.Sentence s : ToyTreebank.sample(n, minLength, maxLength, seed)) {
            List<int[]> tokens = new ArrayList<>();       // {target id, source position or -1}
            render(s.tree(), s, tokens, perm, synonyms, r);
            int[] tgt = new int[tokens.size()];
            List<int[]> links = new ArrayList<>();
            for (int j = 0; j < tgt.length; j++) {
                tgt[j] = tokens.get(j)[0];
                if (tokens.get(j)[1] >= 0) {
                    links.add(new int[] {tokens.get(j)[1], j});
                }
            }
            out.add(new Pair(s.words().clone(), tgt, links));
        }
        return out;
    }

    private static boolean isVerb(String tag) {
        return tag.startsWith("V");
    }

    private static void render(ToyTreebank.Tree t, ToyTreebank.Sentence s, List<int[]> out, int[] perm,
            boolean[] synonyms, SplittableRandom r) {
        int v = sourceVocabulary();
        if (t.preterminal()) {
            int w = s.words()[t.word()];
            String word = ToyTreebank.VOCABULARY.get(w);
            if (word.equals("the") || word.equals("a")) {
                return;
            }
            int id = perm[synonyms[w] && r.nextBoolean() ? v + w : w];
            out.add(new int[] {id, t.word()});
            if (t.label().equals("Vt") || t.label().equals("Vp") || t.label().equals("Vs")) {
                out.add(new int[] {2 * v + (t.label().equals("Vs") ? 1 : 0), t.word()});
            }
            return;
        }
        List<ToyTreebank.Tree> kids = new ArrayList<>(t.children());
        switch (t.label()) {
            case "VP" -> {
                if (!kids.isEmpty() && kids.get(0).preterminal() && isVerb(kids.get(0).label())) {
                    kids.add(kids.remove(0));
                }
            }
            case "PP" -> {
                if (!kids.isEmpty() && kids.get(0).preterminal() && kids.get(0).label().equals("P")) {
                    kids.add(kids.remove(0));
                }
            }
            case "NP" -> {
                for (int i = 0; i + 1 < kids.size(); i++) {
                    if (kids.get(i).preterminal() && kids.get(i).label().equals("Adj")
                            && kids.get(i + 1).preterminal() && kids.get(i + 1).label().equals("N")) {
                        ToyTreebank.Tree a = kids.get(i);
                        kids.set(i, kids.get(i + 1));
                        kids.set(i + 1, a);
                        i++;
                    }
                }
            }
            default -> {
            }
        }
        for (int i = 0; i < kids.size(); i++) {
            ToyTreebank.Tree k = kids.get(i);
            render(k, s, out, perm, synonyms, r);
            if (t.label().equals("S") && k.label().equals("NP") && i == 0) {
                out.add(new int[] {2 * v + 2, -1});
            }
            if (t.label().equals("VP") && k.label().equals("NP")) {
                out.add(new int[] {2 * v + 3, -1});
            }
        }
    }
}
