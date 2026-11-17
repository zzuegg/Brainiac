package io.github.zzuegg.brainiac.experiments.language;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * A synthetic language with morphology, agreement and planted analogies, shared by the static
 * and contextual word-representation experiments (wave 6 items 2.14 and 2.15). Every sentence
 * has five tokens:
 *
 * <pre>
 *   DET  SUBJECT  VERB  DET  OBJECT          e.g.  la dakas vedin lo mirok
 * </pre>
 *
 * <ul>
 *   <li>Nouns are a stem (12 stems) plus a gender suffix ({@code o} / {@code a}) and an
 *       optional plural {@code s}: {@code dako, daka, dakos, dakas}. Stems 0-5 are animate
 *       (subjects and objects of animate verbs), stems 6-11 inanimate (objects only).</li>
 *   <li>Determiners agree with their noun ({@code lo, la, los, las}); verbs agree with the
 *       subject's number ({@code vedi} / {@code vedin}) and select the object's animacy.</li>
 *   <li>Each stem has a partner: an animate subject {@code s} takes object stem
 *       {@code (s + 3) mod 6} or {@code 6 + s} in 60% of sentences, so stems differ in context
 *       too.</li>
 *   <li>Gender and number are therefore distributional: {@code dako - daka + miro ~ mira} is a
 *       planted analogy, and so is {@code dako - dakos + miro ~ miros}.</li>
 *   <li>The plural feminine forms of stems 2, 5, 8 and 11 never occur: they are out of
 *       vocabulary for word-level models, but their character n-grams are not.</li>
 *   <li>Three tokens ({@code nim, vash, tral}) are both singular verbs and masculine singular
 *       inanimate nouns without suffix; only the context tells which, which is what the
 *       contextual tagging task (2.15) measures.</li>
 * </ul>
 */
public final class SyntheticGrammar {

    /** Tag of determiners. */
    public static final int TAG_DET = 0;
    /** Tag of nouns. */
    public static final int TAG_NOUN = 1;
    /** Tag of verbs. */
    public static final int TAG_VERB = 2;
    /** Number of tags. */
    public static final int TAGS = 3;
    /** Tokens per sentence. */
    public static final int LENGTH = 5;

    private static final String[] STEMS = {"dak", "mir", "tov", "bel", "kus", "ren", "zol", "fim", "gar", "lup", "sen",
        "vor"};
    private static final String[] ANIMATE_VERBS = {"ved", "sal", "kor"};
    private static final String[] INANIMATE_VERBS = {"bur", "tem", "pas"};
    private static final String[] AMBIGUOUS = {"nim", "vash", "tral"};
    private static final int[] HELD_OUT = {2, 5, 8, 11};
    private static final double PARTNER = 0.6;

    private final List<String> words = new ArrayList<>();
    private final Map<String, Integer> ids = new HashMap<>();

    /** Creates the vocabulary. */
    public SyntheticGrammar() {
        for (String d : new String[] {"lo", "la", "los", "las"}) {
            add(d);
        }
        for (int s = 0; s < STEMS.length; s++) {
            for (int f = 0; f < 4; f++) {
                if (!(f == 3 && isHeldOut(s))) {
                    add(noun(s, f));
                }
            }
        }
        for (String v : ANIMATE_VERBS) {
            add(v + "i");
            add(v + "in");
        }
        for (String v : INANIMATE_VERBS) {
            add(v + "i");
            add(v + "in");
        }
        for (String a : AMBIGUOUS) {
            add(a);
            add(a + "en");
        }
    }

    private void add(String w) {
        ids.put(w, words.size());
        words.add(w);
    }

    private static boolean isHeldOut(int stem) {
        for (int h : HELD_OUT) {
            if (h == stem) {
                return true;
            }
        }
        return false;
    }

    /**
     * A noun form.
     *
     * @param stem the stem, 0 to 11
     * @param form {@code 2 * plural + feminine}: 0 masculine singular, 1 feminine singular,
     *     2 masculine plural, 3 feminine plural
     * @return the word
     */
    public static String noun(int stem, int form) {
        return STEMS[stem] + ((form & 1) == 0 ? "o" : "a") + (form >= 2 ? "s" : "");
    }

    /** @return the number of noun stems */
    public static int stems() {
        return STEMS.length;
    }

    /** @return the stems whose feminine plural never occurs */
    public static int[] heldOutStems() {
        return HELD_OUT.clone();
    }

    /** @return whether a stem is animate */
    public static boolean animate(int stem) {
        return stem < 6;
    }

    /** @return the vocabulary, indexed by id */
    public List<String> words() {
        return List.copyOf(words);
    }

    /** @return the vocabulary size */
    public int size() {
        return words.size();
    }

    /**
     * @param word a word
     * @return its id, or -1 if out of vocabulary
     */
    public int id(String word) {
        return ids.getOrDefault(word, -1);
    }

    /** @return the ambiguous noun/verb words */
    public static List<String> ambiguous() {
        return List.of(AMBIGUOUS);
    }

    private int randomForm(int stem, SplittableRandom r) {
        int f;
        do {
            f = r.nextInt(4);
        } while (f == 3 && isHeldOut(stem));
        return f;
    }

    /**
     * Samples sentences.
     *
     * @param n the number of sentences
     * @param r the random source
     * @return {@code {ids [n][5], tags [n][5]}}
     */
    public int[][][] sentences(int n, SplittableRandom r) {
        int[][] tokens = new int[n][LENGTH];
        int[][] tags = new int[n][LENGTH];
        String[] dets = {"lo", "la", "los", "las"};
        for (int i = 0; i < n; i++) {
            int subject = r.nextInt(6);
            int sf = randomForm(subject, r);
            boolean plural = sf >= 2;
            int kind = r.nextInt(5);
            String verb;
            int object;
            int of;
            String objectWord;
            if (kind < 2) {
                verb = ANIMATE_VERBS[r.nextInt(3)];
                object = r.nextDouble() < PARTNER ? (subject + 3) % 6 : r.nextInt(6);
                of = randomForm(object, r);
                objectWord = noun(object, of);
                verb = verb + (plural ? "in" : "i");
            } else if (kind < 4) {
                verb = INANIMATE_VERBS[r.nextInt(3)];
                object = r.nextDouble() < PARTNER ? 6 + subject : 6 + r.nextInt(6);
                of = randomForm(object, r);
                objectWord = noun(object, of);
                verb = verb + (plural ? "in" : "i");
            } else {
                // an ambiguous word as the verb, and another (or a regular noun) as the object
                verb = AMBIGUOUS[r.nextInt(3)] + (plural ? "en" : "");
                if (r.nextBoolean()) {
                    of = 0;
                    objectWord = AMBIGUOUS[r.nextInt(3)];
                } else {
                    object = 6 + r.nextInt(6);
                    of = randomForm(object, r);
                    objectWord = noun(object, of);
                }
            }
            tokens[i][0] = ids.get(dets[sf]);
            tokens[i][1] = ids.get(noun(subject, sf));
            tokens[i][2] = ids.get(verb);
            tokens[i][3] = ids.get(dets[of]);
            tokens[i][4] = ids.get(objectWord);
            tags[i] = new int[] {TAG_DET, TAG_NOUN, TAG_VERB, TAG_DET, TAG_NOUN};
        }
        return new int[][][] {tokens, tags};
    }
}
