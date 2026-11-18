package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * A small synthetic retrieval corpus with planted structure, for comparing lexical, dense,
 * late-interaction and learned-sparse retrievers and rerankers.
 *
 * <p>Every document belongs to a topic and holds a rare <em>key</em> term (a name or id that
 * occurs in that document only), three <em>aspect</em> words from its topic's word list and
 * common filler words. Documents come in sibling groups that share two of their three aspect
 * words, so lexical overlap alone is often ambiguous (distractors). Every aspect word has a
 * <em>paraphrase</em> twin that never occurs in a document. Queries come in three kinds:
 * <ul>
 *   <li>{@link Kind#LEXICAL}: the key term plus one aspect word; exact term matching wins;</li>
 *   <li>{@link Kind#PARAPHRASE}: the paraphrase twins of all three aspect words, with no word in
 *       common with the document; only a model that learned the synonyms finds it;</li>
 *   <li>{@link Kind#MULTI_ASPECT}: two aspect words that only the target contains together,
 *       while many documents contain one of them; both terms must be matched.</li>
 * </ul>
 * Queries of the first {@code trainShare} of the documents are for training; the others are for
 * evaluation, over the whole corpus. {@link #grade} gives graded relevance labels for learning
 * to rank.
 */
public final class SyntheticCorpus {

    /** The kind of a query. */
    public enum Kind {
        /** Key term plus one aspect word. */
        LEXICAL,
        /** Paraphrases of all aspect words. */
        PARAPHRASE,
        /** Two aspect words that only the target holds together. */
        MULTI_ASPECT
    }

    /**
     * A query.
     *
     * @param tokens the token ids, padded with 0 to {@link #QUERY_LENGTH}
     * @param target the relevant document
     * @param kind the query kind
     * @param concepts the aspect words the query refers to (original, not paraphrased, ids)
     */
    public record Query(int[] tokens, int target, Kind kind, int[] concepts) {
    }

    /** The token length of every document. */
    public static final int DOC_LENGTH = 8;
    /** The padded token length of every query. */
    public static final int QUERY_LENGTH = 5;
    /** The padding token. */
    public static final int PAD = 0;

    private final int vocabulary;
    private final int[][] documents;
    private final int[] topic;
    private final int[][] aspects;
    private final int[] paraphrase;
    private final List<Query> train;
    private final List<Query> test;

    private SyntheticCorpus(int vocabulary, int[][] documents, int[] topic, int[][] aspects, int[] paraphrase,
            List<Query> train, List<Query> test) {
        this.vocabulary = vocabulary;
        this.documents = documents;
        this.topic = topic;
        this.aspects = aspects;
        this.paraphrase = paraphrase;
        this.train = train;
        this.test = test;
    }

    /**
     * Generates a corpus.
     *
     * @param topics the number of topics
     * @param wordsPerTopic aspect words per topic, at least 4
     * @param docsPerTopic documents per topic, a multiple of 3 (sibling groups of three)
     * @param trainShare the share of documents whose queries are for training
     * @param seed the seed
     * @return the corpus
     */
    public static SyntheticCorpus generate(int topics, int wordsPerTopic, int docsPerTopic, double trainShare,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int fillers = 12;
        int firstWord = 1 + fillers;
        int words = topics * wordsPerTopic;
        int firstParaphrase = firstWord + words;
        int docs = topics * docsPerTopic;
        int firstKey = firstParaphrase + words;
        int vocab = firstKey + docs;
        int[] paraphrase = new int[vocab];
        for (int w = 0; w < vocab; w++) {
            paraphrase[w] = w;
        }
        for (int w = 0; w < words; w++) {
            paraphrase[firstWord + w] = firstParaphrase + w;
        }
        int[][] documents = new int[docs][];
        int[] topicOf = new int[docs];
        int[][] aspects = new int[docs][];
        Set<String> used = new HashSet<>();
        int d = 0;
        for (int t = 0; t < topics; t++) {
            int base = firstWord + t * wordsPerTopic;
            int made = 0;
            while (made < docsPerTopic) {
                int[] group = pick(r, wordsPerTopic, 3);
                List<int[]> siblings = new ArrayList<>();
                siblings.add(group);
                for (int s = 0; s < 2; s++) {
                    int[] sib = group.clone();
                    int pos = r.nextInt(3);
                    int w;
                    do {
                        w = r.nextInt(wordsPerTopic);
                    } while (contains(sib, w));
                    sib[pos] = w;
                    siblings.add(sib);
                }
                boolean fresh = true;
                for (int[] s : siblings) {
                    fresh &= !used.contains(key(s));
                }
                if (!fresh || new HashSet<>(siblings.stream().map(SyntheticCorpus::key).toList()).size() < 3) {
                    continue;
                }
                for (int[] s : siblings) {
                    used.add(key(s));
                    int[] asp = new int[3];
                    for (int i = 0; i < 3; i++) {
                        asp[i] = base + s[i];
                    }
                    Arrays.sort(asp);
                    int[] doc = new int[DOC_LENGTH];
                    doc[0] = firstKey + d;
                    System.arraycopy(asp, 0, doc, 1, 3);
                    for (int i = 4; i < DOC_LENGTH; i++) {
                        doc[i] = 1 + r.nextInt(fillers);
                    }
                    shuffle(doc, r);
                    documents[d] = doc;
                    topicOf[d] = t;
                    aspects[d] = asp;
                    d++;
                }
                made += 3;
            }
        }
        List<Query> train = new ArrayList<>();
        List<Query> test = new ArrayList<>();
        int trainDocs = (int) Math.round(docs * trainShare);
        int[] order = new int[docs];
        for (int i = 0; i < docs; i++) {
            order[i] = i;
        }
        shuffle(order, r);
        for (int k = 0; k < docs; k++) {
            int doc = order[k];
            List<Query> into = k < trainDocs ? train : test;
            int[] asp = aspects[doc];
            int one = asp[r.nextInt(3)];
            into.add(new Query(pad(new int[] {firstKey + doc, one, 1 + r.nextInt(fillers)}, r), doc, Kind.LEXICAL,
                    new int[] {one}));
            into.add(new Query(pad(new int[] {paraphrase[asp[0]], paraphrase[asp[1]], paraphrase[asp[2]],
                    1 + r.nextInt(fillers)}, r), doc, Kind.PARAPHRASE, asp.clone()));
            for (int i = 0; i < 3; i++) {
                int a = asp[i];
                int b = asp[(i + 1) % 3];
                int holders = 0;
                for (int o = 0; o < docs; o++) {
                    holders += contains(aspects[o], a) && contains(aspects[o], b) ? 1 : 0;
                }
                if (holders == 1) {
                    into.add(new Query(pad(new int[] {a, b, 1 + r.nextInt(fillers)}, r), doc, Kind.MULTI_ASPECT,
                            new int[] {a, b}));
                    break;
                }
            }
        }
        return new SyntheticCorpus(vocab, documents, topicOf, aspects, paraphrase, train, test);
    }

    private static int[] pick(SplittableRandom r, int n, int k) {
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            int w;
            do {
                w = r.nextInt(n);
            } while (contains(Arrays.copyOf(out, i), w));
            out[i] = w;
        }
        return out;
    }

    private static String key(int[] s) {
        int[] c = s.clone();
        Arrays.sort(c);
        return Arrays.toString(c);
    }

    private static boolean contains(int[] a, int v) {
        for (int x : a) {
            if (x == v) {
                return true;
            }
        }
        return false;
    }

    private static void shuffle(int[] a, SplittableRandom r) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    private static int[] pad(int[] tokens, SplittableRandom r) {
        shuffle(tokens, r);
        return Arrays.copyOf(tokens, QUERY_LENGTH);
    }

    /** @return the vocabulary size, including the padding token 0 */
    public int vocabulary() {
        return vocabulary;
    }

    /** @return the number of documents */
    public int size() {
        return documents.length;
    }

    /**
     * @param doc a document
     * @return its {@link #DOC_LENGTH} token ids
     */
    public int[] document(int doc) {
        return documents[doc];
    }

    /** @return every document's tokens, {@code [docs][DOC_LENGTH]} */
    public int[][] documents() {
        return documents.clone();
    }

    /**
     * @param doc a document
     * @return its topic
     */
    public int topic(int doc) {
        return topic[doc];
    }

    /**
     * @param word a token id
     * @return its paraphrase twin, or the word itself when it has none
     */
    public int paraphrase(int word) {
        return paraphrase[word];
    }

    /** @return the training queries */
    public List<Query> trainQueries() {
        return train;
    }

    /** @return the evaluation queries */
    public List<Query> testQueries() {
        return test;
    }

    /**
     * @param kind a query kind
     * @return the evaluation queries of that kind
     */
    public List<Query> testQueries(Kind kind) {
        return test.stream().filter(q -> q.kind() == kind).toList();
    }

    /**
     * Graded relevance: 2 for the target, 1 for a document of the target's topic that holds at
     * least half of the query's concepts, 0 otherwise.
     *
     * @param q a query
     * @param doc a document
     * @return the grade
     */
    public int grade(Query q, int doc) {
        if (doc == q.target()) {
            return 2;
        }
        if (topic[doc] != topic[q.target()]) {
            return 0;
        }
        int hit = 0;
        for (int c : q.concepts()) {
            hit += contains(aspects[doc], c) ? 1 : 0;
        }
        return 2 * hit >= q.concepts().length ? 1 : 0;
    }
}
