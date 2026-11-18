package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * A toy treebank from a small probabilistic grammar of English-like sentences with nested clauses
 * ({@code said that S}), coordination of sentences, verb phrases and noun phrases, and
 * prepositional-phrase attachment ambiguity resolved by planted lexical preferences:
 * <ul>
 *   <li>{@code with} + an instrument noun (telescope, knife, ...) attaches to the verb;</li>
 *   <li>{@code with} + an attribute noun (hat, beard, ...) attaches to the noun;</li>
 *   <li>a location phrase ({@code on/in/near the table}) attaches to the verb after a placing verb
 *       ({@code put, placed}) and to the noun otherwise.</li>
 * </ul>
 * Each derivation gives the gold constituency tree (labels {@code S, SBAR, NP, VP, PP} over
 * part-of-speech preterminals) and, by head rules, a labelled dependency tree (heads of PPs are
 * their prepositions, of SBARs their verbs, of coordinations their first conjunct).
 */
public final class ToyTreebank {

    /** Part-of-speech tags. */
    public static final List<String> TAGS = List.of("Det", "N", "Adj", "Name", "Vt", "Vi", "Vs", "Vp", "P", "C",
            "CC");
    /** Phrase labels. */
    public static final List<String> PHRASES = List.of("S", "SBAR", "NP", "VP", "PP");
    /** Dependency labels. */
    public static final List<String> RELATIONS = List.of("root", "nsubj", "obj", "det", "amod", "prep", "pobj",
            "ccomp", "mark", "cc", "conj");

    private static final String[] DET = {"the", "a", "every", "this"};
    private static final String[] PERSON = {"man", "woman", "dog", "girl", "boy", "cat", "teacher", "farmer"};
    private static final String[] INSTRUMENT = {"telescope", "knife", "hammer", "spoon", "stick"};
    private static final String[] ATTRIBUTE = {"hat", "beard", "scarf", "coat", "tail"};
    private static final String[] LOCATION = {"table", "shelf", "floor", "box", "bed"};
    private static final String[] ADJ = {"big", "small", "old", "red", "happy"};
    private static final String[] NAME = {"john", "mary", "alice", "bob"};
    private static final String[] VT = {"saw", "hit", "watched", "cut", "touched", "painted"};
    private static final String[] VI = {"slept", "ran", "laughed", "smiled"};
    private static final String[] VS = {"said", "thought", "knew", "believed"};
    private static final String[] VP = {"put", "placed", "left"};
    private static final String[] LOC_P = {"on", "in", "near"};

    /** The vocabulary; word id {@code i} is {@code VOCABULARY.get(i)}. */
    public static final List<String> VOCABULARY;
    private static final Map<String, Integer> IDS = new HashMap<>();

    static {
        List<String> v = new ArrayList<>();
        for (String[] group : new String[][] {DET, PERSON, INSTRUMENT, ATTRIBUTE, LOCATION, ADJ, NAME, VT, VI, VS, VP,
            LOC_P, {"with", "that", "and"}}) {
            for (String w : group) {
                IDS.put(w, v.size());
                v.add(w);
            }
        }
        VOCABULARY = List.copyOf(v);
    }

    /**
     * A constituency tree: a preterminal has a tag label and a word position, a phrase has
     * children and position {@code -1}.
     *
     * @param label the phrase label or tag
     * @param children the children (empty for a preterminal)
     * @param word the 0-based word position of a preterminal, else {@code -1}
     */
    public record Tree(String label, List<Tree> children, int word) {

        /** @return whether this is a preterminal */
        public boolean preterminal() {
            return word >= 0;
        }

        /** @return the first word position covered */
        public int start() {
            return preterminal() ? word : children.get(0).start();
        }

        /** @return one past the last word position covered */
        public int end() {
            return preterminal() ? word + 1 : children.get(children.size() - 1).end();
        }
    }

    /**
     * A sentence with its gold analyses.
     *
     * @param words word ids
     * @param tags tag ids ({@link #TAGS})
     * @param heads 1-based head of each word, 0 for the root
     * @param relations dependency label ids ({@link #RELATIONS})
     * @param spans labelled constituents {@code {phrase id, start, end}} (end exclusive), outermost
     *     label kept where a span carries two
     * @param ppCases 0-based positions of prepositions in ambiguous {@code V NP PP} contexts
     * @param tree the constituency tree
     */
    public record Sentence(int[] words, int[] tags, int[] heads, int[] relations, List<int[]> spans, int[] ppCases,
            Tree tree) {

        /** @return the length */
        public int length() {
            return words.length;
        }

        /** @return the words as text */
        public String text() {
            StringBuilder s = new StringBuilder();
            for (int w : words) {
                s.append(s.isEmpty() ? "" : " ").append(VOCABULARY.get(w));
            }
            return s.toString();
        }
    }

    private ToyTreebank() {
    }

    /**
     * Samples sentences whose length is in {@code [minLength, maxLength]}.
     *
     * @param count the number of sentences
     * @param minLength the shortest
     * @param maxLength the longest
     * @param seed the seed
     * @return the sentences
     */
    public static List<Sentence> sample(int count, int minLength, int maxLength, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Sentence> out = new ArrayList<>();
        while (out.size() < count) {
            Builder b = new Builder(r, maxLength > 20 ? 1.4 : 1.0);
            Node root = b.sentence(0);
            if (b.words.size() >= minLength && b.words.size() <= maxLength) {
                out.add(b.finish(root));
            }
        }
        return out;
    }

    private static final class Node {
        final String label;
        final List<Node> kids = new ArrayList<>();
        final int word;
        int headChild;
        String rel = "root";

        Node(String label, int word) {
            this.label = label;
            this.word = word;
        }

        int head() {
            return word >= 0 ? word : kids.get(headChild).head();
        }
    }

    private static final class Builder {
        final SplittableRandom r;
        final double depthBoost;
        final List<String> words = new ArrayList<>();
        final List<String> tags = new ArrayList<>();
        final List<Node> ppCases = new ArrayList<>();

        Builder(SplittableRandom r, double depthBoost) {
            this.r = r;
            this.depthBoost = depthBoost;
        }

        Node leaf(String tag, String w, String rel) {
            Node n = new Node(tag, words.size());
            n.rel = rel;
            words.add(w);
            tags.add(tag);
            return n;
        }

        String pick(String[] a) {
            return a[r.nextInt(a.length)];
        }

        Node phrase(String label, int head, String rel, Node... kids) {
            Node n = new Node(label, -1);
            n.kids.addAll(List.of(kids));
            n.headChild = head;
            n.rel = rel;
            return n;
        }

        boolean chance(double p) {
            return r.nextDouble() < p;
        }

        Node sentence(int depth) {
            if (depth == 0 && chance(0.06 * depthBoost)) {
                Node a = clause(1, "root");
                Node and = leaf("CC", "and", "cc");
                Node b = clause(1, "conj");
                return phrase("S", 0, "root", a, and, b);
            }
            return clause(depth, "root");
        }

        Node clause(int depth, String rel) {
            Node subj = np(depth, "nsubj", true);
            Node vp = vp(depth);
            return phrase("S", 1, rel, subj, vp);
        }

        Node vp(int depth) {
            Node v = vpCore(depth);
            if (depth < 2 && chance(0.05 * depthBoost)) {
                Node and = leaf("CC", "and", "cc");
                Node w = vpCore(depth + 1);
                w.rel = "conj";
                return phrase("VP", 0, "root", v, and, w);
            }
            return v;
        }

        Node vpCore(int depth) {
            double u = r.nextDouble();
            if (u < 0.13) {
                return phrase("VP", 0, "root", leaf("Vi", pick(VI), "root"));
            }
            if (u < 0.45) {
                Node v = leaf("Vt", pick(VT), "root");
                return phrase("VP", 0, "root", v, object(depth, true));
            }
            if (u < 0.65) {
                Node v = leaf("Vt", pick(VT), "root");
                Node o = object(depth, false);
                Node pp = pp(depth, "with", INSTRUMENT);
                ppCases.add(pp);
                return phrase("VP", 0, "root", v, o, pp);
            }
            if (u < 0.82 || depth >= 2) {
                Node v = leaf("Vp", pick(VP), "root");
                Node o = object(depth, false);
                Node pp = pp(depth, pick(LOC_P), LOCATION);
                ppCases.add(pp);
                return phrase("VP", 0, "root", v, o, pp);
            }
            Node v = leaf("Vs", pick(VS), "root");
            Node that = leaf("C", "that", "mark");
            Node s = clause(depth + 1, "root");
            Node sbar = phrase("SBAR", 1, "ccomp", that, s);
            return phrase("VP", 0, "root", v, sbar);
        }

        /** A direct object; its own attached PP (if any) is an ambiguous case. */
        Node object(int depth, boolean mayAttach) {
            Node base = baseNp(depth, "obj");
            if (mayAttach && chance(0.45)) {
                Node pp = chance(0.5) ? pp(depth, "with", ATTRIBUTE) : pp(depth, pick(LOC_P), LOCATION);
                ppCases.add(pp);
                return phrase("NP", 0, "obj", base, pp);
            }
            return base;
        }

        Node np(int depth, String rel, boolean subject) {
            Node base = baseNp(depth, rel);
            if (chance(0.12)) {
                Node pp = chance(0.5) ? pp(depth, "with", ATTRIBUTE) : pp(depth, pick(LOC_P), LOCATION);
                base = phrase("NP", 0, rel, base, pp);
            }
            if (subject && depth < 2 && chance(0.07 * depthBoost)) {
                Node and = leaf("CC", "and", "cc");
                Node other = baseNp(depth + 1, "conj");
                base = phrase("NP", 0, rel, base, and, other);
            }
            return base;
        }

        Node baseNp(int depth, String rel) {
            double u = r.nextDouble();
            if (u < 0.18) {
                return phrase("NP", 0, rel, leaf("Name", pick(NAME), "root"));
            }
            Node det = leaf("Det", pick(DET), "det");
            if (u < 0.35) {
                Node adj = leaf("Adj", pick(ADJ), "amod");
                return phrase("NP", 2, rel, det, adj, leaf("N", pick(PERSON), "root"));
            }
            return phrase("NP", 1, rel, det, leaf("N", pick(PERSON), "root"));
        }

        Node pp(int depth, String prep, String[] nouns) {
            Node p = leaf("P", prep, "root");
            Node det = leaf("Det", pick(DET), "det");
            Node n = leaf("N", pick(nouns), "root");
            return phrase("PP", 0, "prep", p, phrase("NP", 1, "pobj", det, n));
        }

        Sentence finish(Node root) {
            int n = words.size();
            int[] w = new int[n];
            int[] t = new int[n];
            int[] heads = new int[n];
            int[] rels = new int[n];
            for (int i = 0; i < n; i++) {
                w[i] = IDS.get(words.get(i));
                t[i] = TAGS.indexOf(tags.get(i));
            }
            heads[root.head()] = 0;
            rels[root.head()] = 0;
            depend(root, heads, rels);
            Map<Long, int[]> spans = new LinkedHashMap<>();
            Tree tree = convert(root, spans);
            int[] cases = new int[ppCases.size()];
            for (int i = 0; i < cases.length; i++) {
                cases[i] = ppCases.get(i).head();
            }
            return new Sentence(w, t, heads, rels, List.copyOf(spans.values()), cases, tree);
        }

        static void depend(Node node, int[] heads, int[] rels) {
            if (node.word >= 0) {
                return;
            }
            int h = node.head();
            for (int i = 0; i < node.kids.size(); i++) {
                Node k = node.kids.get(i);
                if (i != node.headChild) {
                    heads[k.head()] = h + 1;
                    rels[k.head()] = RELATIONS.indexOf(k.rel);
                }
                depend(k, heads, rels);
            }
        }

        static Tree convert(Node node, Map<Long, int[]> spans) {
            if (node.word >= 0) {
                return new Tree(node.label, List.of(), node.word);
            }
            List<Tree> kids = new ArrayList<>();
            for (Node k : node.kids) {
                kids.add(convert(k, spans));
            }
            Tree t = new Tree(node.label, List.copyOf(kids), -1);
            // children are converted first, so an outer phrase on the same span overwrites them
            spans.put((long) t.start() * 1000 + t.end(), new int[] {PHRASES.indexOf(node.label), t.start(), t.end()});
            return t;
        }
    }
}
