package io.github.zzuegg.brainiac.experiments.document;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments;
import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments.Line;
import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments.Page;
import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments.Word;
import io.github.zzuegg.brainiac.experimental.document.ImageToMarkup;
import io.github.zzuegg.brainiac.experimental.document.LayoutEncoder;
import io.github.zzuegg.brainiac.experimental.document.LineRecognizer;
import io.github.zzuegg.brainiac.experimental.nn.LinearChainCrf;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Wave 9 items 9.4.11 and 9.4.12: key/value extraction from {@link GlyphDocuments} forms (a
 * title and three key/value fields; a value sits right of its key, or below it with
 * probability 0.3).
 *
 * <p><b>Parsing</b> ({@link #parse}, 9.4.11): the Donut-style {@link ImageToMarkup} reads the
 * 48x64 page and emits {@code <KEY> v a l u e ...}, after a reading phase (emit all text in
 * reading order) or without it; against two pipelines on the CRNN of {@link GlyphOcr} applied
 * to each word box (the boxes are given, an oracle text detector): rule-based field
 * extraction (nearest word right of, else below, each key) and the {@link LayoutEncoder}
 * tagger. Scores: field-level F1 and exact parse accuracy.
 *
 * <p><b>Tagging</b> ({@link #tagging}, 9.4.12): tag each word O / KEY / VALUE-of-field-k from
 * its token and box. Rows: {@link LayoutEncoder} with 2-D box embeddings, with 1-D positions
 * only, a BiGRU-CRF tagger and the nearest-key rules; with reading order kept or shuffled,
 * with ground-truth or OCR tokens, and with or without masked pretraining. Score: F1 over
 * extracted (field, value text) pairs.
 */
public final class DocumentParsing {

    /** Fields per form. */
    public static final int FIELDS = 3;
    /** Page height of the image models. */
    public static final int HEIGHT = 48;
    /** Page width of the image models. */
    public static final int WIDTH = 64;
    private static final int KEY0 = 1;
    private static final int CHAR0 = KEY0 + GlyphDocuments.KEYS.length;
    private static final int DONUT_VOCAB = CHAR0 + GlyphDocuments.CHARSET.length();
    private static final int NUM = GlyphDocuments.KEYS.length + GlyphDocuments.LEXICON.length;
    private static final int DATE = NUM + 1;
    private static final int UNK = NUM + 2;
    private static final int VOCAB = NUM + 4;
    private static final int TAGS = 2 + GlyphDocuments.KEYS.length;
    private static final int GRID = 64;

    private DocumentParsing() {
    }

    /**
     * Random forms.
     *
     * @param n forms
     * @param noise pixel flip probability
     * @param random the source
     * @return the forms
     */
    public static List<Page> forms(int n, double noise, SplittableRandom random) {
        List<Page> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            out.add(GlyphDocuments.form(FIELDS, 0.3, true, noise, random));
        }
        return out;
    }

    // ------------------------------------------------------------------ parsing (9.4.11)

    /** The parsers of 9.4.11. */
    public enum Parser {
        /** Image-to-markup, half the steps reading, half parsing. */
        DONUT,
        /** Image-to-markup trained on parsing only. */
        DONUT_NO_READING,
        /** CRNN OCR per word box, then nearest-key rules. */
        OCR_RULES,
        /** CRNN OCR per word box, then the 2-D layout tagger. */
        OCR_LAYOUT
    }

    /**
     * Scores of one parser.
     *
     * @param parser the parser
     * @param f1 field-level F1
     * @param exact the share of pages parsed exactly
     * @param millis training and evaluation time
     */
    public record ParseScores(Parser parser, double f1, double exact, long millis) {
    }

    /**
     * Runs one parser.
     *
     * @param parser the parser
     * @param steps training steps (each model of a pipeline gets this many)
     * @param testPages held-out forms
     * @param noise pixel flip probability of the forms
     * @param seed the seed
     * @return the scores
     */
    public static ParseScores parse(Parser parser, int steps, int testPages, double noise, long seed) {
        long start = System.currentTimeMillis();
        List<Page> test = forms(testPages, noise, new SplittableRandom(seed + 99));
        List<String[]> predicted = new ArrayList<>();
        switch (parser) {
            case DONUT, DONUT_NO_READING -> {
                ImageToMarkup m = new ImageToMarkup(HEIGHT, WIDTH, 4, 32, 2, 2, DONUT_VOCAB, seed);
                Optimizer opt = Optimizer.adam(1e-3f);
                SplittableRandom r = new SplittableRandom(seed + 1);
                int reading = parser == Parser.DONUT ? steps / 2 : 0;
                for (int s = 0; s < steps; s++) {
                    List<Page> batch = forms(8, noise, r);
                    int[][] targets = new int[batch.size()][];
                    for (int i = 0; i < targets.length; i++) {
                        targets[i] = s < reading ? readingTarget(batch.get(i)) : parseTarget(batch.get(i));
                    }
                    m.loss(GlyphDocuments.images(batch, HEIGHT, WIDTH), pad(targets)).backward();
                    opt.step(m.parameters());
                }
                for (int from = 0; from < test.size(); from += 16) {
                    List<Page> part = test.subList(from, Math.min(test.size(), from + 16));
                    for (int[] out : m.generate(GlyphDocuments.images(part, HEIGHT, WIDTH), 4 + FIELDS * 8)) {
                        predicted.add(fromTokens(out));
                    }
                }
            }
            default -> {
                LineRecognizer ocr = GlyphOcr.trainRecognizer(LineRecognizer.Encoder.BIGRU, steps, 3, 12, seed);
                TokenTagger tagger = null;
                if (parser == Parser.OCR_LAYOUT) {
                    tagger = layout(true, seed + 2);
                    train(tagger, forms(200, noise, new SplittableRandom(seed + 3)), steps, false,
                            new SplittableRandom(seed + 4));
                }
                for (Page p : test) {
                    String[] texts = ocr(ocr, p);
                    predicted.add(tagger == null ? rules(p.words(), texts)
                            : fields(tagger, p, texts, false, new SplittableRandom(seed)));
                }
            }
        }
        double tp = 0;
        double np = 0;
        double ng = 0;
        int exact = 0;
        for (int i = 0; i < test.size(); i++) {
            String[] gold = test.get(i).fields();
            String[] pred = predicted.get(i);
            boolean same = true;
            for (int k = 0; k < gold.length; k++) {
                np += pred[k] != null ? 1 : 0;
                ng += gold[k] != null ? 1 : 0;
                boolean ok = gold[k] != null && gold[k].equals(pred[k]);
                tp += ok ? 1 : 0;
                same &= java.util.Objects.equals(gold[k], pred[k]);
            }
            exact += same ? 1 : 0;
        }
        return new ParseScores(parser, f1(tp, np, ng), (double) exact / test.size(),
                System.currentTimeMillis() - start);
    }

    private static double f1(double tp, double predicted, double gold) {
        return tp == 0 ? 0 : 2 * tp / (predicted + gold);
    }

    private static int[] chars(String s) {
        int[] out = new int[s.length()];
        for (int i = 0; i < out.length; i++) {
            out[i] = CHAR0 + GlyphDocuments.index(s.charAt(i));
        }
        return out;
    }

    /** All words in reading order, separated by spaces, then end-of-sequence. */
    static int[] readingTarget(Page p) {
        StringBuilder sb = new StringBuilder();
        for (Word w : p.words()) {
            sb.append(sb.isEmpty() ? "" : " ").append(w.text());
        }
        return append(chars(sb.toString()), 0);
    }

    /** {@code <KEY> value-characters ...} in canonical key order, then end-of-sequence. */
    static int[] parseTarget(Page p) {
        List<Integer> out = new ArrayList<>();
        for (int k = 0; k < GlyphDocuments.KEYS.length; k++) {
            if (p.fields()[k] != null) {
                out.add(KEY0 + k);
                for (int c : chars(p.fields()[k])) {
                    out.add(c);
                }
            }
        }
        out.add(0);
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int[] append(int[] a, int v) {
        int[] out = Arrays.copyOf(a, a.length + 1);
        out[a.length] = v;
        return out;
    }

    private static int[][] pad(int[][] rows) {
        int len = 0;
        for (int[] r : rows) {
            len = Math.max(len, r.length);
        }
        int[][] out = new int[rows.length][len];
        for (int i = 0; i < rows.length; i++) {
            Arrays.fill(out[i], -1);
            System.arraycopy(rows[i], 0, out[i], 0, rows[i].length);
        }
        return out;
    }

    private static String[] fromTokens(int[] tokens) {
        String[] out = new String[GlyphDocuments.KEYS.length];
        int key = -1;
        StringBuilder sb = new StringBuilder();
        for (int t : tokens) {
            if (t >= KEY0 && t < CHAR0) {
                if (key >= 0 && !sb.isEmpty()) {
                    out[key] = sb.toString();
                }
                key = t - KEY0;
                sb.setLength(0);
            } else if (t >= CHAR0 && t < DONUT_VOCAB) {
                sb.append(GlyphDocuments.CHARSET.charAt(t - CHAR0));
            }
        }
        if (key >= 0 && !sb.isEmpty()) {
            out[key] = sb.toString();
        }
        return out;
    }

    /**
     * Reads every word box of a page with a line recogniser.
     *
     * @param rec the recogniser
     * @param page the page
     * @return per word its OCR text
     */
    public static String[] ocr(LineRecognizer rec, Page page) {
        List<Line> lines = new ArrayList<>();
        for (Word w : page.words()) {
            int width = w.x1() - w.x0() + 2;
            float[] px = new float[GlyphDocuments.LINE_HEIGHT * width];
            for (int y = 0; y < GlyphDocuments.LINE_HEIGHT; y++) {
                for (int x = 0; x < width; x++) {
                    int py = w.y0() - 1 + y;
                    int pxx = w.x0() - 1 + x;
                    if (py >= 0 && py < page.height() && pxx >= 0 && pxx < page.width()) {
                        px[y * width + x] = page.pixels()[py * page.width() + pxx];
                    }
                }
            }
            lines.add(new Line(width, px, w.text(), new int[0][], false));
        }
        return GlyphOcr.read(rec, lines);
    }

    /** The key index of a word ending in a colon (edit distance at most 1), or -1. */
    private static int key(String text) {
        if (!text.endsWith(":")) {
            return -1;
        }
        String t = text.substring(0, text.length() - 1);
        int best = -1;
        int bestD = 2;
        for (int k = 0; k < GlyphDocuments.KEYS.length; k++) {
            int d = LineRecognizer.editDistance(t, GlyphDocuments.KEYS[k]);
            if (d < bestD) {
                bestD = d;
                best = k;
            }
        }
        return best;
    }

    /**
     * Rule-based extraction: for each key, the nearest non-key word to its right on the same
     * row, else the nearest word starting below it.
     *
     * @param words the words with their boxes
     * @param texts per word its (possibly OCR) text
     * @return per key of {@link GlyphDocuments#KEYS} its value, or null
     */
    public static String[] rules(List<Word> words, String[] texts) {
        String[] out = new String[GlyphDocuments.KEYS.length];
        for (int i = 0; i < words.size(); i++) {
            int k = key(texts[i]);
            if (k < 0) {
                continue;
            }
            Word kw = words.get(i);
            int best = -1;
            int bestD = Integer.MAX_VALUE;
            for (int j = 0; j < words.size(); j++) {
                Word w = words.get(j);
                if (j == i || key(texts[j]) >= 0) {
                    continue;
                }
                int d = Integer.MAX_VALUE;
                if (Math.abs(w.y0() - kw.y0()) <= 1 && w.x0() >= kw.x1()) {
                    d = w.x0() - kw.x1();
                } else if (w.y0() > kw.y0() && w.y0() - kw.y0() <= GlyphDocuments.LINE_HEIGHT
                        && Math.abs(w.x0() - kw.x0()) <= 4) {
                    d = 1000 + w.y0() - kw.y0();
                }
                if (d < bestD) {
                    bestD = d;
                    best = j;
                }
            }
            if (best >= 0) {
                out[k] = texts[best];
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ tagging (9.4.12)

    /** The taggers of 9.4.12. */
    public enum Tagger {
        /** {@link LayoutEncoder} with 2-D box embeddings. */
        LAYOUT_2D,
        /** The same encoder with 1-D positions only. */
        POSITION_1D,
        /** A bidirectional GRU with a CRF on top. */
        BIGRU_CRF,
        /** Nearest-key rules (no training). */
        RULES
    }

    /**
     * Scores of one tagger.
     *
     * @param tagger the tagger
     * @param shuffled whether reading order was shuffled
     * @param ocrTokens whether test tokens came from OCR
     * @param pretrained whether masked pretraining ran first
     * @param f1 F1 over extracted (field, value) pairs
     * @param millis training and evaluation time
     */
    public record TagScores(Tagger tagger, boolean shuffled, boolean ocrTokens, boolean pretrained, double f1,
            long millis) {
    }

    /** Token ids of word texts. */
    static int token(String text) {
        int k = key(text);
        if (k >= 0) {
            return k;
        }
        if (!text.isEmpty() && text.chars().allMatch(Character::isDigit)) {
            return NUM;
        }
        if (text.contains("-")) {
            return DATE;
        }
        int i = Arrays.asList(GlyphDocuments.LEXICON).indexOf(text);
        return i >= 0 ? GlyphDocuments.KEYS.length + i : UNK;
    }

    /** One form as tagger input. */
    private record Encoded(int[] tokens, int[][] boxes, int[] positions, int[] tags, int[] order) {
    }

    private static Encoded encode(Page p, String[] texts, boolean shuffle, SplittableRandom r) {
        int n = p.words().size();
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        if (shuffle) {
            for (int i = n - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
        }
        int[] tok = new int[n];
        int[][] box = new int[n][];
        int[] pos = new int[n];
        int[] tags = new int[n];
        for (int i = 0; i < n; i++) {
            Word w = p.words().get(order[i]);
            tok[i] = token(texts[order[i]]);
            box[i] = new int[] {w.x0(), w.y0(), w.x1(), w.y1()};
            pos[i] = i;
            tags[i] = switch (w.role()) {
                case KEY -> 1;
                case VALUE -> 2 + w.field();
                default -> 0;
            };
        }
        return new Encoded(tok, box, pos, tags, order);
    }

    /** A trainable tagger over fixed-length token sequences. */
    private interface TokenTagger {
        Var loss(int[][] tokens, int[][][] boxes, int[][] positions, int[][] tags);

        int[][] tag(int[][] tokens, int[][][] boxes, int[][] positions);

        Var pretrainLoss(int[][] tokens, int[][][] boxes, int[][] positions, SplittableRandom r);

        List<Var> parameters();
    }

    private static TokenTagger layout(boolean twoD, long seed) {
        LayoutEncoder e = new LayoutEncoder(VOCAB, TAGS, 32, 2, 16, GRID, twoD, seed);
        return new TokenTagger() {
            @Override
            public Var loss(int[][] t, int[][][] b, int[][] p, int[][] tags) {
                return e.tagLoss(t, b, p, tags);
            }

            @Override
            public int[][] tag(int[][] t, int[][][] b, int[][] p) {
                return e.tag(t, b, p);
            }

            @Override
            public Var pretrainLoss(int[][] t, int[][][] b, int[][] p, SplittableRandom r) {
                return e.maskedLoss(t, b, p, 0.15, r);
            }

            @Override
            public List<Var> parameters() {
                return e.parameters();
            }
        };
    }

    /** Token embeddings, a bidirectional GRU and a CRF. */
    private static TokenTagger bigruCrf(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int dim = 32;
        Var table = Var.param(Tensor.normal(Shape.of(VOCAB, dim), 0, 0.3f, r));
        Layer rnn = Recurrent.gru(dim / 2).returnSequences(true).bidirectional(Recurrent.Merge.CONCAT)
                .create(Shape.of(8, dim), r);
        Var w = Var.param(Tensor.normal(Shape.of(dim, TAGS), 0, (float) Math.sqrt(1.0 / dim), r));
        Var bias = Var.param(Tensor.zeros(Shape.of(TAGS)));
        LinearChainCrf crf = new LinearChainCrf(TAGS, r);
        return new TokenTagger() {
            private Var emissions(int[][] t, boolean training) {
                Var x = table.gather(0, flat(t)).reshape(t.length, t[0].length, dim);
                return rnn.forward(x, training).matmul(w).add(bias);
            }

            @Override
            public Var loss(int[][] t, int[][][] b, int[][] p, int[][] tags) {
                return crf.negativeLogLikelihood(emissions(t, true), flat(tags));
            }

            @Override
            public int[][] tag(int[][] t, int[][][] b, int[][] p) {
                return crf.decode(Var.noGrad(() -> emissions(t, false)).value());
            }

            @Override
            public Var pretrainLoss(int[][] t, int[][][] b, int[][] p, SplittableRandom rr) {
                throw new UnsupportedOperationException("no masked pretraining for the BiGRU-CRF");
            }

            @Override
            public List<Var> parameters() {
                List<Var> ps = new ArrayList<>(List.of(table, w, bias));
                ps.addAll(rnn.parameters());
                ps.addAll(crf.parameters());
                return ps;
            }
        };
    }

    private static Tensor flat(int[][] v) {
        float[] f = new float[v.length * v[0].length];
        for (int i = 0; i < v.length; i++) {
            for (int j = 0; j < v[0].length; j++) {
                f[i * v[0].length + j] = v[i][j];
            }
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    private static String[] texts(Page p) {
        String[] t = new String[p.words().size()];
        for (int i = 0; i < t.length; i++) {
            t[i] = p.words().get(i).text();
        }
        return t;
    }

    private static void train(TokenTagger tagger, List<Page> pool, int steps, boolean shuffle, SplittableRandom r) {
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            int b = Math.min(16, pool.size());
            int[][] tok = new int[b][];
            int[][][] box = new int[b][][];
            int[][] pos = new int[b][];
            int[][] tags = new int[b][];
            for (int i = 0; i < b; i++) {
                Page p = pool.get(r.nextInt(pool.size()));
                Encoded e = encode(p, texts(p), shuffle, r);
                tok[i] = e.tokens();
                box[i] = e.boxes();
                pos[i] = e.positions();
                tags[i] = e.tags();
            }
            tagger.loss(tok, box, pos, tags).backward();
            opt.step(tagger.parameters());
        }
    }

    private static void pretrain(TokenTagger tagger, int steps, double noise, SplittableRandom r) {
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            List<Page> batch = forms(16, noise, r);
            int[][] tok = new int[batch.size()][];
            int[][][] box = new int[batch.size()][][];
            int[][] pos = new int[batch.size()][];
            for (int i = 0; i < batch.size(); i++) {
                Encoded e = encode(batch.get(i), texts(batch.get(i)), false, r);
                tok[i] = e.tokens();
                box[i] = e.boxes();
                pos[i] = e.positions();
            }
            tagger.pretrainLoss(tok, box, pos, r).backward();
            opt.step(tagger.parameters());
        }
    }

    /** Tags one page and reads off per key its value text. */
    private static String[] fields(TokenTagger tagger, Page p, String[] texts, boolean shuffle, SplittableRandom r) {
        Encoded e = encode(p, texts, shuffle, r);
        int[] tags = tagger.tag(new int[][] {e.tokens()}, new int[][][] {e.boxes()}, new int[][] {e.positions()})[0];
        String[] out = new String[GlyphDocuments.KEYS.length];
        for (int i = 0; i < tags.length; i++) {
            if (tags[i] >= 2 && out[tags[i] - 2] == null) {
                out[tags[i] - 2] = texts[e.order()[i]];
            }
        }
        return out;
    }

    /**
     * Runs one tagger.
     *
     * @param tagger the tagger
     * @param labelled labelled training forms
     * @param steps tagging steps (batches of 16 drawn from the labelled forms)
     * @param pretrainSteps masked pretraining steps on fresh unlabelled forms (layout encoders only)
     * @param shuffled whether to shuffle reading order (training and test)
     * @param ocr the recogniser for test tokens, or null for ground-truth tokens
     * @param testPages held-out forms
     * @param seed the seed
     * @return the scores
     */
    public static TagScores tagging(Tagger tagger, int labelled, int steps, int pretrainSteps, boolean shuffled,
            LineRecognizer ocr, int testPages, long seed) {
        long start = System.currentTimeMillis();
        double noise = 0.02;
        List<Page> pool = forms(labelled, noise, new SplittableRandom(seed + 1));
        List<Page> test = forms(testPages, noise, new SplittableRandom(seed + 99));
        TokenTagger model = switch (tagger) {
            case LAYOUT_2D -> layout(true, seed);
            case POSITION_1D -> layout(false, seed);
            case BIGRU_CRF -> bigruCrf(seed);
            case RULES -> null;
        };
        boolean pretrained = model != null && tagger != Tagger.BIGRU_CRF && pretrainSteps > 0;
        if (pretrained) {
            pretrain(model, pretrainSteps, noise, new SplittableRandom(seed + 2));
        }
        if (model != null) {
            train(model, pool, steps, shuffled, new SplittableRandom(seed + 3));
        }
        SplittableRandom r = new SplittableRandom(seed + 4);
        double tp = 0;
        double np = 0;
        double ng = 0;
        for (Page p : test) {
            String[] t = ocr == null ? texts(p) : ocr(ocr, p);
            String[] pred = model == null ? rules(p.words(), t) : fields(model, p, t, shuffled, r);
            Set<String> gold = new HashSet<>();
            for (int k = 0; k < p.fields().length; k++) {
                if (p.fields()[k] != null) {
                    gold.add(k + "=" + p.fields()[k]);
                }
            }
            ng += gold.size();
            for (int k = 0; k < pred.length; k++) {
                if (pred[k] != null) {
                    np++;
                    tp += gold.contains(k + "=" + pred[k]) ? 1 : 0;
                }
            }
        }
        return new TagScores(tagger, shuffled, ocr != null, pretrained, f1(tp, np, ng),
                System.currentTimeMillis() - start);
    }
}
