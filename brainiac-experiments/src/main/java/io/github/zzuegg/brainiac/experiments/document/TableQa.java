package io.github.zzuegg.brainiac.experiments.document;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ToyTables;
import io.github.zzuegg.brainiac.experimental.data.ToyTables.Kind;
import io.github.zzuegg.brainiac.experimental.data.ToyTables.Question;
import io.github.zzuegg.brainiac.experimental.data.ToyTables.Table;
import io.github.zzuegg.brainiac.experimental.document.TableEncoder;
import io.github.zzuegg.brainiac.experimental.evolve.GuidedEnumerator;
import io.github.zzuegg.brainiac.experimental.multimodal.FusionDecoder;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 9 item 9.4.13: question answering over {@link ToyTables} (lookup, comparison, count
 * and sum). Training tables have 3-5 rows and 3-4 columns; the test is on tables of that size
 * and on larger ones (6-8 rows, 5 columns).
 *
 * <p>Rows: the TAPAS-style {@link TableEncoder} (row, column and rank embeddings, cell
 * selection and aggregation, trained on answers only); a decoder-only toy LM
 * ({@link FusionDecoder}) that reads the linearised table and the question and emits the
 * answer token; and a semantic parser into a seven-primitive DSL (ROW, GT, ARGMAX, COUNT,
 * SUM, VALUE, NAME over the table) found by {@link GuidedEnumerator}: training questions are
 * grouped by template (entity words replaced by placeholders), and per template the shortest
 * program reproducing the answers of up to ten training questions is kept; at test time the
 * template's program is executed. Score: answer accuracy per question kind and table size.
 */
public final class TableQa {

    private static final List<String> PRIMITIVES = List.of("ROW", "GT", "ARGMAX", "COUNT", "SUM", "VALUE", "NAME");
    private static final int CONTEXT = 5;
    private static final int NAME_CODE = 1000;

    private TableQa() {
    }

    /** The answerers compared. */
    public enum Method {
        /** Cell selection and aggregation. */
        TAPAS,
        /** A toy LM reading the linearised table. */
        LM,
        /** A program per question template, found by enumeration. */
        PARSER
    }

    /**
     * Scores of one answerer.
     *
     * @param method the answerer
     * @param small per kind the accuracy on training-size tables
     * @param large per kind the accuracy on larger tables
     * @param smallAccuracy mean accuracy on training-size tables
     * @param largeAccuracy mean accuracy on larger tables
     * @param millis training and evaluation time
     */
    public record Scores(Method method, Map<Kind, Double> small, Map<Kind, Double> large, double smallAccuracy,
            double largeAccuracy, long millis) {
    }

    /** A table and a question about it. */
    private record Item(Table table, Question question) {
    }

    private static List<Item> items(int n, int minRows, int maxRows, int minCols, int maxCols, SplittableRandom r) {
        List<Item> out = new ArrayList<>();
        Kind[] kinds = Kind.values();
        for (int i = 0; i < n; i++) {
            Table t = ToyTables.table(minRows + r.nextInt(maxRows - minRows + 1),
                    minCols + r.nextInt(maxCols - minCols + 1), r);
            out.add(new Item(t, ToyTables.question(t, kinds[i % kinds.length], r)));
        }
        return out;
    }

    /**
     * Runs one answerer.
     *
     * @param method the answerer
     * @param steps training steps of 16 questions (the parser uses {@code 16 * steps} questions
     *     at most, grouped by template)
     * @param testQuestions questions per test size
     * @param seed the seed
     * @return the scores
     */
    public static Scores run(Method method, int steps, int testQuestions, long seed) {
        long start = System.currentTimeMillis();
        SplittableRandom r = new SplittableRandom(seed);
        List<Item> small = items(testQuestions, 3, 5, 3, 4, new SplittableRandom(seed + 98));
        List<Item> large = items(testQuestions, 6, 8, 5, 5, new SplittableRandom(seed + 99));
        Answerer a = switch (method) {
            case TAPAS -> tapas(steps, r, seed);
            case LM -> lm(steps, r, seed);
            case PARSER -> parser(items(Math.min(16 * steps, 400), 3, 5, 3, 4, r));
        };
        Map<Kind, Double> s = accuracy(a, small);
        Map<Kind, Double> l = accuracy(a, large);
        return new Scores(method, s, l, mean(s), mean(l), System.currentTimeMillis() - start);
    }

    private static double mean(Map<Kind, Double> m) {
        return m.values().stream().mapToDouble(Double::doubleValue).average().orElse(0);
    }

    private interface Answerer {
        String[] answer(List<Item> items);
    }

    private static Map<Kind, Double> accuracy(Answerer a, List<Item> test) {
        String[] got = a.answer(test);
        Map<Kind, double[]> c = new EnumMap<>(Kind.class);
        for (int i = 0; i < test.size(); i++) {
            double[] v = c.computeIfAbsent(test.get(i).question().kind(), k -> new double[2]);
            v[0] += test.get(i).question().answer().equals(got[i]) ? 1 : 0;
            v[1]++;
        }
        Map<Kind, Double> out = new EnumMap<>(Kind.class);
        c.forEach((k, v) -> out.put(k, v[0] / v[1]));
        return out;
    }

    // ------------------------------------------------------------------ TAPAS

    private static Answerer tapas(int steps, SplittableRandom r, long seed) {
        TableEncoder m = new TableEncoder(ToyTables.vocabulary(80), 32, 2, 8, 5, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            List<Item> batch = items(16, 3, 5, 3, 4, r);
            List<TableEncoder.Example> ex = new ArrayList<>();
            List<Question> qs = new ArrayList<>();
            for (Item it : batch) {
                ex.add(m.encode(it.table(), it.question()));
                qs.add(it.question());
            }
            m.loss(ex, qs, 1f).backward();
            opt.step(m.parameters());
        }
        return items -> {
            String[] out = new String[items.size()];
            for (int from = 0; from < items.size(); from += 32) {
                List<TableEncoder.Example> ex = new ArrayList<>();
                for (Item it : items.subList(from, Math.min(items.size(), from + 32))) {
                    ex.add(m.encode(it.table(), it.question()));
                }
                String[] a = m.answer(ex);
                System.arraycopy(a, 0, out, from, a.length);
            }
            return out;
        };
    }

    // ------------------------------------------------------------------ toy LM

    private static Answerer lm(int steps, SplittableRandom r, long seed) {
        List<String> words = new ArrayList<>(ToyTables.vocabulary(80));
        words.addAll(List.of("NAME", "[SEP]", "[ROW]", "[ANS]", "[PAD]"));
        Map<String, Integer> ids = new HashMap<>();
        for (String w : words) {
            ids.putIfAbsent(w, ids.size());
        }
        FusionDecoder m = new FusionDecoder(ids.size(), 32, 2, 4, 96, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            List<Item> batch = items(16, 3, 5, 3, 4, r);
            int[][] seqs = sequences(batch, ids);
            int[][] targets = new int[seqs.length][seqs[0].length];
            for (int i = 0; i < seqs.length; i++) {
                Arrays.fill(targets[i], -1);
                targets[i][answerPosition(batch.get(i))] = ids.get(batch.get(i).question().answer());
            }
            FusionDecoder.loss(m.logits(m.embed(seqs), null, null, true), targets).backward();
            opt.step(m.parameters());
        }
        List<String> byId = new ArrayList<>(ids.keySet());
        byId.sort((a, b) -> Integer.compare(ids.get(a), ids.get(b)));
        return items -> {
            String[] out = new String[items.size()];
            for (int from = 0; from < items.size(); from += 32) {
                List<Item> part = items.subList(from, Math.min(items.size(), from + 32));
                int[][] seqs = sequences(part, ids);
                float[] logits = Var.noGrad(() -> m.logits(m.embed(seqs), null, null, false)).value().toArray();
                int len = seqs[0].length;
                int v = ids.size();
                for (int i = 0; i < part.size(); i++) {
                    int off = (i * len + answerPosition(part.get(i))) * v;
                    int best = 0;
                    for (int k = 1; k < v; k++) {
                        best = logits[off + k] > logits[off + best] ? k : best;
                    }
                    out[from + i] = byId.get(best);
                }
            }
            return out;
        };
    }

    /** {@code question [SEP] header [ROW] cells [ROW] cells ... [ANS]}. */
    private static List<String> linear(Item it) {
        List<String> out = new ArrayList<>(it.question().words());
        out.add("[SEP]");
        out.addAll(Arrays.asList(it.table().header()));
        for (String[] row : it.table().cells()) {
            out.add("[ROW]");
            out.addAll(Arrays.asList(row));
        }
        out.add("[ANS]");
        return out;
    }

    private static int answerPosition(Item it) {
        return linear(it).size() - 1;
    }

    private static int[][] sequences(List<Item> items, Map<String, Integer> ids) {
        int len = 0;
        for (Item it : items) {
            len = Math.max(len, linear(it).size());
        }
        int[][] out = new int[items.size()][len];
        for (int i = 0; i < items.size(); i++) {
            Arrays.fill(out[i], ids.get("[PAD]"));
            List<String> w = linear(items.get(i));
            for (int t = 0; t < w.size(); t++) {
                out[i][t] = ids.get(w.get(t));
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ semantic parser

    /** The question with entity words replaced by placeholders. */
    private static String template(Item it) {
        StringBuilder sb = new StringBuilder();
        for (String w : it.question().words()) {
            String t = Arrays.asList(ToyTables.COLUMNS).contains(w) ? "<C>"
                    : Arrays.asList(ToyTables.NAMES).contains(w) ? "<R>"
                    : w.chars().allMatch(Character::isDigit) ? "<N>" : w;
            sb.append(t).append(' ');
        }
        return sb.toString();
    }

    /** {@code rows, columns, question column, question row, question number, cells..., mode, items...}. */
    private static int[] state(Item it) {
        Table t = it.table();
        int rows = t.rows();
        int cols = t.columns();
        int qCol = -1;
        int qRow = -1;
        int qNum = -1;
        for (String w : it.question().words()) {
            int c = Arrays.asList(t.header()).indexOf(w);
            if (c >= 1) {
                qCol = c;
            }
            for (int r = 0; r < rows; r++) {
                if (t.cells()[r][0].equals(w)) {
                    qRow = r;
                }
            }
            if (w.chars().allMatch(Character::isDigit)) {
                qNum = Integer.parseInt(w);
            }
        }
        int[] s = new int[CONTEXT + rows * cols + 1 + rows];
        s[0] = rows;
        s[1] = cols;
        s[2] = qCol;
        s[3] = qRow;
        s[4] = qNum;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                s[CONTEXT + r * cols + c] = c == 0 ? NAME_CODE + Arrays.asList(ToyTables.NAMES).indexOf(t.cells()[r][0])
                        : t.value(r, c);
            }
        }
        int m = CONTEXT + rows * cols;
        s[m] = 0;
        for (int r = 0; r < rows; r++) {
            s[m + 1 + r] = r;
        }
        return s;
    }

    private static int[] answerState(Item it) {
        int[] s = state(it);
        int m = CONTEXT + s[0] * s[1];
        int[] out = Arrays.copyOf(s, m + 2);
        out[m] = 1;
        String a = it.question().answer();
        out[m + 1] = it.question().numeric() ? Integer.parseInt(a) : NAME_CODE + Arrays.asList(ToyTables.NAMES).indexOf(a);
        return out;
    }

    /** One DSL primitive applied to a state; null when it does not apply. */
    static int[] apply(String op, int[] s) {
        int rows = s[0];
        int cols = s[1];
        int qCol = s[2];
        int qRow = s[3];
        int qNum = s[4];
        int m = CONTEXT + rows * cols;
        if (s[m] != 0) {
            return null;
        }
        int[] sel = Arrays.copyOfRange(s, m + 1, s.length);
        int[] next;
        int[] scalar = null;
        switch (op) {
            case "ROW" -> {
                if (qRow < 0) {
                    return null;
                }
                next = Arrays.stream(sel).filter(r -> r == qRow).toArray();
            }
            case "GT" -> {
                if (qCol < 1 || qNum < 0) {
                    return null;
                }
                next = Arrays.stream(sel).filter(r -> s[CONTEXT + r * cols + qCol] > qNum).toArray();
            }
            case "ARGMAX" -> {
                if (qCol < 1 || sel.length == 0) {
                    return null;
                }
                int best = sel[0];
                for (int r : sel) {
                    best = s[CONTEXT + r * cols + qCol] > s[CONTEXT + best * cols + qCol] ? r : best;
                }
                next = new int[] {best};
            }
            case "COUNT" -> {
                next = null;
                scalar = new int[] {sel.length};
            }
            case "SUM" -> {
                if (qCol < 1) {
                    return null;
                }
                next = null;
                scalar = new int[] {Arrays.stream(sel).map(r -> s[CONTEXT + r * cols + qCol]).sum()};
            }
            case "VALUE" -> {
                if (qCol < 1 || sel.length != 1) {
                    return null;
                }
                next = null;
                scalar = new int[] {s[CONTEXT + sel[0] * cols + qCol]};
            }
            case "NAME" -> {
                if (sel.length != 1) {
                    return null;
                }
                next = null;
                scalar = new int[] {s[CONTEXT + sel[0] * cols]};
            }
            default -> throw new IllegalArgumentException(op);
        }
        int[] out;
        if (scalar != null) {
            out = Arrays.copyOf(s, m + 2);
            out[m] = 1;
            out[m + 1] = scalar[0];
        } else {
            out = Arrays.copyOf(s, m + 1 + next.length);
            System.arraycopy(next, 0, out, m + 1, next.length);
        }
        return out;
    }

    private static Answerer parser(List<Item> train) {
        GuidedEnumerator search = new GuidedEnumerator(PRIMITIVES, TableQa::apply);
        Map<String, List<Item>> groups = new LinkedHashMap<>();
        for (Item it : train) {
            groups.computeIfAbsent(template(it), k -> new ArrayList<>()).add(it);
        }
        Map<String, List<String>> programs = new HashMap<>();
        for (var e : groups.entrySet()) {
            List<Item> ex = e.getValue().subList(0, Math.min(10, e.getValue().size()));
            int[][] in = new int[ex.size()][];
            int[][] out = new int[ex.size()][];
            for (int i = 0; i < ex.size(); i++) {
                in[i] = state(ex.get(i));
                out[i] = answerState(ex.get(i));
            }
            GuidedEnumerator.Result res = search.enumerate(in, out, 3, PRIMITIVES, 200_000);
            if (res.solved()) {
                programs.put(e.getKey(), res.program());
            }
        }
        return items -> {
            String[] answers = new String[items.size()];
            for (int i = 0; i < items.size(); i++) {
                answers[i] = "";
                List<String> p = programs.get(template(items.get(i)));
                if (p == null) {
                    continue;
                }
                int[] s = state(items.get(i));
                for (String op : p) {
                    s = s == null ? null : apply(op, s);
                }
                if (s != null) {
                    int m = CONTEXT + s[0] * s[1];
                    if (s[m] == 1) {
                        int v = s[m + 1];
                        answers[i] = v >= NAME_CODE ? ToyTables.NAMES[v - NAME_CODE] : Integer.toString(v);
                    }
                }
            }
            return answers;
        };
    }
}
