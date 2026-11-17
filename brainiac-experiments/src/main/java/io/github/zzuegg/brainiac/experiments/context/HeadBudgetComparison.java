package io.github.zzuegg.brainiac.experiments.context;

import io.github.zzuegg.brainiac.experimental.llm.DuoHeadCache;
import io.github.zzuegg.brainiac.experimental.llm.KvBudgetPolicy;
import io.github.zzuegg.brainiac.experimental.llm.KvSnapshot;
import io.github.zzuegg.brainiac.experimental.llm.LogitMath;
import io.github.zzuegg.brainiac.experimental.llm.PagedKvCache;
import io.github.zzuegg.brainiac.experimental.llm.PagedLlama;
import io.github.zzuegg.brainiac.experimental.llm.QuestKvStore;
import io.github.zzuegg.brainiac.experimental.llm.ReconstructionKvScorer;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Head-level KV compression on a synthetic two-layer, four-head attention stack whose heads
 * behave like the head types found in trained LLMs:
 * <ul>
 * <li>layer 0: two <em>diffuse</em> heads (attention spread over the whole context) and two
 *     <em>local</em> heads (a narrow window around their own position);</li>
 * <li>layer 1: two <em>retrieval</em> heads (a question's query matches the key of one
 *     "needle" token anywhere in the context and copies its value), one <em>sink</em> head
 *     (the first token plus a local window) and one local head.</li>
 * </ul>
 * Keys carry a rotary-style position (a 2-d angle), a sink flag and content; queries are built
 * so the softmax scores have the stated shapes. A context of {@code length} tokens holds
 * several needles and ends with an 8-token question about one of them (SnapKV's observation
 * window). Every policy keeps at most the same number of (layer, head, token) entries, except
 * Quest, which evicts nothing and reads a page budget per query, and DuoAttention, whose memory
 * follows from its head split. Scores: needle recall (cosine of a retrieval head's output with
 * the needle's value above 0.8), the relative error of all head outputs against the full
 * cache, and the entries kept (or read, for Quest) per head.
 *
 * <p>{@link #run} asks the prompt's own question, {@link #otherQuestions} asks about every other
 * needle after compressing with the prompt's question (KVzip's setting).
 */
public final class HeadBudgetComparison {

    /**
     * One policy's result.
     *
     * @param recall the share of (question, retrieval head) pairs that read the right needle
     * @param error the relative squared error of all head outputs against the full cache
     * @param entries the KV entries kept (for Quest: read per decode step), summed over heads
     */
    public record Outcome(double recall, double error, double entries) {
    }

    private static final int LAYERS = 2;
    private static final int HEADS = 4;
    private static final int DIM = 16;
    private static final int WINDOW = 8;
    private static final float SCALE = 0.25f;
    private static final int NEEDLES = 6;

    private enum Kind { DIFFUSE, LOCAL, SINK, RETRIEVAL }

    private static final Kind[][] KINDS = {
        {Kind.DIFFUSE, Kind.DIFFUSE, Kind.LOCAL, Kind.LOCAL},
        {Kind.RETRIEVAL, Kind.RETRIEVAL, Kind.SINK, Kind.LOCAL}};

    /** One synthetic context. */
    static final class Context {
        final int n;
        final float[][][][] keys;
        final float[][][][] values;
        final int[] needles;
        final float[][] directions;
        final SplittableRandom random;

        Context(int n, long seed) {
            this.n = n;
            this.random = new SplittableRandom(seed);
            keys = new float[LAYERS][HEADS][n][DIM];
            values = new float[LAYERS][HEADS][n][DIM];
            needles = new int[NEEDLES];
            directions = new float[NEEDLES][];
            List<Integer> spots = new ArrayList<>();
            int lo = 4;
            int hi = n - WINDOW - 16;
            while (spots.size() < NEEDLES) {
                int p = lo + random.nextInt(hi - lo);
                if (!spots.contains(p)) {
                    spots.add(p);
                }
            }
            for (int j = 0; j < NEEDLES; j++) {
                needles[j] = spots.get(j);
                directions[j] = unit(DIM - 3);
            }
            for (int l = 0; l < LAYERS; l++) {
                for (int h = 0; h < HEADS; h++) {
                    for (int t = 0; t < n; t++) {
                        float[] k = keys[l][h][t];
                        double angle = angle(t);
                        k[0] = (float) Math.cos(angle);
                        k[1] = (float) Math.sin(angle);
                        k[2] = t == 0 ? 1 : 0;
                        int needle = indexOf(t);
                        for (int c = 3; c < DIM; c++) {
                            k[c] = needle >= 0 && KINDS[l][h] == Kind.RETRIEVAL ? 6 * directions[needle][c - 3]
                                    : (float) (random.nextGaussian() * 0.3);
                        }
                        for (int c = 0; c < DIM; c++) {
                            values[l][h][t][c] = (float) random.nextGaussian();
                        }
                    }
                }
            }
        }

        double angle(int t) {
            return Math.PI / 2 * t / n;
        }

        int indexOf(int t) {
            for (int j = 0; j < NEEDLES; j++) {
                if (needles[j] == t) {
                    return j;
                }
            }
            return -1;
        }

        float[] unit(int d) {
            float[] u = new float[d];
            double s = 0;
            for (int i = 0; i < d; i++) {
                u[i] = (float) random.nextGaussian();
                s += u[i] * u[i];
            }
            for (int i = 0; i < d; i++) {
                u[i] /= (float) Math.sqrt(s);
            }
            return u;
        }

        /** A head's query at a position, asking about one needle (retrieval heads) or none. */
        float[] query(int l, int h, int position, int needle, double noise) {
            float[] q = new float[DIM];
            double angle = angle(position);
            double kappa = switch (KINDS[l][h]) {
                case DIFFUSE -> 3;
                case LOCAL, SINK -> 300;
                case RETRIEVAL -> 0;
            };
            q[0] = (float) (kappa * Math.cos(angle));
            q[1] = (float) (kappa * Math.sin(angle));
            q[2] = KINDS[l][h] == Kind.SINK ? 4 : 0;
            if (KINDS[l][h] == Kind.RETRIEVAL && needle >= 0) {
                for (int c = 3; c < DIM; c++) {
                    q[c] = 1.5f * directions[needle][c - 3];
                }
            }
            for (int c = 3; c < DIM; c++) {
                q[c] += (float) (random.nextGaussian() * noise);
            }
            for (int c = 0; c < DIM; c++) {
                q[c] /= SCALE;   // so that SCALE * q.k is the score designed above
            }
            return q;
        }

        /** The reconstruction pass's query when the model re-reads token t: it looks for t itself. */
        float[] reconstruction(int l, int h, int t) {
            float[] q = query(l, h, t, -1, 0);
            if (KINDS[l][h] == Kind.RETRIEVAL) {
                float[] k = keys[l][h][t];
                double s = 0;
                for (int c = 3; c < DIM; c++) {
                    s += k[c] * k[c];
                }
                for (int c = 3; c < DIM; c++) {
                    q[c] = (float) (1.5 * k[c] / Math.sqrt(s) / SCALE);
                }
            }
            return q;
        }

        /** Keys with the observation window's queries: the question about one needle. */
        KvSnapshot window(int needle) {
            float[][][][] q = new float[LAYERS][HEADS][WINDOW][];
            for (int l = 0; l < LAYERS; l++) {
                for (int h = 0; h < HEADS; h++) {
                    for (int i = 0; i < WINDOW; i++) {
                        q[l][h][i] = query(l, h, n - WINDOW + i, needle, 0.05);
                    }
                }
            }
            return new KvSnapshot(keys, q, SCALE);
        }

        KvSnapshot reconstructionPass() {
            float[][][][] q = new float[LAYERS][HEADS][n][];
            for (int l = 0; l < LAYERS; l++) {
                for (int h = 0; h < HEADS; h++) {
                    for (int t = 0; t < n; t++) {
                        q[l][h][t] = reconstruction(l, h, t);
                    }
                }
            }
            return new KvSnapshot(keys, q, SCALE);
        }

        float[] attend(int l, int h, float[] q, boolean[] keep) {
            float[] w = KvSnapshot.softmax(q, keys[l][h], keep, SCALE);
            float[] o = new float[DIM];
            for (int t = 0; t < n; t++) {
                if (w[t] != 0) {
                    for (int c = 0; c < DIM; c++) {
                        o[c] += w[t] * values[l][h][t][c];
                    }
                }
            }
            return o;
        }
    }

    private HeadBudgetComparison() {
    }

    /** H2O as a head-level policy: cumulative causal attention of every prompt query, plus the window. */
    private static KvBudgetPolicy heavyHitters(Context c) {
        return (s, budget) -> {
            int per = budget / (LAYERS * HEADS);
            boolean[][][] out = new boolean[LAYERS][HEADS][c.n];
            for (int l = 0; l < LAYERS; l++) {
                for (int h = 0; h < HEADS; h++) {
                    double[] score = new double[c.n];
                    boolean[] causal = new boolean[c.n];
                    for (int t = 0; t < c.n; t++) {
                        causal[t] = true;
                        float[] q = t >= c.n - WINDOW ? s.queries()[l][h][t - (c.n - WINDOW)]
                                : c.query(l, h, t, -1, 0.05);
                        float[] w = KvSnapshot.softmax(q, c.keys[l][h], causal, SCALE);
                        for (int j = 0; j <= t; j++) {
                            score[j] += w[j];
                        }
                    }
                    int recent = per / 2;
                    for (int t = c.n - recent; t < c.n; t++) {
                        out[l][h][t] = true;
                    }
                    Integer[] order = new Integer[c.n - recent];
                    for (int i = 0; i < order.length; i++) {
                        order[i] = i;
                    }
                    java.util.Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
                    for (int i = 0; i < per - recent; i++) {
                        out[l][h][order[i]] = true;
                    }
                }
            }
            return out;
        };
    }

    /** StreamingLLM on every head: sinks plus the most recent tokens. */
    private static KvBudgetPolicy streaming(int sinks) {
        return (s, budget) -> {
            int per = budget / (s.layers() * s.heads());
            boolean[][][] out = new boolean[s.layers()][s.heads()][s.length()];
            for (boolean[][] layer : out) {
                for (boolean[] head : layer) {
                    for (int t = 0; t < s.length(); t++) {
                        head[t] = t < sinks || t >= s.length() - (per - sinks);
                    }
                }
            }
            return out;
        };
    }

    /**
     * DuoAttention's gates from calibration contexts (other seeds), then the cache that keeps a
     * quarter of the heads whole.
     *
     * @param length the context length
     * @param seed the seed
     * @param window the streaming heads' window
     * @return the policy
     */
    static DuoHeadCache duo(int length, long seed, int window) {
        List<float[][][]> full = new ArrayList<>();
        List<float[][][]> stream = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            Context c = new Context(length, seed + 1000 + i);
            boolean[] sinkWindow = new boolean[length];
            for (int t = 0; t < length; t++) {
                sinkWindow[t] = t < 4 || t >= length - window;
            }
            for (int j = 0; j < NEEDLES; j += 2) {
                float[][][] f = new float[LAYERS][HEADS][];
                float[][][] s = new float[LAYERS][HEADS][];
                for (int l = 0; l < LAYERS; l++) {
                    for (int h = 0; h < HEADS; h++) {
                        float[] q = c.query(l, h, length, j, 0.05);
                        f[l][h] = c.attend(l, h, q, null);
                        s[l][h] = c.attend(l, h, q, sinkWindow);
                    }
                }
                full.add(f);
                stream.add(s);
            }
        }
        double[][] gates = DuoHeadCache.gates(full, stream, null, 0.05f, 150);
        return DuoHeadCache.identify(gates, 0.25, 4, window);
    }

    /**
     * Compresses each context with every policy, then asks questions.
     *
     * @param length the context length
     * @param budgetShare the kept entries as a share of the full cache
     * @param contexts the number of contexts
     * @param otherNeedles whether to ask about the needles the prompt's question did not name
     * @param seed the seed
     * @return per policy its outcome, full cache first
     */
    static Map<String, Outcome> evaluate(int length, double budgetShare, int contexts, boolean otherNeedles,
            long seed) {
        int budget = (int) (budgetShare * LAYERS * HEADS * length);
        int perHead = budget / (LAYERS * HEADS);
        DuoHeadCache duo = duo(length, seed, Math.max(8, perHead / 2));
        Map<String, double[]> sums = new LinkedHashMap<>();
        for (int i = 0; i < contexts; i++) {
            Context c = new Context(length, seed + i);
            int asked = i % NEEDLES;
            KvSnapshot window = c.window(asked);
            Map<String, boolean[][][]> masks = new LinkedHashMap<>();
            boolean[][][] all = new boolean[LAYERS][HEADS][length];
            for (boolean[][] layer : all) {
                for (boolean[] head : layer) {
                    java.util.Arrays.fill(head, true);
                }
            }
            masks.put("full", all);
            masks.put("streaming (all heads)", streaming(4).select(window, budget));
            masks.put("h2o", heavyHitters(c).select(window, budget));
            masks.put("snapkv (uniform)", KvBudgetPolicy.uniform(WINDOW, 7).select(window, budget));
            masks.put("pyramidkv", KvBudgetPolicy.pyramid(WINDOW, 7, 3).select(window, budget));
            masks.put("ada-kv", KvBudgetPolicy.adaptive(WINDOW, 7, 0.2).select(window, budget));
            masks.put("kvzip", new ReconstructionKvScorer(WINDOW).select(c.reconstructionPass(), budget));
            masks.put("duoattention", duo.select(window, budget));
            masks.put("quest", null);
            for (int j = 0; j < NEEDLES; j++) {
                if (otherNeedles == (j == asked)) {
                    continue;
                }
                float[][][] q = new float[LAYERS][HEADS][];
                float[][][] reference = new float[LAYERS][HEADS][];
                for (int l = 0; l < LAYERS; l++) {
                    for (int h = 0; h < HEADS; h++) {
                        q[l][h] = c.query(l, h, length, j, 0.05);
                        reference[l][h] = c.attend(l, h, q[l][h], null);
                    }
                }
                for (Map.Entry<String, boolean[][][]> e : masks.entrySet()) {
                    double num = 0;
                    double den = 0;
                    double hits = 0;
                    double entries = 0;
                    for (int l = 0; l < LAYERS; l++) {
                        for (int h = 0; h < HEADS; h++) {
                            boolean[] keep = e.getValue() == null
                                    ? QuestKvStore.selectPages(c.keys[l][h], q[l][h], 16, Math.max(1, perHead / 16))
                                    : e.getValue()[l][h];
                            float[] o = c.attend(l, h, q[l][h], keep);
                            for (int d = 0; d < DIM; d++) {
                                num += (o[d] - reference[l][h][d]) * (o[d] - reference[l][h][d]);
                                den += reference[l][h][d] * reference[l][h][d];
                            }
                            for (boolean b : keep) {
                                entries += b ? 1 : 0;
                            }
                            if (KINDS[l][h] == Kind.RETRIEVAL) {
                                hits += cosine(o, c.values[l][h][c.needles[j]]) > 0.8 ? 0.5 : 0;
                            }
                        }
                    }
                    double[] acc = sums.computeIfAbsent(e.getKey(), k -> new double[4]);
                    acc[0] += hits;
                    acc[1] += num / den;
                    acc[2] += entries;
                    acc[3]++;
                }
            }
        }
        Map<String, Outcome> out = new LinkedHashMap<>();
        sums.forEach((k, v) -> out.put(k, new Outcome(v[0] / v[3], v[1] / v[3], v[2] / v[3])));
        return out;
    }

    /**
     * The prompt's own question after compression.
     *
     * @param length the context length
     * @param budgetShare the kept share of the full cache
     * @param seed the seed
     * @return per policy its outcome
     */
    public static Map<String, Outcome> run(int length, double budgetShare, long seed) {
        return evaluate(length, budgetShare, 12, false, seed);
    }

    /**
     * Questions about the other needles, after compressing with the prompt's question.
     *
     * @param length the context length
     * @param budgetShare the kept share of the full cache
     * @param seed the seed
     * @return per policy its outcome
     */
    public static Map<String, Outcome> otherQuestions(int length, double budgetShare, long seed) {
        return evaluate(length, budgetShare, 6, true, seed);
    }

    /**
     * Quest on a real decoder: a random {@link Llama} with sharpened attention (query and key
     * weights scaled up, so decode steps lean on a few far-back tokens) reads a prompt and then
     * a fixed continuation through {@link QuestKvStore} over a {@link PagedKvCache}.
     *
     * @param prompt the prompt length
     * @param decode the tokens decoded
     * @param pageSize the slots per page (and per block of the paged cache)
     * @param pages the pages each decode step reads
     * @param seed the seed
     * @return per mode the share of steps whose top token equals full attention's, and the
     *     bytes read as a share of full attention's: {@code {agreement, bytesShare}}
     */
    public static Map<String, double[]> questOnLlama(int prompt, int decode, int pageSize, int pages, long seed) {
        Llama core = Llama.random(LlamaConfig.small(64, 64, 2, 4, 2), seed, Device.cpu(), DType.FLOAT32);
        for (Map.Entry<String, Var> e : core.namedParameters().entrySet()) {
            if (e.getKey().endsWith("q_proj.weight") || e.getKey().endsWith("k_proj.weight")) {
                e.getValue().assign(e.getValue().value().mul(20f));
            }
        }
        SplittableRandom r = new SplittableRandom(seed);
        int[] text = new int[prompt + decode];
        for (int i = 0; i < text.length; i++) {
            text[i] = r.nextInt(64);
        }
        LlamaConfig c = core.config();
        Map<String, List<float[]>> logits = new LinkedHashMap<>();
        Map<String, double[]> out = new LinkedHashMap<>();
        for (QuestKvStore.Mode mode : QuestKvStore.Mode.values()) {
            QuestKvStore store = new QuestKvStore(PagedKvCache.forModel(c, pageSize, 1024 / pageSize, Device.cpu()), pageSize, pages,
                    c.kvHeads(), c.headDim(), mode);
            PagedLlama model = new PagedLlama(core, store);
            LanguageModel.State s = model.newState(1);
            model.next(s, new int[][] {java.util.Arrays.copyOf(text, prompt)}, null);
            long before = store.bytesRead();
            long fullBefore = store.fullBytesRead();
            List<float[]> steps = new ArrayList<>();
            for (int t = prompt; t < text.length; t++) {
                steps.add(model.next(s, new int[][] {{text[t]}}, null).toArray());
            }
            String name = mode.name().toLowerCase(java.util.Locale.ROOT);
            logits.put(name, steps);
            out.put(name, new double[] {0, (store.bytesRead() - before) / (double) (store.fullBytesRead() - fullBefore)});
        }
        List<float[]> reference = logits.get("all");
        for (Map.Entry<String, List<float[]>> e : logits.entrySet()) {
            int same = 0;
            for (int i = 0; i < reference.size(); i++) {
                same += LogitMath.argmax(reference.get(i)) == LogitMath.argmax(e.getValue().get(i)) ? 1 : 0;
            }
            out.get(e.getKey())[0] = same / (double) reference.size();
        }
        return out;
    }

    private static double cosine(float[] a, float[] b) {
        double d = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            d += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return d / Math.sqrt(na * nb + 1e-12);
    }
}
