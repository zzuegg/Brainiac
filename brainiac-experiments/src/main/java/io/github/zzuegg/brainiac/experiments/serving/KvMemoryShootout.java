package io.github.zzuegg.brainiac.experiments.serving;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.KvBlockStore;
import io.github.zzuegg.brainiac.experimental.llm.PagedLlama;
import io.github.zzuegg.brainiac.experimental.llm.QuestKvStore;
import io.github.zzuegg.brainiac.experimental.serving.CrossLayerKvMerger;
import io.github.zzuegg.brainiac.experimental.serving.LshSampledAttention;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.BiFunction;

/**
 * Serving shootout, KV-memory and sampled-attention rows (8.4.3 and 8.4.4).
 *
 * <p>{@link #crossLayerMerging}: a toy Llama (six layers) whose deep layers are made alike, as
 * the layers of trained LLMs are (layers 2 to 5 share layer 2's attention and MLP weights plus
 * 10% noise, and residual branches are scaled down), reads a prompt; its cache is then
 * compressed and the model decodes a teacher-forced continuation. Rows: the full cache;
 * {@link CrossLayerKvMerger} (MiniCache) on pairs (2,3) and (4,5); KIVI 2-bit (per-channel keys
 * over groups of 32 tokens, per-token values, the last 32 tokens in full precision, the recipe
 * of {@code QuantizedKvCache}); MiniCache + KIVI; and naive training-free layer sharing (layer
 * {@code l + 1} reuses layer {@code l}'s KV, what CLA does once trained). Reported: KV bytes per
 * token, mean KL from the full cache's next-token distribution and top-1 agreement.
 *
 * <p>{@link #sampledAttention}: one head over 1024 cached tokens. On the <em>needle</em> task
 * one key matches the query and the others are noise (sparse attention); on the
 * <em>aggregation</em> task the query is weakly aligned with every key (dense attention, the
 * output is a weighted mean of many values). {@link LshSampledAttention} (MagicPIG) is compared
 * at equal keys touched with exact top-k, Quest pages ({@link QuestKvStore#selectPages}) and a
 * streaming head (sinks plus a recent window, DuoAttention's non-retrieval heads).
 */
public final class KvMemoryShootout {

    private KvMemoryShootout() {
    }

    /**
     * One KV compression method.
     *
     * @param bytesPerToken KV bytes per token, all layers
     * @param kl mean KL(full || method) of the next-token distributions
     * @param top1 share of continuation steps whose top token matches the full cache's
     * @param retention MiniCache's retained share (0 elsewhere)
     */
    public record MemoryRow(double bytesPerToken, double kl, double top1, double retention) {
    }

    /**
     * Builds the toy Llama with alike deep layers.
     *
     * @param seed the seed
     * @return the model
     */
    static Llama alikeLlama(long seed) {
        Llama core = Llama.random(LlamaConfig.small(64, 64, 6, 4, 2), seed, Device.cpu(), DType.FLOAT32);
        Map<String, Var> named = core.namedParameters();
        SplittableRandom r = new SplittableRandom(seed + 5);
        String[] mats = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"};
        for (int l = 0; l < 6; l++) {
            for (String m : mats) {
                Var w = named.get("model.layers." + l + "." + m);
                Tensor base = l >= 3 ? named.get("model.layers.2." + m).value() : w.value();
                float[] f = base.toArray();
                double std = 0;
                for (float x : f) {
                    std += x * x;
                }
                std = Math.sqrt(std / f.length);
                for (int i = 0; i < f.length && l >= 3; i++) {
                    f[i] += (float) (0.1 * std * r.nextGaussian());
                }
                float mult = m.contains("o_proj") || m.contains("down_proj") ? 0.5f
                        : m.contains("q_proj") || m.contains("k_proj") ? 3f : 1f;
                for (int i = 0; i < f.length; i++) {
                    f[i] *= mult;
                }
                w.assign(Tensor.of(base.shape(), f));
            }
        }
        Var norm = named.get("model.norm.weight");
        norm.assign(norm.value().mul(6f));   // sharper next-token distributions
        return core;
    }

    /**
     * Runs the KV-memory rows.
     *
     * @param prompt the prompt length
     * @param decode the continuation length
     * @param seeds the seeds
     * @return method to row, averaged over seeds
     */
    public static Map<String, MemoryRow> crossLayerMerging(int prompt, int decode, long[] seeds) {
        CrossLayerKvMerger mini = CrossLayerKvMerger.of(0.4);
        Map<String, BiFunction<KvBlockStore, Integer, double[]>> methods = new LinkedHashMap<>();
        methods.put("full cache", (s, q) -> new double[] {1, 0});
        methods.put("MiniCache (2,3),(4,5)", (s, q) -> {
            CrossLayerKvMerger.Stats st = mini.compress(s, q, 2);
            return new double[] {st.ratio(), st.retentionShare()};
        });
        methods.put("KIVI 2-bit", (s, q) -> new double[] {kivi(s, q, 0), 0});
        methods.put("MiniCache + KIVI 2-bit", (s, q) -> {
            CrossLayerKvMerger.Stats st = mini.compress(s, q, 2);
            double k = kivi(s, q, 0);
            return new double[] {st.ratio() * k, st.retentionShare()};
        });
        methods.put("naive layer sharing (untrained CLA)", (s, q) -> new double[] {share(s, q, 2), 0});
        Map<String, double[]> sums = new LinkedHashMap<>();
        for (String m : methods.keySet()) {
            sums.put(m, new double[4]);
        }
        for (long seed : seeds) {
            Llama core = alikeLlama(seed);
            LlamaConfig c = core.config();
            double fullBytes = 2.0 * c.layers() * c.kvHeads() * c.headDim() * Float.BYTES;
            SplittableRandom r = new SplittableRandom(seed);
            int[] text = new int[prompt + decode];
            for (int i = 0; i < text.length; i++) {
                text[i] = r.nextInt(64);
            }
            List<float[]> reference = null;
            for (Map.Entry<String, BiFunction<KvBlockStore, Integer, double[]>> e : methods.entrySet()) {
                PagedLlama paged = PagedLlama.of(core, 16, 64);
                KvBlockStore store = paged.store();
                int seq = store.newSequence();
                paged.forward(List.of(PagedLlama.Pass.chain(seq, Arrays.copyOf(text, prompt))), false);
                double[] info = e.getValue().apply(store, seq);
                List<float[]> logits = new ArrayList<>();
                for (int t = prompt; t < text.length; t++) {
                    logits.add(paged.forward(List.of(PagedLlama.Pass.chain(seq, text[t])), false).getFirst()
                            .row(0, 64));
                }
                if (reference == null) {
                    reference = logits;
                }
                double kl = 0;
                int same = 0;
                for (int i = 0; i < logits.size(); i++) {
                    kl += kl(reference.get(i), logits.get(i));
                    same += argmax(reference.get(i)) == argmax(logits.get(i)) ? 1 : 0;
                }
                double[] s = sums.get(e.getKey());
                s[0] += info[0] * fullBytes;
                s[1] += kl / logits.size();
                s[2] += same / (double) logits.size();
                s[3] += info[1];
            }
        }
        Map<String, MemoryRow> out = new LinkedHashMap<>();
        for (Map.Entry<String, double[]> e : sums.entrySet()) {
            double[] s = e.getValue();
            int n = seeds.length;
            out.put(e.getKey(), new MemoryRow(s[0] / n, s[1] / n, s[2] / n, s[3] / n));
        }
        return out;
    }

    /**
     * KIVI 2-bit round trip on every layer of a sequence (keys per channel over groups of 32
     * tokens, values per token, the last 32 tokens kept in full precision).
     *
     * @return stored bytes over float32 bytes
     */
    static double kivi(KvBlockStore store, int seq, int from) {
        Tensor dummy = Tensor.zeros(Shape.of(1, 1, 1));
        int group = 32;
        int bits = 2;
        double stored = 0;
        double full = 0;
        for (int l = from; l < store.layers(); l++) {
            KvBlockStore.View v = store.view(seq, l, dummy);
            Shape shape = v.keys().shape();
            int n = shape.dim(0);
            int heads = shape.dim(1);
            int d = shape.dim(2);
            float[] k = v.keys().toArray();
            float[] val = v.values().toArray();
            int quantized = Math.max(0, (n - group) / group * group);
            int width = heads * d;
            for (int g = 0; g < quantized; g += group) {
                for (int c = 0; c < width; c++) {   // keys: one range per channel per group
                    float lo = Float.POSITIVE_INFINITY;
                    float hi = Float.NEGATIVE_INFINITY;
                    for (int t = g; t < g + group; t++) {
                        lo = Math.min(lo, k[t * width + c]);
                        hi = Math.max(hi, k[t * width + c]);
                    }
                    for (int t = g; t < g + group; t++) {
                        k[t * width + c] = roundTrip(k[t * width + c], lo, hi, bits);
                    }
                }
            }
            for (int t = 0; t < quantized; t++) {   // values: one range per token and head
                for (int h = 0; h < heads; h++) {
                    float lo = Float.POSITIVE_INFINITY;
                    float hi = Float.NEGATIVE_INFINITY;
                    for (int c = 0; c < d; c++) {
                        lo = Math.min(lo, val[t * width + h * d + c]);
                        hi = Math.max(hi, val[t * width + h * d + c]);
                    }
                    for (int c = 0; c < d; c++) {
                        val[t * width + h * d + c] = roundTrip(val[t * width + h * d + c], lo, hi, bits);
                    }
                }
            }
            store.write(seq, l, 0, Tensor.of(shape, k), Tensor.of(shape, val));
            double numbers = 2.0 * quantized * width;
            stored += numbers * bits / 8.0 + 4.0 * (quantized / group * width + quantized * heads)
                    + 2.0 * (n - quantized) * width * 4;
            full += 2.0 * n * width * 4;
        }
        return stored / full;
    }

    private static float roundTrip(float x, float lo, float hi, int bits) {
        int levels = (1 << bits) - 1;
        if (hi <= lo) {
            return x;
        }
        float scale = (hi - lo) / levels;
        return lo + Math.round((x - lo) / scale) * scale;
    }

    /** Layer {@code l + 1} reuses layer {@code l}'s KV for the pairs from {@code from}. */
    static double share(KvBlockStore store, int seq, int from) {
        Tensor dummy = Tensor.zeros(Shape.of(1, 1, 1));
        int layers = store.layers();
        int pairs = 0;
        for (int l = from; l + 1 < layers; l += 2) {
            KvBlockStore.View v = store.view(seq, l, dummy);
            store.write(seq, l + 1, 0, v.keys(), v.values());
            pairs++;
        }
        return (layers - pairs) / (double) layers;
    }

    static double kl(float[] p, float[] q) {
        double[] a = softmax(p);
        double[] b = softmax(q);
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * (Math.log(a[i] + 1e-12) - Math.log(b[i] + 1e-12));
        }
        return s;
    }

    static double[] softmax(float[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : x) {
            max = Math.max(max, v);
        }
        double[] e = new double[x.length];
        double z = 0;
        for (int i = 0; i < x.length; i++) {
            e[i] = Math.exp(x[i] - max);
            z += e[i];
        }
        for (int i = 0; i < x.length; i++) {
            e[i] /= z;
        }
        return e;
    }

    static int argmax(float[] x) {
        int b = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[i] > x[b]) {
                b = i;
            }
        }
        return b;
    }

    // ------------------------------------------------------------------ 8.4.4 sampled attention

    /**
     * One attention method on one task.
     *
     * @param relativeError mean {@code |o - o*| / |o*|} against exact attention
     * @param accuracy share of queries whose output is within 25% relative error (aggregation)
     *     or points at the needle's value with cosine above 0.8 (needle)
     * @param keysTouched mean keys whose logits were computed per query
     */
    public record AttentionRow(double relativeError, double accuracy, double keysTouched) {
    }

    /**
     * Runs the sampled-attention rows on both tasks.
     *
     * @param n cached tokens
     * @param queries queries per task and seed
     * @param seeds the seeds
     * @param bits SimHash bits per table
     * @param tables hash tables
     * @param collisions tables a key must share with the query
     * @return task name to method to row
     */
    public static Map<String, Map<String, AttentionRow>> sampledAttention(int n, int queries, long[] seeds, int bits,
            int tables, int collisions) {
        int d = 32;
        Map<String, Map<String, double[]>> sums = new LinkedHashMap<>();
        for (String task : List.of("needle", "aggregation")) {
            Map<String, double[]> m = new LinkedHashMap<>();
            for (String name : List.of("exact", "MagicPIG (LSH sampling)", "top-k (same keys)", "Quest pages (same keys)",
                    "streaming head (same keys)")) {
                m.put(name, new double[3]);
            }
            sums.put(task, m);
            for (long seed : seeds) {
                SplittableRandom r = new SplittableRandom(seed * 31 + task.length());
                float[][] keys = new float[n][d];
                float[][] values = new float[n][d];
                for (int i = 0; i < n; i++) {
                    keys[i] = unit(d, r);
                    for (int c = 0; c < d; c++) {
                        values[i][c] = (float) r.nextGaussian();
                    }
                }
                LshSampledAttention lsh = LshSampledAttention.of(d, bits, tables, collisions, 4, 16, seed);
                for (int qi = 0; qi < queries; qi++) {
                    float[] q;
                    int needle = 4 + r.nextInt(n - 40);
                    float scale;
                    if (task.equals("needle")) {
                        q = keys[needle].clone();
                        for (int c = 0; c < d; c++) {
                            q[c] += (float) (0.3 * r.nextGaussian() / Math.sqrt(d));
                        }
                        scale = 20;
                    } else {
                        // dense: every key gets some weight, and values follow the keys, so
                        // dropping the low-weight tail biases the mean
                        q = unit(d, r);
                        for (int i = 0; i < n; i++) {
                            System.arraycopy(keys[i], 0, values[i], 0, d);
                        }
                        scale = 8;
                    }
                    LshSampledAttention.Index index = lsh.index(keys, values);
                    float[] exact = attend(q, keys, values, scale, null);
                    float[] approx = index.attend(q, scale);
                    int budget = (int) Math.max(1, Math.round(index.keysTouchedPerQuery()));
                    Map<String, float[]> outs = new LinkedHashMap<>();
                    Map<String, Double> touched = new LinkedHashMap<>();
                    outs.put("exact", exact);
                    touched.put("exact", (double) n);
                    outs.put("MagicPIG (LSH sampling)", approx);
                    touched.put("MagicPIG (LSH sampling)", index.keysTouchedPerQuery());
                    outs.put("top-k (same keys)", attend(q, keys, values, scale, topK(q, keys, budget)));
                    touched.put("top-k (same keys)", (double) budget);
                    outs.put("Quest pages (same keys)", attend(q, keys, values, scale,
                            QuestKvStore.selectPages(keys, q, 16, Math.max(1, budget / 16))));
                    touched.put("Quest pages (same keys)", (double) Math.max(1, budget / 16) * 16);
                    boolean[] stream = new boolean[n];
                    for (int i = 0; i < n; i++) {
                        stream[i] = i < 4 || i >= n - Math.max(1, budget - 4);
                    }
                    outs.put("streaming head (same keys)", attend(q, keys, values, scale, stream));
                    touched.put("streaming head (same keys)", (double) budget);
                    for (Map.Entry<String, float[]> e : outs.entrySet()) {
                        double err = dist(e.getValue(), exact) / norm(exact);
                        boolean ok = task.equals("needle") ? cosine(e.getValue(), values[needle]) > 0.8 : err < 0.25;
                        double[] s = m.get(e.getKey());
                        s[0] += err;
                        s[1] += ok ? 1 : 0;
                        s[2] += touched.get(e.getKey());
                    }
                }
            }
        }
        Map<String, Map<String, AttentionRow>> out = new LinkedHashMap<>();
        double total = (double) queries * seeds.length;
        for (Map.Entry<String, Map<String, double[]>> t : sums.entrySet()) {
            Map<String, AttentionRow> rows = new LinkedHashMap<>();
            for (Map.Entry<String, double[]> e : t.getValue().entrySet()) {
                double[] s = e.getValue();
                rows.put(e.getKey(), new AttentionRow(s[0] / total, s[1] / total, s[2] / total));
            }
            out.put(t.getKey(), rows);
        }
        return out;
    }

    private static boolean[] topK(float[] q, float[][] keys, int k) {
        Integer[] order = new Integer[keys.length];
        double[] s = new double[keys.length];
        for (int i = 0; i < keys.length; i++) {
            order[i] = i;
            s[i] = dot(q, keys[i]);
        }
        Arrays.sort(order, (a, b) -> Double.compare(s[b], s[a]));
        boolean[] keep = new boolean[keys.length];
        for (int i = 0; i < Math.min(k, keys.length); i++) {
            keep[order[i]] = true;
        }
        return keep;
    }

    static float[] attend(float[] q, float[][] keys, float[][] values, float scale, boolean[] keep) {
        double max = Double.NEGATIVE_INFINITY;
        double[] s = new double[keys.length];
        for (int i = 0; i < keys.length; i++) {
            if (keep == null || keep[i]) {
                s[i] = scale * dot(q, keys[i]);
                max = Math.max(max, s[i]);
            }
        }
        double[] acc = new double[values[0].length];
        double z = 0;
        for (int i = 0; i < keys.length; i++) {
            if (keep == null || keep[i]) {
                double w = Math.exp(s[i] - max);
                z += w;
                for (int c = 0; c < acc.length; c++) {
                    acc[c] += w * values[i][c];
                }
            }
        }
        float[] o = new float[acc.length];
        for (int c = 0; c < o.length; c++) {
            o[c] = (float) (acc[c] / z);
        }
        return o;
    }

    private static float[] unit(int d, SplittableRandom r) {
        float[] u = new float[d];
        double n = 0;
        for (int c = 0; c < d; c++) {
            u[c] = (float) r.nextGaussian();
            n += u[c] * u[c];
        }
        for (int c = 0; c < d; c++) {
            u[c] /= (float) Math.sqrt(n);
        }
        return u;
    }

    private static double dot(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static double norm(float[] a) {
        return Math.sqrt(dot(a, a));
    }

    private static double dist(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return Math.sqrt(s);
    }

    private static double cosine(float[] a, float[] b) {
        return dot(a, b) / (norm(a) * norm(b) + 1e-12);
    }
}
