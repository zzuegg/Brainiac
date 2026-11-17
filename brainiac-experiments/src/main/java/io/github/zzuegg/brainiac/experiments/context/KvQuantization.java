package io.github.zzuegg.brainiac.experiments.context;

import io.github.zzuegg.brainiac.experimental.nn.EvictingKvCache;
import io.github.zzuegg.brainiac.experimental.nn.QuantizedKvCache;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * KV-cache compression by quantization against compression by eviction, at equal memory.
 * One attention layer decodes a long synthetic stream whose keys have the outlier channels
 * real models show (a few channels large for every token), and whose queries half the time
 * look up one random earlier token (retrieval) and otherwise are noise. Each method's
 * attention output is compared with float32 attention over the whole cache:
 * {@link QuantizedKvCache} (KIVI) at 2 and 4 bits, 2-bit with per-token keys (the ablation),
 * and H2O heavy-hitter eviction ({@link EvictingKvCache#heavyHitters}) given exactly the bytes
 * 2-bit KIVI holds.
 */
public final class KvQuantization {

    /**
     * One method's outcome.
     *
     * @param relativeError the mean relative L2 error of the attention output over the scored steps
     * @param bytes the bytes held at the end
     */
    public record Outcome(double relativeError, long bytes) {
    }

    private KvQuantization() {
    }

    /**
     * Runs every method on the same stream.
     *
     * @param steps tokens decoded
     * @param seed the seed
     * @return method name to outcome, float32 first
     */
    public static Map<String, Outcome> run(int steps, long seed) {
        int kvHeads = 2;
        int heads = 4;
        int dim = 32;
        int width = kvHeads * dim;
        float scale = (float) (1 / Math.sqrt(dim));
        SplittableRandom r = new SplittableRandom(seed);
        float[] channelScale = new float[width];
        float[] channelShift = new float[width];
        for (int c = 0; c < width; c++) {
            boolean outlier = c % dim < 3;   // three outlier channels per head
            channelScale[c] = outlier ? 6 : 1;
            channelShift[c] = outlier ? (float) (8 * r.nextGaussian()) : 0;
        }
        float[][] k = new float[steps][width];
        float[][] v = new float[steps][width];
        float[][] q = new float[steps][heads * dim];
        for (int t = 0; t < steps; t++) {
            for (int c = 0; c < width; c++) {
                k[t][c] = (float) (r.nextGaussian() * channelScale[c] + channelShift[c]);
                v[t][c] = (float) r.nextGaussian();
            }
            int source = t > 0 && r.nextBoolean() ? r.nextInt(t) : -1;
            for (int h = 0; h < heads; h++) {
                int kh = h / (heads / kvHeads);
                for (int c = 0; c < dim; c++) {
                    int ch = kh * dim + c;
                    boolean outlier = c < 3;
                    float look = source >= 0 && !outlier ? 2.5f * (k[source][ch] - channelShift[ch]) : 0;
                    q[t][h * dim + c] = (float) (look + 0.5 * r.nextGaussian());
                }
            }
        }
        int from = steps / 4;   // score once the cache is long
        QuantizedKvCache full = new QuantizedKvCache(kvHeads, dim, 8, 1, Integer.MAX_VALUE / 2);
        Map<String, QuantizedKvCache> quant = new LinkedHashMap<>();
        quant.put("KIVI 2-bit", new QuantizedKvCache(kvHeads, dim, 2, 32, 32));
        quant.put("KIVI 4-bit", new QuantizedKvCache(kvHeads, dim, 4, 32, 32));
        quant.put("2-bit, per-token keys", new QuantizedKvCache(kvHeads, dim, 2, 32, 32)
                .keyAxis(QuantizedKvCache.KeyAxis.TOKEN));
        Map<String, double[]> err = new LinkedHashMap<>();
        for (String name : quant.keySet()) {
            err.put(name, new double[1]);
        }
        List<float[][]> exactOut = new ArrayList<>();
        for (int t = 0; t < steps; t++) {
            full.append(k[t], v[t]);
            float[] ref = full.attend(q[t], scale);
            exactOut.add(new float[][] {ref});
            for (Map.Entry<String, QuantizedKvCache> e : quant.entrySet()) {
                e.getValue().append(k[t], v[t]);
                if (t >= from) {
                    err.get(e.getKey())[0] += relative(e.getValue().attend(q[t], scale), ref);
                }
            }
        }
        Map<String, Outcome> out = new LinkedHashMap<>();
        out.put("float32", new Outcome(0, QuantizedKvCache.float32Bytes(steps, kvHeads, dim)));
        for (Map.Entry<String, QuantizedKvCache> e : quant.entrySet()) {
            out.put(e.getKey(), new Outcome(err.get(e.getKey())[0] / (steps - from), e.getValue().bytes()));
        }
        long kiviBytes = quant.get("KIVI 2-bit").bytes();
        int budget = (int) (kiviBytes / QuantizedKvCache.float32Bytes(1, kvHeads, dim));
        out.put("H2O at KIVI-2 bytes", heavyHitters(k, v, q, exactOut, budget, from, scale, kvHeads, dim));
        return out;
    }

    /** H2O: keeps {@code budget} float32 tokens, half of them the latest, the rest the most attended. */
    private static Outcome heavyHitters(float[][] k, float[][] v, float[][] q, List<float[][]> exact, int budget,
            int from, float scale, int kvHeads, int dim) {
        EvictingKvCache.EvictionPolicy policy = EvictingKvCache.heavyHitters(budget / 2);
        List<Integer> kept = new ArrayList<>();
        List<Float> attention = new ArrayList<>();
        double err = 0;
        int steps = k.length;
        for (int t = 0; t < steps; t++) {
            kept.add(t);
            attention.add(0f);
            while (kept.size() > budget) {
                List<EvictingKvCache.SlotInfo> info = new ArrayList<>();
                for (int i = 0; i < kept.size(); i++) {
                    int p = kept.get(i);
                    info.add(new EvictingKvCache.SlotInfo(p, t - p, attention.get(i),
                            attention.get(i) / Math.max(1, t - p), 0, 0, 0, kept.size()));
                }
                int drop = policy.evict(info);
                kept.remove(drop);
                attention.remove(drop);
            }
            QuantizedKvCache window = new QuantizedKvCache(kvHeads, dim, 8, 1, Integer.MAX_VALUE / 2);
            for (int p : kept) {
                window.append(k[p], v[p]);
            }
            float[] got = window.attend(q[t], scale);
            float[] w = weights(k, q[t], kept, scale, kvHeads, dim);
            for (int i = 0; i < kept.size(); i++) {
                attention.set(i, attention.get(i) + w[i]);
            }
            if (t >= from) {
                err += relative(got, exact.get(t)[0]);
            }
        }
        return new Outcome(err / (steps - from), QuantizedKvCache.float32Bytes(budget, kvHeads, dim));
    }

    /** Attention weights over the kept tokens, averaged over heads. */
    private static float[] weights(float[][] k, float[] query, List<Integer> kept, float scale, int kvHeads,
            int dim) {
        int heads = query.length / dim;
        float[] mean = new float[kept.size()];
        for (int h = 0; h < heads; h++) {
            int off = (h / (heads / kvHeads)) * dim;
            double[] s = new double[kept.size()];
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < s.length; i++) {
                double dot = 0;
                for (int c = 0; c < dim; c++) {
                    dot += query[h * dim + c] * k[kept.get(i)][off + c];
                }
                s[i] = dot * scale;
                max = Math.max(max, s[i]);
            }
            double sum = 0;
            for (int i = 0; i < s.length; i++) {
                s[i] = Math.exp(s[i] - max);
                sum += s[i];
            }
            for (int i = 0; i < s.length; i++) {
                mean[i] += (float) (s[i] / sum / heads);
            }
        }
        return mean;
    }

    private static double relative(float[] got, float[] want) {
        double num = 0;
        double den = 0;
        for (int i = 0; i < got.length; i++) {
            num += (got[i] - want[i]) * (got[i] - want[i]);
            den += want[i] * want[i];
        }
        return Math.sqrt(num / Math.max(den, 1e-12));
    }
}
