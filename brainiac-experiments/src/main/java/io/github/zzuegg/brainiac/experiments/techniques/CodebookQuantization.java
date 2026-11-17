package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.AdditiveQuant;
import io.github.zzuegg.brainiac.experimental.nn.Gptq;
import io.github.zzuegg.brainiac.experimental.nn.Hqq;
import io.github.zzuegg.brainiac.experimental.nn.LowRankOutlierQuant;
import io.github.zzuegg.brainiac.experimental.nn.RotatedQuant;
import io.github.zzuegg.brainiac.experimental.nn.TrellisQuant;
import io.github.zzuegg.brainiac.experimental.nn.WeightPruning;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Vector and trellis codes against scalar grids at 2 and 3 bits, and a low-rank outlier
 * branch against rotation and smoothing at W4A4, on the {@link LowBitQuantization} harness
 * (a decoder trained on counting, judged by the KL divergence of its next-token distributions
 * from the float model's, with perplexity alongside):
 * <ul>
 * <li>{@link #weightOnly}: {@link TrellisQuant} (QTIP's bit-shift trellis with a computed
 *     Gaussian codebook after Hadamard incoherence processing) and {@link AdditiveQuant}
 *     (AQLM's sum of two codebooks with beam search, codebooks shared by every layer) against
 *     per-column round-to-nearest, {@link Gptq} and {@link Hqq} groups of 16;</li>
 * <li>{@link #w4a4}: {@link LowRankOutlierQuant} (SVDQuant) against plain W4A4, SmoothQuant and
 *     Hadamard rotation ({@link RotatedQuant}), on the decoder and on one layer whose inputs and
 *     weights both carry outlier channels.</li>
 * </ul>
 */
public final class CodebookQuantization {

    private CodebookQuantization() {
    }

    /**
     * Weight-only quantization at {@code bits} bits per weight.
     *
     * @param base the trained decoder (not changed)
     * @param bits 2 or 3
     * @param seed the seed
     * @return per method its outcome, the float model first
     */
    public static Map<String, LowBitQuantization.Outcome> weightOnly(HookedDecoder base, int bits, long seed) {
        Map<String, LowBitQuantization.Quantizer> q = new LinkedHashMap<>();
        q.put("float", (d, x) -> d);
        q.put("RTN per column", (d, x) -> WeightPruning.replaceWeight(d, Gptq.roundToNearest(d.weight().value(),
                bits)));
        q.put("HQQ groups of 16", (d, x) -> WeightPruning.replaceWeight(d, new Hqq(bits, 16).quantize(
                d.weight().value())));
        q.put("GPTQ per column", (d, x) -> WeightPruning.replaceWeight(d, new Gptq(bits).quantize(d.weight().value(),
                x)));
        TrellisQuant qtip = new TrellisQuant(bits, 12, 32, seed);
        q.put("QTIP trellis", (d, x) -> qtip.quantize(d));
        Map<String, LowBitQuantization.Outcome> out = new LinkedHashMap<>();
        q.forEach((k, v) -> out.put(k, LowBitQuantization.perplexity(base, v, seed)));
        out.put("AQLM 2x" + (2 * bits) + "-bit codes / 4 weights (shared)", additive(base,
                new AdditiveQuant(2, 2 * bits, 4, 4, 3, seed), seed));
        return out;
    }

    /** AQLM with one set of codebooks for every projection: record inputs, code jointly, then evaluate. */
    private static LowBitQuantization.Outcome additive(HookedDecoder base, AdditiveQuant aq, long seed) {
        List<Tensor> weights = new ArrayList<>();
        List<Tensor> inputs = new ArrayList<>();
        LowBitQuantization.perplexity(base, (d, x) -> {
            weights.add(d.weight().value());
            inputs.add(x);
            return d;
        }, seed);
        long t0 = System.nanoTime();
        List<Tensor> coded = aq.quantizeAll(weights, inputs);
        double ms = (System.nanoTime() - t0) / 1e6;
        int[] at = {0};
        LowBitQuantization.Outcome o = LowBitQuantization.perplexity(base,
                (d, x) -> WeightPruning.replaceWeight(d, coded.get(at[0]++)), seed);
        return new LowBitQuantization.Outcome(o.error(), o.perplexity(), ms);
    }

    /**
     * The W4A4 methods.
     *
     * @param seed the rotation seed
     * @return method name to quantizer
     */
    static Map<String, LowBitQuantization.Quantizer> w4a4Methods(long seed) {
        Map<String, LowBitQuantization.Quantizer> q = new LinkedHashMap<>();
        q.put("W4A4 RTN", (d, x) -> RotatedQuant.plain(d, 4));
        q.put("W4A4 SmoothQuant", (d, x) -> RotatedQuant.smoothed(d, 4, x, 0.5f));
        q.put("W4A4 Hadamard", (d, x) -> RotatedQuant.hadamard(d, 4, seed));
        q.put("W4A4 SVDQuant rank 4", (d, x) -> LowRankOutlierQuant.of(d, 4, 4, x, 0.5f));
        q.put("W4A4 SVDQuant rank 8", (d, x) -> LowRankOutlierQuant.of(d, 4, 8, x, 0.5f));
        return q;
    }

    /**
     * W4A4 on every projection of the decoder.
     *
     * @param base the trained decoder
     * @param seed the seed
     * @return per method its outcome
     */
    public static Map<String, LowBitQuantization.Outcome> w4a4(HookedDecoder base, long seed) {
        Map<String, LowBitQuantization.Outcome> out = new LinkedHashMap<>();
        w4a4Methods(seed).forEach((k, v) -> out.put(k, LowBitQuantization.perplexity(base, v, seed)));
        return out;
    }

    /**
     * W4A4 on one 128-wide layer whose inputs have two channels 12 times larger than the rest
     * and whose weights have a few large outlier entries.
     *
     * @param seed the seed
     * @return per method the relative output error
     */
    public static Map<String, Double> w4a4OutlierLayer(long seed) {
        int in = 128;
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[512 * in];
        for (int n = 0; n < 512; n++) {
            for (int i = 0; i < in; i++) {
                x[n * in + i] = (float) (r.nextGaussian() * (i % 64 == 3 ? 12 : 1));
            }
        }
        Tensor calib = Tensor.of(Shape.of(256, in), java.util.Arrays.copyOf(x, 256 * in));
        Tensor test = Tensor.of(Shape.of(256, in), java.util.Arrays.copyOfRange(x, 256 * in, 512 * in));
        float[] w = Tensor.normal(Shape.of(in, in), 0, 0.125f, r).toArray();
        for (int k = 0; k < 24; k++) {
            w[r.nextInt(w.length)] *= 25;   // weight outliers
        }
        Dense d = new Dense(in, in, Activation.IDENTITY, Tensor.of(Shape.of(in, in), w), Tensor.zeros(Shape.of(in)));
        float[] ref = d.forward(Var.of(test)).value().toArray();
        Map<String, Double> out = new LinkedHashMap<>();
        for (Map.Entry<String, LowBitQuantization.Quantizer> e : w4a4Methods(seed).entrySet()) {
            Layer q = e.getValue().apply(d, calib);
            float[] got = q.forward(Var.of(test)).value().toArray();
            double num = 0;
            double den = 0;
            for (int i = 0; i < ref.length; i++) {
                num += (got[i] - ref[i]) * (got[i] - ref[i]);
                den += ref[i] * ref[i];
            }
            out.put(e.getKey(), Math.sqrt(num / den));
        }
        return out;
    }
}
