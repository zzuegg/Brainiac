package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.Awq;
import io.github.zzuegg.brainiac.experimental.nn.Fp8Format;
import io.github.zzuegg.brainiac.experimental.nn.Gptq;
import io.github.zzuegg.brainiac.experimental.nn.Hqq;
import io.github.zzuegg.brainiac.experimental.nn.InputRecorder;
import io.github.zzuegg.brainiac.experimental.nn.MicroscalingFormat;
import io.github.zzuegg.brainiac.experimental.nn.RotatedQuant;
import io.github.zzuegg.brainiac.experimental.nn.WeightPruning;
import io.github.zzuegg.brainiac.experiments.language.CountingTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Low-bit post-training quantization of a decoder trained on {@link CountingTasks}, judged by
 * the KL divergence of its next-token distributions from the float model's (the counting task
 * is easy enough that perplexity barely moves, so it is reported alongside), plus one dense
 * layer whose inputs carry the outlier channels of large models:
 * <ul>
 * <li>{@link #w4a4Layer} and {@link #w4a4Decoder}: 4-bit weights and activations with a
 *     random Hadamard rotation ({@link RotatedQuant}, QuaRot), against no rotation and
 *     {@code SmoothQuant} scaling, with round-to-nearest or GPTQ weights;</li>
 * <li>{@link #weightOnly}: {@link Hqq} at 3 and 4 bits against round-to-nearest (symmetric,
 *     asymmetric and the core {@code Tensor.quantized} groups), {@link Gptq} and {@link Awq},
 *     with each method's quantize time;</li>
 * <li>{@link #formats}: {@link MicroscalingFormat} MXFP4 and NVFP4 against core int4 groups of
 *     the same size and per-tensor FP8 E4M3.</li>
 * </ul>
 */
public final class LowBitQuantization {

    /**
     * One method's result.
     *
     * @param error the mean KL divergence from the float decoder's next-token distributions
     *     on fresh sequences (for {@link #w4a4Layer}: the relative output error)
     * @param perplexity the decoder's perplexity on fresh counting sequences (0 for the layer)
     * @param millis the time to quantize every layer
     */
    public record Outcome(double error, double perplexity, double millis) {
    }

    /** Builds a quantized replacement of a projection from its calibration inputs. */
    @FunctionalInterface
    public interface Quantizer {
        /**
         * @param dense the projection
         * @param inputs its calibration inputs {@code [samples, in]}
         * @return the replacement
         */
        Layer apply(Dense dense, Tensor inputs);
    }

    private LowBitQuantization() {
    }

    /**
     * Quantizes every projection of a copy of {@code base} and measures its perplexity.
     *
     * @param base the trained decoder (not changed)
     * @param quantizer the method
     * @param seed the calibration and evaluation seed
     * @return the perplexity and the quantize time
     */
    public static Outcome perplexity(HookedDecoder base, Quantizer quantizer, long seed) {
        HookedDecoder m = base.copy();
        Map<String, InputRecorder> rec = new HashMap<>();
        m.replaceLinears((name, l) -> {
            InputRecorder r = new InputRecorder(l);
            rec.put(name, r);
            return r;
        });
        m.forward(CountingTasks.sequences(new SplittableRandom(seed), 16, 12, m.vocabSize(), 1, 2));
        long t0 = System.nanoTime();
        m.replaceLinears((name, l) -> {
            Tensor x = rec.get(name).inputs();
            int width = x.shape().dim(x.shape().rank() - 1);
            return quantizer.apply((Dense) ((InputRecorder) l).base(), x.reshape((int) (x.shape().size() / width),
                    width));
        });
        double ms = (System.nanoTime() - t0) / 1e6;
        int[][] eval = CountingTasks.sequences(new SplittableRandom(seed + 1), 32, 12, m.vocabSize(), 1, 2);
        float[] p = base.forward(eval).logits().value().toArray();
        float[] q = m.forward(eval).logits().value().toArray();
        int vocab = m.vocabSize();
        double kl = 0;
        int rows = p.length / vocab;
        for (int r = 0; r < rows; r++) {
            double[] lp = logSoftmax(p, r * vocab, vocab);
            double[] lq = logSoftmax(q, r * vocab, vocab);
            for (int i = 0; i < vocab; i++) {
                kl += Math.exp(lp[i]) * (lp[i] - lq[i]);
            }
        }
        return new Outcome(kl / rows, Math.exp(CountingTasks.evaluate(m, 12, seed + 1, 1, 2)), ms);
    }

    private static double[] logSoftmax(float[] x, int from, int n) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, x[from + i]);
        }
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += Math.exp(x[from + i] - max);
        }
        double lse = max + Math.log(sum);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = x[from + i] - lse;
        }
        return out;
    }

    private static Layer weights(Dense d, Tensor w) {
        return WeightPruning.replaceWeight(d, w);
    }

    /** @return the methods of the W4A4 comparison on a decoder */
    private static Map<String, Quantizer> w4a4Methods(long seed) {
        Map<String, Quantizer> q = new LinkedHashMap<>();
        q.put("float", (d, x) -> d);
        q.put("W4A4 RTN", (d, x) -> RotatedQuant.plain(d, 4));
        q.put("W4A4 SmoothQuant", (d, x) -> RotatedQuant.smoothed(d, 4, x, 0.5f));
        q.put("W4A4 Hadamard + RTN", (d, x) -> RotatedQuant.hadamard(d, 4, seed));
        q.put("W4A4 Hadamard + GPTQ", (d, x) -> RotatedQuant.hadamardGptq(d, 4, x, seed));
        return q;
    }

    /**
     * W4A4 on every projection of the decoder.
     *
     * @param base the trained decoder
     * @param seed the seed
     * @return perplexity per method
     */
    public static Map<String, Outcome> w4a4Decoder(HookedDecoder base, long seed) {
        Map<String, Outcome> out = new LinkedHashMap<>();
        w4a4Methods(seed).forEach((k, v) -> out.put(k, perplexity(base, v, seed)));
        return out;
    }

    /**
     * W4A4 on one layer whose inputs have two of 128 channels 12 times larger than the rest, with
     * weights 12 times smaller on those rows (so each outlier channel contributes to the output
     * like an ordinary one).
     *
     * @param seed the seed
     * @return relative output error per method (in {@code error})
     */
    public static Map<String, Outcome> w4a4Layer(long seed) {
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
        for (int i = 0; i < in; i++) {
            for (int c = 0; c < in; c++) {
                w[i * in + c] /= i % 64 == 3 ? 12 : 1;
            }
        }
        Dense d = new Dense(in, in, Activation.IDENTITY, Tensor.of(Shape.of(in, in), w), Tensor.zeros(Shape.of(in)));
        float[] ref = d.forward(Var.of(test)).value().toArray();
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (Map.Entry<String, Quantizer> e : w4a4Methods(seed).entrySet()) {
            if (e.getKey().equals("float")) {
                continue;
            }
            long t0 = System.nanoTime();
            Layer q = e.getValue().apply(d, calib);
            double ms = (System.nanoTime() - t0) / 1e6;
            float[] got = q.forward(Var.of(test)).value().toArray();
            double num = 0;
            double den = 0;
            for (int i = 0; i < ref.length; i++) {
                num += (got[i] - ref[i]) * (got[i] - ref[i]);
                den += ref[i] * ref[i];
            }
            out.put(e.getKey(), new Outcome(Math.sqrt(num / den), 0, ms));
        }
        return out;
    }

    /**
     * Weight-only quantization.
     *
     * @param base the trained decoder
     * @param bits the bits per weight
     * @param seed the seed
     * @return perplexity and quantize time per method
     */
    public static Map<String, Outcome> weightOnly(HookedDecoder base, int bits, long seed) {
        Map<String, Quantizer> q = new LinkedHashMap<>();
        q.put("float", (d, x) -> d);
        q.put("RTN symmetric per column", (d, x) -> weights(d, Gptq.roundToNearest(d.weight().value(), bits)));
        q.put("RTN asymmetric per column", (d, x) -> weights(d, new Hqq(bits, 0).roundToNearest(d.weight().value())));
        q.put("HQQ per column", (d, x) -> weights(d, new Hqq(bits, 0).quantize(d.weight().value())));
        q.put("core int groups of 16", (d, x) -> weights(d, d.weight().value().quantized(bits, 16).unpacked()));
        q.put("RTN asymmetric groups of 16", (d, x) -> weights(d, new Hqq(bits, 16).roundToNearest(
                d.weight().value())));
        q.put("HQQ groups of 16", (d, x) -> weights(d, new Hqq(bits, 16).quantize(d.weight().value())));
        q.put("GPTQ per column", (d, x) -> weights(d, new Gptq(bits).quantize(d.weight().value(), x)));
        q.put("AWQ per column", (d, x) -> weights(d, Awq.quantize(d.weight().value(), x, bits).weight()));
        Map<String, Outcome> out = new LinkedHashMap<>();
        q.forEach((k, v) -> out.put(k, perplexity(base, v, seed)));
        return out;
    }

    /**
     * 4-bit formats on every weight.
     *
     * @param base the trained decoder
     * @param seed the seed
     * @return perplexity per format
     */
    public static Map<String, Outcome> formats(HookedDecoder base, long seed) {
        Map<String, Quantizer> q = new LinkedHashMap<>();
        q.put("float", (d, x) -> d);
        q.put("FP8 E4M3 per tensor", (d, x) -> weights(d, Fp8Format.E4M3.quantize(d.weight().value())));
        q.put("MXFP4 (32 per E8M0 scale)", (d, x) -> weights(d, MicroscalingFormat.MXFP4.quantize(d.weight().value())));
        q.put("core int4 groups of 32", (d, x) -> weights(d, d.weight().value().quantized(4, 32).unpacked()));
        q.put("NVFP4 (16 per E4M3 scale)", (d, x) -> weights(d, MicroscalingFormat.NVFP4.quantize(d.weight().value())));
        q.put("core int4 groups of 16", (d, x) -> weights(d, d.weight().value().quantized(4, 16).unpacked()));
        q.put("int4 per column", (d, x) -> weights(d, Gptq.roundToNearest(d.weight().value(), 4)));
        Map<String, Outcome> out = new LinkedHashMap<>();
        q.forEach((k, v) -> out.put(k, perplexity(base, v, seed)));
        return out;
    }
}
