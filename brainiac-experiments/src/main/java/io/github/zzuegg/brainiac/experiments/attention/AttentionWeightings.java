package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.experimental.nn.GatedAttention;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention.Weighting;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * How attention turns scores into weights, compared on the {@link ArchitectureBench} toy
 * language at equal size and budget: softmax (the baseline), sigmoid attention, Softpick,
 * gated attention, a learned sink logit, learned register slots and talking heads, all in
 * the same pre-norm block with RoPE ({@link RefinedAttention}, except the existing
 * {@link GatedAttention}). Besides the validation loss it measures what the attention maps
 * look like on the validation set:
 *
 * <ul>
 *   <li><b>sink share</b>: the mean weight on the first token, the usual attention sink;</li>
 *   <li><b>mass</b>: the mean total weight per query (below 1 when a head may attend to
 *       nothing: sigmoid, Softpick, sink logit, registers);</li>
 *   <li><b>entropy</b>: of each query's weights renormalized to sum to one, in nats;</li>
 *   <li><b>4-bit error</b>: the relative error of quantizing the final hidden states to 4 bits
 *       per token (absmax), which outlier "massive activations" make large; and the
 *       <b>outlier ratio</b> max|h| / mean|h|.</li>
 * </ul>
 */
public final class AttentionWeightings {

    private AttentionWeightings() {
    }

    /**
     * One configuration's result.
     *
     * @param name the configuration
     * @param validationLoss the validation loss
     * @param sinkShare the mean weight on the first token, or NaN when not observable
     * @param mass the mean total attention weight per query, or NaN
     * @param entropy the mean entropy of the renormalized weights, or NaN
     * @param quantError the relative 4-bit quantization error of the hidden states
     * @param outlierRatio max |h| over mean |h| of the hidden states
     */
    public record Row(String name, double validationLoss, double sinkShare, double mass, double entropy,
            double quantError, double outlierRatio) {
    }

    /**
     * The configurations, by name.
     *
     * @return name to attention spec, softmax first
     */
    public static Map<String, LayerSpec> lineup() {
        RefinedAttention.Spec base = RefinedAttention.of(4).causal(true).rotary(10000f);
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("softmax", base);
        m.put("sigmoid", base.weighting(Weighting.SIGMOID));
        m.put("softpick", base.weighting(Weighting.SOFTPICK));
        m.put("gated", GatedAttention.of(4, true, 10000f));
        m.put("sink logit", base.sinkLogit(true));
        m.put("registers 2", base.registers(2));
        m.put("talking heads", base.talkingHeads(true));
        return m;
    }

    /**
     * Trains the named configurations.
     *
     * @param names the configurations to run, from {@link #lineup()}
     * @param depth the number of blocks
     * @param steps training steps
     * @param seed the seed
     * @return one row per name
     */
    public static List<Row> run(List<String> names, int depth, int steps, long seed) {
        List<Row> rows = new ArrayList<>();
        for (String name : names) {
            LayerSpec attention = lineup().get(name);
            ArchitectureBench.LmResult r = ArchitectureBench.languageModel(name,
                    ArchitectureBench.stack(attention, depth), 32, steps, seed);
            Tensor hidden = r.model().hidden(r.validation(), false).value();   // also refreshes the maps
            double[] maps = mapStatistics(ArchitectureBench.attentions(r.body()));
            float[] h = hidden.toArray();
            rows.add(new Row(name, r.validationLoss(), maps[0], maps[1], maps[2],
                    quantizationError(h, hidden.shape().dim(-1), 4), outlierRatio(h)));
        }
        return rows;
    }

    /**
     * Sink share, mass and entropy of the last attention maps.
     *
     * @param layers the attention layers
     * @return {sink share, mass, entropy}, NaN without layers
     */
    public static double[] mapStatistics(List<RefinedAttention> layers) {
        if (layers.isEmpty()) {
            return new double[] {Double.NaN, Double.NaN, Double.NaN};
        }
        double sink = 0;
        double mass = 0;
        double entropy = 0;
        long rows = 0;
        for (RefinedAttention a : layers) {
            Tensor w = a.lastWeights();
            int keys = w.shape().dim(3);
            int t = w.shape().dim(2);
            int prefix = keys - t;
            float[] f = w.toArray();
            int n = f.length / keys;
            for (int row = 0; row < n; row++) {
                int query = row % t;
                if (query == 0) {
                    continue;                 // the first query has no choice
                }
                int base = row * keys;
                double sum = 0;
                for (int j = 0; j < keys; j++) {
                    sum += f[base + j];
                }
                sink += f[base + prefix];
                mass += sum;
                if (sum > 1e-6) {
                    double hh = 0;
                    for (int j = 0; j < keys; j++) {
                        double p = f[base + j] / sum;
                        hh -= p > 0 ? p * Math.log(p) : 0;
                    }
                    entropy += hh;
                }
                rows++;
            }
        }
        return new double[] {sink / rows, mass / rows, entropy / rows};
    }

    /**
     * Relative error of symmetric absmax quantization per row.
     *
     * @param h the values, rows of {@code width}
     * @param width the row width
     * @param bits the bits per value
     * @return {@code ||h - q(h)|| / ||h||}
     */
    public static double quantizationError(float[] h, int width, int bits) {
        int levels = (1 << (bits - 1)) - 1;
        double err = 0;
        double norm = 0;
        for (int r = 0; r < h.length / width; r++) {
            double max = 0;
            for (int i = 0; i < width; i++) {
                max = Math.max(max, Math.abs(h[r * width + i]));
            }
            double scale = max == 0 ? 1 : max / levels;
            for (int i = 0; i < width; i++) {
                double v = h[r * width + i];
                double q = Math.round(v / scale) * scale;
                err += (v - q) * (v - q);
                norm += v * v;
            }
        }
        return Math.sqrt(err / Math.max(norm, 1e-12));
    }

    /**
     * The largest magnitude over the mean magnitude.
     *
     * @param h the values
     * @return the ratio
     */
    public static double outlierRatio(float[] h) {
        double max = 0;
        double sum = 0;
        for (float v : h) {
            max = Math.max(max, Math.abs(v));
            sum += Math.abs(v);
        }
        return max / (sum / h.length);
    }

    /**
     * Formats rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String table(List<Row> rows) {
        StringBuilder sb = new StringBuilder(String.format("%-14s %9s %7s %6s %8s %8s %8s%n", "weighting",
                "val loss", "sink", "mass", "entropy", "4bit err", "outlier"));
        for (Row r : rows) {
            sb.append(String.format("%-14s %9.4f %7.3f %6.3f %8.3f %8.4f %8.2f%n", r.name(), r.validationLoss(),
                    r.sinkShare(), r.mass(), r.entropy(), r.quantError(), r.outlierRatio()));
        }
        return sb.toString();
    }

    /**
     * Runs the whole lineup with two blocks.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(table(run(new ArrayList<>(lineup().keySet()), 2, 400, 1)));
    }
}
