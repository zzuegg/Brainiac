package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.experimental.nn.BlockRoutedAttention;
import io.github.zzuegg.brainiac.experimental.nn.IndexedSparseAttention;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.NativeSparseAttention;
import io.github.zzuegg.brainiac.experimental.nn.ResidualMixerBlock;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Trainable sparse attention that selects keys per query, against dense attention: DeepSeek
 * Sparse Attention ({@link IndexedSparseAttention}, a lightning indexer picks the top-k
 * tokens), MoBA ({@link BlockRoutedAttention}, a parameter-free gate picks the top-k blocks)
 * and Native Sparse Attention ({@link NativeSparseAttention}), with the core
 * {@code TransformerBlock} as the dense baseline, all in two-block stacks.
 *
 * <ul>
 *   <li>{@link #languageModel}: the {@link ArchitectureBench} toy language (24 tokens), each
 *       sparse layer limited to about a third of the context;</li>
 *   <li>{@link #longRecall}: associative recall trained with 8 pairs (17 tokens) and tested with
 *       8, 16 and 32 pairs (up to 65 tokens, about 4x the training length), where a query must
 *       find one needle among many keys while seeing only a fixed number of them.</li>
 * </ul>
 * Next to the quality the rows report the mean number of keys each query attended to.
 */
public final class SparseSelectionShootout {

    private SparseSelectionShootout() {
    }

    /**
     * One configuration's result.
     *
     * @param name the configuration
     * @param metrics the measured values (validation loss, or recall at each test length)
     * @param attended the mean keys per query in the last evaluation pass (NaN for dense)
     * @param parameters the body's parameter count
     */
    public record Row(String name, double[] metrics, double attended, long parameters) {
    }

    /**
     * The lineup: dense core blocks, DSA, MoBA and NSA.
     *
     * @param depth blocks per body
     * @param heads attention heads
     * @param width the model width
     * @param keys keys per query for DSA ({@code topK}); MoBA gets {@code keys / block} blocks
     * @param block the MoBA and NSA block length
     * @param warmup DSA's dense warm-up passes (per layer)
     * @return name to body spec
     */
    public static Map<String, LayerSpec> lineup(int depth, int heads, int width, int keys, int block, int warmup) {
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("dense (core)", LayerChain.repeat(ArchitectureBench.coreBlock(heads, width), depth));
        m.put("DSA top-" + keys, ArchitectureBench.stack(IndexedSparseAttention.of(heads, keys, warmup), depth));
        m.put("MoBA " + keys / block + "x" + block, ArchitectureBench.stack(BlockRoutedAttention.of(heads, block,
                keys / block), depth));
        m.put("NSA", ArchitectureBench.stack(NativeSparseAttention.of(heads, block, Math.max(1, keys / block - 1),
                block), depth));
        return m;
    }

    /**
     * Toy-language validation loss.
     *
     * @param depth blocks per body
     * @param width the model width (2 heads)
     * @param steps training steps
     * @param seed the seed
     * @return one row per lineup entry, metric = validation loss
     */
    public static List<Row> languageModel(int depth, int width, int steps, long seed) {
        List<Row> rows = new ArrayList<>();
        for (Map.Entry<String, LayerSpec> e : lineup(depth, 2, width, 8, 2, steps / 2).entrySet()) {
            ArchitectureBench.LmResult r = ArchitectureBench.languageModel(e.getKey(), e.getValue(), width, steps,
                    seed);
            rows.add(new Row(e.getKey(), new double[] {r.validationLoss()}, attended(r.body()), r.parameters()));
        }
        return rows;
    }

    /**
     * Recall trained at 8 pairs, tested at 8, 16 and 32 pairs; sparse layers see 8 keys.
     *
     * @param depth blocks per body
     * @param width the model width (2 heads)
     * @param steps training steps
     * @param seed the seed
     * @param names the lineup entries to run, by name prefix
     * @return one row per run entry, metrics = accuracy per test length (128 samples each)
     */
    public static List<Row> longRecall(int depth, int width, int steps, long seed, List<String> names) {
        List<MixerShootout.Task> tests = List.of(MixerShootout.recall(32, 8, 8), MixerShootout.recall(32, 8, 16),
                MixerShootout.recall(32, 8, 32));
        List<Row> rows = new ArrayList<>();
        for (Map.Entry<String, LayerSpec> e : lineup(depth, 2, width, 8, 2, steps / 2).entrySet()) {
            if (names.stream().noneMatch(e.getKey()::startsWith)) {
                continue;
            }
            ArchitectureBench.TinyModel m = ArchitectureBench.trainRetrieval(e.getValue(), width, tests.getFirst(),
                    steps, 0.01f, seed, true);
            double[] acc = new double[tests.size()];
            double att = Double.NaN;
            for (int k = 0; k < acc.length; k++) {
                acc[k] = ArchitectureBench.accuracy(m, tests.get(k).batch(new SplittableRandom(seed + 1000 + k), 128));
                att = attended(m.body());
            }
            rows.add(new Row(e.getKey(), acc, att, ArchitectureBench.count(m.body())));
        }
        return rows;
    }

    /**
     * The mean keys per query of the sparse layers in a body, after its last pass.
     *
     * @param body the body
     * @return the mean, or NaN without sparse layers
     */
    public static double attended(Layer body) {
        List<Double> values = new ArrayList<>();
        collect(body, values);
        return values.stream().mapToDouble(Double::doubleValue).average().orElse(Double.NaN);
    }

    private static void collect(Layer layer, List<Double> out) {
        switch (layer) {
            case IndexedSparseAttention a -> out.add(a.lastAttended());
            case BlockRoutedAttention a -> out.add(a.lastAttended());
            case LayerChain c -> c.layers().forEach(l -> collect(l, out));
            case ResidualMixerBlock b -> collect(b.mixer(), out);
            default -> {
            }
        }
    }

    /**
     * Formats rows.
     *
     * @param title the metric columns' title
     * @param rows the rows
     * @return a text table
     */
    public static String table(String title, List<Row> rows) {
        StringBuilder sb = new StringBuilder(String.format("%-14s %-24s %9s %8s%n", "attention", title, "keys/q",
                "params"));
        for (Row r : rows) {
            StringBuilder m = new StringBuilder();
            for (double v : r.metrics()) {
                m.append(String.format("%8.3f", v));
            }
            sb.append(String.format("%-14s %-24s %9.2f %8d%n", r.name(), m, r.attended(), r.parameters()));
        }
        return sb.toString();
    }

    /**
     * Runs both comparisons.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(table("val loss", languageModel(2, 32, 400, 1)));
        System.out.print(table("recall @ 8/16/32 pairs", longRecall(2, 32, 600, 1, List.of("dense", "DSA", "MoBA", "NSA"))));
    }

}
