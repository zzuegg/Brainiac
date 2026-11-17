package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.experimental.nn.CrossLayerKvBlock;
import io.github.zzuegg.brainiac.experimental.nn.LatentAttention;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.TransformerBlock;

import java.util.ArrayList;
import java.util.List;

/**
 * KV-cache-saving attention against full multi-head attention on the {@link ArchitectureBench}
 * toy language, at equal depth and step budget: validation loss next to the cached numbers
 * per token (keys plus values, all layers) that grow with the context.
 *
 * <ul>
 *   <li>across heads: the core {@link TransformerBlock} with multi-head attention (MHA), with
 *       grouped-query attention (one KV head, GQA), and multi-head latent attention
 *       ({@link LatentAttention}, MLA), which caches one latent plus a small shared rotary key
 *       per token;</li>
 *   <li>across layers: an unshared stack against cross-layer attention (two layers per KV set)
 *       and YOCO (sliding-window lower half, one global KV set for the upper half), all
 *       {@link CrossLayerKvBlock}s.</li>
 * </ul>
 */
public final class KvEfficientAttention {

    private KvEfficientAttention() {
    }

    /**
     * One configuration's result.
     *
     * @param name the configuration
     * @param validationLoss the toy-language validation loss
     * @param cachedPerToken cached numbers per token over all layers
     * @param parameters the body's parameter count
     */
    public record Row(String name, double validationLoss, int cachedPerToken, long parameters) {
    }

    /**
     * MHA, GQA and MLA, each in a two-layer stack.
     *
     * @param width the model width (4 heads)
     * @param steps training steps
     * @param seed the seed
     * @return the rows, MHA first
     */
    public static List<Row> acrossHeads(int width, int steps, long seed) {
        int depth = 2;
        int heads = 4;
        int latent = width / 2;
        int rope = 4;
        List<Row> rows = new ArrayList<>();
        rows.add(row("MHA (core)", LayerChain.repeat(ArchitectureBench.coreBlock(heads, width), depth),
                depth * 2 * width, width, steps, seed));
        rows.add(row("GQA kv=1 (core)", LayerChain.repeat(ArchitectureBench.coreBlock(heads, width).kvHeads(1),
                depth), depth * 2 * (width / heads), width, steps, seed));
        LayerSpec mla = LatentAttention.of(heads, latent, rope);
        rows.add(row("MLA latent=" + latent, ArchitectureBench.stack(mla, depth), depth * (latent + rope), width,
                steps, seed));
        return rows;
    }

    /**
     * An unshared four-layer stack against CLA (pairs) and YOCO (window 8).
     *
     * @param width the model width (4 heads)
     * @param steps training steps
     * @param seed the seed
     * @return the rows, unshared first
     */
    public static List<Row> acrossLayers(int width, int steps, long seed) {
        int depth = 4;
        List<Row> rows = new ArrayList<>();
        for (Object[] c : new Object[][] {{"unshared", CrossLayerKvBlock.Sharing.NONE, 1},
            {"CLA x2", CrossLayerKvBlock.Sharing.CLA, 2}, {"YOCO w=8", CrossLayerKvBlock.Sharing.YOCO, 8}}) {
            CrossLayerKvBlock.Sharing sharing = (CrossLayerKvBlock.Sharing) c[1];
            int arg = (Integer) c[2];
            int cached = new CrossLayerKvBlock(width, 4, depth, sharing, arg, arg, 10000f,
                    new java.util.SplittableRandom(0)).cachedValuesPerToken();
            rows.add(row((String) c[0], CrossLayerKvBlock.of(4, depth, sharing, arg), cached, width, steps, seed));
        }
        return rows;
    }

    private static Row row(String name, LayerSpec body, int cached, int width, int steps, long seed) {
        ArchitectureBench.LmResult r = ArchitectureBench.languageModel(name, body, width, steps, seed);
        return new Row(name, r.validationLoss(), cached, r.parameters());
    }

    /**
     * Formats rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String table(List<Row> rows) {
        StringBuilder sb = new StringBuilder(String.format("%-18s %10s %12s %8s%n", "attention", "val loss",
                "cache/token", "params"));
        for (Row r : rows) {
            sb.append(String.format("%-18s %10.4f %12d %8d%n", r.name(), r.validationLoss(), r.cachedPerToken(),
                    r.parameters()));
        }
        return sb.toString();
    }

    /**
     * Prints both comparisons.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(table(acrossHeads(32, 400, 1)));
        System.out.print(table(acrossLayers(32, 400, 1)));
    }
}
