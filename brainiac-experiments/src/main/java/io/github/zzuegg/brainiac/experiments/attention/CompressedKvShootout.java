package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.experimental.nn.CompressedBlockAttention;
import io.github.zzuegg.brainiac.experimental.nn.InfiniAttention;
import io.github.zzuegg.brainiac.experimental.nn.LatentAttention;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepSeek-V4's compressed attention ({@link CompressedBlockAttention}: CSA with small
 * blocks and top-k selection, HCA with large blocks and dense attention, alternating by
 * layer) against the other ways to shrink the KV cache, on the {@link ArchitectureBench} toy
 * language at equal depth and step budget: grouped-query attention (core
 * {@code TransformerBlock} with one KV head), multi-head latent attention
 * ({@link LatentAttention}) and a compressive memory ({@link InfiniAttention}, the trainable
 * counterpart of {@code CompressiveKvMemory}: a constant-size linear memory plus local
 * attention), with full multi-head attention as the baseline. Reported: validation loss, the
 * cached numbers per token that grow with the context (all layers), and the constant part
 * (windows and fixed memories).
 */
public final class CompressedKvShootout {

    private CompressedKvShootout() {
    }

    /**
     * One configuration's result.
     *
     * @param name the configuration
     * @param validationLoss the toy-language validation loss
     * @param cachedPerToken cached numbers per token over all layers (growing part)
     * @param constantCache cached numbers that do not grow with the context
     * @param parameters the body's parameter count
     */
    public record Row(String name, double validationLoss, double cachedPerToken, int constantCache,
            long parameters) {
    }

    /**
     * Runs the comparison in two-layer stacks with 4 heads.
     *
     * @param width the model width
     * @param steps training steps
     * @param seed the seed
     * @param withSingleModes whether to add CSA-only and HCA-only stacks
     * @return MHA, GQA, MLA, compressive memory, CSA/HCA (and CSA-only, HCA-only)
     */
    public static List<Row> run(int width, int steps, long seed, boolean withSingleModes) {
        int depth = 2;
        int heads = 4;
        int kv = width / 2;
        int window = 4;
        List<Row> rows = new ArrayList<>();
        rows.add(row("MHA (core)", LayerChain.repeat(ArchitectureBench.coreBlock(heads, width), depth),
                depth * 2.0 * width, 0, width, steps, seed));
        rows.add(row("GQA kv=1 (core)", LayerChain.repeat(ArchitectureBench.coreBlock(heads, width).kvHeads(1),
                depth), depth * 2.0 * (width / heads), 0, width, steps, seed));
        int latent = width / 4;
        rows.add(row("MLA latent=" + latent, ArchitectureBench.stack(LatentAttention.of(heads, latent, 4), depth),
                depth * (latent + 4.0), 0, width, steps, seed));
        int headSize = width / heads;
        int segment = 8;
        rows.add(row("compressive (Infini)", ArchitectureBench.stack(InfiniAttention.spec(heads, headSize, segment),
                depth), 0, depth * (heads * headSize * (headSize + 1) + segment * 2 * width), width, steps, seed));
        int csa = 4;
        int hca = 8;
        rows.add(row("CSA/HCA (V4)", CompressedBlockAttention.alternating(depth, heads, kv, csa, 2, hca, window),
                (double) kv / csa + (double) kv / hca, depth * window * kv, width, steps, seed));
        if (withSingleModes) {
            rows.add(row("CSA only", ArchitectureBench.stack(CompressedBlockAttention.of(heads, kv,
                    CompressedBlockAttention.Mode.CSA, csa, 2, window), depth), depth * (double) kv / csa,
                    depth * window * kv, width, steps, seed));
            rows.add(row("HCA only", ArchitectureBench.stack(CompressedBlockAttention.of(heads, kv,
                    CompressedBlockAttention.Mode.HCA, hca, 0, window), depth), depth * (double) kv / hca,
                    depth * window * kv, width, steps, seed));
        }
        return rows;
    }

    private static Row row(String name, LayerSpec body, double cached, int constant, int width, int steps,
            long seed) {
        ArchitectureBench.LmResult r = ArchitectureBench.languageModel(name, body, width, steps, seed);
        return new Row(name, r.validationLoss(), cached, constant, r.parameters());
    }

    /**
     * Formats rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String table(List<Row> rows) {
        StringBuilder sb = new StringBuilder(String.format("%-22s %9s %12s %10s %8s%n", "attention", "val loss",
                "cache/token", "constant", "params"));
        for (Row r : rows) {
            sb.append(String.format("%-22s %9.4f %12.2f %10d %8d%n", r.name(), r.validationLoss(), r.cachedPerToken(),
                    r.constantCache(), r.parameters()));
        }
        return sb.toString();
    }

    /**
     * Prints the comparison.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(table(run(32, 400, 1, true)));
    }
}
