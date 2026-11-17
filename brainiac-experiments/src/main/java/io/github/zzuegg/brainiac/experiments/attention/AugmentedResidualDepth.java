package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.experimental.nn.AugmentedResidualBlock;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.ManifoldHyperConnection;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.ResidualMixerBlock;
import io.github.zzuegg.brainiac.experimental.nn.ValueResidualBlock;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Learned augmented residuals (LAuReL, {@link AugmentedResidualBlock}) in deep stacks, next
 * to the other residual-path refinements of {@link DeepStackRefinements}: the
 * {@link ArchitectureBench} toy language with the plain residual against LAuReL's
 * residual-weight (RW), low-rank (LR), both, and previous-activation (PA) forms, value
 * residual learning ({@link ValueResidualBlock}) and manifold-constrained hyper-connections
 * ({@link ManifoldHyperConnection} around every block), at equal depth, width and step budget.
 */
public final class AugmentedResidualDepth {

    private AugmentedResidualDepth() {
    }

    /**
     * One configuration's result.
     *
     * @param name the configuration
     * @param depth the number of blocks
     * @param validationLoss the validation loss
     * @param parameters the body's parameter count
     */
    public record Row(String name, int depth, double validationLoss, long parameters) {
    }

    /**
     * The lineup at one depth (causal RoPE attention, 2 heads, SwiGLU FFNs).
     *
     * @param depth the number of blocks
     * @return name to body spec
     */
    public static Map<String, LayerSpec> lineup(int depth) {
        LayerSpec attention = RefinedAttention.of(2).causal(true).rotary(10000f);
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("plain residual", AugmentedResidualBlock.of(attention, depth, AugmentedResidualBlock.Variant.NONE));
        m.put("LAuReL-RW", AugmentedResidualBlock.of(attention, depth, AugmentedResidualBlock.Variant.RW));
        m.put("LAuReL-LR", AugmentedResidualBlock.of(attention, depth, AugmentedResidualBlock.Variant.LR));
        m.put("LAuReL-RW+LR", AugmentedResidualBlock.of(attention, depth, AugmentedResidualBlock.Variant.RW_LR));
        m.put("LAuReL-PA", AugmentedResidualBlock.of(attention, depth, AugmentedResidualBlock.Variant.PA));
        m.put("value residual", ValueResidualBlock.of(2, depth, true));
        m.put("mHC", LayerChain.repeat(ManifoldHyperConnection.of(ResidualMixerBlock.of(attention)), depth));
        return m;
    }

    /**
     * Trains named lineup entries at one depth.
     *
     * @param names the entries
     * @param depth the number of blocks
     * @param width the model width
     * @param steps training steps
     * @param seed the seed
     * @return one row per name
     */
    public static List<Row> run(List<String> names, int depth, int width, int steps, long seed) {
        Map<String, LayerSpec> lineup = lineup(depth);
        List<Row> rows = new ArrayList<>();
        for (String n : names) {
            ArchitectureBench.LmResult r = ArchitectureBench.languageModel(n, lineup.get(n), width, steps, seed);
            rows.add(new Row(n, depth, r.validationLoss(), r.parameters()));
        }
        return rows;
    }

    /**
     * Formats rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String table(List<Row> rows) {
        StringBuilder sb = new StringBuilder(String.format("%-16s %6s %9s %8s%n", "residual", "depth", "val loss",
                "params"));
        for (Row r : rows) {
            sb.append(String.format("%-16s %6d %9.4f %8d%n", r.name(), r.depth(), r.validationLoss(), r.parameters()));
        }
        return sb.toString();
    }

    /**
     * Runs depths 8 and 16.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        List<Row> rows = new ArrayList<>();
        for (int depth : new int[] {8, 16}) {
            rows.addAll(run(new ArrayList<>(lineup(depth).keySet()), depth, 32, 300, 1));
        }
        System.out.print(table(rows));
    }
}
