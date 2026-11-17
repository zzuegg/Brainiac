package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.ValueResidualBlock;
import io.github.zzuegg.brainiac.nn.TransformerBlock;

import java.util.ArrayList;
import java.util.List;

/**
 * Value residual learning (ResFormer) in deep stacks: the {@link ArchitectureBench} toy
 * language with a {@link ValueResidualBlock} stack against the same stack without the value
 * shortcut and against a stack of core {@link TransformerBlock}s, at equal depth, width and
 * step budget. Deeper stacks over-smooth more, so the gap should grow with depth.
 */
public final class DeepStackRefinements {

    private DeepStackRefinements() {
    }

    /**
     * One configuration's result.
     *
     * @param name the configuration
     * @param depth the number of layers
     * @param validationLoss the validation loss
     * @param parameters the body's parameter count
     */
    public record Row(String name, int depth, double validationLoss, long parameters) {
    }

    /**
     * Trains the three stacks at one depth.
     *
     * @param depth the number of layers
     * @param width the model width (4 heads)
     * @param steps training steps
     * @param seed the seed
     * @param includeCore whether to train the core-block stack as well
     * @return value residual, plain, and (optionally) core rows
     */
    public static List<Row> run(int depth, int width, int steps, long seed, boolean includeCore) {
        List<Row> rows = new ArrayList<>();
        ArchitectureBench.LmResult vr = ArchitectureBench.languageModel("value residual",
                ValueResidualBlock.of(4, depth, true), width, steps, seed);
        rows.add(new Row(vr.name(), depth, vr.validationLoss(), vr.parameters()));
        ArchitectureBench.LmResult plain = ArchitectureBench.languageModel("plain",
                ValueResidualBlock.of(4, depth, false), width, steps, seed);
        rows.add(new Row(plain.name(), depth, plain.validationLoss(), plain.parameters()));
        if (includeCore) {
            ArchitectureBench.LmResult core = ArchitectureBench.languageModel("core TransformerBlock",
                    LayerChain.repeat(ArchitectureBench.coreBlock(4, width), depth), width, steps, seed);
            rows.add(new Row(core.name(), depth, core.validationLoss(), core.parameters()));
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
        StringBuilder sb = new StringBuilder(String.format("%-22s %6s %9s %8s%n", "stack", "depth", "val loss",
                "params"));
        for (Row r : rows) {
            sb.append(String.format("%-22s %6d %9.4f %8d%n", r.name(), r.depth(), r.validationLoss(), r.parameters()));
        }
        return sb.toString();
    }

    /**
     * Runs depths 8 and 16.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        List<Row> rows = new ArrayList<>(run(8, 32, 300, 1, true));
        rows.addAll(run(16, 32, 300, 1, true));
        System.out.print(table(rows));
    }
}
