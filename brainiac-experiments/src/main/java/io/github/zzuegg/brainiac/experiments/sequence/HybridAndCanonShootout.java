package io.github.zzuegg.brainiac.experiments.sequence;

import io.github.zzuegg.brainiac.experimental.nn.CanonLayer;
import io.github.zzuegg.brainiac.experimental.nn.HybridLinearStack;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.ParallelHybridBlock;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.ResidualMixerBlock;
import io.github.zzuegg.brainiac.experiments.attention.ArchitectureBench;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Horizontal information flow between tokens, two ways.
 *
 * <ul>
 *   <li><b>Canon layers</b> ({@link CanonLayer}): associative recall with a single
 *       transformer block and no previous-token feature. Plain attention then cannot bind a
 *       value to the key before it in one layer; a short causal convolution before attention
 *       and before the MLP can. Compared with and without canon layers, around the
 *       {@link RefinedAttention} block and around the core transformer block.</li>
 *   <li><b>Hymba</b> ({@link ParallelHybridBlock}, attention and Mamba heads in parallel with
 *       meta tokens) against the sequential Kimi-style 3:1 {@link HybridLinearStack}: the
 *       {@link MixerShootout} recall and recent-recall tasks and the
 *       {@link ArchitectureBench} toy-language loss.</li>
 * </ul>
 */
public final class HybridAndCanonShootout {

    private HybridAndCanonShootout() {
    }

    /**
     * The canon comparison: one block each, with and without canon layers.
     *
     * @param width the model width
     * @return name to body spec
     */
    public static Map<String, LayerSpec> canonLineup(int width) {
        RefinedAttention.Spec attention = RefinedAttention.of(2).causal(true).rotary(10000f);
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("block", ResidualMixerBlock.of(attention));
        m.put("block + canon", ResidualMixerBlock.of(attention).canon(4));
        m.put("core block", ArchitectureBench.coreBlock(2, width));
        m.put("canon + core block", LayerChain.of(CanonLayer.of(4), ArchitectureBench.coreBlock(2, width)));
        return m;
    }

    /**
     * Associative recall (8 keys, 8 values, 4 pairs) without a previous-token feature.
     *
     * @param names entries of {@link #canonLineup}
     * @param width the model width
     * @param steps training steps
     * @param seed the seed
     * @return the accuracy per name
     */
    public static Map<String, Double> canon(List<String> names, int width, int steps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        MixerShootout.Task task = MixerShootout.recall(8, 8, 4);
        for (String n : names) {
            out.put(n, ArchitectureBench.retrieval(canonLineup(width).get(n), width, task, List.of(task), steps, 0.01f,
                    seed, false)[0]);
        }
        return out;
    }

    /**
     * Hymba against the sequential hybrid.
     *
     * @return name to body spec
     */
    public static Map<String, LayerSpec> hybridLineup() {
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("Hymba x2", ParallelHybridBlock.of(2, 2, 4));
        m.put("Kimi 3:1 x4", HybridLinearStack.of(2, 4, 3));
        return m;
    }

    /**
     * One hybrid's results.
     *
     * @param name the hybrid
     * @param parameters its parameter count
     * @param recall the recall accuracy
     * @param recentRecall the recent-recall accuracy
     * @param validationLoss the toy-language loss
     */
    public record HybridRow(String name, long parameters, double recall, double recentRecall,
            double validationLoss) {
    }

    /**
     * Trains each hybrid on the shoot-out tasks and the toy language; a step count of 0 skips
     * that measurement (reported as NaN).
     *
     * @param recallSteps steps for associative recall
     * @param recentSteps steps for recent recall
     * @param lmSteps steps for the toy language
     * @param seed the seed
     * @return one row per hybrid
     */
    public static List<HybridRow> hybrids(int recallSteps, int recentSteps, int lmSteps, long seed) {
        List<HybridRow> rows = new ArrayList<>();
        for (Map.Entry<String, LayerSpec> e : hybridLineup().entrySet()) {
            MixerShootout.Result recall = MixerShootout.run(e.getKey(), e.getValue(), MixerShootout.recall(8, 8, 4),
                    16, recallSteps, 0.01f, seed);
            double recent = recentSteps == 0 ? Double.NaN : MixerShootout.run(e.getKey(), e.getValue(),
                    MixerShootout.recentRecall(3, 4, 4), 16, recentSteps, 0.01f, seed).accuracy();
            double lm = lmSteps == 0 ? Double.NaN
                    : ArchitectureBench.languageModel(e.getKey(), e.getValue(), 32, lmSteps, seed).validationLoss();
            rows.add(new HybridRow(e.getKey(), recall.parameters(), recall.accuracy(), recent, lm));
        }
        return rows;
    }

    /**
     * Runs both comparisons.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.println(canon(new ArrayList<>(canonLineup(32).keySet()), 32, 400, 1));
        for (HybridRow r : hybrids(300, 300, 400, 1)) {
            System.out.println(r);
        }
    }
}
