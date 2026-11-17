package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.experimental.nn.InterleavedRopeStack;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention.Weighting;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.experiments.techniques.PositionShootout;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Length generalization of attention variants: train at one length, test at longer ones.
 *
 * <ul>
 *   <li><b>Exact lag</b> (as in {@link PositionShootout}): output the token two steps back, at
 *       every step. It needs relative positions. Stick-breaking attention (no
 *       position encoding, recency built in) against RoPE and NoPE softmax attention and
 *       sigmoid attention.</li>
 *   <li><b>Associative recall with more distractors</b>: trained with 4 key-value pairs, tested
 *       with 8 and 16, so the attention has to stay sharp over two to four times as many
 *       keys. Scalable-Softmax ({@code s log n} logit scaling) and sigmoid attention against
 *       plain softmax, and the iRoPE stack (local RoPE layer, then a NoPE layer with the
 *       {@code log n} temperature) against all-RoPE and all-NoPE stacks.</li>
 * </ul>
 */
public final class LengthGeneralisation {

    private LengthGeneralisation() {
    }

    /**
     * One configuration's accuracies.
     *
     * @param name the configuration
     * @param lengths the test lengths (lag) or pair counts (recall)
     * @param accuracy the accuracy at each
     */
    public record Row(String name, int[] lengths, double[] accuracy) {
    }

    /**
     * The exact-lag lineup, used in two-block stacks (stick-breaking attention, with no
     * positions, builds a lag of {@code k} from {@code k} "attend to the nearest earlier step"
     * layers).
     *
     * @return name to attention spec
     */
    public static Map<String, LayerSpec> lagLineup() {
        RefinedAttention.Spec base = RefinedAttention.of(4).causal(true);
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("softmax RoPE", base.rotary(10000f));
        m.put("softmax NoPE", base);
        m.put("stick-breaking", base.weighting(Weighting.STICK_BREAKING));
        m.put("sigmoid RoPE", base.rotary(10000f).weighting(Weighting.SIGMOID));
        return m;
    }

    /**
     * The recall lineup: single blocks with NoPE attention, and two-layer position stacks.
     *
     * @return name to body spec
     */
    public static Map<String, LayerSpec> recallLineup() {
        RefinedAttention.Spec base = RefinedAttention.of(2).causal(true);
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("softmax", ArchitectureBench.stack(base, 1));
        m.put("SSMax", ArchitectureBench.stack(base.scalableSoftmax(true), 1));
        m.put("sigmoid", ArchitectureBench.stack(base.weighting(Weighting.SIGMOID), 1));
        m.put("all RoPE x2", InterleavedRopeStack.of(2, 2, 0, false, 0));
        m.put("all NoPE x2", InterleavedRopeStack.of(2, 2, 1, false, 0));
        m.put("iRoPE x2", InterleavedRopeStack.of(2, 2, 2, true, 4));
        return m;
    }

    /**
     * Trains the named lag configurations (two blocks) at length 12 with lag 2 and tests at
     * 12, 24 and 48.
     *
     * @param names configurations from {@link #lagLineup()}
     * @param steps training steps
     * @param seed the seed
     * @return one row per name
     */
    public static List<Row> lag(List<String> names, int steps, long seed) {
        int[] lengths = {12, 24, 48};
        List<Row> rows = new ArrayList<>();
        for (String n : names) {
            rows.add(new Row(n, lengths, ArchitectureBench.lag(ArchitectureBench.stack(lagLineup().get(n), 2), 32, 2,
                    12, lengths, steps, seed)));
        }
        return rows;
    }

    /**
     * Trains the named recall configurations with 4 pairs and tests with 4, 8 and 16.
     *
     * @param names configurations from {@link #recallLineup()}
     * @param steps training steps
     * @param seed the seed
     * @return one row per name
     */
    public static List<Row> recall(List<String> names, int steps, long seed) {
        int[] pairs = {4, 8, 16};
        List<MixerShootout.Task> tests = new ArrayList<>();
        for (int p : pairs) {
            tests.add(MixerShootout.recall(16, 8, p));
        }
        List<Row> rows = new ArrayList<>();
        for (String n : names) {
            rows.add(new Row(n, pairs, ArchitectureBench.retrieval(recallLineup().get(n), 16, tests.get(0), tests,
                    steps, 0.01f, seed, true)));
        }
        return rows;
    }

    /**
     * Formats rows.
     *
     * @param title the first column's title
     * @param rows the rows
     * @return a text table
     */
    public static String table(String title, List<Row> rows) {
        StringBuilder sb = new StringBuilder(String.format("%-16s", title));
        for (int l : rows.getFirst().lengths()) {
            sb.append(String.format(" %7d", l));
        }
        sb.append('\n');
        for (Row r : rows) {
            sb.append(String.format("%-16s", r.name()));
            for (double a : r.accuracy()) {
                sb.append(String.format(" %7.3f", a));
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    /**
     * Runs both lineups.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(table("lag @ length", lag(new ArrayList<>(lagLineup().keySet()), 400, 1)));
        System.out.print(table("recall @ pairs", recall(new ArrayList<>(recallLineup().keySet()), 400, 1)));
    }
}
