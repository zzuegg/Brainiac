package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.experimental.nn.ContextualPositionAttention;
import io.github.zzuegg.brainiac.experimental.nn.ForgettingAttention;
import io.github.zzuegg.brainiac.experimental.nn.PathPositionAttention;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experiments.attention.ArchitectureBench;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Position encodings that depend on the content, against fixed ones, on tasks where token
 * indices are the wrong ruler. Every model is one pre-norm block around the attention
 * ({@link ArchitectureBench#stack}), trained at one length and tested at longer ones.
 *
 * <ul>
 *   <li><b>Flip-flop</b> ({@link #flipFlop}): a stream of "write 0/1" and "ignore 0/1"
 *       instructions ends in "read"; the answer is the last written bit. The distance to the
 *       last write varies and distractors look alike, so the layer must track state.
 *       {@link PathPositionAttention} (PaTH, accumulated Householder transforms) and PaTH-FoX
 *       against RoPE, NoPE and {@link ForgettingAttention}.</li>
 *   <li><b>Count the k-th word</b> ({@link #kthWord}): words of 1 to 3 letters separated by a
 *       separator, then a query "k"; the answer is the first letter of the k-th last word.
 *       Word lengths vary, so the target sits at a varying token distance but a fixed
 *       separator count. {@link ContextualPositionAttention} (CoPE) against RoPE and
 *       stick-breaking attention.</li>
 * </ul>
 */
public final class DataDependentPositions {

    private static final int LETTERS = 8;

    private DataDependentPositions() {
    }

    /**
     * One configuration's accuracies.
     *
     * @param name the configuration
     * @param lengths the test lengths
     * @param accuracy the accuracy at each
     */
    public record Row(String name, int[] lengths, double[] accuracy) {
    }

    /**
     * The flip-flop language: tokens {@code w0 w1 i0 i1 r} (0-4); the first instruction is a
     * write, later ones write with probability 0.2; the last token is {@code r}.
     *
     * @param length the sequence length
     * @return the task (answer class 0 or 1)
     */
    public static MixerShootout.Task flipFlop(int length) {
        return new MixerShootout.Task("flipflop(" + length + ")", 5, length, (random, in) -> {
            int[] target = new int[in.length];
            for (int b = 0; b < in.length; b++) {
                int bit = random.nextInt(2);
                in[b][0] = bit;
                for (int i = 1; i < length - 1; i++) {
                    int v = random.nextInt(2);
                    if (random.nextDouble() < 0.2) {
                        in[b][i] = v;
                        bit = v;
                    } else {
                        in[b][i] = 2 + v;
                    }
                }
                in[b][length - 1] = 4;
                target[b] = bit;
            }
            return target;
        });
    }

    /**
     * The k-th-word task: letters 0-7, separator 8, queries 9-11 for {@code k = 1, 2, 3}.
     * Sequences are {@code word sep word sep ... word sep q}, filled to the length (the first
     * word may be cut off at the front).
     *
     * @param length the sequence length
     * @return the task (answer = a letter)
     */
    public static MixerShootout.Task kthWord(int length) {
        return new MixerShootout.Task("kthWord(" + length + ")", LETTERS + 4, length, (random, in) -> {
            int[] target = new int[in.length];
            for (int b = 0; b < in.length; b++) {
                int k = 1 + random.nextInt(3);
                in[b][length - 1] = LETTERS + k;
                int pos = length - 2;
                int word = 0;
                while (pos >= 0) {
                    in[b][pos--] = LETTERS;                       // separator
                    word++;
                    int len = 1 + random.nextInt(3);
                    int first = -1;
                    for (int c = 0; c < len && pos >= 0; c++) {
                        first = random.nextInt(LETTERS);
                        in[b][pos--] = first;
                    }
                    if (word == k) {
                        target[b] = first;
                    }
                }
            }
            return target;
        });
    }

    /**
     * The flip-flop lineup.
     *
     * @return name to attention spec
     */
    public static Map<String, LayerSpec> flipFlopLineup() {
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("PaTH", PathPositionAttention.of(2, false));
        m.put("PaTH-FoX", PathPositionAttention.of(2, true));
        m.put("RoPE", RefinedAttention.of(2).causal(true).rotary(10000f));
        m.put("NoPE", RefinedAttention.of(2).causal(true));
        m.put("FoX", ForgettingAttention.of(2, true));
        return m;
    }

    /**
     * The k-th-word lineup.
     *
     * @return name to attention spec
     */
    public static Map<String, LayerSpec> countingLineup() {
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("CoPE", ContextualPositionAttention.of(2));
        m.put("CoPE sep-keys", ContextualPositionAttention.of(2, 16, true));
        m.put("RoPE", RefinedAttention.of(2).causal(true).rotary(10000f));
        m.put("stick-breaking", RefinedAttention.of(2).causal(true).weighting(RefinedAttention.Weighting.STICK_BREAKING));
        return m;
    }

    /**
     * Trains named entries of a lineup on a task at the first length and tests at all.
     *
     * @param lineup name to attention spec
     * @param names the entries to run
     * @param task builds the task for a length
     * @param lengths the lengths, the first one for training
     * @param steps training steps
     * @param seed the seed
     * @return one row per name (accuracy on 128 fresh samples per length)
     */
    public static List<Row> run(Map<String, LayerSpec> lineup, List<String> names,
            java.util.function.IntFunction<MixerShootout.Task> task, int[] lengths, int steps, long seed) {
        List<MixerShootout.Task> tests = new ArrayList<>();
        for (int l : lengths) {
            tests.add(task.apply(l));
        }
        List<Row> rows = new ArrayList<>();
        for (String n : names) {
            ArchitectureBench.TinyModel m = ArchitectureBench.trainRetrieval(ArchitectureBench.stack(lineup.get(n), 1),
                    16, tests.getFirst(), steps, 0.01f, seed, true);
            double[] acc = new double[tests.size()];
            for (int k = 0; k < acc.length; k++) {
                acc[k] = ArchitectureBench.accuracy(m, tests.get(k).batch(new SplittableRandom(seed + 1000 + k), 128));
            }
            rows.add(new Row(n, lengths, acc));
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
        System.out.print(table("flip-flop @ len", run(flipFlopLineup(), new ArrayList<>(flipFlopLineup().keySet()),
                DataDependentPositions::flipFlop, new int[] {16, 32, 48}, 400, 1)));
        System.out.print(table("k-th word @ len", run(countingLineup(), new ArrayList<>(countingLineup().keySet()),
                DataDependentPositions::kthWord, new int[] {16, 24, 32}, 400, 1)));
    }
}
