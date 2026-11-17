package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.experimental.nn.MultiTokenAttention;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.TalkingHeadsAttention;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Multi-token attention ({@link MultiTokenAttention}: convolutions over the query, key and
 * head axes of the attention logits) against multi-head attention and talking-heads
 * attention, on a lookup that needs two tokens' evidence at once, and on the
 * {@link ArchitectureBench} toy language.
 *
 * <p>The lookup ({@link #blockLookup}): the context is a list of blocks
 * {@code x y z label}, then two query letters {@code A B}; the answer is the label of the
 * one block that contains both. In standard attention each weight depends on one key, so a
 * single layer cannot weigh "A here and B two tokens earlier" into one choice; MTA's key
 * convolution sums the match evidence of a block's letters onto its label.
 */
public final class MultiTokenLookup {

    private static final int LETTERS = 8;

    private MultiTokenLookup() {
    }

    /**
     * The block lookup task: labels {@code 0..blocks-1}, letters {@code blocks..blocks+7}.
     *
     * @param blocks the blocks per sequence
     * @return the task (answer = a label)
     */
    public static MixerShootout.Task blockLookup(int blocks) {
        int length = 4 * blocks + 2;
        return new MixerShootout.Task("blocks(" + blocks + ")", blocks + LETTERS, length, (random, in) -> {
            int[] target = new int[in.length];
            for (int b = 0; b < in.length; b++) {
                int[] labels = new int[blocks];
                for (int i = 0; i < blocks; i++) {
                    labels[i] = i;
                }
                for (int i = blocks - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    int t = labels[i];
                    labels[i] = labels[j];
                    labels[j] = t;
                }
                int answer = random.nextInt(blocks);
                int a = random.nextInt(LETTERS);
                int c = (a + 1 + random.nextInt(LETTERS - 1)) % LETTERS;
                for (int k = 0; k < blocks; k++) {
                    int[] letters = new int[3];
                    do {
                        for (int s = 0; s < 3; s++) {
                            letters[s] = random.nextInt(LETTERS);
                        }
                        if (k == answer) {
                            int pa = random.nextInt(3);
                            int pc = (pa + 1 + random.nextInt(2)) % 3;
                            letters[pa] = a;
                            letters[pc] = c;
                        }
                    } while (k != answer && contains(letters, a) && contains(letters, c));
                    for (int s = 0; s < 3; s++) {
                        in[b][4 * k + s] = blocks + letters[s];
                    }
                    in[b][4 * k + 3] = labels[k];
                }
                in[b][length - 2] = blocks + a;
                in[b][length - 1] = blocks + c;
                target[b] = labels[answer];
            }
            return target;
        });
    }

    private static boolean contains(int[] letters, int v) {
        for (int l : letters) {
            if (l == v) {
                return true;
            }
        }
        return false;
    }

    /**
     * The lineup, one causal attention layer each (2 heads, RoPE).
     *
     * @return name to attention spec
     */
    public static Map<String, LayerSpec> lineup() {
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("MTA 2x7", MultiTokenAttention.of(2, 2, 7, true));
        m.put("MHA", RefinedAttention.of(2).causal(true).rotary(10000f));
        m.put("talking heads", TalkingHeadsAttention.of(2, true, 10000f));
        return m;
    }

    /**
     * One configuration's result.
     *
     * @param name the configuration
     * @param lookup the block-lookup accuracy
     * @param validationLoss the toy-language validation loss (NaN when not run)
     */
    public record Row(String name, double lookup, double validationLoss) {
    }

    /**
     * Trains named lineup entries on the lookup and on the toy language.
     *
     * @param names the lineup entries
     * @param blocks blocks per lookup sequence
     * @param steps lookup training steps
     * @param lmSteps toy-language training steps, 0 to skip
     * @param seed the seed
     * @return one row per entry
     */
    public static List<Row> run(List<String> names, int blocks, int steps, int lmSteps, long seed) {
        MixerShootout.Task task = blockLookup(blocks);
        List<Row> rows = new ArrayList<>();
        for (Map.Entry<String, LayerSpec> e : lineup().entrySet()) {
            if (!names.contains(e.getKey())) {
                continue;
            }
            double acc = ArchitectureBench.retrieval(ArchitectureBench.stack(e.getValue(), 1), 16, task,
                    List.of(task), steps, 0.01f, seed, true)[0];
            double loss = lmSteps > 0 ? ArchitectureBench.languageModel(e.getKey(),
                    ArchitectureBench.stack(e.getValue(), 1), 16, lmSteps, seed).validationLoss() : Double.NaN;
            rows.add(new Row(e.getKey(), acc, loss));
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
        StringBuilder sb = new StringBuilder(String.format("%-14s %8s %9s%n", "attention", "lookup", "val loss"));
        for (Row r : rows) {
            sb.append(String.format("%-14s %8.3f %9.4f%n", r.name(), r.lookup(), r.validationLoss()));
        }
        return sb.toString();
    }

    /**
     * Runs the comparison.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(table(run(new ArrayList<>(lineup().keySet()), 4, 800, 400, 1)));
    }
}
