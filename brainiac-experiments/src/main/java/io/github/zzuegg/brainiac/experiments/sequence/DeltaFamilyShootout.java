package io.github.zzuegg.brainiac.experiments.sequence;

import io.github.zzuegg.brainiac.experimental.nn.DeltaProductLayer;
import io.github.zzuegg.brainiac.experimental.nn.GatedDeltaNet;
import io.github.zzuegg.brainiac.experimental.nn.KimiDeltaAttention;
import io.github.zzuegg.brainiac.experimental.nn.LogLinearAttention;
import io.github.zzuegg.brainiac.experimental.nn.Mamba2Layer;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The 2025 refinements of delta-rule and linear-attention mixers on the {@link MixerShootout}
 * diagnostics:
 *
 * <ul>
 *   <li><b>Kimi Delta Attention</b> ({@link KimiDeltaAttention}, channel-wise decay,
 *       chunkwise) against Gated DeltaNet (scalar decay) and Mamba-2 on associative recall;</li>
 *   <li><b>log-linear attention</b> ({@link LogLinearAttention}, {@code O(log T)} Fenwick
 *       states) against the same layer with one state (gated linear attention) and softmax
 *       attention, on recall with many pairs, where one fixed-size state runs out of
 *       capacity;</li>
 *   <li><b>DeltaProduct</b> ({@link DeltaProductLayer}) on state tracking: composing
 *       permutations of the non-abelian group S3 ({@link #permutations}) and parity, at the
 *       training length and twice it. One Householder step per token ({@code n_h = 1}, a
 *       DeltaNet with negative eigenvalues) can only reflect; two can rotate.</li>
 * </ul>
 */
public final class DeltaFamilyShootout {

    /** The six elements of S3 as images of (0, 1, 2); index 0 is the identity. */
    private static final int[][] S3 = {{0, 1, 2}, {1, 0, 2}, {0, 2, 1}, {2, 1, 0}, {1, 2, 0}, {2, 0, 1}};

    private DeltaFamilyShootout() {
    }

    /**
     * Composition of random S3 permutations: tokens are group elements, the answer is the
     * product of all of them (a state-tracking task in NC1, beyond diagonal linear RNNs).
     *
     * @param length the number of permutations
     * @return the task (6 tokens, 6 classes)
     */
    public static MixerShootout.Task permutations(int length) {
        return new MixerShootout.Task("S3(" + length + ")", 6, length, (random, in) -> {
            int[] target = new int[in.length];
            for (int b = 0; b < in.length; b++) {
                int[] state = {0, 1, 2};
                for (int i = 0; i < length; i++) {
                    int g = random.nextInt(6);
                    in[b][i] = g;
                    int[] next = new int[3];
                    for (int c = 0; c < 3; c++) {
                        next[c] = S3[g][state[c]];
                    }
                    state = next;
                }
                target[b] = index(state);
            }
            return target;
        });
    }

    private static int index(int[] p) {
        for (int i = 0; i < S3.length; i++) {
            if (S3[i][0] == p[0] && S3[i][1] == p[1] && S3[i][2] == p[2]) {
                return i;
            }
        }
        throw new IllegalStateException();
    }

    /**
     * The recall lineup.
     *
     * @return name to mixer spec
     */
    public static Map<String, LayerSpec> recallLineup() {
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("softmax attention", RefinedAttention.of(2).causal(true).rotary(10000f));
        m.put("Gated DeltaNet", GatedDeltaNet.of(2, GatedDeltaNet.Gating.SCALAR, 4));
        m.put("KDA", KimiDeltaAttention.of(2, 4, 4));
        m.put("Mamba-2", Mamba2Layer.of(2, 8, 8, 4, Mamba2Layer.Variant.REAL));
        m.put("gated linear", LogLinearAttention.of(2, false));
        m.put("log-linear", LogLinearAttention.of(2, true));
        return m;
    }

    /**
     * The state-tracking lineup.
     *
     * @return name to mixer spec
     */
    public static Map<String, LayerSpec> trackingLineup() {
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("softmax attention", RefinedAttention.of(2).causal(true).rotary(10000f));
        m.put("Gated DeltaNet", GatedDeltaNet.of(2, GatedDeltaNet.Gating.SCALAR, 4));
        m.put("DeltaProduct n=1", DeltaProductLayer.of(1, 1));
        m.put("DeltaProduct n=2", DeltaProductLayer.of(1, 2));
        return m;
    }

    /**
     * Runs named entries of a lineup on tasks.
     *
     * @param lineup name to spec
     * @param names the entries to run
     * @param tasks the tasks
     * @param width the model width
     * @param steps Adam steps per run
     * @param seed the seed
     * @return the results, mixer-major
     */
    public static List<MixerShootout.Result> run(Map<String, LayerSpec> lineup, List<String> names,
            List<MixerShootout.Task> tasks, int width, int steps, long seed) {
        Map<String, LayerSpec> chosen = new LinkedHashMap<>();
        for (String n : names) {
            chosen.put(n, lineup.get(n));
        }
        return MixerShootout.run(chosen, tasks, width, steps, 0.01f, seed);
    }

    /**
     * Looks up one result.
     *
     * @param results the results
     * @param mixer the mixer name
     * @param task the task name
     * @return the accuracy
     */
    public static double accuracy(List<MixerShootout.Result> results, String mixer, String task) {
        for (MixerShootout.Result r : results) {
            if (r.mixer().equals(mixer) && r.task().equals(task)) {
                return r.accuracy();
            }
        }
        throw new IllegalArgumentException(mixer + " on " + task);
    }

    /**
     * Prints both comparisons.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(MixerShootout.table(run(recallLineup(), new ArrayList<>(recallLineup().keySet()),
                List.of(MixerShootout.recall(8, 8, 4), MixerShootout.recall(16, 16, 12)), 32, 400, 1)));
        System.out.print(MixerShootout.table(run(trackingLineup(), new ArrayList<>(trackingLineup().keySet()),
                List.of(permutations(3), permutations(4), MixerShootout.parity(8)), 16, 600, 1)));
    }
}
