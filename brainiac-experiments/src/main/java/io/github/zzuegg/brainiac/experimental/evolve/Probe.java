package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Probe: bottom-up search guided by a probabilistic grammar learned during the search,
 * with no pretraining. Primitive costs are {@code -log2 p(op)}, starting uniform. Whenever
 * the search makes a <em>partial solution</em>, a program that is right on a set of
 * examples no earlier partial solution was right on (nor on a subset of such a set), the grammar is
 * updated, {@code p(op) ∝ p_uniform^(1 - fit(op))} with {@code fit(op)} the largest share
 * of examples solved by a partial solution using {@code op}, and the search restarts with
 * the new costs. Every restart's cost counts against the budget.
 *
 * <p>Paper: Barke, Peleg and Polikarpova, "Just-in-Time Learning for Bottom-Up Enumerative
 * Synthesis" (OOPSLA 2020), <a href="https://doi.org/10.1145/3428295">doi:10.1145/3428295</a>.
 */
public final class Probe {

    /**
     * A search outcome.
     *
     * @param program the program found, or {@code null}
     * @param evaluated primitive applications over all restarts
     * @param restarts grammar updates
     * @param probabilities the final grammar, one probability per primitive
     */
    public record Result(List<String> program, long evaluated, int restarts, double[] probabilities) {
        /** @return whether a program was found */
        public boolean solved() {
            return program != null;
        }
    }

    private final BottomUpSynthesis search;
    private final int maxRestarts;

    /**
     * Creates the search.
     *
     * @param search the bottom-up search
     * @param maxRestarts the most grammar updates per task
     */
    public Probe(BottomUpSynthesis search, int maxRestarts) {
        this.search = search;
        this.maxRestarts = maxRestarts;
    }

    /**
     * Synthesises with just-in-time grammar learning.
     *
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @param maxLength the longest program
     * @param budget the most primitive applications over all restarts
     * @return the outcome
     */
    public Result search(int[][] inputs, int[][] outputs, int maxLength, long budget) {
        List<String> prims = search.primitives();
        int k = prims.size();
        int examples = inputs.length;
        double[] fit = new double[k];
        double[] p = new double[k];
        Arrays.fill(p, 1.0 / k);
        List<boolean[]> solvedSets = new ArrayList<>();
        boolean[] start = solvedBy(inputs, outputs);
        if (count(start) > 0) {
            solvedSets.add(start);
        }
        long spent = 0;
        int restarts = 0;
        while (true) {
            double[] costs = new double[k];
            for (int i = 0; i < k; i++) {
                costs[i] = -Math.log(p[i]) / Math.log(2);
            }
            boolean learn = restarts < maxRestarts;
            BottomUpSynthesis.Listener listener = !learn ? null : (program, value) -> {
                boolean[] s = solvedBy(value, outputs);
                if (count(s) == 0) {
                    return false;
                }
                for (boolean[] old : solvedSets) {
                    if (subset(s, old)) {
                        return false;
                    }
                }
                solvedSets.add(s);
                double share = count(s) / (double) examples;
                for (String op : program) {
                    int i = prims.indexOf(op);
                    fit[i] = Math.max(fit[i], share);
                }
                return true;
            };
            BottomUpSynthesis.Result r = search.search(inputs, outputs, maxLength, costs, null, listener, budget - spent);
            spent += r.evaluated();
            if (r.solved()) {
                return new Result(r.program(), spent, restarts, p);
            }
            if (!r.stopped() || spent >= budget) {
                return new Result(null, spent, restarts, p);
            }
            restarts++;
            double z = 0;
            for (int i = 0; i < k; i++) {
                p[i] = Math.pow(1.0 / k, 1 - fit[i]);
                z += p[i];
            }
            for (int i = 0; i < k; i++) {
                p[i] /= z;
            }
        }
    }

    private static boolean[] solvedBy(int[][] value, int[][] outputs) {
        boolean[] s = new boolean[value.length];
        for (int e = 0; e < value.length; e++) {
            s[e] = Arrays.equals(value[e], outputs[e]);
        }
        return s;
    }

    private static int count(boolean[] s) {
        int c = 0;
        for (boolean b : s) {
            c += b ? 1 : 0;
        }
        return c;
    }

    private static boolean subset(boolean[] a, boolean[] b) {
        for (int i = 0; i < a.length; i++) {
            if (a[i] && !b[i]) {
                return false;
            }
        }
        return true;
    }
}
