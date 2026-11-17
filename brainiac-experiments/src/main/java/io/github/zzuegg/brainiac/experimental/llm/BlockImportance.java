package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.Arrays;
import java.util.List;

/**
 * Layer removal by block importance (ShortGPT, Men et al. 2024,
 * <a href="https://arxiv.org/abs/2403.03853">arXiv:2403.03853</a>): many blocks of a trained
 * decoder barely change the residual stream. A block's importance is
 * {@code BI = 1 - E[cos(h_in, h_out)]} over calibration tokens; removing the blocks of lowest
 * importance shrinks the model with little loss, often more gracefully than width pruning.
 */
public final class BlockImportance {

    private BlockImportance() {
    }

    /**
     * @param model the decoder
     * @param calibration {@code [batch][steps]} calibration tokens
     * @return the importance of each block
     */
    public static double[] scores(HookedDecoder model, int[][] calibration) {
        List<Var> hidden = model.forward(calibration).hidden();
        int dim = model.settings().dim();
        double[] bi = new double[model.layers()];
        for (int l = 0; l < model.layers(); l++) {
            float[] a = hidden.get(l).value().toArray();
            float[] b = hidden.get(l + 1).value().toArray();
            int rows = a.length / dim;
            double sum = 0;
            for (int r = 0; r < rows; r++) {
                double dot = 0;
                double na = 0;
                double nb = 0;
                for (int i = 0; i < dim; i++) {
                    dot += (double) a[r * dim + i] * b[r * dim + i];
                    na += (double) a[r * dim + i] * a[r * dim + i];
                    nb += (double) b[r * dim + i] * b[r * dim + i];
                }
                sum += dot / Math.sqrt(na * nb + 1e-12);
            }
            bi[l] = 1 - sum / rows;
        }
        return bi;
    }

    /**
     * @param scores the importances
     * @param count how many to remove
     * @return the least important blocks, ascending by index
     */
    public static int[] leastImportant(double[] scores, int count) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        int[] out = new int[count];
        for (int i = 0; i < count; i++) {
            out[i] = order[i];
        }
        Arrays.sort(out);
        return out;
    }

    /**
     * Skips the {@code count} least important blocks of {@code model}.
     *
     * @param model the decoder; its skipped blocks are set
     * @param calibration calibration tokens
     * @param count blocks to remove
     * @return the removed blocks
     */
    public static int[] prune(HookedDecoder model, int[][] calibration, int count) {
        model.skip();
        int[] drop = leastImportant(scores(model, calibration), count);
        model.skip(drop);
        return drop;
    }
}
