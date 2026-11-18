package io.github.zzuegg.brainiac.experimental.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * Black-box access to a classifier: it returns logits (score-based attacks) or only the
 * predicted label (decision-based attacks) and counts the queries, one per input row. A
 * batch of rows costs as many queries as it has rows, so attacks may evaluate candidates in
 * batches for speed without hiding their query cost. Counts can be attributed to the
 * example being attacked.
 */
public final class QueryOracle {

    private final UnaryOperator<Var> model;
    private long total;
    private final long[] perExample;

    /**
     * Wraps a model.
     *
     * @param model the logits function
     * @param examples the number of attacked examples whose queries are counted separately
     */
    public QueryOracle(UnaryOperator<Var> model, int examples) {
        this.model = Objects.requireNonNull(model);
        this.perExample = new long[examples];
    }

    /**
     * Logits of a batch, counted against example ids.
     *
     * @param rows the inputs
     * @param owner the attacked example each row belongs to
     * @return the logits, one row each
     */
    public float[][] logits(float[][] rows, int[] owner) {
        for (int o : owner) {
            perExample[o]++;
        }
        total += rows.length;
        Tensor z = model.apply(Var.of(Tensor.of(rows))).value();
        int k = z.shape().dim(1);
        float[] v = z.toArray();
        float[][] out = new float[rows.length][k];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(v, i * k, out[i], 0, k);
        }
        return out;
    }

    /**
     * Predicted labels of a batch, counted against example ids.
     *
     * @param rows the inputs
     * @param owner the attacked example each row belongs to
     * @return the argmax labels
     */
    public int[] labels(float[][] rows, int[] owner) {
        float[][] z = logits(rows, owner);
        int[] out = new int[z.length];
        for (int i = 0; i < z.length; i++) {
            out[i] = argmax(z[i]);
        }
        return out;
    }

    /** @return all queries so far */
    public long total() {
        return total;
    }

    /**
     * @param example an attacked example
     * @return its queries so far
     */
    public long queries(int example) {
        return perExample[example];
    }

    /**
     * The margin loss {@code z_y - max_{j != y} z_j}; negative means misclassified.
     *
     * @param z logits
     * @param y the true class
     * @return the margin
     */
    public static float margin(float[] z, int y) {
        float other = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < z.length; j++) {
            if (j != y) {
                other = Math.max(other, z[j]);
            }
        }
        return z[y] - other;
    }

    /**
     * The index of the largest entry.
     *
     * @param z values
     * @return the argmax
     */
    public static int argmax(float[] z) {
        int best = 0;
        for (int j = 1; j < z.length; j++) {
            best = z[j] > z[best] ? j : best;
        }
        return best;
    }
}
