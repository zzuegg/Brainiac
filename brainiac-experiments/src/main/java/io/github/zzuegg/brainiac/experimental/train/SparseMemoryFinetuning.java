package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Sparse memory fine-tuning (Lin et al. 2025, "Continual Learning via Sparse Memory
 * Finetuning", <a href="https://arxiv.org/abs/2510.15103">arXiv:2510.15103</a>): to learn new
 * facts without forgetting, update only the memory-layer slots that the new data uses
 * <em>much more</em> than the pretraining data did. Slots are ranked by TF-IDF, term
 * frequency the number of new examples that read the slot and inverse document frequency
 * {@code log((N + 1) / (df + 1))} over {@code N} background examples, and only the top
 * {@code t} value rows of the memory table receive their gradient; every other parameter
 * stays frozen.
 *
 * <p>An {@link Optimizer} wrapper: give it the memory table (for example
 * {@code ProductKeyMemory.values()}), record which rows background examples read
 * ({@link #recordBackground}), call {@link #select} with the rows each new example reads,
 * and step as usual. Which rows an example reads is taken from the rows of the table that
 * received a gradient ({@link #accessedRows}).
 */
public final class SparseMemoryFinetuning implements Optimizer {

    private final Var table;
    private final int trainable;
    private final Optimizer inner;
    private final int rows;
    private final int width;
    private final int[] documentFrequency;
    private int documents;
    private boolean[] mask;

    /**
     * @param table the memory value table {@code [slots, valueSize]}
     * @param trainable how many slots each new batch may update, {@code t}
     * @param inner the optimizer applied to the selected rows, plain SGD in the paper
     */
    public SparseMemoryFinetuning(Var table, int trainable, Optimizer inner) {
        this.table = table;
        this.trainable = trainable;
        this.inner = inner;
        this.rows = table.value().shape().dim(0);
        this.width = Math.toIntExact(table.value().size() / rows);
        this.documentFrequency = new int[rows];
    }

    /**
     * @param table a memory table after a backward pass
     * @return the rows with a non-zero gradient, the slots that were read
     */
    public static int[] accessedRows(Var table) {
        Tensor g = table.grad();
        if (g == null) {
            return new int[0];
        }
        int n = g.shape().dim(0);
        int w = Math.toIntExact(g.size() / n);
        float[] v = g.toArray();
        List<Integer> out = new ArrayList<>();
        for (int r = 0; r < n; r++) {
            for (int k = 0; k < w; k++) {
                if (v[r * w + k] != 0) {
                    out.add(r);
                    break;
                }
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Counts one background (pretraining) example.
     *
     * @param accessed the rows it read
     */
    public void recordBackground(int[] accessed) {
        documents++;
        for (int r : accessed) {
            documentFrequency[r]++;
        }
    }

    /**
     * Chooses the trainable slots for a new batch by TF-IDF.
     *
     * @param accessedPerExample the rows each new example read
     * @return the chosen rows
     */
    public int[] select(List<int[]> accessedPerExample) {
        double[] score = new double[rows];
        for (int[] example : accessedPerExample) {
            for (int r : example) {
                score[r] += 1;
            }
        }
        for (int r = 0; r < rows; r++) {
            score[r] *= Math.log((documents + 1.0) / (documentFrequency[r] + 1.0));
        }
        Integer[] order = new Integer[rows];
        for (int r = 0; r < rows; r++) {
            order[r] = r;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
        mask = new boolean[rows];
        List<Integer> chosen = new ArrayList<>();
        for (int i = 0; i < Math.min(trainable, rows) && score[order[i]] > 0; i++) {
            mask[order[i]] = true;
            chosen.add(order[i]);
        }
        return chosen.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public void step(List<Var> parameters) {
        for (Var p : parameters) {
            if (p != table) {
                p.zeroGrad();
            }
        }
        Tensor g = table.grad();
        if (g == null || mask == null) {
            table.zeroGrad();
            return;
        }
        float[] v = g.toArray();
        for (int r = 0; r < rows; r++) {
            if (!mask[r]) {
                java.util.Arrays.fill(v, r * width, (r + 1) * width, 0f);
            }
        }
        table.setGrad(Tensor.of(g.shape(), v));
        inner.step(List.of(table));
    }

    @Override
    public float learningRate() {
        return inner.learningRate();
    }

    @Override
    public void learningRate(float learningRate) {
        inner.learningRate(learningRate);
    }

    @Override
    public String toString() {
        return "SparseMemoryFinetuning(t=" + trainable + ", " + inner + ")";
    }
}
