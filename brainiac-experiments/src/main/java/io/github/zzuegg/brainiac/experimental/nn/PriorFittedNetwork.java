package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.TabularPrior;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A prior-fitted network for small tables, after TabPFN (Hollmann et al., "TabPFN: A Transformer
 * That Solves Small Tabular Classification Problems in a Second", 2022,
 * <a href="https://arxiv.org/abs/2207.01848">arXiv:2207.01848</a>; Müller et al., "Transformers
 * Can Do Bayesian Inference", 2021, <a href="https://arxiv.org/abs/2112.10510">arXiv:2112.10510</a>).
 * The network is trained once, on many synthetic datasets drawn from a prior
 * ({@link TabularPrior}), to predict held-out labels from labelled rows in context. On a new
 * table it then classifies in a single forward pass, with no gradient steps: an amortised
 * approximation of the posterior predictive under the prior.
 * <ul>
 *   <li>every row is one token: its features (standardised with the context rows' statistics,
 *       zero-padded to {@code maxFeatures} and rescaled by {@code maxFeatures / features}) through
 *       a linear map, plus a label embedding (0, 1 or "unknown" for query rows);</li>
 *   <li>core {@link TransformerBlock}s run with an additive attention mask: every row attends to
 *       the context (training) rows only, so query rows never see each other;</li>
 *   <li>a head maps each query token to two logits.</li>
 * </ul>
 * This is the TabPFN v1 layout; v2's alternating attention over columns is left out.
 */
public final class PriorFittedNetwork {

    private static final float MASKED = -1e9f;

    private final int maxFeatures;
    private final int dim;
    private final Var embed;
    private final Var embedBias;
    private final Var labels;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final LayerNorm norm;
    private final Var head;
    private final Var headBias;
    private final Optimizer optimizer;

    /**
     * Creates an untrained network.
     *
     * @param maxFeatures the largest number of columns it accepts
     * @param dim the token width
     * @param heads attention heads, dividing {@code dim}
     * @param blocks transformer blocks
     * @param lr the Adam learning rate of {@link #trainStep}
     * @param random the initialization source
     */
    public PriorFittedNetwork(int maxFeatures, int dim, int heads, int blocks, float lr, SplittableRandom random) {
        this.maxFeatures = maxFeatures;
        this.dim = dim;
        this.embed = ClassicOps.matrix(random, maxFeatures, dim);
        this.embedBias = TabularOps.zeros(dim);
        this.labels = TabularOps.normal(random, 1f, 3, dim);
        for (int b = 0; b < blocks; b++) {
            this.blocks.add((TransformerBlock) TransformerBlock.of(heads).ffnSize(2 * dim)
                    .create(Shape.of(8, dim), random));
        }
        this.norm = LayerNorm.of(dim);
        this.head = ClassicOps.matrix(random, dim, 2);
        this.headBias = TabularOps.zeros(2);
        this.optimizer = Optimizer.adam(lr);
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, embedBias, labels));
        for (TransformerBlock b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(norm.parameters());
        p.add(head);
        p.add(headBias);
        return p;
    }

    /** @return the optimizer of {@link #trainStep} */
    public Optimizer optimizer() {
        return optimizer;
    }

    /**
     * Query logits for a batch of tasks of equal shape.
     *
     * @param tasks the datasets; each has the same number of rows and at most {@code maxFeatures} columns
     * @param context the leading rows of each task that are labelled context; the rest are queries
     * @param training the training flag
     * @return {@code [tasks * queries, 2]} logits, task by task
     */
    public Var logits(List<TabularPrior.Task> tasks, int context, boolean training) {
        int b = tasks.size();
        int rows = tasks.get(0).rows();
        float[] x = new float[b * rows * maxFeatures];
        float[] ids = new float[b * rows];
        for (int t = 0; t < b; t++) {
            TabularPrior.Task task = tasks.get(t);
            int f = task.x()[0].length;
            if (f > maxFeatures || task.rows() != rows) {
                throw new IllegalArgumentException("tasks need " + rows + " rows and at most " + maxFeatures
                        + " columns");
            }
            double rescale = maxFeatures / (double) f;
            for (int j = 0; j < f; j++) {
                double mean = 0;
                double sq = 0;
                for (int i = 0; i < context; i++) {
                    mean += task.x()[i][j];
                    sq += task.x()[i][j] * task.x()[i][j];
                }
                mean /= context;
                double sd = Math.sqrt(Math.max(1e-8, sq / context - mean * mean));
                for (int i = 0; i < rows; i++) {
                    double z = Math.max(-5, Math.min(5, (task.x()[i][j] - mean) / sd));
                    x[(t * rows + i) * maxFeatures + j] = (float) (z * rescale);
                }
            }
            for (int i = 0; i < rows; i++) {
                ids[t * rows + i] = i < context ? task.y()[i] : 2;
            }
        }
        Var h = Var.of(Tensor.of(Shape.of(b, rows, maxFeatures), x)).matmul(embed).add(embedBias)
                .add(labels.gather(0, Tensor.of(Shape.of(b * rows), ids)).reshape(b, rows, dim));
        float[] mask = new float[rows * rows];
        for (int i = 0; i < rows; i++) {
            for (int j = context; j < rows; j++) {
                mask[i * rows + j] = MASKED;
            }
        }
        Tensor bias = Tensor.of(Shape.of(1, 1, rows, rows), mask);
        for (TransformerBlock block : blocks) {
            h = block.forward(List.of(h), bias, training);
        }
        int queries = rows - context;
        Var q = h.slice(1, context, rows).reshape(b * queries, dim);
        return norm.forward(q).matmul(head).add(headBias);
    }

    /**
     * One Adam step of cross-entropy on the query labels of a batch of tasks.
     *
     * @param tasks the datasets, of equal shape
     * @param context labelled context rows per task
     * @return the loss
     */
    public float trainStep(List<TabularPrior.Task> tasks, int context) {
        int rows = tasks.get(0).rows();
        float[] y = new float[tasks.size() * (rows - context)];
        for (int t = 0; t < tasks.size(); t++) {
            for (int i = context; i < rows; i++) {
                y[t * (rows - context) + i - context] = tasks.get(t).y()[i];
            }
        }
        Var loss = Functional.crossEntropy(logits(tasks, context, true), Tensor.of(Shape.of(y.length), y));
        loss.backward();
        optimizer.step(parameters());
        return loss.value().toArray()[0];
    }

    /**
     * Pre-trains on datasets drawn from {@link TabularPrior#sample(int, int, SplittableRandom)}
     * with a random number of columns (2 to {@code maxFeatures}) and a random context size.
     *
     * @param steps training steps
     * @param batch tasks per step
     * @param rows rows per task
     * @param seed the seed of the prior
     */
    public void pretrain(int steps, int batch, int rows, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int f = 2 + r.nextInt(maxFeatures - 1);
            int context = rows / 4 + r.nextInt(rows / 2);
            List<TabularPrior.Task> tasks = new ArrayList<>();
            for (int b = 0; b < batch; b++) {
                tasks.add(TabularPrior.sample(rows, f, r));
            }
            trainStep(tasks, context);
        }
    }

    /**
     * Classifies new rows in context, with no training.
     *
     * @param trainX labelled rows {@code [n][features]}
     * @param trainY their 0/1 labels
     * @param testX the rows to classify
     * @return the probability of label 1 for each test row
     */
    public double[] predictProbability(double[][] trainX, int[] trainY, double[][] testX) {
        int n = trainX.length;
        double[][] x = new double[n + testX.length][];
        int[] y = new int[n + testX.length];
        System.arraycopy(trainX, 0, x, 0, n);
        System.arraycopy(testX, 0, x, n, testX.length);
        System.arraycopy(trainY, 0, y, 0, n);
        float[] l = Var.noGrad(() -> logits(List.of(new TabularPrior.Task(x, y)), n, false)).value().toArray();
        double[] p = new double[testX.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = 1 / (1 + Math.exp(l[2 * i] - l[2 * i + 1]));
        }
        return p;
    }

    @Override
    public String toString() {
        return "PriorFittedNetwork(" + maxFeatures + " features, " + dim + " wide, " + blocks.size() + " blocks)";
    }
}
