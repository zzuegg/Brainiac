package io.github.zzuegg.brainiac.experimental.recsys;

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
 * A TabICL-lite tabular in-context learner (after Qu et al., ICML 2025, "TabICL: A Tabular
 * Foundation Model for In-Context Learning on Large Data", <a href="https://arxiv.org/abs/2502.05564">
 * arXiv:2502.05564</a>), pre-trained on synthetic tasks from {@link TabularPrior} like the
 * {@code PriorFittedNetwork} but with TabICL's three-stage architecture:
 * <ol>
 *   <li><em>Column-wise embedding</em>: every column's cells are embedded by an induced set
 *       transformer (ISAB, Lee et al. 2019): {@code m} learned inducing points attend to the
 *       column's <em>training</em> cells, then every cell attends back to the inducing points,
 *       so each cell's embedding depends on its column's distribution with cost linear in the
 *       rows and without test-row leakage;</li>
 *   <li><em>Row-wise interaction</em>: a transformer over each row's cell embeddings plus a
 *       {@code [CLS]} token, with rotary position embeddings (RoPE, Su et al. 2021) on the column
 *       axis so columns stay distinguishable; the {@code [CLS]} output is the row embedding;</li>
 *   <li><em>Dataset-level ICL</em>: training rows get a label embedding, test rows an "unknown"
 *       one, and transformer blocks run over rows with a mask under which every row attends only
 *       to training rows; a head maps test rows to two class logits.</li>
 * </ol>
 * Pre-training follows a size curriculum: the rows per synthetic task grow linearly over the
 * run. Unlike the PFN, the column count is not fixed: any table with any number of columns is
 * embedded column by column.
 */
public final class ColumnRowIcl {

    private static final float MASKED = -1e9f;

    private final int dim;
    private final int heads;
    private final Var cellIn;
    private final Var cellBias;
    private final Var inducing;
    private final Var[] isab;
    private final LayerNorm columnNorm;
    private final Var cls;
    private final Var[] rowAttention;
    private final Var rowFfn1;
    private final Var rowFfn2;
    private final LayerNorm rowNorm1;
    private final LayerNorm rowNorm2;
    private final Var labels;
    private final List<TransformerBlock> icl = new ArrayList<>();
    private final LayerNorm norm;
    private final Var head;
    private final Var headBias;
    private final Optimizer optimizer;

    /**
     * Creates the model.
     *
     * @param dim the embedding width, divisible by {@code heads} with an even head width
     * @param heads attention heads of the row and ICL stages
     * @param inducingPoints inducing points of the column stage
     * @param iclBlocks transformer blocks of the ICL stage
     * @param lr the Adam learning rate
     * @param random the initializer
     */
    public ColumnRowIcl(int dim, int heads, int inducingPoints, int iclBlocks, float lr, SplittableRandom random) {
        if (dim % heads != 0 || (dim / heads) % 2 != 0) {
            throw new IllegalArgumentException("dim / heads must be an even integer");
        }
        this.dim = dim;
        this.heads = heads;
        this.cellIn = RecsysOps.matrix(random, 1, dim);
        this.cellBias = RecsysOps.normal(random, 0.1f, dim);
        this.inducing = RecsysOps.normal(random, 1f, inducingPoints, dim);
        this.isab = new Var[] {RecsysOps.matrix(random, dim, dim), RecsysOps.matrix(random, dim, dim),
            RecsysOps.matrix(random, dim, dim), RecsysOps.matrix(random, dim, dim), RecsysOps.matrix(random, dim, dim),
            RecsysOps.matrix(random, dim, dim)};
        this.columnNorm = LayerNorm.of(dim);
        this.cls = RecsysOps.normal(random, 1f, 1, dim);
        this.rowAttention = new Var[] {RecsysOps.matrix(random, dim, dim), RecsysOps.matrix(random, dim, dim),
            RecsysOps.matrix(random, dim, dim), RecsysOps.matrix(random, dim, dim)};
        this.rowFfn1 = RecsysOps.matrix(random, dim, 2 * dim);
        this.rowFfn2 = RecsysOps.matrix(random, 2 * dim, dim);
        this.rowNorm1 = LayerNorm.of(dim);
        this.rowNorm2 = LayerNorm.of(dim);
        this.labels = RecsysOps.normal(random, 1f, 3, dim);
        for (int b = 0; b < iclBlocks; b++) {
            icl.add((TransformerBlock) TransformerBlock.of(heads).ffnSize(2 * dim).create(Shape.of(8, dim), random));
        }
        this.norm = LayerNorm.of(dim);
        this.head = RecsysOps.matrix(random, dim, 2);
        this.headBias = RecsysOps.zeros(2);
        this.optimizer = Optimizer.adam(lr);
    }

    /** @return every trainable parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(cellIn, cellBias, inducing));
        p.addAll(List.of(isab));
        p.addAll(columnNorm.parameters());
        p.add(cls);
        p.addAll(List.of(rowAttention));
        p.add(rowFfn1);
        p.add(rowFfn2);
        p.addAll(rowNorm1.parameters());
        p.addAll(rowNorm2.parameters());
        p.add(labels);
        for (TransformerBlock b : icl) {
            p.addAll(b.parameters());
        }
        p.addAll(norm.parameters());
        p.add(head);
        p.add(headBias);
        return p;
    }

    private Var split(Var x, int batch, int tokens) {
        return x.reshape(batch, tokens, heads, dim / heads).permute(0, 2, 1, 3);
    }

    private static Var rotateHalf(Var x) {
        int d = x.shape().dim(-1);
        return Var.concat(x.shape().rank() - 1, x.slice(x.shape().rank() - 1, d / 2, d).neg(),
                x.slice(x.shape().rank() - 1, 0, d / 2));
    }

    /** RoPE on {@code [batch, heads, tokens, headDim]}. */
    private static Var rope(Var x) {
        int t = x.shape().dim(2);
        int d = x.shape().dim(3);
        float[] cos = new float[t * d];
        float[] sin = new float[t * d];
        for (int p = 0; p < t; p++) {
            for (int k = 0; k < d / 2; k++) {
                double a = p * Math.pow(10000, -2.0 * k / d);
                cos[p * d + k] = (float) Math.cos(a);
                cos[p * d + k + d / 2] = (float) Math.cos(a);
                sin[p * d + k] = (float) Math.sin(a);
                sin[p * d + k + d / 2] = (float) Math.sin(a);
            }
        }
        return x.mul(Tensor.of(Shape.of(t, d), cos)).add(rotateHalf(x).mul(Tensor.of(Shape.of(t, d), sin)));
    }

    /** Column stage: {@code [b, rows, f]} standardised cells to {@code [b * rows, f, dim]}. */
    private Var columns(Tensor x, int context) {
        int b = x.shape().dim(0);
        int rows = x.shape().dim(1);
        int f = x.shape().dim(2);
        int bf = b * f;
        int m = inducing.shape().dim(0);
        Var e = Var.of(x).permute(0, 2, 1).reshape(bf, rows, 1).matmul(cellIn).add(cellBias).reshape(bf, 1, rows, dim);
        Var ctx = e.slice(2, 0, context);
        float scale = (float) (1 / Math.sqrt(dim));
        Var iq = inducing.reshape(1, 1, m, dim).expand(Shape.of(bf, 1, m, dim));
        Var h = Functional.attention(iq.matmul(isab[0]), ctx.matmul(isab[1]), ctx.matmul(isab[2]), false, scale)
                .add(iq);
        Var out = Functional.attention(e.matmul(isab[3]), h.matmul(isab[4]), h.matmul(isab[5]), false, scale);
        Var cells = columnNorm.forward(e.add(out)).reshape(b, f, rows, dim).permute(0, 2, 1, 3);
        return cells.reshape(b * rows, f, dim);
    }

    /** Row stage: {@code [n, f, dim]} cells to {@code [n, dim]} row embeddings. */
    private Var rows(Var cells) {
        int n = cells.shape().dim(0);
        int t = cells.shape().dim(1) + 1;
        Var x = Var.concat(1, cls.reshape(1, 1, dim).expand(Shape.of(n, 1, dim)), cells);
        Var q = rope(split(x.matmul(rowAttention[0]), n, t));
        Var k = rope(split(x.matmul(rowAttention[1]), n, t));
        Var v = split(x.matmul(rowAttention[2]), n, t);
        Var a = Functional.attention(q, k, v, false, (float) (1 / Math.sqrt(dim / heads))).permute(0, 2, 1, 3)
                .reshape(n, t, dim).matmul(rowAttention[3]);
        Var r = rowNorm1.forward(x.add(a));
        r = rowNorm2.forward(r.add(r.matmul(rowFfn1).relu().matmul(rowFfn2)));
        return r.slice(1, 0, 1).reshape(n, dim);
    }

    /**
     * Logits of the test rows of a batch of tasks sharing their row and column counts.
     *
     * @param tasks tasks, the first {@code context} rows of each labelled
     * @param context training rows per task
     * @param training whether blocks run in training mode
     * @return {@code [tasks * (rows - context), 2]}
     */
    public Var logits(List<TabularPrior.Task> tasks, int context, boolean training) {
        int b = tasks.size();
        int rows = tasks.get(0).rows();
        int f = tasks.get(0).x()[0].length;
        float[] x = new float[b * rows * f];
        float[] ids = new float[b * rows];
        for (int t = 0; t < b; t++) {
            TabularPrior.Task task = tasks.get(t);
            if (task.rows() != rows || task.x()[0].length != f) {
                throw new IllegalArgumentException("tasks in a batch need equal rows and columns");
            }
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
                    x[(t * rows + i) * f + j] = (float) Math.max(-5, Math.min(5, (task.x()[i][j] - mean) / sd));
                }
            }
            for (int i = 0; i < rows; i++) {
                ids[t * rows + i] = i < context ? task.y()[i] : 2;
            }
        }
        Var rowEmb = rows(columns(Tensor.of(Shape.of(b, rows, f), x), context)).reshape(b, rows, dim);
        Var h = rowEmb.add(labels.gather(0, Tensor.of(Shape.of(b * rows), ids)).reshape(b, rows, dim));
        float[] mask = new float[rows * rows];
        for (int i = 0; i < rows; i++) {
            for (int j = context; j < rows; j++) {
                mask[i * rows + j] = MASKED;
            }
        }
        Tensor bias = Tensor.of(Shape.of(1, 1, rows, rows), mask);
        for (TransformerBlock block : icl) {
            h = block.forward(List.of(h), bias, training);
        }
        int queries = rows - context;
        return norm.forward(h.slice(1, context, rows).reshape(b * queries, dim)).matmul(head).add(headBias);
    }

    /**
     * One gradient step on a batch of tasks.
     *
     * @param tasks tasks with equal shapes
     * @param context labelled rows per task
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
     * Pre-trains on prior tasks with a size curriculum: task rows grow linearly from
     * {@code minRows} to {@code maxRows}; feature counts are drawn from 2 to {@code maxFeatures}.
     *
     * @param steps gradient steps
     * @param batch tasks per step
     * @param minRows rows per task at the start
     * @param maxRows rows per task at the end
     * @param maxFeatures the largest feature count drawn
     * @param seed the seed
     */
    public void pretrain(int steps, int batch, int minRows, int maxRows, int maxFeatures, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int rows = minRows + (int) ((long) (maxRows - minRows) * s / Math.max(1, steps - 1));
            rows = Math.max(8, rows / 8 * 8);
            int f = 2 + r.nextInt(Math.max(1, maxFeatures - 1));
            int context = rows / 4 + r.nextInt(rows / 2);
            List<TabularPrior.Task> tasks = new ArrayList<>();
            for (int b = 0; b < batch; b++) {
                tasks.add(TabularPrior.sample(rows, f, r));
            }
            trainStep(tasks, context);
        }
    }

    /**
     * Classifies test rows in context.
     *
     * @param trainX labelled rows
     * @param trainY their 0/1 labels
     * @param testX rows to classify
     * @return {@code P(y = 1)} per test row
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
        return "ColumnRowIcl(" + dim + " wide, " + icl.size() + " ICL blocks)";
    }
}
