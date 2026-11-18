package io.github.zzuegg.brainiac.experimental.document;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ToyTables;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Table question answering by cell selection and aggregation, in the style of TAPAS (Herzig
 * et al., ACL 2020): no logical form is generated. The question and the flattened table
 * (one token per cell, headers included) form one sequence, and each token's embedding adds
 * learned embeddings of its segment (question or table), column index, row index and the
 * numeric rank of its value within its column. Each body cell also adds the word embeddings
 * of its column header and of its row's name cell (a table-aware cell embedding: without it
 * the encoder must learn two attention hops, header to column and name to row, which it did
 * not within the toy budget). A bidirectional core {@link TransformerBlock}
 * stack encodes it; a cell-selection head scores every body cell and an aggregation head on
 * the first token picks an operator ({@code NONE}, {@code COUNT}, {@code SUM},
 * {@code AVERAGE}).
 *
 * <p>Training is weakly supervised as in the paper. Questions whose answer is a cell are
 * trained on that cell's coordinates and the {@code NONE} operator. Questions with a scalar
 * answer get only the number: the loss is a Huber loss between the answer and the
 * <em>expected</em> result under the soft selection, {@code sum_op p(op) result_op(p_select)}
 * over the non-{@code NONE} operators ({@code COUNT = sum p_c}, {@code SUM = sum p_c v_c},
 * {@code AVERAGE = SUM / COUNT}), which is differentiable, plus {@code -log(1 - p(NONE))}.
 * At inference the most likely operator is applied: {@code NONE} returns the most likely
 * cell, and the scalar operators return the same soft expectation the loss fits, rounded
 * (thresholding the selection at one half, as the paper does, answered every count with 0
 * here: the soft count is met by many cells of small probability).
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2004.02349">arXiv:2004.02349</a>.
 */
public final class TableEncoder {

    /**
     * Tokens of one question over one table, ready for the encoder.
     *
     * @param tokens word ids
     * @param segments 0 for the question, 1 for the table
     * @param rows row index plus one (0 outside the body)
     * @param columns column index plus one (0 in the question)
     * @param ranks numeric rank plus one (0 for names and outside the body)
     * @param values cell values (0 for names and outside the body)
     * @param cells whether each token is a body cell
     * @param texts the token texts
     * @param headers per body cell its column header's word id, -1 elsewhere
     * @param names per body cell its row name's word id, -1 elsewhere
     */
    public record Example(int[] tokens, int[] segments, int[] rows, int[] columns, int[] ranks, float[] values,
            boolean[] cells, String[] texts, int[] headers, int[] names) {
    }

    private final Map<String, Integer> vocabulary = new HashMap<>();
    private final int vocab;
    private final int dim;
    private final int maxRows;
    private final int maxColumns;
    private final Var words;
    private final Var segments;
    private final Var rowTable;
    private final Var columnTable;
    private final Var rankTable;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final Var selectW;
    private final Var selectB;
    private final Var aggW;
    private final Var aggB;

    /**
     * Creates the model.
     *
     * @param words the word vocabulary (unknown words map to one extra id)
     * @param dim the model width
     * @param layers transformer blocks
     * @param maxRows the most table rows (larger indices are clamped)
     * @param maxColumns the most table columns
     * @param seed the seed
     */
    public TableEncoder(List<String> words, int dim, int layers, int maxRows, int maxColumns, long seed) {
        for (String w : words) {
            vocabulary.putIfAbsent(w, vocabulary.size());
        }
        vocabulary.putIfAbsent("[CLS]", vocabulary.size());
        vocabulary.putIfAbsent("[SEP]", vocabulary.size());
        vocabulary.putIfAbsent("[UNK]", vocabulary.size());
        this.vocab = vocabulary.size();
        this.dim = dim;
        this.maxRows = maxRows;
        this.maxColumns = maxColumns;
        SplittableRandom r = new SplittableRandom(seed);
        this.words = table(r, vocab);
        this.segments = table(r, 2);
        this.rowTable = table(r, maxRows + 1);
        this.columnTable = table(r, maxColumns + 1);
        this.rankTable = table(r, ToyTables.MAX_VALUE + 2);
        for (int i = 0; i < layers; i++) {
            blocks.add((TransformerBlock) TransformerBlock.of(4).ffnSize(2 * dim).create(Shape.of(8, dim), r));
        }
        this.selectW = Var.param(Tensor.normal(Shape.of(dim, 1), 0, (float) Math.sqrt(1.0 / dim), r));
        this.selectB = Var.param(Tensor.zeros(Shape.of(1)));
        this.aggW = Var.param(Tensor.normal(Shape.of(dim, 4), 0, (float) Math.sqrt(1.0 / dim), r));
        this.aggB = Var.param(Tensor.zeros(Shape.of(4)));
    }

    private Var table(SplittableRandom r, int n) {
        return Var.param(Tensor.normal(Shape.of(n, dim), 0, 0.3f, r));
    }

    private int id(String w) {
        return vocabulary.getOrDefault(w, vocabulary.get("[UNK]"));
    }

    /**
     * Flattens a question and a table: {@code [CLS] question [SEP] header cells, then body
     * cells row by row}.
     *
     * @param table the table
     * @param question the question
     * @return the example
     */
    public Example encode(ToyTables.Table table, ToyTables.Question question) {
        List<int[]> meta = new ArrayList<>();
        List<String> texts = new ArrayList<>();
        meta.add(new int[] {id("[CLS]"), 0, 0, 0, 0, 0});
        texts.add("[CLS]");
        for (String w : question.words()) {
            meta.add(new int[] {id(w), 0, 0, 0, 0, 0});
            texts.add(w);
        }
        meta.add(new int[] {id("[SEP]"), 0, 0, 0, 0, 0});
        texts.add("[SEP]");
        for (int c = 0; c < table.columns(); c++) {
            meta.add(new int[] {id(table.header()[c]), 1, 0, c + 1, 0, 0});
            texts.add(table.header()[c]);
        }
        List<Float> values = new ArrayList<>();
        for (int i = 0; i < meta.size(); i++) {
            values.add(0f);
        }
        for (int row = 0; row < table.rows(); row++) {
            for (int c = 0; c < table.columns(); c++) {
                int rank = c == 0 ? 0 : table.rank(row, c) + 1;
                meta.add(new int[] {id(table.cells()[row][c]), 1, row + 1, c + 1, rank, 1, id(table.header()[c]),
                    id(table.cells()[row][0])});
                texts.add(table.cells()[row][c]);
                values.add(c == 0 ? 0f : (float) table.value(row, c));
            }
        }
        int n = meta.size();
        int[][] cols = new int[7][n];
        float[] v = new float[n];
        boolean[] cell = new boolean[n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 5; k++) {
                cols[k][i] = meta.get(i)[k];
            }
            cell[i] = meta.get(i)[5] == 1;
            cols[5][i] = cell[i] ? meta.get(i)[6] : -1;
            cols[6][i] = cell[i] ? meta.get(i)[7] : -1;
            v[i] = values.get(i);
        }
        return new Example(cols[0], cols[1], cols[2], cols[3], cols[4], v, cell, texts.toArray(new String[0]),
                cols[5], cols[6]);
    }

    private Var lookup(Var t, List<Example> batch, int which, int len, int max) {
        int b = batch.size();
        float[] f = new float[b * len];
        for (int i = 0; i < b; i++) {
            Example e = batch.get(i);
            int[] src = switch (which) {
                case 0 -> e.tokens();
                case 1 -> e.segments();
                case 2 -> e.rows();
                case 3 -> e.columns();
                default -> e.ranks();
            };
            for (int k = 0; k < src.length; k++) {
                f[i * len + k] = Math.min(max, src[k]);
            }
        }
        return t.gather(0, Tensor.of(Shape.of(b * len), f)).reshape(b, len, dim);
    }

    /** Word embeddings of the body cells' headers ({@code names == false}) or row names, zero elsewhere. */
    private Var context(List<Example> batch, int len, boolean names) {
        int b = batch.size();
        float[] f = new float[b * len];
        float[] m = new float[b * len];
        for (int i = 0; i < b; i++) {
            int[] src = names ? batch.get(i).names() : batch.get(i).headers();
            for (int k = 0; k < src.length; k++) {
                f[i * len + k] = Math.max(0, src[k]);
                m[i * len + k] = src[k] >= 0 ? 1 : 0;
            }
        }
        return words.gather(0, Tensor.of(Shape.of(b * len), f)).reshape(b, len, dim)
                .mul(Tensor.of(Shape.of(b, len, 1), m));
    }

    /** The encoder outputs, the cell-selection logits {@code [b, len]} and the aggregation logits {@code [b, 4]}. */
    private Var[] run(List<Example> batch, boolean training) {
        int b = batch.size();
        int len = 0;
        for (Example e : batch) {
            len = Math.max(len, e.tokens().length);
        }
        Var x = lookup(words, batch, 0, len, vocab - 1).add(lookup(segments, batch, 1, len, 1))
                .add(lookup(rowTable, batch, 2, len, maxRows)).add(lookup(columnTable, batch, 3, len, maxColumns))
                .add(lookup(rankTable, batch, 4, len, ToyTables.MAX_VALUE + 1))
                .add(context(batch, len, false)).add(context(batch, len, true));
        float[] bias = new float[b * len];
        for (int i = 0; i < b; i++) {
            for (int k = batch.get(i).tokens().length; k < len; k++) {
                bias[i * len + k] = -1e9f;
            }
        }
        Tensor mask = Tensor.of(Shape.of(b, 1, 1, len), bias);
        for (TransformerBlock blk : blocks) {
            x = blk.forward(List.of(x), mask, training);
        }
        Var select = x.matmul(selectW).add(selectB).reshape(b, len);
        Var agg = x.slice(1, 0, 1).reshape(b, dim).matmul(aggW).add(aggB);
        return new Var[] {select, agg};
    }

    private static int width(List<Example> batch) {
        int len = 0;
        for (Example e : batch) {
            len = Math.max(len, e.tokens().length);
        }
        return len;
    }

    /**
     * The weakly supervised loss of a batch.
     *
     * @param batch encoded examples
     * @param questions their questions (for the answers, and the cell coordinates of
     *     questions whose answer is a cell)
     * @param huberDelta the Huber loss threshold for scalar answers
     * @return the mean loss
     */
    public Var loss(List<Example> batch, List<ToyTables.Question> questions, float huberDelta) {
        int b = batch.size();
        int len = width(batch);
        Var[] out = run(batch, true);
        Var select = out[0];
        Var aggLogp = Activation.LOG_SOFTMAX.forward(out[1]);
        float[] cellMask = new float[b * len];
        float[] vals = new float[b * len];
        float[] selTarget = new float[b * len];
        float[] selWeight = new float[b * len];
        float[] isCell = new float[b];
        float[] isScalar = new float[b];
        float[] answer = new float[b];
        float[] noneOneHot = new float[b * 4];
        int cellQuestions = 0;
        for (int i = 0; i < b; i++) {
            Example e = batch.get(i);
            ToyTables.Question q = questions.get(i);
            boolean cellAnswer = q.aggregation() == ToyTables.Aggregation.NONE;
            int bodyStart = e.tokens().length - countCells(e);
            int columns = columnsOf(e);
            for (int k = 0; k < e.tokens().length; k++) {
                if (e.cells()[k]) {
                    cellMask[i * len + k] = 1;
                    vals[i * len + k] = e.values()[k];
                    if (cellAnswer) {
                        int cellIndex = k - bodyStart;
                        boolean gold = q.gold()[cellIndex / columns][cellIndex % columns];
                        selTarget[i * len + k] = gold ? 1 : 0;
                        selWeight[i * len + k] = 1;
                    }
                }
            }
            if (cellAnswer) {
                isCell[i] = 1;
                noneOneHot[i * 4] = 1;
                cellQuestions++;
            } else {
                isScalar[i] = 1;
                answer[i] = (float) q.value();
            }
        }
        Tensor mask = Tensor.of(Shape.of(b, len), cellMask);
        Var loss = Var.of(Tensor.zeros(Shape.of()));
        if (cellQuestions > 0) {
            // binary cross-entropy with logits over the body cells, and -log p(NONE)
            Tensor y = Tensor.of(Shape.of(b, len), selTarget);
            Tensor w = Tensor.of(Shape.of(b, len), selWeight);
            float cells = 0;
            for (float x : selWeight) {
                cells += x;
            }
            Var bce = select.softplus().sub(select.mul(y)).mul(w).sum().mul(1f / Math.max(1f, cells) * cellQuestions / b);
            Var none = aggLogp.mul(Tensor.of(Shape.of(b, 4), noneOneHot)).sum().mul(-1f / b);
            loss = loss.add(bce).add(none);
        }
        if (cellQuestions < b) {
            Var p = select.sigmoid().mul(mask);
            Var count = p.sum(1, false);
            Var sum = p.mul(Tensor.of(Shape.of(b, len), vals)).sum(1, false);
            Var avg = sum.div(count.add(1e-3f));
            Var agg = aggLogp.exp();
            Var notNone = agg.slice(1, 1, 4).sum(1, false).add(1e-6f);
            Var expected = agg.slice(1, 1, 2).reshape(b).mul(count).add(agg.slice(1, 2, 3).reshape(b).mul(sum))
                    .add(agg.slice(1, 3, 4).reshape(b).mul(avg)).div(notNone);
            Var err = expected.sub(Tensor.of(Shape.of(b), answer)).abs();
            Var huber = Var.where(err.value().greater(Tensor.full(Shape.of(), huberDelta, err.value().device())),
                    err.mul(huberDelta).sub(0.5f * huberDelta * huberDelta), err.square().mul(0.5f));
            Tensor scalar = Tensor.of(Shape.of(b), isScalar);
            loss = loss.add(huber.mul(scalar).sum().mul(1f / b))
                    .add(notNone.log().mul(scalar).sum().mul(-1f / b));
        }
        return loss;
    }

    private static int countCells(Example e) {
        int c = 0;
        for (boolean x : e.cells()) {
            c += x ? 1 : 0;
        }
        return c;
    }

    private static int columnsOf(Example e) {
        int c = 0;
        for (int i = 0; i < e.tokens().length; i++) {
            if (e.cells()[i]) {
                c = Math.max(c, e.columns()[i]);
            }
        }
        return c;
    }

    /**
     * Answers questions.
     *
     * @param batch encoded examples
     * @return per example the answer text (a cell or a number)
     */
    public String[] answer(List<Example> batch) {
        Var[] out = Var.noGrad(() -> run(batch, false));
        int len = width(batch);
        float[] sel = out[0].value().toArray();
        float[] agg = out[1].value().toArray();
        String[] answers = new String[batch.size()];
        for (int i = 0; i < batch.size(); i++) {
            Example e = batch.get(i);
            int op = 0;
            for (int k = 1; k < 4; k++) {
                op = agg[i * 4 + k] > agg[i * 4 + op] ? k : op;
            }
            int best = -1;
            double count = 0;
            double sum = 0;
            for (int k = 0; k < e.tokens().length; k++) {
                if (!e.cells()[k]) {
                    continue;
                }
                float z = sel[i * len + k];
                if (best < 0 || z > sel[i * len + best]) {
                    best = k;
                }
                double p = 1 / (1 + Math.exp(-z));
                count += p;
                sum += p * e.values()[k];
            }
            answers[i] = switch (op) {
                case 0 -> best < 0 ? "" : e.texts()[best];
                case 1 -> Long.toString(Math.round(count));
                case 2 -> Long.toString(Math.round(sum));
                default -> Long.toString(Math.round(count == 0 ? 0 : sum / count));
            };
        }
        return answers;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(words, segments, rowTable, columnTable, rankTable));
        for (TransformerBlock blk : blocks) {
            p.addAll(blk.parameters());
        }
        p.addAll(List.of(selectW, selectB, aggW, aggB));
        return p;
    }

    @Override
    public String toString() {
        return "TableEncoder(" + blocks.size() + " blocks, " + dim + " wide)";
    }
}
