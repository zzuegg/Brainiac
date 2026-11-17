package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * A Sudoku-like toy for the recursive reasoners: complete a 4x4 Latin square (every digit
 * once per row and column) with some cells blanked out. Inputs are one-hot cells (blank or a
 * digit), outputs are four logits per cell. {@link #train} runs deep supervision for any
 * model that returns one prediction per supervision step, so the Tiny Recursive Model, the
 * Hierarchical Reasoning Model and a plain MLP train with the same loop.
 */
public final class LatinSquarePuzzles {

    /** Cells per puzzle. */
    public static final int CELLS = 16;
    /** One-hot input width. */
    public static final int INPUTS = CELLS * 5;
    /** Output logits. */
    public static final int OUTPUTS = CELLS * 4;

    /**
     * A puzzle.
     *
     * @param givens the cells, -1 for blank
     * @param solution the full square
     */
    public record Puzzle(int[] givens, int[] solution) {
    }

    /**
     * Accuracy after each supervision step.
     *
     * @param blankAccuracy per step, the fraction of blank cells filled right
     * @param solved per step, the fraction of puzzles solved completely
     */
    public record Evaluation(double[] blankAccuracy, double[] solved) {
    }

    private LatinSquarePuzzles() {
    }

    /** @return all 576 Latin squares of order 4 */
    public static List<int[]> allSquares() {
        List<int[]> out = new ArrayList<>();
        fill(new int[CELLS], 0, out);
        return out;
    }

    private static void fill(int[] g, int cell, List<int[]> out) {
        if (cell == CELLS) {
            out.add(g.clone());
            return;
        }
        int r = cell / 4;
        int c = cell % 4;
        for (int d = 0; d < 4; d++) {
            boolean ok = true;
            for (int k = 0; k < 4 && ok; k++) {
                ok = !(k < c && g[r * 4 + k] == d) && !(k < r && g[k * 4 + c] == d);
            }
            if (ok) {
                g[cell] = d;
                fill(g, cell + 1, out);
            }
        }
    }

    /**
     * @param square a solution
     * @param blanks how many cells to hide
     * @param random the random source
     * @return a puzzle
     */
    public static Puzzle blank(int[] square, int blanks, SplittableRandom random) {
        int[] g = square.clone();
        int hidden = 0;
        while (hidden < blanks) {
            int k = random.nextInt(CELLS);
            if (g[k] >= 0) {
                g[k] = -1;
                hidden++;
            }
        }
        return new Puzzle(g, square);
    }

    static Tensor inputs(List<Puzzle> batch) {
        float[] x = new float[batch.size() * INPUTS];
        for (int i = 0; i < batch.size(); i++) {
            int[] g = batch.get(i).givens();
            for (int c = 0; c < CELLS; c++) {
                x[i * INPUTS + c * 5 + g[c] + 1] = 1;
            }
        }
        return Tensor.of(Shape.of(batch.size(), INPUTS), x);
    }

    static Tensor targets(List<Puzzle> batch) {
        float[] y = new float[batch.size() * CELLS];
        for (int i = 0; i < batch.size(); i++) {
            for (int c = 0; c < CELLS; c++) {
                y[i * CELLS + c] = batch.get(i).solution()[c];
            }
        }
        return Tensor.of(Shape.of(batch.size() * CELLS), y);
    }

    /**
     * Trains with deep supervision: a cross-entropy loss on every step's prediction.
     *
     * @param model maps a batch to one logits tensor {@code [batch, 64]} per step
     * @param parameters the model's parameters
     * @param train the training puzzles
     * @param steps optimizer steps
     * @param batchSize puzzles per step
     * @param seed the seed
     */
    public static void train(Function<Var, List<Var>> model, List<Var> parameters, List<Puzzle> train, int steps,
            int batchSize, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adamW(3e-3f, 1e-4f);
        for (int s = 0; s < steps; s++) {
            List<Puzzle> batch = new ArrayList<>();
            for (int i = 0; i < batchSize; i++) {
                batch.add(train.get(random.nextInt(train.size())));
            }
            Tensor ids = targets(batch);
            Var loss = null;
            for (Var logits : model.apply(Var.of(inputs(batch)))) {
                Var l = Functional.crossEntropy(logits.reshape(batch.size() * CELLS, 4), ids);
                loss = loss == null ? l : loss.add(l);
            }
            loss.backward();
            opt.step(parameters);
        }
    }

    /**
     * @param model the model
     * @param test the puzzles
     * @return accuracy after every supervision step
     */
    public static Evaluation evaluate(Function<Var, List<Var>> model, List<Puzzle> test) {
        List<Var> outs = model.apply(Var.of(inputs(test)));
        double[] acc = new double[outs.size()];
        double[] solved = new double[outs.size()];
        for (int k = 0; k < outs.size(); k++) {
            float[] v = outs.get(k).value().toArray();
            int right = 0;
            int blanks = 0;
            int full = 0;
            for (int i = 0; i < test.size(); i++) {
                boolean all = true;
                for (int c = 0; c < CELLS; c++) {
                    if (test.get(i).givens()[c] >= 0) {
                        continue;
                    }
                    int best = 0;
                    for (int d = 1; d < 4; d++) {
                        if (v[i * OUTPUTS + c * 4 + d] > v[i * OUTPUTS + c * 4 + best]) {
                            best = d;
                        }
                    }
                    blanks++;
                    boolean ok = best == test.get(i).solution()[c];
                    right += ok ? 1 : 0;
                    all &= ok;
                }
                full += all ? 1 : 0;
            }
            acc[k] = (double) right / blanks;
            solved[k] = (double) full / test.size();
        }
        return new Evaluation(acc, solved);
    }

    /**
     * Splits all squares into train and test puzzles.
     *
     * @param blanks cells hidden per puzzle
     * @param trainSquares how many squares go to training
     * @param seed the seed
     * @return {train, test}
     */
    public static List<List<Puzzle>> split(int blanks, int trainSquares, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<int[]> squares = allSquares();
        for (int i = squares.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int[] t = squares.get(i);
            squares.set(i, squares.get(j));
            squares.set(j, t);
        }
        List<Puzzle> train = new ArrayList<>();
        List<Puzzle> test = new ArrayList<>();
        for (int i = 0; i < squares.size(); i++) {
            if (i < trainSquares) {
                for (int k = 0; k < 4; k++) {
                    train.add(blank(squares.get(i), blanks, random));
                }
            } else {
                test.add(blank(squares.get(i), blanks, random));
            }
        }
        return List.of(train, test);
    }
}
