package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.HierarchicalReasoningModel;
import io.github.zzuegg.brainiac.experimental.nn.TinyRecursiveModel;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * CompressARC (Liao and Gu 2025, "ARC-AGI Without Pretraining",
 * <a href="https://arxiv.org/abs/2512.06104">arXiv:2512.06104</a>): solve each grid puzzle
 * alone, with no training set, by <em>compressing</em> it. A fresh network decodes the whole
 * puzzle (every demonstration input and output and the test input) from a per-puzzle latent
 * with a KL penalty, minimizing reconstruction plus KL, a description length. The test
 * output is never shown; whatever the shortest description predicts there is the answer.
 * The decoder is equivariant like the paper's: its layers mix information only through
 * means over examples, over input/output, along rows and along columns, and through a
 * horizontal mirror (plus a learned embedding telling inputs from outputs), so the same rule costs the same bits everywhere.
 *
 * <p>Two benchmarks. ARC-like 5x5 tasks with three demonstrations (colour mapping, mirror,
 * row fill, column fill), against copying the input. {@link LatinSquarePuzzles} posed with
 * three solved demonstrations each, against the {@link TinyRecursiveModel} and the
 * {@link HierarchicalReasoningModel} trained on 1600 puzzles.
 */
public final class CompressArc {

    private CompressArc() {
    }

    /**
     * A puzzle: demonstration pairs and a test pair, grids of colour indices.
     *
     * @param inputs every example's input grid {@code [example][cell]}, the test last
     * @param outputs every example's output grid, the test last (the answer, never trained on)
     * @param height grid height
     * @param width grid width
     * @param colors the number of colours
     * @param scored which test-output cells are scored; {@code null} scores all
     */
    public record Puzzle(int[][] inputs, int[][] outputs, int height, int width, int colors, boolean[] scored) {
    }

    /** The CompressARC solver's settings. */
    private static final int CHANNELS = 16;
    private static final int LAYERS = 3;

    /**
     * Solves one puzzle by compression.
     *
     * @param p the puzzle
     * @param steps optimizer steps
     * @param beta the weight of the KL term per latent value
     * @param seed the seed
     * @return the predicted test output grid
     */
    public static int[] solve(Puzzle p, int steps, float beta, long seed) {
        int e = p.inputs().length;
        int h = p.height();
        int w = p.width();
        int k = p.colors();
        SplittableRandom r = new SplittableRandom(seed);
        Shape latent = Shape.of(e, 2, h, w, CHANNELS);
        Var mu = Var.param(Tensor.normal(latent, 0, 0.1f, r));
        Var logSigma = Var.param(Tensor.full(latent, -1f));
        List<Var[]> layers = new ArrayList<>();
        float scale = (float) (1 / Math.sqrt(CHANNELS));
        for (int l = 0; l < LAYERS; l++) {
            Var[] ws = new Var[7];
            for (int i = 0; i < 6; i++) {
                ws[i] = Var.param(Tensor.normal(Shape.of(CHANNELS, CHANNELS), 0, scale, r));
            }
            ws[6] = Var.param(Tensor.zeros(Shape.of(CHANNELS)));
            layers.add(ws);
        }
        Var head = Var.param(Tensor.normal(Shape.of(CHANNELS, k), 0, scale, r));
        Var headBias = Var.param(Tensor.zeros(Shape.of(k)));
        Var role = Var.param(Tensor.normal(Shape.of(1, 2, 1, 1, CHANNELS), 0, 1, r));
        layers.add(new Var[] {role});
        List<Var> params = new ArrayList<>(List.of(mu, logSigma, head, headBias));
        for (Var[] ws : layers) {
            params.addAll(List.of(ws));
        }
        // targets and the mask of known cells: all demonstration grids and the test input
        int cells = e * 2 * h * w;
        float[] target = new float[cells * k];
        float[] known = new float[cells];
        for (int x = 0; x < e; x++) {
            for (int io = 0; io < 2; io++) {
                int[] grid = io == 0 ? p.inputs()[x] : p.outputs()[x];
                boolean visible = x < e - 1 || io == 0;
                for (int c = 0; c < h * w; c++) {
                    int at = ((x * 2 + io) * h * w) + c;
                    target[at * k + grid[c]] = 1;
                    known[at] = visible ? 1 : 0;
                }
            }
        }
        double knownCount = 0;
        for (float v : known) {
            knownCount += v;
        }
        Tensor targetT = Tensor.of(Shape.of(cells, k), target);
        Tensor knownT = Tensor.of(Shape.of(cells, 1), known);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            Var z = mu.add(logSigma.exp().mul(Var.of(Tensor.normal(latent, 0, 1, r))));
            Var logp = Activation.LOG_SOFTMAX.forward(decode(z, layers, head, headBias, e, h, w, k));
            Var recon = logp.mul(targetT).mul(knownT).sum().div((float) knownCount).neg();
            Var kl = mu.square().add(logSigma.mul(2).exp()).sub(1).sub(logSigma.mul(2)).mul(0.5f).sum()
                    .div((float) knownCount);
            recon.add(kl.mul(beta)).backward();
            opt.step(params);
        }
        float[] logits = decode(mu, layers, head, headBias, e, h, w, k).value().toArray();
        int[] out = new int[h * w];
        int base = ((e - 1) * 2 + 1) * h * w;
        for (int c = 0; c < h * w; c++) {
            int best = 0;
            for (int j = 1; j < k; j++) {
                best = logits[(base + c) * k + j] > logits[(base + c) * k + best] ? j : best;
            }
            out[c] = best;
        }
        return out;
    }

    /** The equivariant decoder: residual layers mixing cells only through means and a mirror. */
    private static Var decode(Var z, List<Var[]> layers, Var head, Var headBias, int e, int h, int w, int k) {
        Var x = z.add(layers.get(layers.size() - 1)[0]);   // which grids are inputs, which outputs
        for (Var[] ws : layers.subList(0, layers.size() - 1)) {
            Var u = lin(x, ws[0], e, 2, h, w)
                    .add(lin(x.mean(1, true), ws[1], e, 1, h, w))
                    .add(lin(x.mean(3, true), ws[2], e, 2, h, 1))
                    .add(lin(x.mean(2, true), ws[3], e, 2, 1, w))
                    .add(lin(x.mean(0, true), ws[4], 1, 2, h, w))
                    .add(lin(x.flip(3), ws[5], e, 2, h, w))
                    .add(ws[6]);
            x = x.add(u.tanh());
        }
        return x.reshape(e * 2 * h * w, CHANNELS).matmul(head).add(headBias);
    }

    private static Var lin(Var x, Var weight, int a, int b, int c, int d) {
        return x.reshape(a * b * c * d, CHANNELS).matmul(weight).reshape(a, b, c, d, CHANNELS);
    }

    /**
     * ARC-like tasks: colour mapping, horizontal mirror, row fill and column fill, on 5x5
     * grids with 5 colours (0 the background) and three demonstrations each.
     *
     * @param perFamily tasks per family
     * @param seed the seed
     * @return the tasks
     */
    public static List<Puzzle> arcLike(int perFamily, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = 5;
        List<Puzzle> out = new ArrayList<>();
        for (int family = 0; family < 4; family++) {
            for (int t = 0; t < perFamily; t++) {
                int[] perm = {0, 1, 2, 3, 4};
                for (int i = 4; i > 1; i--) {
                    int j = 1 + r.nextInt(i);
                    int tmp = perm[i];
                    perm[i] = perm[j];
                    perm[j] = tmp;
                }
                int[][] in = new int[4][];
                int[][] res = new int[4][];
                for (int x = 0; x < 4; x++) {
                    in[x] = new int[n * n];
                    res[x] = new int[n * n];
                    for (int row = 0; row < n; row++) {
                        int fill = r.nextInt(3) == 0 ? 0 : 1 + r.nextInt(4);
                        for (int col = 0; col < n; col++) {
                            int c = row * n + col;
                            switch (family) {
                                case 0, 1 -> in[x][c] = r.nextInt(2) == 0 ? 0 : 1 + r.nextInt(4);
                                case 2 -> in[x][c] = col == 0 ? fill : 0;
                                default -> in[x][c] = 0;
                            }
                        }
                    }
                    if (family == 3) {
                        for (int col = 0; col < n; col++) {
                            in[x][col] = r.nextInt(3) == 0 ? 0 : 1 + r.nextInt(4);
                        }
                    }
                    for (int row = 0; row < n; row++) {
                        for (int col = 0; col < n; col++) {
                            int c = row * n + col;
                            res[x][c] = switch (family) {
                                case 0 -> perm[in[x][c]];
                                case 1 -> in[x][row * n + n - 1 - col];
                                case 2 -> in[x][row * n];
                                default -> in[x][col];
                            };
                        }
                    }
                }
                out.add(new Puzzle(in, res, n, n, 5, null));
            }
        }
        return out;
    }

    /**
     * Latin-square puzzles posed with three solved demonstrations: grids of colour 0 to 3
     * (digits) and 4 (blank); only the blanks of the test output are scored.
     *
     * @param tests the test puzzles
     * @param demos puzzles to draw demonstrations from
     * @param seed the seed
     * @return the puzzles
     */
    public static List<Puzzle> latin(List<LatinSquarePuzzles.Puzzle> tests, List<LatinSquarePuzzles.Puzzle> demos,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Puzzle> out = new ArrayList<>();
        for (LatinSquarePuzzles.Puzzle t : tests) {
            int[][] in = new int[4][];
            int[][] res = new int[4][];
            for (int x = 0; x < 4; x++) {
                LatinSquarePuzzles.Puzzle p = x < 3 ? demos.get(r.nextInt(demos.size())) : t;
                in[x] = new int[16];
                for (int c = 0; c < 16; c++) {
                    in[x][c] = p.givens()[c] < 0 ? 4 : p.givens()[c];
                }
                res[x] = p.solution().clone();
            }
            boolean[] scored = new boolean[16];
            for (int c = 0; c < 16; c++) {
                scored[c] = t.givens()[c] < 0;
            }
            out.add(new Puzzle(in, res, 4, 4, 5, scored));
        }
        return out;
    }

    /**
     * @param p a puzzle
     * @param prediction a predicted test output
     * @return whether every scored cell is right
     */
    public static boolean solved(Puzzle p, int[] prediction) {
        int[] truth = p.outputs()[p.outputs().length - 1];
        for (int c = 0; c < truth.length; c++) {
            if ((p.scored() == null || p.scored()[c]) && truth[c] != prediction[c]) {
                return false;
            }
        }
        return true;
    }

    /**
     * @param p a puzzle
     * @param prediction a predicted test output
     * @return the share of scored cells that are right
     */
    public static double cellAccuracy(Puzzle p, int[] prediction) {
        int[] truth = p.outputs()[p.outputs().length - 1];
        int right = 0;
        int total = 0;
        for (int c = 0; c < truth.length; c++) {
            if (p.scored() == null || p.scored()[c]) {
                total++;
                right += truth[c] == prediction[c] ? 1 : 0;
            }
        }
        return (double) right / total;
    }
}
