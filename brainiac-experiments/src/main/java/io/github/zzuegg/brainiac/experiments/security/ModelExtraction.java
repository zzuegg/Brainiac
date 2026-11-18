package io.github.zzuegg.brainiac.experiments.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ClientSplits;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.security.PredictionPoisoning;
import io.github.zzuegg.brainiac.experimental.security.TransferAttacks;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Wave 11 item 11.1.11: stealing digit classifiers through a prediction API with the blocks of
 * {@link io.github.zzuegg.brainiac.experimental.security.ModelExtraction}, and defending it
 * with {@link PredictionPoisoning}.
 *
 * <ul>
 *   <li>{@link #equationSolving}: a softmax regression recovered from {@code d + 1}
 *       confidence queries.</li>
 *   <li>{@link #extraction}: Knockoff Nets (random and adaptive query selection, with
 *       probabilities or labels only), Jacobian-based augmentation from 100 seed digits, and
 *       Knockoff against the top-1-preserving and rounding defences, against a copy trained on
 *       the same number of true-labelled digits. The attacker's pool mixes {@link ShapeImages},
 *       round-font {@link ShiftSuites} digits under random styles, and uniform noise.</li>
 * </ul>
 */
public final class ModelExtraction {

    /**
     * Equation-solving result.
     *
     * @param queries queries used
     * @param agreement top-1 agreement on digit test images
     * @param noiseAgreement top-1 agreement on uniform-noise inputs
     */
    public record Solved(int queries, double agreement, double noiseAgreement) {
    }

    /**
     * One extraction row.
     *
     * @param attack the attack and defence
     * @param budget queries (labelled samples for {@code TRUE_LABELS})
     * @param agreement top-1 agreement with the victim on digit test images
     * @param accuracy the copy's test accuracy
     */
    public record Row(String attack, int budget, double agreement, double accuracy) {
    }

    private static final int[] COPY = {64, 64, 10};
    private static final int STEPS = 400;

    private ModelExtraction() {
    }

    static ToyTasks.Data[] data(long seed) {
        return new ToyTasks.Data[] {ToyTasks.digits(10, 300, 8, 0.05, seed, seed + 1),
                ToyTasks.digits(10, 30, 8, 0.05, seed, seed + 2)};
    }

    static float[][] oneHot(int[] y, int classes) {
        float[][] t = new float[y.length][classes];
        for (int i = 0; i < y.length; i++) {
            t[i][y[i]] = 1;
        }
        return t;
    }

    static double accuracy(UnaryOperator<Var> f, ToyTasks.Data test) {
        int[] p = TransferAttacks.predict(f, test.x());
        int ok = 0;
        for (int i = 0; i < p.length; i++) {
            ok += p[i] == test.y()[i] ? 1 : 0;
        }
        return ok / (double) p.length;
    }

    static float[][] noise(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] out = new float[n][64];
        for (float[] row : out) {
            for (int j = 0; j < 64; j++) {
                row[j] = (float) r.nextDouble();
            }
        }
        return out;
    }

    /**
     * The attacker's out-of-distribution pool.
     *
     * @param n images
     * @param seed the seed
     * @return a third shapes, a third styled round-font digits, a third uniform noise, shuffled
     */
    public static float[][] pool(int n, long seed) {
        int third = n / 3;
        List<float[]> out = new ArrayList<>();
        ClientSplits.Labelled shapes = ClientSplits.fromImages(new ShapeImages(8, 0.1f).generate(third, seed), 6);
        out.addAll(List.of(shapes.x()));
        ShiftSuites.Style[] styles = ShiftSuites.Style.values();
        int per = Math.max(1, third / styles.length);
        int[] classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        for (int s = 0; s < styles.length; s++) {
            ShiftSuites.Split split = ShiftSuites.digits(per, classes, ShiftSuites.Font.ROUND, styles[s], 0.05,
                    seed + 1 + s);
            float[] v = split.x().toArray();
            for (int i = 0; i < per; i++) {
                float[] row = new float[64];
                System.arraycopy(v, i * 64, row, 0, 64);
                out.add(row);
            }
        }
        out.addAll(List.of(noise(n - out.size(), seed + 99)));
        SplittableRandom r = new SplittableRandom(seed + 7);
        for (int i = out.size() - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            float[] t = out.get(i);
            out.set(i, out.get(j));
            out.set(j, t);
        }
        return out.toArray(new float[0][]);
    }

    /**
     * Equation solving against a softmax regression.
     *
     * @param seed the seed
     * @return the result with {@code d + 1 = 65} queries
     */
    public static Solved equationSolving(long seed) {
        ToyTasks.Data[] d = data(seed);
        SimpleMlp victim = ModelExtractionBlocks.train(d[0].x(), oneHot(d[0].y(), 10), new int[] {64, 10}, STEPS,
                seed);
        UnaryOperator<Var> f = victim::forward;
        double[][] w = ModelExtractionBlocks.equationSolving(ModelExtractionBlocks.of(f), 64, 65, seed + 5);
        UnaryOperator<Var> copy = ModelExtractionBlocks.linear(w);
        return new Solved(65, ModelExtractionBlocks.agreement(f, copy, d[1].x()),
                ModelExtractionBlocks.agreement(f, copy, noise(500, seed + 6)));
    }

    /**
     * Extraction attacks and defences against an MLP victim.
     *
     * @param budgets query budgets
     * @param seed the seed
     * @return rows TRUE_LABELS, KNOCKOFF_RANDOM, KNOCKOFF_ADAPTIVE, KNOCKOFF_LABELS, JACOBIAN,
     *     KNOCKOFF_TOP1_DEFENDED and KNOCKOFF_ROUNDED for each budget, then the honest-user
     *     accuracy of the undefended and both defended services ({@code budget} 0)
     */
    public static List<Row> extraction(int[] budgets, long seed) {
        ToyTasks.Data[] d = data(seed);
        SimpleMlp victim = ModelExtractionBlocks.train(d[0].x(), oneHot(d[0].y(), 10), COPY, STEPS, seed);
        UnaryOperator<Var> f = victim::forward;
        io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api api = ModelExtractionBlocks.of(f);
        io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api labels = ModelExtractionBlocks
                .labelOnly(api);
        io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api top1 = PredictionPoisoning.defend(api,
                PredictionPoisoning.Method.TOP1_PRESERVING, 0.9);
        io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api rounded = PredictionPoisoning.defend(api,
                PredictionPoisoning.Method.ROUNDING, 1);
        int max = 0;
        for (int b : budgets) {
            max = Math.max(max, b);
        }
        float[][] pool = pool(Math.max(3 * max, 600), seed + 11);
        float[][] seedSet = new float[100][];
        System.arraycopy(d[0].x(), 0, seedSet, 0, 100);
        List<Row> out = new ArrayList<>();
        for (int b : budgets) {
            int n = Math.min(b, d[0].x().length);
            float[][] tx = new float[n][];
            int[] ty = new int[n];
            SplittableRandom r = new SplittableRandom(seed + b);
            for (int i = 0; i < n; i++) {
                int j = r.nextInt(d[0].x().length);
                tx[i] = d[0].x()[j];
                ty[i] = d[0].y()[j];
            }
            out.add(row("TRUE_LABELS", b, f, ModelExtractionBlocks.train(tx, oneHot(ty, 10), COPY, STEPS, seed + 1),
                    d[1]));
            out.add(row("KNOCKOFF_RANDOM", b, f, ModelExtractionBlocks.knockoff(api, pool, b, false, COPY, STEPS,
                    seed + 2), d[1]));
            out.add(row("KNOCKOFF_ADAPTIVE", b, f, ModelExtractionBlocks.knockoff(api, pool, b, true, COPY, STEPS,
                    seed + 3), d[1]));
            out.add(row("KNOCKOFF_LABELS", b, f, ModelExtractionBlocks.knockoff(labels, pool, b, false, COPY, STEPS,
                    seed + 4), d[1]));
            out.add(row("JACOBIAN", b, f, ModelExtractionBlocks.jacobianAugmentation(api, seedSet, 6, 0.1f,
                    Math.max(b, seedSet.length), COPY, STEPS / 2, seed + 5), d[1]));
            out.add(row("KNOCKOFF_TOP1_DEFENDED", b, f, ModelExtractionBlocks.knockoff(top1, pool, b, false, COPY,
                    STEPS, seed + 2), d[1]));
            out.add(row("KNOCKOFF_ROUNDED", b, f, ModelExtractionBlocks.knockoff(rounded, pool, b, false, COPY, STEPS,
                    seed + 2), d[1]));
        }
        out.add(new Row("HONEST_UNDEFENDED", 0, 1, served(api, d[1])));
        out.add(new Row("HONEST_TOP1_DEFENDED", 0, 1, served(top1, d[1])));
        out.add(new Row("HONEST_ROUNDED", 0, 1, served(rounded, d[1])));
        return out;
    }

    private static double served(io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api api,
            ToyTasks.Data test) {
        float[][] p = api.query(test.x());
        int ok = 0;
        for (int i = 0; i < p.length; i++) {
            int best = 0;
            for (int c = 1; c < p[i].length; c++) {
                if (p[i][c] > p[i][best]) {
                    best = c;
                }
            }
            ok += best == test.y()[i] ? 1 : 0;
        }
        return ok / (double) p.length;
    }

    private static Row row(String name, int budget, UnaryOperator<Var> victim, SimpleMlp copy, ToyTasks.Data test) {
        UnaryOperator<Var> c = copy::forward;
        return new Row(name, budget, ModelExtractionBlocks.agreement(victim, c, test.x()), accuracy(c, test));
    }

    /** Short alias for the block of the same simple name. */
    private static final class ModelExtractionBlocks {

        private ModelExtractionBlocks() {
        }

        static SimpleMlp train(float[][] x, float[][] t, int[] sizes, int steps, long seed) {
            return io.github.zzuegg.brainiac.experimental.security.ModelExtraction.train(x, t, sizes, steps, seed);
        }

        static io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api of(UnaryOperator<Var> f) {
            return io.github.zzuegg.brainiac.experimental.security.ModelExtraction.of(f);
        }

        static io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api labelOnly(
                io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api api) {
            return io.github.zzuegg.brainiac.experimental.security.ModelExtraction.labelOnly(api);
        }

        static double[][] equationSolving(io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api api,
                int features, int queries, long seed) {
            return io.github.zzuegg.brainiac.experimental.security.ModelExtraction.equationSolving(api, features,
                    queries, seed);
        }

        static UnaryOperator<Var> linear(double[][] w) {
            return io.github.zzuegg.brainiac.experimental.security.ModelExtraction.linear(w);
        }

        static SimpleMlp knockoff(io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api api,
                float[][] pool, int budget, boolean adaptive, int[] sizes, int steps, long seed) {
            return io.github.zzuegg.brainiac.experimental.security.ModelExtraction.knockoff(api, pool, budget, adaptive,
                    sizes, steps, seed);
        }

        static SimpleMlp jacobianAugmentation(io.github.zzuegg.brainiac.experimental.security.ModelExtraction.Api api,
                float[][] seedSet, int rounds, float lambda, int budget, int[] sizes, int steps, long seed) {
            return io.github.zzuegg.brainiac.experimental.security.ModelExtraction.jacobianAugmentation(api, seedSet,
                    rounds, lambda, budget, sizes, steps, seed);
        }

        static double agreement(UnaryOperator<Var> a, UnaryOperator<Var> b, float[][] x) {
            return io.github.zzuegg.brainiac.experimental.security.ModelExtraction.agreement(a, b, x);
        }
    }
}
