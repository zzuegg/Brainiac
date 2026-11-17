package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ExpertChoiceMoe;
import io.github.zzuegg.brainiac.experimental.nn.FineGrainedMoe;
import io.github.zzuegg.brainiac.experimental.nn.ReluRoutedMoe;
import io.github.zzuegg.brainiac.experimental.nn.SiluMlpExpert;
import io.github.zzuegg.brainiac.experimental.nn.SparseUpcycling;
import io.github.zzuegg.brainiac.experimental.nn.ZeroComputationMoe;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Mixture-of-experts refinements on a regression task made for experts: inputs fall into
 * four regimes (by the nearest of four random directions), and each regime has its own
 * random teacher network, so experts can specialize by regime.
 *
 * <ul>
 *   <li><b>Routing</b>: {@link FineGrainedMoe} top-2 token choice (the baseline) against
 *       {@link ExpertChoiceMoe} expert choice (perfectly even loads) and token choice with a
 *       capacity factor and the router z-loss, with and without a balance loss. Reported:
 *       validation MSE and the load imbalance (coefficient of variation of tokens per
 *       expert) on the validation batch.</li>
 *   <li><b>Upcycling</b>: a dense SiLU FFN trained for {@code N} steps, then trained for
 *       {@code M} more steps as is, or upcycled into a top-2 MoE of four copies
 *       ({@link SparseUpcycling}), or drop-upcycled (half of each copy's hidden units
 *       re-initialized); against a MoE trained from scratch at the same total compute
 *       ({@code N / 2 + M} steps, as a top-2 step costs two dense steps). The dense row
 *       keeps training at half the per-step compute.</li>
 *   <li><b>Compute per token</b>: {@link ReluRoutedMoe} (ReMoE, ReLU routing with an adaptive L1
 *       penalty aiming at two experts per token on average) and {@link ZeroComputationMoe}
 *       (MoE++, zero, copy and constant experts beside the FFN experts) against top-2
 *       {@link FineGrainedMoe} and expert choice. Reported: validation MSE, load imbalance,
 *       the mean and variance of FFN experts per token (the compute).</li>
 * </ul>
 */
public final class MoeRefinements {

    private static final int WIDTH = 8;
    private static final int REGIMES = 4;
    private static final int TEACHER_HIDDEN = 16;

    private MoeRefinements() {
    }

    /**
     * One configuration's result.
     *
     * @param name the configuration
     * @param validationMse the mean squared error on 1024 fresh samples
     * @param loadImbalance the coefficient of variation of expert loads (NaN for dense)
     * @param dropped the share of dropped token-choice assignments (0 elsewhere)
     */
    public record Row(String name, double validationMse, double loadImbalance, double dropped) {
    }

    /**
     * One router's result with its compute per token.
     *
     * @param name the configuration
     * @param validationMse the mean squared error on 1024 fresh samples
     * @param loadImbalance the coefficient of variation of expert loads
     * @param ffnPerToken the mean number of FFN experts a token ran through
     * @param ffnVariance the variance of that number over tokens (0 for fixed top-k)
     */
    public record ComputeRow(String name, double validationMse, double loadImbalance, double ffnPerToken,
            double ffnVariance) {
    }

    /** The regime teacher: directions and per-regime networks. */
    private record Teacher(float[] directions, float[][] w, float[][] v) {

        static Teacher create() {
            SplittableRandom r = new SplittableRandom(42);
            float[] dirs = Tensor.normal(Shape.of(WIDTH, REGIMES), 0, 1, r).toArray();
            float[][] w = new float[REGIMES][];
            float[][] v = new float[REGIMES][];
            for (int k = 0; k < REGIMES; k++) {
                w[k] = Tensor.normal(Shape.of(WIDTH, TEACHER_HIDDEN), 0, 0.6f, r).toArray();
                v[k] = Tensor.normal(Shape.of(TEACHER_HIDDEN, WIDTH), 0, 0.6f, r).toArray();
            }
            return new Teacher(dirs, w, v);
        }

        Tensor[] batch(SplittableRandom random, int n) {
            float[] x = Tensor.normal(Shape.of(n, WIDTH), 0, 1, random).toArray();
            float[] y = new float[n * WIDTH];
            for (int i = 0; i < n; i++) {
                int best = 0;
                double bestScore = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < REGIMES; k++) {
                    double s = 0;
                    for (int d = 0; d < WIDTH; d++) {
                        s += x[i * WIDTH + d] * directions[d * REGIMES + k];
                    }
                    if (s > bestScore) {
                        bestScore = s;
                        best = k;
                    }
                }
                for (int h = 0; h < TEACHER_HIDDEN; h++) {
                    double a = 0;
                    for (int d = 0; d < WIDTH; d++) {
                        a += x[i * WIDTH + d] * w[best][d * TEACHER_HIDDEN + h];
                    }
                    a = Math.tanh(a);
                    for (int o = 0; o < WIDTH; o++) {
                        y[i * WIDTH + o] += (float) (a * v[best][h * WIDTH + o]);
                    }
                }
            }
            return new Tensor[] {Tensor.of(Shape.of(n, WIDTH), x), Tensor.of(Shape.of(n, WIDTH), y)};
        }
    }

    private static final Teacher TEACHER = Teacher.create();

    /**
     * Compares the routers, all with 8 experts of hidden width 16 and two experts per token
     * on average.
     *
     * @param steps Adam steps on batches of 128
     * @param seed the seed
     * @return FineGrainedMoe top-2, expert choice, token choice with capacity and z-loss, and
     *     the latter with a balance loss
     */
    public static List<Row> routers(int steps, long seed) {
        List<Row> rows = new ArrayList<>();
        rows.add(trainMoe("top-2 (FineGrainedMoe)", () -> FineGrainedMoe.of(WIDTH, 8, 0, 16, 2,
                new SplittableRandom(seed)), steps, seed));
        rows.add(trainMoe("expert choice c=2", () -> ExpertChoiceMoe.of(WIDTH, 8, 16,
                ExpertChoiceMoe.Routing.EXPERT_CHOICE, 2, 2f, new SplittableRandom(seed)).zLossWeight(1e-3f), steps,
                seed));
        rows.add(trainMoe("token choice cf=1.25 + z", () -> ExpertChoiceMoe.of(WIDTH, 8, 16,
                ExpertChoiceMoe.Routing.TOKEN_CHOICE, 2, 1.25f, new SplittableRandom(seed)).zLossWeight(1e-3f), steps,
                seed));
        rows.add(trainMoe("token choice + z + balance", () -> ExpertChoiceMoe.of(WIDTH, 8, 16,
                ExpertChoiceMoe.Routing.TOKEN_CHOICE, 2, 1.25f, new SplittableRandom(seed)).zLossWeight(1e-3f)
                .balanceWeight(1e-2f), steps, seed));
        return rows;
    }

    /**
     * Compares upcycling with continued dense training and with a MoE from scratch.
     *
     * @param denseSteps steps of dense pre-training {@code N}
     * @param continuedSteps further steps {@code M}
     * @param seed the seed
     * @return dense continued, upcycled, drop-upcycled, MoE from scratch
     */
    public static List<Row> upcycling(int denseSteps, int continuedSteps, long seed) {
        List<Row> rows = new ArrayList<>();
        SplittableRandom data = new SplittableRandom(seed);
        SiluMlpExpert dense = new SiluMlpExpert(WIDTH, 32, WIDTH, new SplittableRandom(seed));
        train(dense, denseSteps, data);
        Tensor w1 = dense.w1().value();
        Tensor b1 = dense.b1().value();
        Tensor w2 = dense.w2().value();
        SiluMlpExpert continued = new SiluMlpExpert(w1, b1, w2);
        train(continued, continuedSteps, new SplittableRandom(seed + 1));
        rows.add(new Row("dense continued", mse(continued), Double.NaN, 0));
        for (float drop : new float[] {0f, 0.5f}) {
            FineGrainedMoe moe = SparseUpcycling.upcycle(new SiluMlpExpert(w1, b1, w2), 4, 2, drop,
                    new SplittableRandom(seed + 2));
            train(moe, continuedSteps, new SplittableRandom(seed + 1));
            rows.add(new Row(drop == 0 ? "upcycled" : "drop-upcycled 0.5", mse(moe),
                    ExpertChoiceMoe.loadImbalance(moe.expertLoad()), 0));
        }
        FineGrainedMoe scratch = FineGrainedMoe.of(WIDTH, 4, 0, 32, 2, new SplittableRandom(seed + 3));
        train(scratch, denseSteps / 2 + continuedSteps, new SplittableRandom(seed));
        rows.add(new Row("MoE from scratch", mse(scratch), ExpertChoiceMoe.loadImbalance(scratch.expertLoad()), 0));
        return rows;
    }

    /**
     * Compares ReLU routing (ReMoE) and zero-computation experts (MoE++) with top-2 token
     * choice and expert choice, all with 8 FFN experts of hidden width 16 and two FFN experts
     * per token on average (MoE++ adds 2 zero, 1 copy and 1 constant expert to the pool).
     *
     * @param steps Adam steps on batches of 128
     * @param seed the seed
     * @return top-2, expert choice, ReMoE, MoE++
     */
    public static List<ComputeRow> computePerToken(int steps, long seed) {
        List<ComputeRow> rows = new ArrayList<>();
        FineGrainedMoe topK = FineGrainedMoe.of(WIDTH, 8, 0, 16, 2, new SplittableRandom(seed));
        train(topK, steps, new SplittableRandom(seed));
        rows.add(new ComputeRow("top-2 (FineGrainedMoe)", mse(topK), ExpertChoiceMoe.loadImbalance(topK.expertLoad()),
                2, 0));
        ExpertChoiceMoe ec = ExpertChoiceMoe.of(WIDTH, 8, 16, ExpertChoiceMoe.Routing.EXPERT_CHOICE, 2, 2f,
                new SplittableRandom(seed));
        train(ec, steps, new SplittableRandom(seed));
        double ecMse = mse(ec);
        double[] ecStats = meanVariance(ec.expertsPerToken());
        rows.add(new ComputeRow("expert choice c=2", ecMse, ExpertChoiceMoe.loadImbalance(ec.expertLoad()), ecStats[0],
                ecStats[1]));
        ReluRoutedMoe remoe = ReluRoutedMoe.of(WIDTH, 8, 16, 2, new SplittableRandom(seed));
        train(remoe, steps, new SplittableRandom(seed));
        double reMse = mse(remoe);
        double[] reStats = meanVariance(remoe.expertsPerToken());
        rows.add(new ComputeRow("ReMoE (ReLU, k=2)", reMse, ExpertChoiceMoe.loadImbalance(remoe.expertLoad()),
                reStats[0], reStats[1]));
        ZeroComputationMoe plus = new ZeroComputationMoe(WIDTH, 8, 2, 1, 1, 16, 2, new SplittableRandom(seed));
        train(plus, steps, new SplittableRandom(seed));
        double plusMse = mse(plus);
        int[] load = plus.expertLoad();
        rows.add(new ComputeRow("MoE++ (8 FFN + 4 zero-compute)", plusMse,
                ExpertChoiceMoe.loadImbalance(java.util.Arrays.copyOf(load, 8)), plus.ffnExpertsPerToken(), Double.NaN));
        return rows;
    }

    private static double[] meanVariance(int[] values) {
        double mean = 0;
        for (int v : values) {
            mean += v;
        }
        mean /= values.length;
        double var = 0;
        for (int v : values) {
            var += (v - mean) * (v - mean);
        }
        return new double[] {mean, var / values.length};
    }

    /**
     * Formats compute rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String computeTable(List<ComputeRow> rows) {
        StringBuilder sb = new StringBuilder(String.format("%-32s %9s %9s %9s %9s%n", "model", "val MSE", "load CV",
                "FFN/tok", "var"));
        for (ComputeRow r : rows) {
            sb.append(String.format("%-32s %9.4f %9.3f %9.3f %9.3f%n", r.name(), r.validationMse(), r.loadImbalance(),
                    r.ffnPerToken(), r.ffnVariance()));
        }
        return sb.toString();
    }

    /**
     * The validation MSE of always predicting zero, the scale of the targets.
     *
     * @return the mean squared target
     */
    public static double zeroPredictorMse() {
        Tensor[] b = TEACHER.batch(new SplittableRandom(4242), 1024);
        return b[1].square().mean().item();
    }

    private static Row trainMoe(String name, Supplier<Layer> make, int steps, long seed) {
        Layer moe = make.get();
        train(moe, steps, new SplittableRandom(seed));
        double mse = mse(moe);
        int[] load = moe instanceof FineGrainedMoe f ? f.expertLoad() : ((ExpertChoiceMoe) moe).expertLoad();
        double dropped = moe instanceof ExpertChoiceMoe e ? e.droppedFraction() : 0;
        return new Row(name, mse, ExpertChoiceMoe.loadImbalance(load), dropped);
    }

    private static void train(Layer layer, int steps, SplittableRandom data) {
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = TEACHER.batch(data, 128);
            Var loss = layer.forward(Var.of(b[0]), true).sub(b[1]).square().mean();
            Var aux = layer.auxiliaryLoss();
            (aux == null ? loss : loss.add(aux)).backward();
            opt.step(layer.parameters());
        }
    }

    private static double mse(Layer layer) {
        Tensor[] b = TEACHER.batch(new SplittableRandom(4242), 1024);
        return layer.forward(Var.of(b[0]), false).sub(b[1]).square().mean().value().item();
    }

    /**
     * Formats rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String table(List<Row> rows) {
        StringBuilder sb = new StringBuilder(String.format("%-28s %9s %9s %8s%n", "model", "val MSE", "load CV",
                "dropped"));
        for (Row r : rows) {
            sb.append(String.format("%-28s %9.4f %9.3f %8.3f%n", r.name(), r.validationMse(), r.loadImbalance(),
                    r.dropped()));
        }
        return sb.toString();
    }

    /**
     * Runs both comparisons.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(table(routers(800, 1)));
        System.out.print(table(upcycling(800, 400, 1)));
        System.out.print(computeTable(computePerToken(800, 1)));
    }
}
