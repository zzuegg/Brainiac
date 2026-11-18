package io.github.zzuegg.brainiac.experiments.compression;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.compress.ExpertMerging;
import io.github.zzuegg.brainiac.experimental.compress.ExpertPruning;
import io.github.zzuegg.brainiac.experimental.nn.FineGrainedMoe;
import io.github.zzuegg.brainiac.experimental.nn.SiluMlpExpert;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Halving the experts of a trained {@link FineGrainedMoe} (wave 8 item 8.5.7) on a regime
 * regression task like {@code attention.MoeRefinements}' (inputs of width 8 fall into one of
 * {@code regimes} regions, each with its own small tanh teacher network): with 8 experts and 4
 * regimes experts are redundant, with 8 regimes they specialise. Rows: the full layer, expert
 * pruning by subset search, dropping the least-used experts, MC-SMoE merging, all experts
 * averaged into one, and a dense SiLU MLP with the MoE's active width trained from scratch.
 * Used by {@link PruningShootout#moe}.
 */
final class MoeCompression {

    private static final int WIDTH = 8;
    private static final int EXPERTS = 8;
    private static final int HIDDEN = 16;
    private static final int TOP_K = 2;

    private MoeCompression() {
    }

    /** A regime teacher. */
    private record Teacher(int regimes, float[] directions, float[][] w, float[][] v) {

        static Teacher create(int regimes, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            float[] dirs = Tensor.normal(Shape.of(WIDTH, regimes), 0, 1, r).toArray();
            float[][] w = new float[regimes][];
            float[][] v = new float[regimes][];
            for (int k = 0; k < regimes; k++) {
                w[k] = Tensor.normal(Shape.of(WIDTH, HIDDEN), 0, 0.6f, r).toArray();
                v[k] = Tensor.normal(Shape.of(HIDDEN, WIDTH), 0, 0.6f, r).toArray();
            }
            return new Teacher(regimes, dirs, w, v);
        }

        Tensor[] batch(SplittableRandom random, int n) {
            float[] x = Tensor.normal(Shape.of(n, WIDTH), 0, 1, random).toArray();
            float[] y = new float[n * WIDTH];
            for (int i = 0; i < n; i++) {
                int best = 0;
                double bestScore = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < regimes; k++) {
                    double s = 0;
                    for (int d = 0; d < WIDTH; d++) {
                        s += x[i * WIDTH + d] * directions[d * regimes + k];
                    }
                    if (s > bestScore) {
                        bestScore = s;
                        best = k;
                    }
                }
                for (int h = 0; h < HIDDEN; h++) {
                    double a = 0;
                    for (int d = 0; d < WIDTH; d++) {
                        a += x[i * WIDTH + d] * w[best][d * HIDDEN + h];
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

    static List<CompactRow> run(int[] regimeCounts, int steps, long seed) {
        List<CompactRow> rows = new ArrayList<>();
        for (int regimes : regimeCounts) {
            Teacher teacher = Teacher.create(regimes, seed + regimes);
            SplittableRandom r = new SplittableRandom(seed);
            List<Layer> experts = new ArrayList<>();
            for (int i = 0; i < EXPERTS; i++) {
                experts.add(new SiluMlpExpert(WIDTH, HIDDEN, WIDTH, r));
            }
            FineGrainedMoe moe = new FineGrainedMoe(WIDTH, WIDTH, TOP_K, experts, List.of(), r).balanceWeight(0.01f);
            train(moe, teacher, steps, seed);
            Tensor calib = teacher.batch(new SplittableRandom(seed + 5), 512)[0];
            Tensor[] val = teacher.batch(new SplittableRandom(seed + 77), 1024);
            String tag = regimes + " regimes: ";
            int keep = EXPERTS / 2;
            rows.add(row(tag + "full 8", moe, val, EXPERTS));
            rows.add(row(tag + "search-prune 4", ExpertPruning.reduce(moe,
                    ExpertPruning.search(moe, keep, TOP_K, calib), TOP_K), val, keep));
            rows.add(row(tag + "drop least-used 4", ExpertPruning.reduce(moe,
                    ExpertPruning.mostUsed(moe, keep, calib), TOP_K), val, keep));
            rows.add(row(tag + "merge 4", ExpertMerging.merge(moe, keep, TOP_K, calib), val, keep));
            rows.add(row(tag + "average all 1", ExpertMerging.averageAll(moe), val, 1));
            SiluMlpExpert dense = new SiluMlpExpert(WIDTH, TOP_K * HIDDEN, WIDTH, new SplittableRandom(seed + 9));
            train(dense, teacher, steps, seed);
            rows.add(row(tag + "dense active-width", dense, val, 1));
        }
        return rows;
    }

    private static void train(Layer layer, Teacher teacher, int steps, long seed) {
        SplittableRandom data = new SplittableRandom(seed + 1);
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = teacher.batch(data, 128);
            Var loss = layer.forward(Var.of(b[0]), true).sub(b[1]).square().mean();
            Var aux = layer.auxiliaryLoss();
            (aux == null ? loss : loss.add(aux)).backward();
            opt.step(layer.parameters());
        }
    }

    private static CompactRow row(String name, Layer layer, Tensor[] val, int experts) {
        double mse = Var.noGrad(() -> layer.forward(Var.of(val[0])).sub(val[1]).square().mean().value().item());
        long params = 0;
        for (Var p : layer.parameters()) {
            params += p.shape().size();
        }
        return new CompactRow(name, mse, experts, params, -1);
    }
}
