package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.nn.DartsCell;
import io.github.zzuegg.brainiac.experimental.nn.DartsCell.Candidate;
import io.github.zzuegg.brainiac.experimental.train.DartsSearch;
import io.github.zzuegg.brainiac.experimental.train.WeightSharingNas;
import io.github.zzuegg.brainiac.experimental.train.ZeroCostProxies;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Neural architecture search on the {@link DartsCell} space: a two-node cell (three edges),
 * each edge one of {@code ZERO, IDENTITY, LINEAR, RELU, TANH, SQUARE} (216 architectures),
 * between a linear stem and a linear head, on three-arm spirals.
 *
 * <ul>
 *   <li>{@link #proxyCorrelations}: how well each {@link ZeroCostProxies} score at
 *       initialization ranks architectures by their accuracy after full training (Spearman).</li>
 *   <li>{@link #search}: the test accuracy of the architecture each method picks, after
 *       training it from scratch, and the search cost in training steps: random search,
 *       first-order DARTS ({@link DartsSearch}), ENAS ({@link WeightSharingNas}) and
 *       zero-cost ranking of the whole space by NASWOT.</li>
 * </ul>
 */
public final class ArchitectureSearchShootout {

    /** The search space's operations. */
    public static final List<Candidate> CANDIDATES = List.of(Candidate.ZERO, Candidate.IDENTITY, Candidate.LINEAR,
            Candidate.RELU, Candidate.TANH, Candidate.SQUARE);

    private static final int NODES = 2;
    private static final int EDGES = 3;
    private static final int WIDTH = 16;
    private static final int TRAIN_STEPS = 300;

    /** The search methods. */
    public enum Method {
        /** Train random architectures from scratch, keep the best on validation. */
        RANDOM,
        /** First-order DARTS. */
        DARTS,
        /** ENAS: shared weights plus a REINFORCE controller. */
        ENAS,
        /** Rank the whole space by NASWOT at initialization. */
        NASWOT
    }

    /**
     * A search's outcome.
     *
     * @param architecture the chosen operation per edge
     * @param testAccuracy its test accuracy after training from scratch
     * @param searchSteps training steps (or network evaluations for zero-cost) the search spent
     */
    public record Outcome(List<Candidate> architecture, double testAccuracy, long searchSteps) {
    }

    private ArchitectureSearchShootout() {
    }

    private record Splits(ToyTasks.Data train, ToyTasks.Data valid, ToyTasks.Data test) {
    }

    private static Splits data(long seed) {
        return new Splits(ToyTasks.spiral(100, 3, 0.2, seed), ToyTasks.spiral(60, 3, 0.2, seed + 1),
                ToyTasks.spiral(100, 3, 0.2, seed + 2));
    }

    private static Model model(int[] arch, long seed) {
        return Model.sequential().input(2).dense(WIDTH, Activation.IDENTITY)
                .add(WeightSharingNas.SharedCell.spec(NODES, CANDIDATES, arch)).dense(3, Activation.IDENTITY)
                .seed(seed).build();
    }

    private static double train(Model m, Splits d, ToyTasks.Data eval) {
        Optimizer opt = Optimizer.adam(0.02f);
        Tensor x = d.train().inputs();
        Tensor y = d.train().ids();
        for (int s = 0; s < TRAIN_STEPS; s++) {
            Functional.crossEntropy(m.forward(Var.of(x), true), y).backward();
            opt.step(m.parameters());
        }
        return ToyTasks.accuracy(m.forward(Var.of(eval.inputs())).value(), eval.y());
    }

    private static int[] decode(int index) {
        int[] a = new int[EDGES];
        for (int e = 0; e < EDGES; e++) {
            a[e] = index % CANDIDATES.size();
            index /= CANDIDATES.size();
        }
        return a;
    }

    private static List<Candidate> names(int[] arch) {
        List<Candidate> out = new ArrayList<>();
        for (int k : arch) {
            out.add(CANDIDATES.get(k));
        }
        return out;
    }

    /**
     * Spearman correlations of the proxies with trained test accuracy over random architectures.
     *
     * @param architectures how many distinct random architectures to train
     * @param seed the seed
     * @return {@code {gradNorm, snip, synflow, naswot}} correlations
     */
    public static double[] proxyCorrelations(int architectures, long seed) {
        Splits d = data(seed);
        SplittableRandom r = new SplittableRandom(seed);
        Set<Integer> picked = new LinkedHashSet<>();
        int space = (int) Math.pow(CANDIDATES.size(), EDGES);
        while (picked.size() < architectures) {
            picked.add(r.nextInt(space));
        }
        int n = picked.size();
        double[][] proxies = new double[4][n];
        double[] acc = new double[n];
        Tensor bx = d.train().slice(0, 64).inputs();
        Tensor by = d.train().slice(0, 64).ids();
        int i = 0;
        for (int index : picked) {
            int[] arch = decode(index);
            Model m = model(arch, seed);
            proxies[0][i] = ZeroCostProxies.gradNorm(m, bx, by, Functional::crossEntropy);
            proxies[1][i] = ZeroCostProxies.snip(m, bx, by, Functional::crossEntropy);
            proxies[2][i] = ZeroCostProxies.synflow(m, Shape.of(2));
            proxies[3][i] = ZeroCostProxies.naswot(m, bx);
            acc[i] = train(m, d, d.test());
            i++;
        }
        double[] out = new double[4];
        for (int p = 0; p < 4; p++) {
            out[p] = ZeroCostProxies.spearman(proxies[p], acc);
        }
        return out;
    }

    /**
     * Runs one search method and trains its pick from scratch.
     *
     * @param method the method
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome search(Method method, long seed) {
        Splits d = data(seed);
        long[] cost = {0};
        int[] arch = switch (method) {
            case RANDOM -> {
                SplittableRandom r = new SplittableRandom(seed);
                int[] best = null;
                double bestAcc = -1;
                for (int k = 0; k < 6; k++) {
                    int[] a = decode(r.nextInt((int) Math.pow(CANDIDATES.size(), EDGES)));
                    double acc = train(model(a, seed), d, d.valid());
                    cost[0] += TRAIN_STEPS;
                    if (acc > bestAcc) {
                        bestAcc = acc;
                        best = a;
                    }
                }
                yield best;
            }
            case DARTS -> {
                DartsCell cell = DartsCell.of(WIDTH, NODES, new SplittableRandom(seed),
                        CANDIDATES.toArray(new Candidate[0]));
                Model m = Model.sequential().input(2).dense(WIDTH, Activation.IDENTITY).layer(cell)
                        .dense(3, Activation.IDENTITY).seed(seed).build();
                Dataset train = Dataset.of(d.train().x(), toArray(d.train().oneHot()));
                Dataset valid = Dataset.of(d.valid().x(), toArray(d.valid().oneHot()));
                int epochs = 25;
                DartsSearch.search(m, cell.architecture(), train, valid, Loss.crossEntropyFromLogits(0),
                        Optimizer.adam(0.02f), Optimizer.adam(0.03f), epochs, 32);
                cost[0] += 2L * epochs * ((d.train().size() + 31) / 32);
                List<Candidate> derived = cell.derive();
                int[] a = new int[EDGES];
                for (int e = 0; e < EDGES; e++) {
                    a[e] = CANDIDATES.indexOf(derived.get(e));
                }
                yield a;
            }
            case ENAS -> {
                WeightSharingNas.SharedCell cell = new WeightSharingNas.SharedCell(WIDTH, NODES, CANDIDATES,
                        new SplittableRandom(seed));
                Model m = Model.sequential().input(2).dense(WIDTH, Activation.IDENTITY).layer(cell)
                        .dense(3, Activation.IDENTITY).seed(seed).build();
                WeightSharingNas nas = new WeightSharingNas(cell, 0.5f, seed);
                Optimizer opt = Optimizer.adam(0.02f);
                SplittableRandom r = new SplittableRandom(seed + 3);
                Tensor vx = d.valid().inputs();
                int[] a = nas.search(30, 20, () -> {
                    int from = r.nextInt(d.train().size() - 32);
                    ToyTasks.Data b = d.train().slice(from, from + 32);
                    Functional.crossEntropy(m.forward(Var.of(b.inputs()), true), b.ids()).backward();
                    opt.step(m.parameters());
                }, 10, c -> ToyTasks.accuracy(m.forward(Var.of(vx)).value(), d.valid().y()));
                cost[0] += nas.weightSteps();
                yield a;
            }
            case NASWOT -> {
                int space = (int) Math.pow(CANDIDATES.size(), EDGES);
                Tensor bx = d.train().slice(0, 64).inputs();
                int[] best = null;
                double bestScore = Double.NEGATIVE_INFINITY;
                for (int index = 0; index < space; index++) {
                    int[] a = decode(index);
                    double s = ZeroCostProxies.naswot(model(a, seed), bx);
                    cost[0]++;
                    if (s > bestScore) {
                        bestScore = s;
                        best = a;
                    }
                }
                yield best;
            }
        };
        return new Outcome(names(arch), train(model(arch, seed + 100), d, d.test()), cost[0]);
    }

    private static float[][] toArray(Tensor t) {
        int n = t.shape().dim(0);
        int k = t.shape().dim(1);
        float[] v = t.toArray();
        float[][] out = new float[n][k];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, i * k, out[i], 0, k);
        }
        return out;
    }
}
