package io.github.zzuegg.brainiac.experiments.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ClientSplits;
import io.github.zzuegg.brainiac.experimental.ml.ClusteringMetrics;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.privacy.ClusteredFederation;
import io.github.zzuegg.brainiac.experimental.privacy.EnsembleDistillationAggregator;
import io.github.zzuegg.brainiac.experimental.privacy.FedBn;
import io.github.zzuegg.brainiac.experimental.privacy.FedDyn;
import io.github.zzuegg.brainiac.experimental.privacy.FederatedAlgorithms;
import io.github.zzuegg.brainiac.experimental.privacy.FederatedClient;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;
import java.util.function.IntFunction;

/**
 * Wave 12 item 12.2.12: federated aggregation follow-ups on digits with 20 clients, using the
 * {@link FederatedAlgorithms.Aggregator} hook and the wave 11 client set-up of
 * {@link FederatedShootout}.
 *
 * <ul>
 *   <li>{@link #permutedLabels}: four client groups with permuted labels; IFCA
 *       ({@link ClusteredFederation}) against a single FedAvg model, plus the adjusted Rand index of
 *       the recovered clusters.</li>
 *   <li>{@link #featureShift}: four client groups whose inputs are rescaled, shifted or inverted;
 *       FedBN ({@link FedBn}, a per-feature input affine kept local) against FedAvg on the same
 *       model and local-only training.</li>
 *   <li>{@link #labelSkew}: Dirichlet label skew; FedDyn ({@link FedDyn}) and ensemble distillation
 *       (FedDF, {@link EnsembleDistillationAggregator} through the aggregator hook) against FedAvg
 *       and SCAFFOLD.</li>
 * </ul>
 *
 * <p>Every row reports the mean and the worst client accuracy (each client scored on test data of
 * its own group, or, for label skew, on the test classes weighted by its label distribution) and
 * the first round (checked every 5) at which the mean reaches a target.
 */
public final class FederatedFollowUps {

    private FederatedFollowUps() {
    }

    /**
     * One federated method in one setting.
     *
     * @param setting the heterogeneity
     * @param method the method
     * @param meanAccuracy the mean client accuracy after the last round
     * @param worstClient the lowest client accuracy
     * @param roundsToTarget the first checked round with mean accuracy at the target, or -1
     * @param note an extra figure (the cluster ARI for IFCA), or {@code NaN}
     */
    public record FedRow(String setting, String method, double meanAccuracy, double worstClient, int roundsToTarget,
            double note) {
    }

    private static final int CLASSES = 10;
    private static final float LR = 0.3f;

    private static int[] argmaxRows(Tensor scores) {
        int n = scores.shape().dim(0);
        int c = scores.shape().dim(1);
        float[] v = scores.toArray();
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int j = 1; j < c; j++) {
                if (v[i * c + j] > v[i * c + best]) {
                    best = j;
                }
            }
            out[i] = best;
        }
        return out;
    }

    private static double accuracy(Tensor scores, int[] y) {
        int[] p = argmaxRows(scores);
        int right = 0;
        for (int i = 0; i < y.length; i++) {
            right += p[i] == y[i] ? 1 : 0;
        }
        return (double) right / y.length;
    }

    private static FedRow summary(String setting, String method, double[] acc, int reached, double note) {
        double mean = 0;
        double worst = 1;
        for (double a : acc) {
            mean += a / acc.length;
            worst = Math.min(worst, a);
        }
        return new FedRow(setting, method, mean, worst, reached, note);
    }

    private static int[][] iidParts(int n, int clients, long seed) {
        return ClientSplits.iid(n, clients, seed);
    }

    // ------------------------------------------------------------------ permuted labels

    /**
     * IFCA against FedAvg when client groups permute the labels.
     *
     * @param clients the clients
     * @param groups the label permutations (clusters)
     * @param rounds rounds
     * @param localSteps local steps
     * @param participation the share of clients per round
     * @param seed the seed
     * @return FedAvg, IFCA with {@code groups} models, IFCA with 2 models
     */
    public static List<FedRow> permutedLabels(int clients, int groups, int rounds, int localSteps,
            double participation, long seed) {
        ClientSplits.Labelled[] d = FederatedShootout.data(FederatedShootout.Task.DIGITS, 1600, seed);
        int[][] parts = iidParts(d[0].size(), clients, seed + 3);
        int[][] perms = ClientSplits.labelPermutations(clients, CLASSES, groups, seed + 4);
        ClientSplits.Labelled[] local = new ClientSplits.Labelled[clients];
        ClientSplits.Labelled[] tests = new ClientSplits.Labelled[clients];
        int[] group = new int[clients];
        for (int k = 0; k < clients; k++) {
            local[k] = d[0].subset(parts[k]).relabel(perms[k]);
            tests[k] = d[1].relabel(perms[k]);
            group[k] = k % groups;
        }
        String setting = groups + " permuted-label groups";
        List<FedRow> rows = new ArrayList<>();
        // FedAvg
        SimpleMlp global = FederatedShootout.net(CLASSES, seed);
        List<FederatedClient> fc = new ArrayList<>();
        for (int k = 0; k < clients; k++) {
            fc.add(FederatedShootout.client(FederatedShootout.net(CLASSES, seed), local[k], LR, seed * 1000 + k));
        }
        FederatedAlgorithms avg = FederatedAlgorithms.fedAvg(global.parameters(), fc, seed);
        for (int r = 0; r < rounds; r++) {
            avg.round(localSteps, participation);
        }
        double[] acc = new double[clients];
        for (int k = 0; k < clients; k++) {
            acc[k] = accuracy(global.predict(tests[k].inputs()), tests[k].y());
        }
        rows.add(summary(setting, "FedAvg (one model)", acc, -1, Double.NaN));
        for (int models : new int[] {groups, 2}) {
            List<SimpleMlp> nets = new ArrayList<>();
            List<List<Var>> params = new ArrayList<>();
            for (int m = 0; m < models; m++) {
                SimpleMlp n = FederatedShootout.net(CLASSES, seed + 17 * (m + 1));
                nets.add(n);
                params.add(n.parameters());
            }
            List<FederatedClient> cc = new ArrayList<>();
            for (int k = 0; k < clients; k++) {
                cc.add(FederatedShootout.client(FederatedShootout.net(CLASSES, seed), local[k], LR,
                        seed * 1000 + k));
            }
            ClusteredFederation ifca = ClusteredFederation.of(params, cc, 2, seed);
            for (int r = 0; r < rounds; r++) {
                ifca.round(localSteps, participation);
            }
            int[] assigned = new int[clients];
            for (int k = 0; k < clients; k++) {
                assigned[k] = ifca.assign(k);
                acc[k] = accuracy(nets.get(assigned[k]).predict(tests[k].inputs()), tests[k].y());
            }
            rows.add(summary(setting, "IFCA (" + models + " models)", acc, -1, ClusteringMetrics.ari(group,
                    assigned)));
        }
        return rows;
    }

    // ------------------------------------------------------------------ feature shift

    /** A tanh MLP with a per-feature input affine (the "normalisation" parameters FedBN keeps local). */
    private static final class AffineMlp {
        private final List<Var> params;

        AffineMlp(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            params = List.of(Var.param(Tensor.full(Shape.of(1, 64), 1f)), Var.param(Tensor.zeros(Shape.of(1, 64))),
                    Var.param(Tensor.normal(Shape.of(64, 32), 0, 0.125f, r)), Var.param(Tensor.zeros(Shape.of(1, 32))),
                    Var.param(Tensor.normal(Shape.of(32, CLASSES), 0, (float) Math.sqrt(1.0 / 32), r)),
                    Var.param(Tensor.zeros(Shape.of(1, CLASSES))));
        }

        Var forward(Var x) {
            return x.mul(params.get(0)).add(params.get(1)).matmul(params.get(2)).add(params.get(3)).tanh()
                    .matmul(params.get(4)).add(params.get(5));
        }

        Tensor predict(Tensor x) {
            return Var.noGrad(() -> forward(Var.of(x))).value();
        }

        static final boolean[] NORMALISATION = {true, true, false, false, false, false};
    }

    private static FederatedClient affineClient(AffineMlp net, ClientSplits.Labelled local, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(16, local.size());
        return new FederatedClient(net.params, local.size(), () -> {
            float[][] x = new float[b][];
            float[] y = new float[b];
            for (int i = 0; i < b; i++) {
                int j = r.nextInt(local.size());
                x[i] = local.x()[j];
                y[i] = local.y()[j];
            }
            return Functional.crossEntropy(net.forward(Var.of(Tensor.of(x))), Tensor.of(Shape.of(b), y));
        }, LR);
    }

    private static final float[][] SHIFTS = {{1, 0}, {-1, 1}, {0.5f, 0.3f}, {2, -0.5f}};

    private static ClientSplits.Labelled shift(ClientSplits.Labelled data, int group) {
        float a = SHIFTS[group % SHIFTS.length][0];
        float b = SHIFTS[group % SHIFTS.length][1];
        float[][] x = new float[data.size()][];
        for (int i = 0; i < x.length; i++) {
            x[i] = new float[data.x()[i].length];
            for (int j = 0; j < x[i].length; j++) {
                x[i][j] = a * data.x()[i][j] + b;
            }
        }
        return new ClientSplits.Labelled(x, data.y(), data.classes());
    }

    /**
     * FedBN against FedAvg and local training when client groups see differently scaled inputs
     * (identity, inverted, half contrast, double contrast).
     *
     * @param clients the clients
     * @param rounds rounds
     * @param localSteps local steps
     * @param participation the share of clients per round
     * @param seed the seed
     * @return FedAvg, FedBN, local only
     */
    public static List<FedRow> featureShift(int clients, int rounds, int localSteps, double participation, long seed) {
        ClientSplits.Labelled[] d = FederatedShootout.data(FederatedShootout.Task.DIGITS, 1600, seed);
        int[][] parts = iidParts(d[0].size(), clients, seed + 3);
        ClientSplits.Labelled[] local = new ClientSplits.Labelled[clients];
        ClientSplits.Labelled[] tests = new ClientSplits.Labelled[clients];
        for (int k = 0; k < clients; k++) {
            local[k] = shift(d[0].subset(parts[k]), k % SHIFTS.length);
            tests[k] = shift(d[1], k % SHIFTS.length);
        }
        String setting = "feature shift (4 input transforms)";
        List<FedRow> rows = new ArrayList<>();
        double[] acc = new double[clients];
        // FedAvg on the same model
        AffineMlp global = new AffineMlp(seed);
        List<AffineMlp> nets = new ArrayList<>();
        List<FederatedClient> fc = new ArrayList<>();
        for (int k = 0; k < clients; k++) {
            AffineMlp n = new AffineMlp(seed);
            nets.add(n);
            fc.add(affineClient(n, local[k], seed * 1000 + k));
        }
        FederatedAlgorithms avg = FederatedAlgorithms.fedAvg(global.params, fc, seed);
        for (int r = 0; r < rounds; r++) {
            avg.round(localSteps, participation);
        }
        for (int k = 0; k < clients; k++) {
            acc[k] = accuracy(global.predict(tests[k].inputs()), tests[k].y());
        }
        rows.add(summary(setting, "FedAvg", acc, -1, Double.NaN));
        // FedBN
        AffineMlp g2 = new AffineMlp(seed);
        List<AffineMlp> nets2 = new ArrayList<>();
        List<FederatedClient> fc2 = new ArrayList<>();
        for (int k = 0; k < clients; k++) {
            AffineMlp n = new AffineMlp(seed);
            nets2.add(n);
            fc2.add(affineClient(n, local[k], seed * 1000 + k));
        }
        FedBn bn = FedBn.of(g2.params, fc2, AffineMlp.NORMALISATION, seed);
        for (int r = 0; r < rounds; r++) {
            bn.round(localSteps, participation);
        }
        for (int k = 0; k < clients; k++) {
            bn.personalize(k);
            acc[k] = accuracy(nets2.get(k).predict(tests[k].inputs()), tests[k].y());
        }
        rows.add(summary(setting, "FedBN (input affine local)", acc, -1, Double.NaN));
        // local only, with the same number of steps a client gets on average
        int steps = (int) Math.round(rounds * localSteps * participation);
        for (int k = 0; k < clients; k++) {
            AffineMlp n = new AffineMlp(seed);
            affineClient(n, local[k], seed * 1000 + k).train(steps);
            acc[k] = accuracy(n.predict(tests[k].inputs()), tests[k].y());
        }
        rows.add(summary(setting, "local only", acc, -1, Double.NaN));
        return rows;
    }

    // ------------------------------------------------------------------ label skew

    /**
     * FedDyn and FedDF against FedAvg and SCAFFOLD under Dirichlet label skew.
     *
     * @param alpha the Dirichlet concentration
     * @param clients the clients
     * @param rounds rounds
     * @param localSteps local steps
     * @param participation the share of clients per round
     * @param target the mean accuracy whose first round is reported
     * @param seed the seed
     * @return FedAvg, SCAFFOLD, FedDyn, FedDF
     */
    public static List<FedRow> labelSkew(double alpha, int clients, int rounds, int localSteps, double participation,
            double target, long seed) {
        ClientSplits.Labelled[] d = FederatedShootout.data(FederatedShootout.Task.DIGITS, 1600, seed);
        int[][] parts = ClientSplits.dirichlet(d[0].y(), CLASSES, clients, alpha, 4, seed + 3);
        double[][] mix = new double[clients][CLASSES];
        for (int k = 0; k < clients; k++) {
            for (int i : parts[k]) {
                mix[k][d[0].y()[i]] += 1.0 / parts[k].length;
            }
        }
        ToyTasks.Data pool = ToyTasks.digits(CLASSES, 40, 8, 0.08, seed, seed + 77);
        Tensor unlabelled = pool.inputs();
        String setting = "label skew, alpha " + alpha;
        List<FedRow> rows = new ArrayList<>();
        rows.add(skewRun(setting, "FedAvg", d, parts, mix, rounds, localSteps, participation, target, seed,
                (g, c) -> {
                    FederatedAlgorithms a = FederatedAlgorithms.fedAvg(g.parameters(), c, seed);
                    return r -> a.round(localSteps, participation);
                }));
        rows.add(skewRun(setting, "SCAFFOLD", d, parts, mix, rounds, localSteps, participation, target, seed,
                (g, c) -> {
                    FederatedAlgorithms a = FederatedAlgorithms.scaffold(g.parameters(), c, seed);
                    return r -> a.round(localSteps, participation);
                }));
        rows.add(skewRun(setting, "FedDyn (alpha 0.01)", d, parts, mix, rounds, localSteps, participation, target,
                seed, (g, c) -> {
                    FedDyn a = FedDyn.of(g.parameters(), c, 0.01f, seed);
                    return r -> a.round(localSteps, participation);
                }));
        rows.add(skewRun(setting, "FedDF (FedAvg + distillation hook)", d, parts, mix, rounds, localSteps,
                participation, target, seed, (g, c) -> {
                    FederatedAlgorithms a = FederatedAlgorithms.fedAvg(g.parameters(), c, seed)
                            .aggregator(new EnsembleDistillationAggregator(g.parameters(), g::forward, unlabelled, 20,
                                    32, 1e-3f, 1f, seed));
                    return r -> a.round(localSteps, participation);
                }));
        return rows;
    }

    private interface SkewFactory {
        IntFunction<Float> create(SimpleMlp global, List<FederatedClient> clients);
    }

    private static FedRow skewRun(String setting, String name, ClientSplits.Labelled[] d, int[][] parts,
            double[][] mix, int rounds, int localSteps, double participation, double target, long seed,
            SkewFactory factory) {
        SimpleMlp global = FederatedShootout.net(CLASSES, seed);
        List<FederatedClient> fc = new ArrayList<>();
        for (int k = 0; k < parts.length; k++) {
            fc.add(FederatedShootout.client(FederatedShootout.net(CLASSES, seed), d[0].subset(parts[k]), LR,
                    seed * 1000 + k));
        }
        IntFunction<Float> round = factory.create(global, fc);
        Supplier<double[]> score = () -> {
            int[] p = argmaxRows(global.predict(d[1].inputs()));
            double[] right = new double[CLASSES];
            double[] count = new double[CLASSES];
            for (int i = 0; i < p.length; i++) {
                count[d[1].y()[i]]++;
                right[d[1].y()[i]] += p[i] == d[1].y()[i] ? 1 : 0;
            }
            double[] acc = new double[parts.length];
            for (int k = 0; k < parts.length; k++) {
                for (int c = 0; c < CLASSES; c++) {
                    acc[k] += mix[k][c] * (count[c] > 0 ? right[c] / count[c] : 0);
                }
            }
            return acc;
        };
        int reached = -1;
        for (int r = 0; r < rounds; r++) {
            round.apply(r);
            if (reached < 0 && (r % 5 == 4 || r == rounds - 1)) {
                double mean = 0;
                for (double a : score.get()) {
                    mean += a / parts.length;
                }
                if (mean >= target) {
                    reached = r + 1;
                }
            }
        }
        return summary(setting, name, score.get(), reached, Double.NaN);
    }
}
