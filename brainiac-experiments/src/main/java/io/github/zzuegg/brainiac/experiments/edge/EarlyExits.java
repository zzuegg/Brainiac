package io.github.zzuegg.brainiac.experiments.edge;

import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.edge.EarlyExitNetwork;
import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Wave 12 item 12.2.11: early-exit classifiers judged by failure prediction. An
 * {@link EarlyExitNetwork} is trained jointly (BranchyNet) or with a frozen backbone (shallow-deep
 * networks), with and without per-exit temperature scaling ({@link Calibration}), and compared
 * with static networks truncated at each depth. Each row reports the accuracy and mean MACs
 * (as a share of the full network) of the max-probability exit rule at one threshold, the area
 * under the accuracy-compute curve, the overthinking rate, the mean ECE over exits and the mean
 * EEFP over exits, so ECE and EEFP can be compared as predictors of the area.
 */
public final class EarlyExits {

    private EarlyExits() {
    }

    /**
     * One network or exit policy on one task.
     *
     * @param task the task
     * @param method the network and training
     * @param accuracy the test accuracy (of the exit rule for early-exit networks)
     * @param macsShare the mean MACs per input over the full network's
     * @param area the area under the accuracy-compute curve ({@code NaN} for static networks)
     * @param overthinking the share of inputs some earlier exit gets right but the last gets wrong
     * @param ece the mean expected calibration error over exits
     * @param eefp the mean EEFP over exits
     */
    public record ExitRow(String task, String method, double accuracy, double macsShare, double area,
            double overthinking, double ece, double eefp) {
    }

    private record Split(Tensor x, int[] y) {
    }

    private static int[] labels(Tensor ids) {
        float[] f = ids.toArray();
        int[] y = new int[f.length];
        for (int i = 0; i < f.length; i++) {
            y[i] = (int) f[i];
        }
        return y;
    }

    /**
     * Runs one task.
     *
     * @param task {@code "shapes"} (the CNN on 16×16 shape images) or {@code "digits"} (an MLP)
     * @param train training examples
     * @param steps training steps per phase
     * @param threshold the max-probability exit threshold
     * @param seed the seed
     * @return rows for static truncations, joint, joint with temperature scaling, and frozen
     */
    public static List<ExitRow> run(String task, int train, int steps, double threshold, long seed) {
        boolean shapes = task.equals("shapes");
        Split tr;
        Split va;
        Split te;
        int classes;
        int[] widths;
        if (shapes) {
            ShapeImages gen = new ShapeImages(16, 0.15f);
            Tensor[] a = gen.generate(train, seed);
            Tensor[] b = gen.generate(500, seed + 1);
            Tensor[] c = gen.generate(1000, seed + 2);
            tr = new Split(a[0], labels(a[1]));
            va = new Split(b[0], labels(b[1]));
            te = new Split(c[0], labels(c[1]));
            classes = ShapeImages.CLASSES;
            widths = new int[] {8, 16, 32};
        } else {
            classes = 10;
            ToyTasks.Data a = ToyTasks.digits(classes, train / classes, 8, 0.1, seed, seed + 11);
            ToyTasks.Data b = ToyTasks.digits(classes, 50, 8, 0.1, seed, seed + 12);
            ToyTasks.Data c = ToyTasks.digits(classes, 100, 8, 0.1, seed, seed + 13);
            tr = new Split(a.inputs(), a.y());
            va = new Split(b.inputs(), b.y());
            te = new Split(c.inputs(), c.y());
            widths = new int[] {24, 24, 24, 24};
        }
        float lr = shapes ? 1e-2f : 3e-3f;
        int batch = shapes ? 32 : 64;
        List<ExitRow> rows = new ArrayList<>();
        int exits = widths.length;
        EarlyExitNetwork probe = build(shapes, widths, classes, seed);
        long[] fullMacs = probe.macs();
        double full = fullMacs[exits - 1];
        // static networks truncated at each depth, trained on their last exit only
        for (int k = 1; k <= exits; k++) {
            int[] w = Arrays.copyOf(widths, k);
            EarlyExitNetwork net = build(shapes, w, classes, seed + k);
            double[] weights = new double[k];
            weights[k - 1] = 1;
            net.trainJoint(tr.x(), tr.y(), weights, steps, batch, lr, seed);
            double[][][] p = net.probabilities(te.x(), ones(k));
            double acc = Calibration.accuracy(p[k - 1], te.y());
            rows.add(new ExitRow(task, "static, depth " + k, acc, net.macs()[k - 1] / full, Double.NaN, Double.NaN,
                    Calibration.expectedCalibrationError(p[k - 1], te.y(), 10), Double.NaN));
        }
        // joint (BranchyNet), equal weights
        EarlyExitNetwork joint = build(shapes, widths, classes, seed);
        double[] equal = new double[exits];
        Arrays.fill(equal, 1);
        joint.trainJoint(tr.x(), tr.y(), equal, steps, batch, lr, seed);
        rows.add(row(task, "joint (BranchyNet)", joint.probabilities(te.x(), ones(exits)), te.y(), threshold,
                joint.macs()));
        double[][][] valLogits = joint.logitArrays(va.x());
        double[] temps = new double[exits];
        for (int e = 0; e < exits; e++) {
            temps[e] = Calibration.fitTemperature(valLogits[e], va.y());
        }
        rows.add(row(task, "joint + temperature scaling", joint.probabilities(te.x(), temps), te.y(), threshold,
                joint.macs()));
        // shallow-deep: backbone first, then heads on frozen features
        EarlyExitNetwork frozen = build(shapes, widths, classes, seed);
        frozen.trainFrozen(tr.x(), tr.y(), steps, batch, lr, seed);
        rows.add(row(task, "frozen backbone (SDN)", frozen.probabilities(te.x(), ones(exits)), te.y(), threshold,
                frozen.macs()));
        return rows;
    }

    private static EarlyExitNetwork build(boolean shapes, int[] widths, int classes, long seed) {
        return shapes ? EarlyExitNetwork.cnn(1, 16, widths, classes, seed)
                : EarlyExitNetwork.mlp(64, widths, classes, seed);
    }

    private static double[] ones(int n) {
        double[] t = new double[n];
        Arrays.fill(t, 1);
        return t;
    }

    private static ExitRow row(String task, String method, double[][][] probs, int[] labels, double threshold,
            long[] macs) {
        EarlyExitNetwork.Outcome o = EarlyExitNetwork.evaluate(probs, labels, EarlyExitNetwork.Rule.MAX_PROB,
                threshold, macs);
        double ece = 0;
        for (double[][] p : probs) {
            ece += Calibration.expectedCalibrationError(p, labels, 10) / probs.length;
        }
        return new ExitRow(task, method + ", max-prob " + threshold, o.accuracy(), o.meanMacs() / macs[macs.length - 1],
                EarlyExitNetwork.accuracyComputeArea(probs, labels, macs), EarlyExitNetwork.overthinking(probs, labels),
                ece, EarlyExitNetwork.meanEefp(probs, labels));
    }
}
