package io.github.zzuegg.brainiac.experiments.statistics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;
import io.github.zzuegg.brainiac.experimental.stats.Multicalibration;
import io.github.zzuegg.brainiac.experimental.train.GroupRobustTraining;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Calibration on subgroups (item 12.1.16): multicalibration with the eight groups built from two
 * binary attributes and their intersections, and multiaccuracy boosting with a ridge or tree
 * auditor that never sees group labels, against global temperature scaling, global isotonic
 * recalibration, per-group Platt scaling with the four intersections known, and a GroupDRO-trained
 * logistic model. The base model is a main-effects logistic regression made overconfident (logits
 * times 1.6, as an overfitted network would be) on data whose intersection groups differ in base
 * rate and in the slope of one feature, so it is miscalibrated both globally and per group.
 * Scored on a large test set by overall ECE, worst-intersection ECE and AUROC.
 */
public final class FairCalibration {

    private FairCalibration() {
    }

    private record Data(double[][] x, int[] y, int[] group) {
    }

    // intercept shift and slope multiplier on x0 per intersection group (a, b) = 00, 01, 10, 11
    private static final double[] SHIFT = {0, 0.4, -0.7, 1.3};
    private static final double[] SLOPE = {1, 0.5, 1, 2.2};

    private static Data sample(int n, SplittableRandom r) {
        double[][] x = new double[n][7];
        int[] y = new int[n];
        int[] g = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 5; j++) {
                x[i][j] = r.nextGaussian();
            }
            int a = r.nextDouble() < 0.3 ? 1 : 0;
            int b = r.nextDouble() < 0.3 ? 1 : 0;
            x[i][5] = a;
            x[i][6] = b;
            g[i] = 2 * a + b;
            double z = SLOPE[g[i]] * 0.9 * x[i][0] + 0.8 * x[i][1] - 0.5 * x[i][2] + 0.3 * x[i][3] + SHIFT[g[i]] - 0.3;
            y[i] = r.nextDouble() < 1 / (1 + Math.exp(-z)) ? 1 : 0;
        }
        return new Data(x, y, g);
    }

    private static double[][] withBias(double[][] x) {
        double[][] f = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            f[i] = new double[x[i].length + 1];
            System.arraycopy(x[i], 0, f[i], 0, x[i].length);
            f[i][x[i].length] = 1;
        }
        return f;
    }

    private static double[] logits(double[] beta, double[][] x, double scale) {
        double[][] f = withBias(x);
        double[] z = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < beta.length; j++) {
                z[i] += beta[j] * f[i][j];
            }
            z[i] *= scale;
        }
        return z;
    }

    private static double[] sigmoid(double[] z) {
        double[] p = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            p[i] = 1 / (1 + Math.exp(-z[i]));
        }
        return p;
    }

    private static boolean[][] groups(Data d) {
        // a, not a, b, not b, and the four intersections
        boolean[][] g = new boolean[8][d.y().length];
        for (int i = 0; i < d.y().length; i++) {
            int a = d.group()[i] / 2;
            int b = d.group()[i] % 2;
            g[a == 1 ? 0 : 1][i] = true;
            g[b == 1 ? 2 : 3][i] = true;
            g[4 + d.group()[i]][i] = true;
        }
        return g;
    }

    private static double[] toDouble(int[] y) {
        double[] v = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            v[i] = y[i];
        }
        return v;
    }

    /**
     * Runs the comparison.
     *
     * @param train training examples for the base model
     * @param cal calibration examples
     * @param test test examples
     * @param seeds seeds averaged over
     * @return {@code {overall ECE, worst-intersection ECE, AUROC}} per method
     */
    public static Map<String, double[]> run(int train, int cal, int test, int seeds) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int seed = 0; seed < seeds; seed++) {
            SplittableRandom r = new SplittableRandom(1000 + seed);
            Data tr = sample(train, r);
            Data ca = sample(cal, r);
            Data te = sample(test, r);
            double[] beta = Recalibrators.logistic(withBias(tr.x()), toDouble(tr.y()), 1e-3);
            double over = 1.6;
            double[] zc = logits(beta, ca.x(), over);
            double[] zt = logits(beta, te.x(), over);
            double[] pc = sigmoid(zc);
            double[] pt = sigmoid(zt);
            Map<String, double[]> preds = new LinkedHashMap<>();
            preds.put("uncalibrated", pt);
            // global temperature: a single slope on the logit
            double[][] fz = new double[zc.length][];
            for (int i = 0; i < zc.length; i++) {
                fz[i] = new double[] {zc[i]};
            }
            double tSlope = Recalibrators.logistic(fz, toDouble(ca.y()), 0)[0];
            double[] ztT = zt.clone();
            for (int i = 0; i < ztT.length; i++) {
                ztT[i] *= tSlope;
            }
            preds.put("temperature", sigmoid(ztT));
            // global isotonic
            double[][] iso = Recalibrators.isotonic(pc, toDouble(ca.y()));
            double[] pIso = new double[pt.length];
            for (int i = 0; i < pt.length; i++) {
                pIso[i] = isoLookup(iso, pt[i]);
            }
            preds.put("isotonic", pIso);
            // per-group Platt with the four intersections known
            double[] pg = new double[pt.length];
            for (int g = 0; g < 4; g++) {
                List<double[]> rows = new ArrayList<>();
                List<Double> ys = new ArrayList<>();
                for (int i = 0; i < zc.length; i++) {
                    if (ca.group()[i] == g) {
                        rows.add(new double[] {zc[i], 1});
                        ys.add((double) ca.y()[i]);
                    }
                }
                double[] yy = ys.stream().mapToDouble(Double::doubleValue).toArray();
                double[] ab = Recalibrators.logistic(rows.toArray(new double[0][]), yy, 1e-3);
                for (int i = 0; i < zt.length; i++) {
                    if (te.group()[i] == g) {
                        pg[i] = 1 / (1 + Math.exp(-(ab[0] * zt[i] + ab[1])));
                    }
                }
            }
            preds.put("per-group Platt", pg);
            preds.put("GroupDRO", groupDro(tr, te, seed));
            Multicalibration mc = Multicalibration.multicalibrate(pc, ca.y(), groups(ca), 10, 0.02, 0.005, 400);
            preds.put("MULTICALIBRATE", mc.apply(pt, groups(te), null));
            Multicalibration ridge = Multicalibration.multiaccuracy(pc, ca.y(), ca.x(),
                    Multicalibration.Auditor.RIDGE, 0.5, 1e-3, 30);
            preds.put("MULTIACCURACY ridge", ridge.apply(pt, null, te.x()));
            Multicalibration tree = Multicalibration.multiaccuracy(pc, ca.y(), ca.x(),
                    Multicalibration.Auditor.TREE, 0.5, 1e-3, 30);
            preds.put("MULTIACCURACY tree", tree.apply(pt, null, te.x()));
            for (Map.Entry<String, double[]> e : preds.entrySet()) {
                double[] p = e.getValue();
                double worst = 0;
                for (int g = 0; g < 4; g++) {
                    List<Double> gp = new ArrayList<>();
                    List<Integer> gy = new ArrayList<>();
                    for (int i = 0; i < p.length; i++) {
                        if (te.group()[i] == g) {
                            gp.add(p[i]);
                            gy.add(te.y()[i]);
                        }
                    }
                    worst = Math.max(worst, Scores.ece(gp.stream().mapToDouble(Double::doubleValue).toArray(),
                            gy.stream().mapToInt(Integer::intValue).toArray(), 10));
                }
                double[] acc = out.computeIfAbsent(e.getKey(), k -> new double[3]);
                acc[0] += Scores.ece(p, te.y(), 10) / seeds;
                acc[1] += worst / seeds;
                acc[2] += Scores.auc(p, te.y()) / seeds;
            }
        }
        return out;
    }

    private static double isoLookup(double[][] iso, double s) {
        double[] xs = iso[0];
        double[] ys = iso[1];
        if (s <= xs[0]) {
            return ys[0];
        }
        for (int k = 1; k < xs.length; k++) {
            if (s <= xs[k]) {
                double t = (s - xs[k - 1]) / (xs[k] - xs[k - 1]);
                return ys[k - 1] + t * (ys[k] - ys[k - 1]);
            }
        }
        return ys[ys.length - 1];
    }

    /** A logistic model on the same features trained with GroupDRO over the four intersections. */
    private static double[] groupDro(Data tr, Data te, int seed) {
        int d = tr.x()[0].length;
        Var w = TinyMlp.linear(d, 1, seed);
        Var b = Var.param(Tensor.zeros(Shape.of(1, 1)));
        List<Var> xs = new ArrayList<>();
        List<Tensor> ys = new ArrayList<>();
        for (int g = 0; g < 4; g++) {
            List<double[]> rows = new ArrayList<>();
            List<Float> yv = new ArrayList<>();
            for (int i = 0; i < tr.y().length; i++) {
                if (tr.group()[i] == g) {
                    rows.add(tr.x()[i]);
                    yv.add((float) tr.y()[i]);
                }
            }
            xs.add(Var.of(TinyMlp.matrix(rows.toArray(new double[0][]))));
            float[] t = new float[yv.size()];
            for (int i = 0; i < t.length; i++) {
                t[i] = yv.get(i);
            }
            ys.add(Tensor.of(Shape.of(t.length, 1), t));
        }
        GroupRobustTraining.GroupDro dro = new GroupRobustTraining.GroupDro(4, 0.05);
        Optimizer opt = Optimizer.adam(0.05f);
        for (int step = 0; step < 300; step++) {
            List<Var> losses = new ArrayList<>();
            for (int g = 0; g < 4; g++) {
                Var z = xs.get(g).matmul(w).add(b);
                losses.add(z.softplus().sub(z.mul(ys.get(g))).mean());
            }
            dro.loss(losses).backward();
            opt.step(List.of(w, b));
        }
        float[] z = Var.noGrad(() -> Var.of(TinyMlp.matrix(te.x())).matmul(w).add(b)).value().toArray();
        double[] p = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            p[i] = 1 / (1 + Math.exp(-z[i]));
        }
        return p;
    }
}
