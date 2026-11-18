package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.interp.ConceptErasure.Eraser;
import io.github.zzuegg.brainiac.experimental.interp.ModelStitcher;
import io.github.zzuegg.brainiac.experimental.nn.LinearProbe;
import io.github.zzuegg.brainiac.experimental.nn.PersonaVector;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 9 item 9.4.3: linear concept erasure on the spurious-colour task of
 * {@link CausalAndFairness}. A bar (horizontal = class 0, vertical = class 1) is drawn in
 * one of two colour channels, and in training the colour agrees with the class 90% of the
 * time; at test time it is independent. An MLP (128-32-2) trained by ERM picks up the
 * colour. Its hidden layer is the representation; the concept to erase is the colour.
 *
 * <p>Each eraser ({@link io.github.zzuegg.brainiac.experimental.interp.ConceptErasure}) is
 * fitted on 600 colour-balanced rows with colour labels (the oracle-free LEACE on 60 of
 * them). Then, on held-out balanced rows, a linear probe and an MLP probe try to recover the
 * colour from erased representations (lower is better, chance is 0.5), and a linear head for
 * the class is retrained on erased representations of the spurious training set and scored
 * on balanced test rows: mean and worst-group accuracy (groups are class x colour). The mean
 * squared change of the representation measures the damage.
 *
 * <p>Rows: no erasure, LEACE, oracle-free LEACE, INLP (2 and 8 iterations), the mean
 * difference direction of {@link PersonaVector} projected out, and subtracting the
 * colour's steering vector from colour-1 rows (which needs the colour at test time); the
 * GroupDRO model of {@code CausalAndFairness} is the training-time baseline.
 */
public final class ConceptErasure {

    private ConceptErasure() {
    }

    /** Rows of the coloured-bar task. */
    record Coloured(Tensor x, int[] y, int[] colour) {
    }

    /** The coloured bars of {@code CausalAndFairness}, regenerated here. */
    static Coloured coloured(int n, double agreement, long seed) {
        Tensor[] g = new ShapeImages(8, 0.25f).generate(4 * n, seed);
        float[] img = g[0].toArray();
        float[] ids = g[1].toArray();
        SplittableRandom r = new SplittableRandom(seed + 1);
        float[] out = new float[n * 128];
        int[] y = new int[n];
        int[] colour = new int[n];
        int k = 0;
        for (int i = 0; i < ids.length && k < n; i++) {
            if (ids[i] > 1) {
                continue;
            }
            int cls = (int) ids[i];
            int c = r.nextDouble() < agreement ? cls : 1 - cls;
            System.arraycopy(img, i * 64, out, k * 128 + c * 64, 64);
            for (int p = 0; p < 64; p++) {
                out[k * 128 + (1 - c) * 64 + p] = (float) (0.25 * r.nextGaussian());
            }
            y[k] = cls;
            colour[k] = c;
            k++;
        }
        return new Coloured(Tensor.of(Shape.of(n, 128), out), y, colour);
    }

    private static Tensor ids(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return Tensor.of(Shape.of(v.length), f);
    }

    private static float[][] hidden(Model m, Tensor x) {
        Tensor h = ModelStitcher.activation(m, 0, x);
        int n = h.shape().dim(0);
        int d = h.shape().dim(1);
        float[] f = h.toArray();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(f, i * d, out[i], 0, d);
        }
        return out;
    }

    /**
     * Scores of one eraser.
     *
     * @param method the eraser
     * @param linearProbe best linear probe accuracy on the colour after erasure
     * @param mlpProbe MLP probe accuracy on the colour after erasure
     * @param taskAccuracy mean test accuracy of a class head retrained on erased representations
     * @param worstGroup its worst-group accuracy
     * @param change the mean squared change of the representation
     */
    public record Scores(String method, double linearProbe, double mlpProbe, double taskAccuracy, double worstGroup,
            double change) {
    }

    /**
     * Runs every eraser.
     *
     * @param seed the seed
     * @return one row per eraser, then the GroupDRO baseline (probes not applicable, -1)
     */
    public static List<Scores> run(long seed) {
        Coloured train = coloured(1000, 0.9, seed);
        Coloured fit = coloured(600, 0.5, seed + 10);
        Coloured test = coloured(1000, 0.5, seed + 20);
        Model m = Model.sequential().input(128).dense(32, Activation.RELU).dense(2, Activation.IDENTITY).seed(seed)
                .build();
        Optimizer opt = Optimizer.adam(1e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        Tensor ty = ids(train.y());
        for (int s = 0; s < 400; s++) {
            int f = r.nextInt(1000 - 100);
            Functional.crossEntropy(m.forward(Var.of(train.x().slice(0, f, f + 100)), true), ty.slice(0, f, f + 100))
                    .backward();
            opt.step(m.parameters());
        }
        float[][] hTrain = hidden(m, train.x());
        float[][] hFit = hidden(m, fit.x());
        float[][] hTest = hidden(m, test.x());
        int d = hFit[0].length;
        List<Eraser> erasers = new ArrayList<>();
        erasers.add(io.github.zzuegg.brainiac.experimental.interp.ConceptErasure.none(d));
        erasers.add(io.github.zzuegg.brainiac.experimental.interp.ConceptErasure.leace(hFit, fit.colour(), 2));
        int[] few = io.github.zzuegg.brainiac.experimental.interp.ConceptErasure.subset(hFit.length, 60, seed + 5);
        float[][] fewX = new float[few.length][];
        int[] fewZ = new int[few.length];
        for (int i = 0; i < few.length; i++) {
            fewX[i] = hFit[few[i]];
            fewZ[i] = fit.colour()[few[i]];
        }
        Eraser oracleFree = io.github.zzuegg.brainiac.experimental.interp.ConceptErasure.leace(fewX, fewZ, 2);
        erasers.add(new Eraser(oracleFree.mean(), oracleFree.correction(), "LEACE (60 labels)"));
        erasers.add(io.github.zzuegg.brainiac.experimental.interp.ConceptErasure.inlp(hFit, fit.colour(), 2, 2, 200));
        erasers.add(io.github.zzuegg.brainiac.experimental.interp.ConceptErasure.inlp(hFit, fit.colour(), 2, 8, 200));
        List<float[]> with = new ArrayList<>();
        List<float[]> without = new ArrayList<>();
        for (int i = 0; i < hFit.length; i++) {
            (fit.colour()[i] == 1 ? with : without).add(hFit[i]);
        }
        PersonaVector pv = PersonaVector.fromActivations(0, with.toArray(new float[0][]), without.toArray(new float[0][]));
        float[] u = pv.unit();
        double[][] proj = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                proj[i][j] = (double) u[i] * u[j];
            }
        }
        erasers.add(new Eraser(new double[d], proj, "mean difference (PersonaVector)"));
        List<Scores> rows = new ArrayList<>();
        for (Eraser e : erasers) {
            rows.add(score(e.name(), e.apply(hFit), fit.colour(), e.apply(hTrain), train, e.apply(hTest), test,
                    e.meanSquaredChange(hTest), seed));
        }
        float[] steer = pv.vector().direction();
        rows.add(score("steering vector subtraction (needs colour)", subtract(hFit, fit.colour(), steer), fit.colour(),
                subtract(hTrain, train.colour(), steer), train, subtract(hTest, test.colour(), steer), test,
                change(hTest, subtract(hTest, test.colour(), steer)), seed));
        CausalAndFairness.RobustScores dro = CausalAndFairness.spurious(CausalAndFairness.RobustMethod.GROUP_DRO, seed);
        rows.add(new Scores("GroupDRO (training-time baseline)", -1, -1, dro.average(), dro.worstGroup(), 0));
        return rows;
    }

    private static float[][] subtract(float[][] x, int[] z, float[] v) {
        float[][] out = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i].clone();
            if (z[i] == 1) {
                for (int j = 0; j < v.length; j++) {
                    out[i][j] -= v[j];
                }
            }
        }
        return out;
    }

    private static double change(float[][] a, float[][] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                s += (a[i][j] - b[i][j]) * (a[i][j] - b[i][j]);
            }
        }
        return s / a.length;
    }

    private static Scores score(String name, float[][] fitX, int[] fitZ, float[][] trainX, Coloured train,
            float[][] testX, Coloured test, double change, long seed) {
        LinearProbe probe = LinearProbe.fit(fitX, fitZ, 2, 300, 0.05, 1e-4);
        double linear = probe.accuracy(testX, test.colour());
        double mlp = mlpProbe(fitX, fitZ, testX, test.colour(), seed);
        LinearProbe head = LinearProbe.fit(trainX, train.y(), 2, 300, 0.05, 1e-4);
        double[] hit = new double[4];
        double[] cnt = new double[4];
        int correct = 0;
        for (int i = 0; i < testX.length; i++) {
            boolean ok = head.predict(testX[i]) == test.y()[i];
            int g = 2 * test.y()[i] + test.colour()[i];
            hit[g] += ok ? 1 : 0;
            cnt[g]++;
            correct += ok ? 1 : 0;
        }
        double worst = 1;
        for (int g = 0; g < 4; g++) {
            if (cnt[g] > 0) {
                worst = Math.min(worst, hit[g] / cnt[g]);
            }
        }
        return new Scores(name, linear, mlp, (double) correct / testX.length, worst, change);
    }

    private static double mlpProbe(float[][] x, int[] z, float[][] testX, int[] testZ, long seed) {
        int d = x[0].length;
        Model m = Model.sequential().input(d).dense(32, Activation.RELU).dense(2, Activation.IDENTITY)
                .seed(seed + 77).build();
        Optimizer opt = Optimizer.adam(1e-2f);
        Tensor tx = Tensor.of(x);
        Tensor tz = ids(z);
        for (int s = 0; s < 300; s++) {
            Functional.crossEntropy(m.forward(Var.of(tx), true), tz).backward();
            opt.step(m.parameters());
        }
        float[] p = Var.noGrad(() -> m.forward(Var.of(Tensor.of(testX)))).value().argmax(1).toArray();
        int hit = 0;
        for (int i = 0; i < testZ.length; i++) {
            hit += p[i] == testZ[i] ? 1 : 0;
        }
        return (double) hit / testZ.length;
    }
}
