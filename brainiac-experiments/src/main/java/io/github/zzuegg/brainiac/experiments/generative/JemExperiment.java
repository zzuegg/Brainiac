package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.llm.ContaminationDetector;
import io.github.zzuegg.brainiac.experimental.nn.EnergyRefiner;
import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.experimental.train.JointEnergyModel;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * JEM against a plain classifier and an {@link EnergyRefiner} (wave 6, item 4.8) on 8x8
 * {@link ShapeImages}: the same MLP trained by cross-entropy alone and by
 * {@link JointEnergyModel} (cross-entropy plus persistent-SGLD maximum likelihood of
 * {@code p(x)}). Scores: test accuracy, expected calibration error ({@link Calibration}), OOD
 * AUROC against pixel-shuffled test images (max-softmax score for every model, and JEM's own
 * density {@code logsumexp f(x)}), and sample quality: the mean squared distance per pixel from
 * JEM's persistent-chain samples (and fresh 100-step Langevin chains from uniform noise) to the nearest training image, against
 * that of the uniform noise itself.
 */
@SuppressWarnings("exports")
public final class JemExperiment {
    private static final int SIDE = 8;
    private static final int FEATURES = SIDE * SIDE;

    private JemExperiment() {
    }

    private static Model mlp(long seed) {
        return Model.sequential().input(FEATURES).dense(128, Activation.SILU).dense(128, Activation.SILU)
                .dense(ShapeImages.CLASSES).seed(seed).build();
    }

    private static int[] labels(Tensor ids) {
        float[] f = ids.toArray();
        int[] out = new int[f.length];
        for (int i = 0; i < f.length; i++) {
            out[i] = (int) f[i];
        }
        return out;
    }

    private static Tensor flat(Tensor images) {
        return images.reshape(Shape.of(images.shape().dim(0), FEATURES));
    }

    private static double[][] rows(Tensor t, int cols) {
        float[] v = t.toArray();
        double[][] out = new double[v.length / cols][cols];
        for (int i = 0; i < out.length; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = v[i * cols + j];
            }
        }
        return out;
    }

    private static double[][] softmax(double[][] logits) {
        double[][] p = new double[logits.length][];
        for (int i = 0; i < p.length; i++) {
            p[i] = Calibration.softmax(logits[i], 1);
        }
        return p;
    }

    private static double[] maxProb(double[][] p) {
        double[] out = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            out[i] = p[i][Calibration.argmax(p[i])];
        }
        return out;
    }

    /**
     * Trains the three models and scores them.
     *
     * @param iterations steps of 64 images
     * @param seed the seed
     * @return named metrics
     */
    public static Map<String, Double> run(int iterations, long seed) {
        ShapeImages data = new ShapeImages(SIDE, 0.1f);
        Tensor[] train = data.generate(2000, seed);
        Tensor[] test = data.generate(1000, seed + 1);
        Tensor trainX = flat(train[0]);
        int[] trainY = labels(train[1]);
        Tensor testX = flat(test[0]);
        int[] testY = labels(test[1]);
        float[] trainV = trainX.toArray();

        Model plain = mlp(seed);
        JointEnergyModel jem = JointEnergyModel.of(mlp(seed), FEATURES, 1000, -0.3f, 1.3f, seed).sgld(10, 1f, 0.01f)
                .energyWeights(0.2f, 0.1f);
        EnergyRefiner refiner = EnergyRefiner.of(FEATURES, ShapeImages.CLASSES, 128, 5, 0.1f, seed);
        Optimizer plainOpt = Optimizer.adam(2e-3f);
        Optimizer jemOpt = Optimizer.adam(2e-3f);
        Optimizer refOpt = Optimizer.adam(3e-3f);
        JointEnergyModel ce = JointEnergyModel.of(plain, FEATURES, 1, 0, 1, seed).energyWeights(0, 0);
        SplittableRandom r = new SplittableRandom(seed);
        int b = 64;
        for (int it = 0; it < iterations; it++) {
            float[] xb = new float[b * FEATURES];
            int[] yb = new int[b];
            float[] onehot = new float[b * ShapeImages.CLASSES];
            for (int i = 0; i < b; i++) {
                int k = r.nextInt(trainY.length);
                System.arraycopy(trainV, k * FEATURES, xb, i * FEATURES, FEATURES);
                yb[i] = trainY[k];
                onehot[i * ShapeImages.CLASSES + yb[i]] = 1;
            }
            Tensor x = Tensor.of(Shape.of(b, FEATURES), xb);
            ce.loss(x, yb, r).backward();
            plainOpt.step(plain.parameters());
            jem.loss(x, yb, r).backward();
            Nets.clip(jem.classifier().parameters(), 5f);
            jemOpt.step(jem.classifier().parameters());
            refiner.loss(Var.of(x), Tensor.of(Shape.of(b, ShapeImages.CLASSES), onehot),
                    Var.of(Tensor.zeros(Shape.of(b, ShapeImages.CLASSES))), 5).backward();
            refOpt.step(refiner.parameters());
        }
        // OOD: every test image with its pixels shuffled (same brightness histogram, no shape)
        float[] testV = testX.toArray();
        float[] ood = new float[testV.length];
        SplittableRandom shuffle = new SplittableRandom(seed + 2);
        for (int i = 0; i < testY.length; i++) {
            int[] perm = new int[FEATURES];
            for (int p = 0; p < FEATURES; p++) {
                int q = shuffle.nextInt(p + 1);
                perm[p] = perm[q];
                perm[q] = p;
            }
            for (int p = 0; p < FEATURES; p++) {
                ood[i * FEATURES + p] = testV[i * FEATURES + perm[p]];
            }
        }
        Tensor oodX = Tensor.of(testX.shape(), ood);
        Map<String, Double> out = new LinkedHashMap<>();
        Model[] classifiers = {plain, jem.classifier()};
        String[] names = {"plain", "JEM"};
        for (int m = 0; m < 2; m++) {
            double[][] p = softmax(rows(classifiers[m].forward(Var.of(testX), false).value(), ShapeImages.CLASSES));
            double[][] po = softmax(rows(classifiers[m].forward(Var.of(oodX), false).value(), ShapeImages.CLASSES));
            out.put(names[m] + " accuracy", Calibration.accuracy(p, testY));
            out.put(names[m] + " ECE", Calibration.expectedCalibrationError(p, testY, 10));
            out.put(names[m] + " OOD AUROC maxprob", ContaminationDetector.auroc(maxProb(p), maxProb(po)));
        }
        double[] eIn = jem.energy(testX);
        double[] eOut = jem.energy(oodX);
        for (int i = 0; i < eIn.length; i++) {
            eIn[i] = -eIn[i];
            eOut[i] = -eOut[i];
        }
        out.put("JEM OOD AUROC density", ContaminationDetector.auroc(eIn, eOut));
        // the refiner's outputs regress one-hot targets: read them as logits with a temperature
        // fitted on the training set
        double[][] refTrain = rows(refiner.forward(Var.of(trainX)).value(), ShapeImages.CLASSES);
        double temperature = Calibration.fitTemperature(refTrain, trainY);
        double[][] pr = Calibration.withTemperature(rows(refiner.forward(Var.of(testX)).value(),
                ShapeImages.CLASSES), temperature);
        double[][] pro = Calibration.withTemperature(rows(refiner.forward(Var.of(oodX)).value(),
                ShapeImages.CLASSES), temperature);
        out.put("EnergyRefiner accuracy", Calibration.accuracy(pr, testY));
        out.put("EnergyRefiner ECE", Calibration.expectedCalibrationError(pr, testY, 10));
        out.put("EnergyRefiner OOD AUROC maxprob", ContaminationDetector.auroc(maxProb(pr), maxProb(pro)));
        // samples: the plain classifier's confidence on JEM's samples, against uniform noise
        SplittableRandom sr = new SplittableRandom(seed + 3);
        Tensor noise = jem.sample(200, 0, sr);
        Tensor samples = jem.sample(200, 100, sr);
        out.put("noise distance", nearest(noise, trainV));
        out.put("JEM fresh sample distance", nearest(samples, trainV));
        out.put("JEM buffer sample distance", nearest(jem.buffer(200), trainV));
        return out;
    }

    /** The mean squared distance per pixel from each sample to its nearest training image. */
    private static double nearest(Tensor samples, float[] train) {
        float[] s = samples.toArray();
        int n = s.length / FEATURES;
        int m = train.length / FEATURES;
        double total = 0;
        for (int i = 0; i < n; i++) {
            double best = Double.MAX_VALUE;
            for (int j = 0; j < m; j++) {
                double d = 0;
                for (int p = 0; p < FEATURES && d < best; p++) {
                    double e = s[i * FEATURES + p] - train[j * FEATURES + p];
                    d += e * e;
                }
                best = Math.min(best, d);
            }
            total += best / FEATURES / n;
        }
        return total;
    }
}
