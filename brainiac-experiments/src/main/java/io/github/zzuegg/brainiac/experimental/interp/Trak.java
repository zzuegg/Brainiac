package io.github.zzuegg.brainiac.experimental.interp;

import io.github.zzuegg.brainiac.tensor.Linalg;

import java.util.SplittableRandom;

/**
 * TRAK: data attribution with random projections (Park et al., ICML 2023). Influence
 * functions need inverse Hessians and leave-one-out retraining is too expensive; TRAK
 * linearises the model around its trained weights, projects the per-example gradients of
 * the model output function (the margin {@code log p_y - log(1 - p_y)}) to {@code k}
 * dimensions with a Gaussian matrix, and scores training example {@code j} for target
 * {@code x} with the influence formula of the linearised logistic model,
 *
 * <pre>
 * tau(x)_j = phi(x)^T (Phi^T Phi)^(-1) phi_j * Q_j,    Q_j = 1 - p_j(correct),
 * </pre>
 *
 * <p>averaged over a few independently trained models: the kernel term is averaged over
 * models and multiplied by the averaged {@code Q}, as in the paper. Attributions are judged
 * by the {@link #linearDatamodelingScore linear datamodeling score}: the rank correlation
 * between the summed attributions of a random training subset and the actual output of a
 * model retrained on that subset.
 *
 * <p>Per-example gradients come from any source; the experiments use
 * {@code TrainingDataAttribution.perExampleGradients} with the margin as the "loss".
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2303.14186">arXiv:2303.14186</a>.
 */
public final class Trak {

    private final int dim;
    private final long seed;
    private double[][] kernelSum;
    private double[] qSum;
    private int models;

    /**
     * Creates an empty ensemble.
     *
     * @param projection the projected dimension {@code k}
     * @param seed the seed of the projection matrices (one per model)
     */
    public Trak(int projection, long seed) {
        this.dim = projection;
        this.seed = seed;
    }

    /**
     * Projects gradients with a Gaussian matrix of entries {@code N(0, 1/k)}.
     *
     * @param gradients {@code [n][p]}
     * @param k the projected dimension
     * @param seed the matrix seed
     * @return {@code [n][k]}
     */
    public static double[][] project(double[][] gradients, int k, long seed) {
        int p = gradients[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        double scale = 1 / Math.sqrt(k);
        double[][] proj = new double[p][k];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < k; j++) {
                proj[i][j] = r.nextGaussian() * scale;
            }
        }
        return InterpMath.matmul(gradients, proj);
    }

    /**
     * Adds one trained model.
     *
     * @param trainGradients per training example the gradient of the output function
     * @param targetGradients per target example the gradient of the output function
     * @param trainQ per training example {@code 1 - p(correct class)}
     */
    public void addModel(double[][] trainGradients, double[][] targetGradients, double[] trainQ) {
        long s = seed + 104729L * models;
        double[][] phi = project(trainGradients, dim, s);
        double[][] phiX = project(targetGradients, dim, s);
        double[][] gram = InterpMath.crossProduct(phi, phi, 1);
        double trace = 0;
        for (int i = 0; i < dim; i++) {
            trace += gram[i][i];
        }
        for (int i = 0; i < dim; i++) {
            gram[i][i] += 1e-6 * trace / dim + 1e-12;
        }
        double[][] inv = InterpMath.matrix(Linalg.inverse(InterpMath.tensor(gram)));
        double[][] kern = InterpMath.matmul(InterpMath.matmul(phiX, inv), InterpMath.transpose(phi));
        if (kernelSum == null) {
            kernelSum = new double[kern.length][kern[0].length];
            qSum = new double[trainQ.length];
        }
        for (int i = 0; i < kern.length; i++) {
            for (int j = 0; j < kern[0].length; j++) {
                kernelSum[i][j] += kern[i][j];
            }
        }
        for (int j = 0; j < trainQ.length; j++) {
            qSum[j] += trainQ[j];
        }
        models++;
    }

    /** @return the models added */
    public int models() {
        return models;
    }

    /**
     * The attribution scores.
     *
     * @return {@code [targets][train]}: positive means the training example raises the
     *     target's margin
     */
    public double[][] scores() {
        double[][] out = new double[kernelSum.length][kernelSum[0].length];
        for (int i = 0; i < out.length; i++) {
            for (int j = 0; j < out[0].length; j++) {
                out[i][j] = kernelSum[i][j] / models * (qSum[j] / models);
            }
        }
        return out;
    }

    /**
     * The linear datamodeling score of attributions: for each target, the Spearman
     * correlation over subsets between the summed attributions of the subset's examples and
     * the actual output of a model trained on that subset, averaged over targets.
     *
     * @param scores {@code [targets][train]} attributions
     * @param subsets {@code [subsets][train]} membership
     * @param outputs {@code [subsets][targets]} the retrained models' outputs (margins)
     * @return the mean rank correlation
     */
    public static double linearDatamodelingScore(double[][] scores, boolean[][] subsets, double[][] outputs) {
        double total = 0;
        for (int t = 0; t < scores.length; t++) {
            double[] predicted = new double[subsets.length];
            double[] actual = new double[subsets.length];
            for (int s = 0; s < subsets.length; s++) {
                double sum = 0;
                for (int j = 0; j < subsets[s].length; j++) {
                    if (subsets[s][j]) {
                        sum += scores[t][j];
                    }
                }
                predicted[s] = sum;
                actual[s] = outputs[s][t];
            }
            total += InterpMath.spearman(predicted, actual);
        }
        return total / scores.length;
    }
}
