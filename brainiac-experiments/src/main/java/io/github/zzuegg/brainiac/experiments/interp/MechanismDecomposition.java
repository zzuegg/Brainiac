package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ParameterDecomposition;
import io.github.zzuegg.brainiac.experimental.nn.SparseAutoencoder;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Stochastic parameter decomposition on the toy model of superposition (TMS): five sparse
 * features squeezed through two dimensions and read back out,
 * {@code y = relu(W^T (W x) + b)}, with {@code W} used twice. Each feature has a planted
 * mechanism, the rank-one slice {@code e_k W_k} of the weights. SPD
 * ({@link ParameterDecomposition}) decomposes {@code W} into more components than features
 * with causal-importance gates; it should find one component per feature and use about
 * one component per input. The baseline is a sparse autoencoder on the hidden activations,
 * which can only find directions in activation space, not pieces of the weights: its
 * directions are compared with the feature directions {@code W_k}.
 *
 * <p>Bushnaq et al. 2025, "Stochastic Parameter Decomposition",
 * <a href="https://arxiv.org/abs/2506.20790">arXiv:2506.20790</a>; Elhage et al. 2022,
 * "Toy Models of Superposition", <a href="https://arxiv.org/abs/2209.10652">arXiv:2209.10652</a>.
 */
public final class MechanismDecomposition {

    /**
     * The outcome.
     *
     * @param tmsLoss the toy model's reconstruction loss
     * @param mechanisms the planted mechanisms (features)
     * @param spdRecovered mechanisms matched by an SPD component with cosine above 0.9
     * @param spdActivePerInput mean components with importance above 0.1 per single-feature input
     * @param faithfulness relative error of the summed components against {@code W}
     * @param saeRecovered feature directions matched by an SAE direction with cosine above 0.9
     */
    public record Outcome(double tmsLoss, int mechanisms, int spdRecovered, double spdActivePerInput,
            double faithfulness, int saeRecovered) {
    }

    private static final int FEATURES = 5;
    private static final int HIDDEN = 2;
    private static final int BATCH = 128;

    private MechanismDecomposition() {
    }

    private static Tensor batch(SplittableRandom r, int n, double p) {
        float[] x = new float[n * FEATURES];
        for (int i = 0; i < x.length; i++) {
            x[i] = r.nextDouble() < p ? (float) r.nextDouble() : 0;
        }
        return Tensor.of(Shape.of(n, FEATURES), x);
    }

    /**
     * Runs the experiment.
     *
     * @param spdSteps decomposition training steps
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int spdSteps, long seed) {
        return run(spdSteps, 0.05, 0.02f, 10f, seed);
    }

    /**
     * Runs the experiment with explicit settings.
     *
     * @param spdSteps decomposition training steps
     * @param minimality the weight of the importance-minimality penalty
     * @param learningRate the decomposition learning rate
     * @param reconstruction the weight of the stochastic reconstruction loss
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int spdSteps, double minimality, float learningRate, float reconstruction, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Var w = Var.param(Tensor.normal(Shape.of(FEATURES, HIDDEN), 0, 0.5f, r));
        Var b = Var.param(Tensor.zeros(Shape.of(FEATURES)));
        Optimizer opt = Optimizer.adam(0.02f);
        double tmsLoss = 0;
        for (int s = 0; s < 1500; s++) {
            Var x = Var.of(batch(r, 256, 0.05));
            Var y = x.matmul(w).matmul(w.transpose()).add(b).relu();
            Var loss = y.sub(x).square().mean();
            loss.backward();
            opt.step(java.util.List.of(w, b));
            tmsLoss = loss.value().item();
        }
        Tensor wt = w.value();
        Tensor bias = b.value();
        ParameterDecomposition spd = ParameterDecomposition.of(FEATURES, HIDDEN, 2 * FEATURES, 8, seed + 1);
        Optimizer so = Optimizer.adam(learningRate);
        for (int s = 0; s < spdSteps; s++) {
            Var x = Var.of(batch(r, BATCH, 0.1));
            Var target = x.matmul(Var.of(wt)).matmul(Var.of(wt.transpose())).add(bias).relu();
            Var t = spd.gateLogits(x);
            Var mask = ParameterDecomposition.stochasticMask(ParameterDecomposition.lowerLeaky(t), r);
            Var masked = spd.applyTransposed(spd.apply(x, mask), mask).add(bias).relu();
            float minWeight = (float) (minimality * Math.min(1, s / (0.3 * spdSteps)));
            Var loss = spd.faithfulness(wt).add(masked.sub(target).square().sum(1, false).mean().mul(reconstruction))
                    .add(ParameterDecomposition.minimality(ParameterDecomposition.upperLeaky(t), 1f).mul(minWeight));
            loss.backward();
            so.step(spd.parameters());
        }
        float[] wf = wt.to(DType.FLOAT32).toArray();
        int recovered = 0;
        for (int k = 0; k < FEATURES; k++) {
            float[][] truth = new float[FEATURES][HIDDEN];
            for (int j = 0; j < HIDDEN; j++) {
                truth[k][j] = wf[k * HIDDEN + j];
            }
            double best = 0;
            for (int c = 0; c < spd.components(); c++) {
                best = Math.max(best, cosine(truth, spd.component(c)));
            }

            recovered += best > 0.9 ? 1 : 0;
        }
        float[] single = new float[FEATURES * FEATURES];
        for (int k = 0; k < FEATURES; k++) {
            single[k * FEATURES + k] = 0.75f;
        }
        float[] g = spd.importance(Var.of(Tensor.of(Shape.of(FEATURES, FEATURES), single))).value()
                .to(DType.FLOAT32).toArray();
        double active = 0;
        for (float v : g) {
            active += v > 0.1 ? 1.0 / FEATURES : 0;
        }
        double err = spd.weight().sub(wt).square().sum().value().item() / wt.square().sum().item();
        // baseline: an SAE on the two-dimensional hidden activations
        float[][] hidden = new float[3000][];
        for (int i = 0; i < hidden.length; i++) {
            Tensor x = batch(r, 1, 0.2);
            hidden[i] = x.matmul(wt).to(DType.FLOAT32).toArray();
        }
        SparseAutoencoder sae = SparseAutoencoder.topK(HIDDEN, 2 * FEATURES, 1, seed + 2);
        sae.train(hidden, 1500, 128, 3e-3f, seed + 3);
        int saeRecovered = 0;
        for (int k = 0; k < FEATURES; k++) {
            float[][] dir = {{wf[k * HIDDEN], wf[k * HIDDEN + 1]}};
            double best = 0;
            for (int f = 0; f < sae.features(); f++) {
                best = Math.max(best, cosine(dir, new float[][] {sae.direction(f)}));
            }
            saeRecovered += best > 0.9 ? 1 : 0;
        }
        return new Outcome(tmsLoss, FEATURES, recovered, active, Math.sqrt(err), saeRecovered);
    }

    private static double cosine(float[][] a, float[][] b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                dot += a[i][j] * b[i][j];
                na += a[i][j] * a[i][j];
                nb += b[i][j] * b[i][j];
            }
        }
        return na == 0 || nb == 0 ? 0 : Math.abs(dot) / Math.sqrt(na * nb);
    }
}
