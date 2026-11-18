package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FourierNeuralOperator1d;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * PDE-Refiner (Lippe, Veeling, Perdikaris, Turner and Brandstetter, NeurIPS 2023): a neural
 * solver that predicts the next state of a PDE and then refines it in a few denoising steps,
 * so it learns the low-amplitude, high-frequency parts of the solution that a plain one-step
 * MSE model neglects and that make long rollouts diverge.
 *
 * <p>One grid network {@code f(u_t, u~, k)} serves every step {@code k}; its input channels
 * are the current state, the noisy estimate and the step {@code k / K} as a constant channel:
 *
 * <ul>
 *   <li>{@code k = 0}: predict {@code u_{t+1} - u_t} from the current state (noisy input 0);
 *   <li>{@code k = 1..K}: with noise level {@code sigma_k = sigmaMin^(k/K)} (the paper's
 *       exponentially decreasing schedule), the input is {@code u~ = u_{t+1} + sigma_k eps} and
 *       the target is {@code eps}; at inference {@code u <- u~ - sigma_k eps_hat}.
 * </ul>
 *
 * <p>The default network is the experimental {@link FourierNeuralOperator1d} (the paper uses a
 * U-Net; any {@code [b, n, 3] -> [b, n, 1]} {@link Layer} works). With {@code K = 0} the class
 * is the plain one-step MSE baseline with the same network.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2308.05732">arXiv:2308.05732</a>.
 */
public final class PdeRefiner {

    private final int points;
    private final int refinements;
    private final double minNoise;
    private final Layer network;

    private PdeRefiner(int points, Layer network, int refinements, double minNoise) {
        this.points = points;
        this.refinements = refinements;
        this.minNoise = minNoise;
        this.network = network;
    }

    /**
     * Creates the refiner with a Fourier neural operator backbone.
     *
     * @param points the grid points
     * @param width the FNO width
     * @param modes the FNO modes
     * @param depth the FNO layers
     * @param refinements the refinement steps {@code K}, 0 for a one-step model
     * @param minNoise the smallest noise level {@code sigmaMin}, for example 1e-3
     * @param seed the seed
     * @return the refiner
     */
    public static PdeRefiner of(int points, int width, int modes, int depth, int refinements, double minNoise,
            long seed) {
        return new PdeRefiner(points,
                new FourierNeuralOperator1d(3, 1, width, modes, depth, new SplittableRandom(seed)),
                refinements, minNoise);
    }

    /**
     * Creates the refiner with any grid network.
     *
     * @param points the grid points
     * @param network {@code [b, points, 3] -> [b, points, 1]}
     * @param refinements the refinement steps
     * @param minNoise the smallest noise level
     * @return the refiner
     */
    public static PdeRefiner of(int points, Layer network, int refinements, double minNoise) {
        return new PdeRefiner(points, network, refinements, minNoise);
    }

    /**
     * @param k a refinement step, 1 to K
     * @return its noise level
     */
    public double sigma(int k) {
        return Math.pow(minNoise, (double) k / refinements);
    }

    /** @return the network's parameters */
    public List<Var> parameters() {
        return network.parameters();
    }

    private Var apply(Tensor current, Tensor noisy, float[] steps) {
        int b = current.shape().dim(0);
        float[] k = new float[b * points];
        for (int i = 0; i < b; i++) {
            java.util.Arrays.fill(k, i * points, (i + 1) * points, steps[i] / Math.max(1, refinements));
        }
        Tensor in = Tensor.concat(2, current.reshape(b, points, 1), noisy.reshape(b, points, 1),
                Tensor.of(Shape.of(b, points, 1), k));
        return network.forward(Var.of(in)).reshape(b, points);
    }

    /**
     * The training loss for a batch of transitions, with a random step {@code k} per example.
     *
     * @param current {@code [b, points]}
     * @param next {@code [b, points]}
     * @param random the source of steps and noise
     * @return the mean squared error
     */
    public Var loss(Tensor current, Tensor next, RandomGenerator random) {
        int b = current.shape().dim(0);
        float[] steps = new float[b];
        float[] noisy = new float[b * points];
        float[] target = new float[b * points];
        float[] nx = next.toArray();
        float[] cur = current.toArray();
        for (int i = 0; i < b; i++) {
            int k = refinements == 0 ? 0 : random.nextInt(refinements + 1);
            steps[i] = k;
            double s = k == 0 ? 0 : sigma(k);
            for (int j = 0; j < points; j++) {
                int idx = i * points + j;
                if (k == 0) {
                    target[idx] = nx[idx] - cur[idx];
                } else {
                    float eps = (float) random.nextGaussian();
                    noisy[idx] = (float) (nx[idx] + s * eps);
                    target[idx] = eps;
                }
            }
        }
        return apply(current, Tensor.of(Shape.of(b, points), noisy), steps)
                .sub(Var.of(Tensor.of(Shape.of(b, points), target))).square().mean();
    }

    /**
     * One solver step: predict, then refine.
     *
     * @param current {@code [b, points]}
     * @param random the source of refinement noise
     * @return the next state {@code [b, points]}
     */
    public Tensor step(Tensor current, RandomGenerator random) {
        return Var.noGrad(() -> {
            int b = current.shape().dim(0);
            Tensor pred = current.add(apply(current, Tensor.zeros(Shape.of(b, points)), new float[b]).value());
            for (int k = 1; k <= refinements; k++) {
                float s = (float) sigma(k);
                Tensor noisy = pred.add(Tensor.normal(Shape.of(b, points), 0, 1, random).mul(s));
                float[] steps = new float[b];
                java.util.Arrays.fill(steps, k);
                Tensor eps = apply(current, noisy, steps).value();
                pred = noisy.sub(eps.mul(s));
            }
            return Var.of(pred);
        }).value();
    }

    /**
     * An autoregressive rollout.
     *
     * @param initial {@code [b, points]}
     * @param steps the steps
     * @param random the source of refinement noise
     * @return states {@code [steps][b * points]}
     */
    public float[][] rollout(Tensor initial, int steps, RandomGenerator random) {
        float[][] out = new float[steps][];
        Tensor u = initial;
        for (int t = 0; t < steps; t++) {
            u = step(u, random);
            out[t] = u.toArray();
        }
        return out;
    }

    @Override
    public String toString() {
        return "PdeRefiner(points=" + points + ", K=" + refinements + ", network=" + network + ")";
    }
}
