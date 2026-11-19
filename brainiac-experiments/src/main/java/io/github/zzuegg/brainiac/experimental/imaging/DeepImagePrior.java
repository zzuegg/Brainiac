package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.UNet2d;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.SplittableRandom;

/**
 * The deep image prior (Ulyanov, Vedaldi &amp; Lempitsky, CVPR 2018,
 * <a href="https://arxiv.org/abs/1711.10925">arXiv:1711.10925</a>): a randomly initialised
 * encoder-decoder ({@link UNet2d} with skips, sigmoid output) fitted to one degraded image from a
 * fixed noise input (perturbed slightly every step, the paper's input-noise regularisation),
 * minimising {@code |H G(z) - y|^2} for the forward operator {@code H} ({@link Task#DENOISING},
 * {@link Task#INPAINTING}, {@link Task#SUPER_RESOLUTION}) with Adam at 0.005 (at 0.01 the sigmoid
 * output saturates and fitting stalls). The network reproduces image structure
 * before noise, so the result depends on when fitting stops:
 * <ul>
 *   <li>{@link Stop#FIXED_ITERATIONS}: the output after the full budget (overfits the noise);</li>
 *   <li>{@link Stop#ORACLE_STOP}: the output with the best PSNR against the clean image
 *       (needs the ground truth; the ceiling);</li>
 *   <li>{@link Stop#ES_WMV} (Wang et al., ICCV 2021): the running variance of the last
 *       {@code W} outputs; fitting stops once it has not reached a new minimum for
 *       {@code patience} steps, and the output at the minimum is returned (no ground truth).</li>
 * </ul>
 */
public final class DeepImagePrior {

    /** The forward operators. */
    public enum Task {
        /** {@code H = I}. */
        DENOISING,
        /** {@code H} keeps the observed pixels. */
        INPAINTING,
        /** {@code H} averages 2x2 blocks. */
        SUPER_RESOLUTION
    }

    /** The stopping rules. */
    public enum Stop {
        /** The full budget. */
        FIXED_ITERATIONS,
        /** The best PSNR against the truth. */
        ORACLE_STOP,
        /** Early stopping by windowed moving variance. */
        ES_WMV
    }

    /**
     * One fit's outputs under the three stopping rules.
     *
     * @param fixed the output after the budget
     * @param oracle the output at the oracle stop
     * @param oracleIteration the oracle stop's iteration
     * @param esWmv the output at ES-WMV's stop
     * @param esWmvIteration ES-WMV's iteration (the variance minimum)
     */
    public record Result(double[] fixed, double[] oracle, int oracleIteration, double[] esWmv, int esWmvIteration) {

        /**
         * The output of one rule.
         *
         * @param stop the rule
         * @return the image
         */
        public double[] get(Stop stop) {
            return switch (stop) {
                case FIXED_ITERATIONS -> fixed;
                case ORACLE_STOP -> oracle;
                case ES_WMV -> esWmv;
            };
        }
    }

    private DeepImagePrior() {
    }

    private static Var apply(Task task, Var x, Tensor mask) {
        return switch (task) {
            case DENOISING -> x;
            case INPAINTING -> x.mul(mask);
            case SUPER_RESOLUTION -> Functional.pool(x, Functional.PoolKind.AVG, new int[] {2, 2}, new int[] {2, 2},
                    new int[] {0, 0}, new int[] {0, 0}, new int[] {1, 1}, false, false, 1f);
        };
    }

    /**
     * Fits the prior to one observation.
     *
     * @param task the forward operator
     * @param observation {@code y}: the noisy image, the masked image or the low-resolution image
     * @param mask the observed pixels (inpainting), else {@code null}
     * @param side the output side
     * @param truth the clean image, for the oracle stop and nothing else
     * @param iterations the budget
     * @param base the U-Net's base channels
     * @param window ES-WMV's window
     * @param patience ES-WMV's patience
     * @param seed the seed
     * @return the outputs
     */
    public static Result fit(Task task, double[] observation, boolean[] mask, int side, double[] truth, int iterations,
            int base, int window, int patience, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int inChannels = 8;
        UNet2d net = new UNet2d(inChannels, 1, base, 2, true, 0, r);
        Tensor z = Tensor.uniform(Shape.of(1, inChannels, side, side), 0, 0.1f, r);
        int obsSide = task == Task.SUPER_RESOLUTION ? side / 2 : side;
        Tensor y = Tensor.of(Shape.of(1, 1, obsSide, obsSide), ImageMath.toFloat(observation));
        Tensor m = null;
        if (mask != null) {
            float[] mv = new float[mask.length];
            for (int i = 0; i < mv.length; i++) {
                mv[i] = mask[i] ? 1 : 0;
            }
            m = Tensor.of(Shape.of(1, 1, side, side), mv);
        }
        Optimizer opt = Optimizer.adam(0.005f);
        double best = Double.NEGATIVE_INFINITY;
        double[] oracle = null;
        int oracleIt = 0;
        Deque<double[]> recent = new ArrayDeque<>();
        double bestVar = Double.MAX_VALUE;
        double[] esOut = null;
        int esIt = -1;
        int sinceBest = 0;
        boolean esDone = false;
        double[] last = null;
        for (int it = 0; it < iterations; it++) {
            Tensor input = z.add(Tensor.normal(z.shape(), 0, 1f / 30, r));
            Var out = net.forward(Var.of(input)).sigmoid();
            Var loss = apply(task, out, m).sub(Var.of(y)).square().mean();
            loss.backward();
            opt.step(net.parameters());
            double[] img = ImageMath.toDouble(out.value().toArray());
            last = img;
            if (truth != null) {
                double p = ImageMath.psnr(img, truth);
                if (p > best) {
                    best = p;
                    oracle = img;
                    oracleIt = it;
                }
            }
            if (!esDone) {
                recent.addLast(img);
                if (recent.size() > window) {
                    recent.removeFirst();
                }
                if (recent.size() == window) {
                    double v = windowVariance(recent);
                    if (v < bestVar) {
                        bestVar = v;
                        esOut = img;
                        esIt = it;
                        sinceBest = 0;
                    } else if (++sinceBest >= patience) {
                        esDone = true;
                    }
                }
            }
        }
        if (esOut == null) {
            esOut = last;
            esIt = iterations - 1;
        }
        return new Result(last, oracle == null ? last : oracle, oracleIt, esOut, esIt);
    }

    private static double windowVariance(Deque<double[]> window) {
        int n = window.peekFirst().length;
        double[] mean = new double[n];
        for (double[] x : window) {
            for (int i = 0; i < n; i++) {
                mean[i] += x[i] / window.size();
            }
        }
        double v = 0;
        for (double[] x : window) {
            for (int i = 0; i < n; i++) {
                double d = x[i] - mean[i];
                v += d * d;
            }
        }
        return v / (n * window.size());
    }
}
