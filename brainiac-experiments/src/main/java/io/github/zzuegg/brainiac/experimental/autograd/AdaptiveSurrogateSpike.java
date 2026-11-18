package io.github.zzuegg.brainiac.experimental.autograd;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A spike nonlinearity whose surrogate gradient has an adjustable width and asymmetry, after
 * the adaptive asymmetric surrogate gradients of A²SG (ICML 2026). The forward pass is the
 * Heaviside step {@code [v > 0]} like {@link SurrogateSpike} (whose default is unchanged); the
 * backward pass is the derivative of the clipped ramp {@code f(v) = v / (2β) + h}, that is
 * {@code 1 / (2β)} on the window {@code [-β(1 - a), β(1 + a)]} and 0 outside, so its integral
 * is always 1. The asymmetry {@code a ∈ (-1, 1)} moves the window above or below the
 * threshold (the offset {@code h} only moves {@code f}, not its derivative).
 *
 * <p>The width {@code β} can be:
 * <ul>
 *   <li>{@link Mode#FIXED};</li>
 *   <li>{@link Mode#SCHEDULED}: geometric from wide ({@code βmax}, informative gradients for
 *       many neurons early) to sharp ({@code βmin}, closer to the true step late) over
 *       training progress;</li>
 *   <li>{@link Mode#GRID}: A²SG's selection, every backward pass scoring each width of a small
 *       grid by the spatial variation of the resulting gradients {@code δ = g · f'(v)},
 *       {@code Var(δ) / Mean|δ|}, and using the lowest-scoring width from the next step on;</li>
 *   <li>{@link Mode#ADAPTIVE}: a local variant, {@code κ} times the spread (standard
 *       deviation) of the membrane potentials seen since the last {@link #adapt}, times the
 *       schedule factor, so the window keeps a stable share of neurons inside it.</li>
 * </ul>
 */
public final class AdaptiveSurrogateSpike implements Op {

    /** How the width changes. */
    public enum Mode {
        /** constant width */
        FIXED,
        /** wide to sharp over training */
        SCHEDULED,
        /** proportional to the membrane-potential spread, times the schedule */
        ADAPTIVE,
        /** chosen from a grid by the gradient-variation score */
        GRID
    }

    private final Mode mode;
    private final float widthMax;
    private final float widthMin;
    private final float kappa;
    private final float asymmetry;
    private float width;
    private double count;
    private double sum;
    private double sumSquares;
    private final double[] gridScore = new double[GRID_SIZE];
    private static final int GRID_SIZE = 6;

    private AdaptiveSurrogateSpike(Mode mode, float widthMax, float widthMin, float kappa, float asymmetry) {
        if (!(widthMin > 0 && widthMax >= widthMin) || !(asymmetry > -1 && asymmetry < 1)) {
            throw new IllegalArgumentException("need 0 < widthMin <= widthMax and -1 < asymmetry < 1");
        }
        this.mode = mode;
        this.widthMax = widthMax;
        this.widthMin = widthMin;
        this.kappa = kappa;
        this.asymmetry = asymmetry;
        this.width = mode == Mode.FIXED ? widthMin : widthMax;
    }

    /**
     * A constant rectangular surrogate.
     *
     * @param width the half-width {@code β}
     * @param asymmetry {@code a}
     * @return the operation
     */
    public static AdaptiveSurrogateSpike fixed(float width, float asymmetry) {
        return new AdaptiveSurrogateSpike(Mode.FIXED, width, width, 1, asymmetry);
    }

    /**
     * A surrogate whose width shrinks over training.
     *
     * @param widthMax the initial half-width
     * @param widthMin the final half-width
     * @param asymmetry {@code a}
     * @return the operation
     */
    public static AdaptiveSurrogateSpike scheduled(float widthMax, float widthMin, float asymmetry) {
        return new AdaptiveSurrogateSpike(Mode.SCHEDULED, widthMax, widthMin, 1, asymmetry);
    }

    /**
     * A surrogate whose width follows the membrane-potential spread.
     *
     * @param widthMax the largest half-width
     * @param widthMin the smallest half-width
     * @param kappa the width per standard deviation of the potentials
     * @param asymmetry {@code a}
     * @return the operation
     */
    public static AdaptiveSurrogateSpike adaptive(float widthMax, float widthMin, float kappa, float asymmetry) {
        return new AdaptiveSurrogateSpike(Mode.ADAPTIVE, widthMax, widthMin, kappa, asymmetry);
    }

    /**
     * A surrogate whose width is picked from a geometric grid by A²SG's variation score.
     *
     * @param widthMax the largest half-width
     * @param widthMin the smallest half-width
     * @param asymmetry {@code a}
     * @return the operation
     */
    public static AdaptiveSurrogateSpike grid(float widthMax, float widthMin, float asymmetry) {
        return new AdaptiveSurrogateSpike(Mode.GRID, widthMax, widthMin, 1, asymmetry);
    }

    private float gridWidth(int k) {
        return (float) (widthMax * Math.pow(widthMin / (double) widthMax, k / (double) (GRID_SIZE - 1)));
    }

    /**
     * Spikes where {@code v > 0}.
     *
     * @param v the membrane potential minus the threshold
     * @return 0/1 spikes with this surrogate's gradient
     */
    public Var spike(Var v) {
        return Var.apply(this, v);
    }

    /**
     * Updates the width after a training step.
     *
     * @param progress the share of training done, in {@code [0, 1]}
     */
    public void adapt(double progress) {
        double p = Math.max(0, Math.min(1, progress));
        double schedule = Math.pow(widthMin / (double) widthMax, p);
        switch (mode) {
            case FIXED -> { }
            case SCHEDULED -> width = (float) (widthMax * schedule);
            case GRID -> {
                int best = 0;
                for (int k = 1; k < GRID_SIZE; k++) {
                    best = gridScore[k] < gridScore[best] ? k : best;
                }
                if (gridScore[best] > 0) {
                    width = gridWidth(best);
                }
                java.util.Arrays.fill(gridScore, 0);
            }
            case ADAPTIVE -> {
                if (count > 0) {
                    double mean = sum / count;
                    double std = Math.sqrt(Math.max(0, sumSquares / count - mean * mean));
                    double w = kappa * std * schedule;
                    width = (float) Math.max(widthMin, Math.min(widthMax, w));
                }
            }
        }
        count = 0;
        sum = 0;
        sumSquares = 0;
    }

    /** @return the current half-width {@code β} */
    public float width() {
        return width;
    }

    @Override
    public Tensor forward(Tensor... in) {
        if (mode == Mode.ADAPTIVE) {
            float[] v = in[0].toArray();
            for (float x : v) {
                sum += x;
                sumSquares += x * x;
            }
            count += v.length;
        }
        return in[0].step();
    }

    @Override
    public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
        float lo = -width * (1 - asymmetry);
        float hi = width * (1 + asymmetry);
        float[] v = in[0].toArray();
        float[] g = grad.toArray();
        float[] d = new float[v.length];
        float scale = 1 / (2 * width);
        for (int i = 0; i < v.length; i++) {
            d[i] = v[i] >= lo && v[i] <= hi ? g[i] * scale : 0;
        }
        if (mode == Mode.GRID) {
            for (int k = 0; k < GRID_SIZE; k++) {
                float w = gridWidth(k);
                float l = -w * (1 - asymmetry);
                float h = w * (1 + asymmetry);
                double s1 = 0;
                double s2 = 0;
                double abs = 0;
                for (int i = 0; i < v.length; i++) {
                    double delta = v[i] >= l && v[i] <= h ? g[i] / (2 * w) : 0;
                    s1 += delta;
                    s2 += delta * delta;
                    abs += Math.abs(delta);
                }
                double mean = s1 / v.length;
                double var = s2 / v.length - mean * mean;
                gridScore[k] += abs > 0 ? var / (abs / v.length) : Double.POSITIVE_INFINITY;
            }
        }
        return new Tensor[] {Tensor.of(in[0].shape(), d)};
    }

    @Override
    public String name() {
        return "adaptive surrogate spike";
    }
}
