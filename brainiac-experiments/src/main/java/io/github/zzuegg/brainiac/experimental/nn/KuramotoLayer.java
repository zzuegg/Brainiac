package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Artificial Kuramoto oscillatory neurons, AKOrN (Miyato et al., "Artificial Kuramoto
 * oscillatory neurons", <a href="https://arxiv.org/abs/2410.13821">arXiv:2410.13821</a>, ICLR
 * 2025). Every token carries {@code oscillators} unit vectors of dimension {@code n} instead of
 * scalar activations; they evolve for a number of steps by a generalized Kuramoto model,
 *
 * <pre>
 * x_i &lt;- normalize( x_i + gamma ( Omega_i x_i + P_(x_i)( c_i + sum_j J_ij x_j ) ) ),
 * P_x(y) = y - (y . x) x
 * </pre>
 *
 * <p>where {@code Omega_i} is a learned antisymmetric matrix (the natural frequency, a
 * rotation), {@code c_i} is the conditional stimulus computed from the input token (a linear
 * map), and {@code J} is a learned coupling, here factorized as a token-mixing matrix (between
 * the {@code tokens} positions) times a channel-mixing matrix. The projection keeps each update
 * tangent to the sphere, so the update follows the gradient of the energy
 * {@code E = -sum_ij x_i^T J_ij x_j - sum_i c_i^T x_i}: oscillators that are coupled
 * synchronize, and groups that are not drift apart, which is how AKOrN binds features into
 * objects. Training backpropagates through the unrolled steps.
 *
 * <p>Input {@code [batch, tokens, features]}, output the final oscillator states
 * {@code [batch, tokens, oscillators * n]}. The initial states are random unit vectors drawn
 * from the layer's own seeded generator. {@link #synchronize} exposes the dynamics with an
 * external, per-sample coupling (for example a fixed local "convolutional" coupling over image
 * pixels), the form used for object discovery.
 */
public final class KuramotoLayer implements Layer {
    private final int features;
    private final int tokens;
    private final int oscillators;
    private final int n;
    private final int steps;
    private final float gamma;
    private final Var stimulus;
    private final Var stimulusBias;
    private final Var tokenCoupling;
    private final Var channelCoupling;
    private final Var omega;
    private final SplittableRandom init;

    private KuramotoLayer(int features, int tokens, int oscillators, int n, int steps, float gamma,
            RandomGenerator random) {
        if (n < 2 || oscillators < 1 || steps < 1) {
            throw new IllegalArgumentException("need n >= 2, oscillators >= 1, steps >= 1");
        }
        this.features = features;
        this.tokens = tokens;
        this.oscillators = oscillators;
        this.n = n;
        this.steps = steps;
        this.gamma = gamma;
        int d = oscillators * n;
        this.stimulus = Var.param(Init.XAVIER.create(Shape.of(features, d), features, d, random));
        this.stimulusBias = Var.param(Tensor.zeros(Shape.of(d)));
        this.tokenCoupling = Var.param(Init.XAVIER.create(Shape.of(tokens, tokens), tokens, tokens, random));
        this.channelCoupling = Var.param(Init.XAVIER.create(Shape.of(d, d), d, d, random));
        this.omega = Var.param(Init.XAVIER.create(Shape.of(oscillators, n, n), n, n, random).mul(0.1f));
        this.init = new SplittableRandom(random.nextLong());
    }

    /**
     * Creates a layer.
     *
     * @param features input features per token
     * @param tokens tokens (fixed, since the token coupling is a learned matrix)
     * @param oscillators oscillators per token
     * @param n the dimension of each oscillator (2 is the classic phase oscillator)
     * @param steps Kuramoto steps
     * @param gamma the step size
     * @param seed the seed
     * @return the layer
     */
    public static KuramotoLayer of(int features, int tokens, int oscillators, int n, int steps, float gamma,
            long seed) {
        return new KuramotoLayer(features, tokens, oscillators, n, steps, gamma, new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders; input {@code [tokens, features]}.
     *
     * @param oscillators oscillators per token
     * @param n oscillator dimension
     * @param steps Kuramoto steps
     * @param gamma the step size
     * @return the spec
     */
    public static LayerSpec spec(int oscillators, int n, int steps, float gamma) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), oscillators * n);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new KuramotoLayer(input.dim(1), input.dim(0), oscillators, n, steps, gamma, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        List<Var> trace = trajectory(input);
        return trace.get(trace.size() - 1);
    }

    /**
     * Runs the dynamics and keeps every step.
     *
     * @param input {@code [batch, tokens, features]}
     * @return the oscillator states after each step, {@code [batch, tokens, oscillators * n]}
     */
    public List<Var> trajectory(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(1) != tokens || s.dim(2) != features) {
            throw new IllegalArgumentException("KuramotoLayer expects [batch, " + tokens + ", " + features + "], got "
                    + s);
        }
        int b = s.dim(0);
        Var c = input.matmul(stimulus).add(stimulusBias);
        Var x = Var.of(randomStates(b, tokens, oscillators, n, init));
        Var antisym = omega.sub(omega.permute(0, 2, 1));
        List<Var> out = new ArrayList<>();
        for (int k = 0; k < steps; k++) {
            Var coupled = tokenCoupling.matmul(x).matmul(channelCoupling);
            x = step(x, c.add(coupled), antisym, gamma, oscillators, n);
            out.add(x);
        }
        return out;
    }

    /**
     * One Kuramoto update for every token.
     *
     * @param x states {@code [batch, tokens, oscillators * n]}
     * @param drive stimulus plus coupling, same shape
     * @param antisymmetric natural frequencies {@code [oscillators, n, n]} (antisymmetric), or {@code null}
     * @param gamma the step size
     * @param oscillators oscillators per token
     * @param n oscillator dimension
     * @return the new states
     */
    public static Var step(Var x, Var drive, Var antisymmetric, float gamma, int oscillators, int n) {
        Shape s = x.shape();
        int b = s.dim(0);
        int t = s.dim(1);
        Var xs = x.reshape(b * t, oscillators, n);
        Var ys = drive.reshape(b * t, oscillators, n);
        Var tangent = ys.sub(xs.mul(ys.mul(xs).sum(2, true)));
        Var update = tangent;
        if (antisymmetric != null) {
            Var rotation = xs.reshape(b * t, oscillators, 1, n).matmul(antisymmetric.permute(0, 2, 1))
                    .reshape(b * t, oscillators, n);
            update = update.add(rotation);
        }
        Var next = xs.add(update.mul(gamma));
        next = next.div(next.square().sum(2, true).add(1e-8f).sqrt());
        return next.reshape(b, t, oscillators * n);
    }

    /**
     * The dynamics with an external coupling per sample, for example a local image coupling.
     *
     * @param x0 initial states {@code [batch, tokens, oscillators * n]} (unit oscillators)
     * @param coupling {@code [batch, tokens, tokens]}
     * @param stimulus the stimulus {@code [batch, tokens, oscillators * n]}, or {@code null}
     * @param steps steps
     * @param gamma the step size
     * @param oscillators oscillators per token
     * @param n oscillator dimension
     * @return the final states
     */
    public static Var synchronize(Var x0, Tensor coupling, Var stimulus, int steps, float gamma, int oscillators,
            int n) {
        Var x = x0;
        Var j = Var.of(coupling);
        for (int k = 0; k < steps; k++) {
            Var drive = j.matmul(x);
            if (stimulus != null) {
                drive = drive.add(stimulus);
            }
            x = step(x, drive, null, gamma, oscillators, n);
        }
        return x;
    }

    /**
     * Random unit oscillators.
     *
     * @param b batch
     * @param t tokens
     * @param oscillators oscillators per token
     * @param n oscillator dimension
     * @param random the random source
     * @return {@code [b, t, oscillators * n]}
     */
    public static Tensor randomStates(int b, int t, int oscillators, int n, RandomGenerator random) {
        float[] v = new float[b * t * oscillators * n];
        for (int o = 0; o < b * t * oscillators; o++) {
            double norm = 0;
            for (int k = 0; k < n; k++) {
                double g = random.nextGaussian();
                v[o * n + k] = (float) g;
                norm += g * g;
            }
            norm = Math.sqrt(norm);
            for (int k = 0; k < n; k++) {
                v[o * n + k] /= (float) norm;
            }
        }
        return Tensor.of(Shape.of(b, t, oscillators * n), v);
    }

    /**
     * The Kuramoto energy {@code -sum x^T J x - sum c^T x} of a state for an external coupling,
     * per sample; it falls as the oscillators synchronize.
     *
     * @param x states {@code [batch, tokens, d]}
     * @param coupling {@code [batch, tokens, tokens]}
     * @return {@code [batch]}
     */
    public static Tensor energy(Tensor x, Tensor coupling) {
        Var xv = Var.of(x);
        Var jx = Var.of(coupling).matmul(xv);
        return xv.mul(jx).sum(2, false).sum(1, false).neg().value();
    }

    @Override
    public List<Var> parameters() {
        return List.of(stimulus, stimulusBias, tokenCoupling, channelCoupling, omega);
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        return Shape.of(inputShape.dim(0), inputShape.dim(1), oscillators * n);
    }
}
