package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A deep Koopman autoencoder (Lusch, Kutz and Brunton, 2018): an encoder learns observables
 * {@code z = phi(x)} in which the dynamics are linear, {@code z_{t+1} = K z_t}, and a decoder
 * maps back. Training combines reconstruction {@code |x - dec(enc x)|^2}, linear prediction in
 * latent space {@code |enc(x_{t+s}) - K^s enc(x_t)|^2} and multi-step prediction in state
 * space {@code |x_{t+s} - dec(K^s enc x_t)|^2}. Forecasts iterate {@code K} in the latent
 * space; its eigenvalues are the learned Koopman spectrum.
 *
 * <p>As a {@link Layer}, {@code forward} advances states {@code [b, n]} by one step.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1712.09707">arXiv:1712.09707</a>.
 */
public final class KoopmanAutoencoder implements Layer {

    private final int latent;
    private final SimpleMlp encoder;
    private final SimpleMlp decoder;
    private final Var koopman;

    private KoopmanAutoencoder(int state, int latent, int width, long seed) {
        this.latent = latent;
        this.encoder = SimpleMlp.tanh(seed, state, width, width, latent);
        this.decoder = SimpleMlp.tanh(seed + 1, latent, width, width, state);
        SplittableRandom r = new SplittableRandom(seed + 2);
        float[] k = new float[latent * latent];
        for (int i = 0; i < latent; i++) {
            for (int j = 0; j < latent; j++) {
                k[i * latent + j] = (i == j ? 1f : 0f) + 0.01f * (float) r.nextGaussian();
            }
        }
        this.koopman = Var.param(Tensor.of(Shape.of(latent, latent), k));
    }

    /**
     * Creates the autoencoder.
     *
     * @param state the state size
     * @param latent the observables {@code k}
     * @param width the hidden width of encoder and decoder
     * @param seed the seed
     * @return the model
     */
    public static KoopmanAutoencoder of(int state, int latent, int width, long seed) {
        return new KoopmanAutoencoder(state, latent, width, seed);
    }

    /**
     * @param x states {@code [b, n]}
     * @return observables {@code [b, k]}
     */
    public Var encode(Var x) {
        return encoder.forward(x);
    }

    /**
     * @param z observables {@code [b, k]}
     * @return states {@code [b, n]}
     */
    public Var decode(Var z) {
        return decoder.forward(z);
    }

    /**
     * One linear step in latent space ({@code z K}, row vectors).
     *
     * @param z observables
     * @return {@code K z}
     */
    public Var advance(Var z) {
        return z.matmul(koopman);
    }

    @Override
    public Var forward(Var x) {
        return decode(advance(encode(x)));
    }

    /**
     * The training loss on windows of consecutive states.
     *
     * @param windows {@code [b, steps + 1, n]}
     * @param latentWeight weight of the latent linearity term
     * @return reconstruction + multi-step prediction + weighted latent linearity
     */
    public Var loss(Tensor windows, float latentWeight) {
        int b = windows.shape().dim(0);
        int steps = windows.shape().dim(1) - 1;
        int n = windows.shape().dim(2);
        Var x0 = Var.of(windows.slice(1, 0, 1).reshape(b, n));
        Var z = encode(x0);
        Var loss = decode(z).sub(x0).square().mean();
        for (int s = 1; s <= steps; s++) {
            Tensor xs = windows.slice(1, s, s + 1).reshape(b, n);
            z = advance(z);
            loss = loss.add(decode(z).sub(xs).square().mean().div(steps));
            loss = loss.add(z.sub(encode(Var.of(xs))).square().mean().mul(latentWeight / steps));
        }
        return loss;
    }

    /**
     * Forecasts by iterating the latent linear map.
     *
     * @param x0 the initial state
     * @param steps the steps
     * @return states {@code [steps + 1][n]}
     */
    public double[][] forecast(double[] x0, int steps) {
        return Var.noGrad(() -> {
            double[][] out = new double[steps + 1][];
            out[0] = x0.clone();
            Var z = encode(Var.of(HamiltonianNetwork.tensor(new double[][] {x0})));
            for (int t = 1; t <= steps; t++) {
                z = advance(z);
                float[] x = decode(z).value().toArray();
                out[t] = new double[x.length];
                for (int i = 0; i < x.length; i++) {
                    out[t][i] = x[i];
                }
            }
            return out;
        });
    }

    /** @return the learned Koopman matrix's eigenvalues {@code [k][2]} */
    public double[][] eigenvalues() {
        float[] k = koopman.value().toArray();
        double[][] m = new double[latent][latent];
        for (int i = 0; i < latent; i++) {
            for (int j = 0; j < latent; j++) {
                m[i][j] = k[j * latent + i];
            }
        }
        return DynamicModeDecomposition.eigenvalues(m);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(encoder.parameters());
        p.addAll(decoder.parameters());
        p.add(koopman);
        return p;
    }

    @Override
    public String toString() {
        return "KoopmanAutoencoder(latent=" + latent + ")";
    }
}
