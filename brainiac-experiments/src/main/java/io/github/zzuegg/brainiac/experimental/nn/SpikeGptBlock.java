package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.autograd.SurrogateSpike;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A SpikeGPT block (Zhu et al., <a href="https://arxiv.org/abs/2302.13939">2302.13939</a>):
 * an RWKV block whose channel mix is made of leaky integrate-and-fire neurons, so the
 * feed-forward half of the network talks in binary spikes. The time mix is
 * {@link Rwkv7TimeMix}; the channel mix projects up, feeds the current into LIF neurons whose
 * membranes integrate <em>across tokens</em> (the token axis is the neurons' time axis, as in
 * SpikeGPT), and projects the spike trains back down:
 *
 * <pre>
 * x   = x + TimeMix(LN(x))
 * u_t = beta u_(t-1) + W_up LN(x)_t - theta s_(t-1),   s_t = [u_t &gt; theta]
 * x   = x + W_down s
 * </pre>
 *
 * <p>The step function is trained with a {@link SurrogateSpike} gradient. With
 * {@code spiking = false} the channel mix is RWKV's dense squared-ReLU instead, the
 * non-spiking baseline. Input and output are {@code [batch, steps, features]};
 * {@link #spikeRate()} reports the share of neurons that fired in the last forward pass, the
 * quantity that sets the energy cost on neuromorphic hardware.
 */
public final class SpikeGptBlock implements Layer {
    private final int features;
    private final int hidden;
    private final boolean spiking;
    private final float beta;
    private final float threshold;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final Rwkv7TimeMix timeMix;
    private final Var up;
    private final Var down;
    private double lastSpikeRate = Double.NaN;

    /**
     * Creates a block.
     *
     * @param features the width
     * @param heads the time-mix heads
     * @param rank the time-mix low-rank size
     * @param hidden the channel-mix neurons
     * @param spiking LIF neurons if true, RWKV's squared ReLU if false
     * @param random the random source
     */
    public SpikeGptBlock(int features, int heads, int rank, int hidden, boolean spiking, RandomGenerator random) {
        this.features = features;
        this.hidden = hidden;
        this.spiking = spiking;
        this.beta = 0.5f;
        this.threshold = 0.5f;
        this.norm1 = LayerNorm.of(features);
        this.norm2 = LayerNorm.of(features);
        this.timeMix = new Rwkv7TimeMix(features, heads, rank, random);
        this.up = MixerOps.weight(random, features, hidden);
        this.down = Var.param(Tensor.normal(Shape.of(hidden, features), 0, (float) (0.5 / Math.sqrt(hidden)), random));
    }

    /**
     * Describes the block for {@code Model.sequential().add(...)}.
     *
     * @param heads the time-mix heads
     * @param rank the time-mix low-rank size
     * @param hidden the channel-mix neurons
     * @param spiking whether the channel mix spikes
     * @return the spec
     */
    public static LayerSpec of(int heads, int rank, int hidden, boolean spiking) {
        return MixerOps.sameShape((in, r) -> new SpikeGptBlock(in.dim(-1), heads, rank, hidden, spiking, r));
    }

    private Var lif(Var current) {
        int b = current.shape().dim(0);
        int steps = current.shape().dim(1);
        Var u = Var.of(Tensor.zeros(Shape.of(b, hidden)).to(current.value().device()));
        Var spikes = null;
        List<Var> out = new ArrayList<>();
        double fired = 0;
        for (int t = 0; t < steps; t++) {
            u = u.mul(beta).add(current.slice(1, t, t + 1).reshape(b, hidden));
            if (spikes != null) {
                u = u.sub(spikes.mul(threshold));   // soft reset
            }
            spikes = SurrogateSpike.spike(u.sub(threshold), 5f);
            fired += spikes.value().sum().item();
            out.add(spikes.reshape(b, 1, hidden));
        }
        lastSpikeRate = fired / ((double) b * steps * hidden);
        return Var.concat(1, out.toArray(Var[]::new));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "SpikeGptBlock");
        Var x = input.add(timeMix.forward(norm1.forward(input)));
        Var current = norm2.forward(x).matmul(up);
        Var activity = spiking ? lif(current) : current.relu().square();
        return x.add(activity.matmul(down));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(norm1.parameters());
        p.addAll(norm2.parameters());
        p.addAll(timeMix.parameters());
        p.add(up);
        p.add(down);
        return p;
    }

    /** @return the share of channel-mix neurons that spiked in the last forward pass; NaN when not spiking */
    public double spikeRate() {
        return lastSpikeRate;
    }

    @Override
    public String toString() {
        return "SpikeGptBlock(" + features + ", hidden=" + hidden + (spiking ? ", spiking)" : ")");
    }
}
