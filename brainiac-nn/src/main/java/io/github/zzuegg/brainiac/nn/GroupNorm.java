package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Normalizes groups of channels of each sample over their channels and positions, then
 * scales and shifts per channel. One group is like layer normalization over a whole
 * image; one group per channel is instance normalization ({@link #instance(int)}). Works
 * on {@code [channels, spatial...]} and on plain {@code [features]}.
 */
public final class GroupNorm implements Layer {

    private final int groups;
    private final int channels;
    private final float epsilon;
    private final Var gamma;
    private final Var beta;

    /**
     * Creates a group normalization.
     *
     * @param groups the number of groups; must divide {@code channels}
     * @param channels the number of channels
     * @param epsilon added to the variance for stability
     * @param affine whether to learn a scale and shift per channel
     */
    public GroupNorm(int groups, int channels, float epsilon, boolean affine) {
        if (groups < 1 || channels < 1 || channels % groups != 0) {
            throw new IllegalArgumentException(groups + " groups must divide " + channels + " channels");
        }
        if (!(epsilon > 0)) {
            throw new IllegalArgumentException("epsilon must be positive, got " + epsilon);
        }
        this.groups = groups;
        this.channels = channels;
        this.epsilon = epsilon;
        this.gamma = affine ? Var.param(Tensor.ones(Shape.of(channels))) : null;
        this.beta = affine ? Var.param(Tensor.zeros(Shape.of(channels))) : null;
    }

    /**
     * Instance normalization: every channel of every sample on its own.
     *
     * @param channels the number of channels
     * @return the layer
     */
    public static GroupNorm instance(int channels) {
        return new GroupNorm(channels, channels, 1e-5f, true);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() < 2 || s.dim(1) != channels) {
            throw new IllegalArgumentException(this + " expects [batch, " + channels + ", ...], got " + s);
        }
        int n = s.dim(0);
        int perGroup = Math.toIntExact(s.size() / ((long) n * groups));
        Var x = input.reshape(n * groups, perGroup);
        Var centered = x.sub(x.mean(1, true));
        Var y = centered.div(centered.square().mean(1, true).add(epsilon).sqrt()).reshape(s.toArray());
        if (gamma != null) {
            int[] cs = Functional.channelShape(channels, s.rank() - 2);
            y = y.mul(gamma.reshape(cs)).add(beta.reshape(cs));
        }
        return y;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (gamma != null) {
            p.add(gamma);
            p.add(beta);
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return groups == channels ? "InstanceNorm(" + channels + ")" : "GroupNorm(" + groups + ", " + channels + ")";
    }

    static Config write(GroupNorm g) {
        return Config.builder().put("groups", g.groups).put("channels", g.channels).put("epsilon", g.epsilon)
                .put("affine", g.gamma != null).build();
    }

    static GroupNorm read(Config c) {
        return new GroupNorm(c.getInt("groups"), c.getInt("channels"), c.getFloat("epsilon"), c.getBoolean("affine"));
    }
}
