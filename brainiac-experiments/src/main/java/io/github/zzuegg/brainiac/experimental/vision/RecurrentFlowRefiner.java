package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GruCell;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * RAFT-lite: optical flow by recurrent refinement over a {@link CorrelationVolume} (Teed and
 * Deng, 2020). A shared CNN encodes both frames at half resolution (feature vectors scaled to
 * a common length, so a perfect match correlates at 4 from the start); a context CNN encodes the
 * first frame into the initial hidden state and context features. Starting from zero flow,
 * every iteration looks up the correlations around each pixel's current target, encodes them
 * with the current flow (a 1x1 convolution), updates a per-pixel GRU ({@link GruCell} applied
 * at every pixel, a 1x1 convolutional GRU) and adds the flow increment predicted from the
 * hidden state. The weights are tied across iterations, so more iterations at test time cost
 * nothing to train. The loss supervises every iteration, weighted {@code gamma^(N-i)}, at full
 * resolution (bilinear upsampling of the doubled flow).
 *
 * <p>Paper: Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
 * (2020), <a href="https://arxiv.org/abs/2003.12039">arXiv:2003.12039</a>.
 */
public final class RecurrentFlowRefiner {

    private final int features;
    private final int hidden;
    private final int context;
    private final int levels;
    private final int radius;
    private final Var[] encoder;
    private final Var[] contextEncoder;
    private final Var motion;
    private final Var motionBias;
    private final GruCell gru;
    private final Var head1;
    private final Var head1Bias;
    private final Var head2;
    private final Var head2Bias;

    /**
     * Creates the model.
     *
     * @param features the matching feature channels
     * @param hidden the GRU state channels
     * @param levels the correlation pyramid levels
     * @param radius the lookup radius
     * @param random the source of initial weights
     */
    public RecurrentFlowRefiner(int features, int hidden, int levels, int radius, RandomGenerator random) {
        this.features = features;
        this.hidden = hidden;
        this.context = 8;
        this.levels = levels;
        this.radius = radius;
        this.encoder = encoderWeights(random, features);
        this.contextEncoder = encoderWeights(random, hidden + context);
        int k = 2 * radius + 1;
        int motionIn = levels * k * k + 2;
        this.motion = VisionOps.convWeight(random, 24, motionIn, 1);
        this.motionBias = VisionOps.zeros(24);
        this.gru = new GruCell(24 + context, hidden, random);
        this.head1 = VisionOps.convWeight(random, 16, hidden, 3);
        this.head1Bias = VisionOps.zeros(16);
        this.head2 = Var.param(Tensor.normal(Shape.of(2, 16, 3, 3), 0, 0.01f, random));
        this.head2Bias = VisionOps.zeros(2);
    }

    private static Var[] encoderWeights(RandomGenerator r, int out) {
        return new Var[] {VisionOps.convWeight(r, 16, 1, 3), VisionOps.zeros(16), VisionOps.convWeight(r, out, 16, 3),
            VisionOps.zeros(out), VisionOps.convWeight(r, out, out, 3), VisionOps.zeros(out)};
    }

    /** Scales feature vectors to a common length, so correlations are scaled cosines (4 at a perfect match). */
    private Var unit(Var f) {
        float length = (float) (2 * Math.pow(features, 0.25));
        return f.div(f.square().sum(1, true).add(1e-6f).sqrt()).mul(length);
    }

    private static Var encode(Var[] w, Var x) {
        Var h = VisionOps.conv(x, w[0], w[1]).relu();
        h = VisionOps.conv(h, w[2], w[3], 2, 1).relu();
        return VisionOps.conv(h, w[4], w[5]);
    }

    /**
     * Estimates flow.
     *
     * @param first the first frames, {@code [batch, 1, h, w]} with even sides
     * @param second the second frames
     * @param iterations the refinement iterations
     * @return the full-resolution flow {@code [batch, 2, h, w]} after each iteration
     */
    public List<Var> predict(Var first, Var second, int iterations) {
        int b = first.shape().dim(0);
        int h = first.shape().dim(2);
        int w = first.shape().dim(3);
        Var f1 = unit(encode(encoder, first));
        Var f2 = unit(encode(encoder, second));
        int lh = f1.shape().dim(2);
        int lw = f1.shape().dim(3);
        int hw = lh * lw;
        CorrelationVolume corr = new CorrelationVolume(f1, f2, levels);
        Var ctx = encode(contextEncoder, first);
        Var state = ctx.slice(1, 0, hidden).tanh().permute(0, 2, 3, 1).reshape(b * hw, hidden);
        Var ctxFeat = ctx.slice(1, hidden, hidden + context).relu();
        Var flow = Var.of(Tensor.zeros(Shape.of(b, 2, lh, lw)));
        List<Var> out = new ArrayList<>();
        for (int it = 0; it < iterations; it++) {
            float[] f = flow.value().toArray();
            float[] targets = new float[f.length];
            for (int i = 0; i < b; i++) {
                for (int p = 0; p < hw; p++) {
                    targets[(2 * i) * hw + p] = p % lw + f[(2 * i) * hw + p];
                    targets[(2 * i + 1) * hw + p] = p / lw + f[(2 * i + 1) * hw + p];
                }
            }
            Var looked = corr.lookup(targets, radius);
            Var detached = flow.detach();
            Var m = VisionOps.conv(Var.concat(1, looked, detached), motion, motionBias).relu();
            Var x = Var.concat(1, m, ctxFeat).permute(0, 2, 3, 1).reshape(b * hw, 24 + context);
            state = gru.step(x, state);
            Var hs = state.reshape(b, lh, lw, hidden).permute(0, 3, 1, 2);
            Var delta = VisionOps.conv(VisionOps.conv(hs, head1, head1Bias).relu(), head2, head2Bias);
            flow = detached.add(delta);
            out.add(Functional.resize(flow, new int[] {h, w}, Functional.Interpolation.LINEAR, false).mul(
                    (float) w / lw));
        }
        return out;
    }

    /**
     * The sequence loss: the mean end-point error (or its square) of every iteration,
     * weighted {@code gamma^(N - 1 - i)}. RAFT uses the plain (L1-like) error; the squared
     * error avoids the all-zero solution that the plain error's median pull favours when most
     * pixels are static, which matters on short training budgets.
     *
     * @param predictions the per-iteration flows {@code [batch, 2, h, w]}
     * @param truth the true flow {@code [batch, 2, h, w]}
     * @param gamma the decay, such as 0.8
     * @param squared whether to average the squared end-point error
     * @return the loss
     */
    public static Var sequenceLoss(List<Var> predictions, Tensor truth, float gamma, boolean squared) {
        int n = predictions.size();
        Var loss = null;
        for (int i = 0; i < n; i++) {
            Var d = predictions.get(i).sub(truth).square().sum(1, false);
            Var epe = (squared ? d : d.add(1e-6f).sqrt()).mean().mul((float) Math.pow(gamma, n - 1 - i));
            loss = loss == null ? epe : loss.add(epe);
        }
        return loss;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(encoder));
        p.addAll(List.of(contextEncoder));
        p.addAll(List.of(motion, motionBias));
        p.addAll(gru.parameters());
        p.addAll(List.of(head1, head1Bias, head2, head2Bias));
        return p;
    }

    @Override
    public String toString() {
        return "RecurrentFlowRefiner(" + features + " features, " + hidden + " hidden, " + levels + " levels, radius "
                + radius + ")";
    }
}
