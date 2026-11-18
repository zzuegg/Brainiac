package io.github.zzuegg.brainiac.experimental.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.SlotAttention;
import io.github.zzuegg.brainiac.experimental.vision.VisionOps;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Object-centric video dynamics: SlotFormer (Wu et al., ICLR 2023,
 * <a href="https://arxiv.org/abs/2210.05861">arXiv:2210.05861</a>). A {@link SlotAttention}
 * encoder decomposes every frame into {@code K} slots, each frame's slots initialised with the
 * previous frame's (as in SAVi) so that slot {@code k} keeps following the same object; a
 * spatial-broadcast decoder renders each slot to an image and a mask logit, and the masks
 * (softmax over slots) composite the frame. The dynamics act on slots, not pixels: a transformer
 * over the slots of the last {@code T} frames, with a learned embedding per frame position,
 * predicts the next frame's slots (as a residual update of the last ones), and rolls out
 * autoregressively on its own predictions. Because every token is one object, interactions such
 * as collisions are attention between slots, and identity is kept by construction.
 *
 * <p>{@link Dynamics#PER_SLOT_MLP} is the no-interaction ablation: each slot is advanced by an
 * MLP of its own last two states only.
 *
 * <p>Training is in two stages, as in the paper: {@link #trainAutoencoder} learns the slot
 * encoder and decoder on frames; {@link #trainDynamics} then freezes them and trains the
 * dynamics with a rollout loss on slots plus the pixel loss of the decoded predictions.
 */
public final class SlotDynamics {

    /** The dynamics network. */
    public enum Dynamics {
        /** A transformer over all slots of the history. */
        TRANSFORMER,
        /** An MLP per slot of its last two states (no interaction). */
        PER_SLOT_MLP
    }

    private final int side;
    private final int slots;
    private final int width;
    private final int history;
    private final Dynamics kind;
    private final Var conv1;
    private final Var conv1b;
    private final Var conv2;
    private final Var conv2b;
    private final Var posIn;
    private final SlotAttention attention;
    private final Var posOut;
    private final SimpleMlp render;
    private final Var timeEmb;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final Var head;
    private final SimpleMlp slotMlp;
    private final Optimizer aeOptimizer = Optimizer.adam(2e-3f);
    private final Optimizer dynOptimizer = Optimizer.adam(1e-3f);
    private final Tensor positions;

    private SlotDynamics(int side, int slots, int width, int history, Dynamics kind, long seed) {
        this.side = side;
        this.slots = slots;
        this.width = width;
        this.history = history;
        this.kind = kind;
        SplittableRandom r = new SplittableRandom(seed);
        conv1 = VisionOps.convWeight(r, 16, 1, 3);
        conv1b = VisionOps.zeros(16);
        conv2 = VisionOps.convWeight(r, 16, 16, 3);
        conv2b = VisionOps.zeros(16);
        posIn = Var.param(Tensor.normal(Shape.of(4, 16), 0, 0.5f, r));
        attention = new SlotAttention(16, slots, width, 3, r);
        posOut = Var.param(Tensor.normal(Shape.of(4, width), 0, 0.5f, r));
        render = SimpleMlp.of(seed + 1, v -> v.mul(v.sigmoid()), width, 32, 32, 2);
        timeEmb = Var.param(Tensor.normal(Shape.of(history, width), 0, 0.1f, r));
        for (int l = 0; l < 2; l++) {
            blocks.add((TransformerBlock) TransformerBlock.of(2).ffnSize(2 * width)
                    .create(Shape.of(1, history * slots, width), r));
        }
        head = Var.param(Tensor.normal(Shape.of(width, width), 0, (float) (0.1 / Math.sqrt(width)), r));
        slotMlp = SimpleMlp.of(seed + 2, v -> v.mul(v.sigmoid()), 2 * width, 64, width);
        float[] pos = new float[side * side * 4];
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                int i = (y * side + x) * 4;
                pos[i] = x / (side - 1f);
                pos[i + 1] = y / (side - 1f);
                pos[i + 2] = 1 - pos[i];
                pos[i + 3] = 1 - pos[i + 1];
            }
        }
        positions = Tensor.of(Shape.of(side * side, 4), pos);
    }

    /**
     * Creates a model.
     *
     * @param side the frame side
     * @param slots slots per frame {@code K} (objects plus background)
     * @param width the slot width
     * @param history frames of slots the dynamics see {@code T}
     * @param kind the dynamics network
     * @param seed the seed
     * @return the model
     */
    public static SlotDynamics of(int side, int slots, int width, int history, Dynamics kind, long seed) {
        return new SlotDynamics(side, slots, width, history, kind, seed);
    }

    private List<Var> autoencoderParameters() {
        List<Var> p = new ArrayList<>(List.of(conv1, conv1b, conv2, conv2b, posIn, posOut));
        p.addAll(attention.parameters());
        p.addAll(render.parameters());
        return p;
    }

    private List<Var> dynamicsParameters() {
        List<Var> p = new ArrayList<>();
        if (kind == Dynamics.TRANSFORMER) {
            p.add(timeEmb);
            p.add(head);
            for (TransformerBlock b : blocks) {
                p.addAll(b.parameters());
            }
        } else {
            p.addAll(slotMlp.parameters());
        }
        return p;
    }

    /** Pixel features {@code [b, side^2, 16]} of frames {@code [b, side, side]}. */
    private Var features(Var frames) {
        int b = frames.shape().dim(0);
        Var x = frames.reshape(b, 1, side, side);
        Var h = VisionOps.conv(VisionOps.conv(x, conv1, conv1b).relu(), conv2, conv2b).relu();
        return h.reshape(b, 16, side * side).permute(0, 2, 1).add(Var.of(positions).matmul(posIn));
    }

    /**
     * Slots of one frame.
     *
     * @param frames {@code [b, side, side]}
     * @param initial the previous frame's slots, or {@code null}
     * @return {@code [b, K, width]}
     */
    public Var encode(Var frames, Var initial) {
        return attention.forward(features(frames), initial, true);
    }

    /**
     * Slots of every frame of clips, each frame initialised with the previous frame's slots.
     *
     * @param clips {@code [b, T, side, side]}
     * @return per frame {@code [b, K, width]}
     */
    public List<Var> encodeSequence(Tensor clips) {
        int t = clips.shape().dim(1);
        int b = clips.shape().dim(0);
        List<Var> out = new ArrayList<>();
        Var prev = null;
        for (int f = 0; f < t; f++) {
            Var s = encode(Var.of(clips.slice(1, f, f + 1).reshape(b, side, side)), prev);
            out.add(s);
            prev = s.detach();
        }
        return out;
    }

    /**
     * Renders slots.
     *
     * @param s {@code [b, K, width]}
     * @return {@code {image [b, side^2], masks [b, K, side^2]}}
     */
    public Var[] decode(Var s) {
        int b = s.shape().dim(0);
        int p = side * side;
        Var grid = s.reshape(b * slots, 1, width).add(Var.of(positions).matmul(posOut).reshape(1, p, width));
        Var out = render.forward(grid.reshape(b * slots * p, width)).reshape(b, slots, p, 2);
        Var value = out.slice(3, 0, 1).reshape(b, slots, p);
        Var logits = out.slice(3, 1, 2).reshape(b, slots, p);
        Var masks = Activation.SOFTMAX.forward(logits.permute(0, 2, 1)).permute(0, 2, 1);
        return new Var[] {value.mul(masks).sum(1, false), masks};
    }

    /**
     * One autoencoder step on clips (every frame reconstructed from its slots).
     *
     * @param clips {@code [b, T, side, side]}
     * @return the reconstruction error
     */
    public float trainAutoencoder(Tensor clips) {
        int b = clips.shape().dim(0);
        int t = clips.shape().dim(1);
        List<Var> s = encodeSequence(clips);
        Var loss = null;
        for (int f = 0; f < t; f++) {
            Var l = decode(s.get(f))[0].sub(clips.slice(1, f, f + 1).reshape(b, side * side)).square().mean();
            loss = loss == null ? l : loss.add(l);
        }
        loss = loss.mul(1f / t);
        float v = loss.value().item();
        loss.backward();
        aeOptimizer.step(autoencoderParameters());
        return v;
    }

    /**
     * The next slots from the last {@code T} frames of slots.
     *
     * @param past at least {@code T} frames of slots, {@code [b, K, width]} each
     * @return the predicted next slots
     */
    public Var step(List<Var> past) {
        int n = past.size();
        Var last = past.get(n - 1);
        int b = last.shape().dim(0);
        if (kind == Dynamics.PER_SLOT_MLP) {
            Var prev = n >= 2 ? past.get(n - 2) : last;
            Var in = Var.concat(2, last, last.sub(prev)).reshape(b * slots, 2 * width);
            return last.add(slotMlp.forward(in).reshape(b, slots, width));
        }
        List<Var> window = new ArrayList<>();
        for (int i = 0; i < history; i++) {
            Var s = past.get(Math.max(0, n - history + i));
            window.add(s.add(timeEmb.slice(0, i, i + 1).reshape(1, 1, width)));
        }
        Var h = Var.concat(1, window.toArray(new Var[0]));
        for (TransformerBlock blk : blocks) {
            h = blk.forward(h);
        }
        Var out = h.slice(1, (history - 1) * slots, history * slots);
        return last.add(out.matmul(head));
    }

    /**
     * One dynamics step: the encoder and decoder frozen, the history encoded, then {@code rollout}
     * predicted frames, each fed back, scored against the encoded slots and the true frames.
     *
     * @param clips {@code [b, T + rollout, side, side]}
     * @param rollout predicted frames per clip
     * @return the loss
     */
    public float trainDynamics(Tensor clips, int rollout) {
        int b = clips.shape().dim(0);
        int t = clips.shape().dim(1);
        List<Var> target = Var.noGrad(() -> {
            List<Var> e = encodeSequence(clips);
            List<Var> d = new ArrayList<>();
            for (Var v : e) {
                d.add(Var.of(v.value()));
            }
            return d;
        });
        List<Var> past = new ArrayList<>(target.subList(0, t - rollout));
        Var loss = null;
        for (int k = 0; k < rollout; k++) {
            Var next = step(past);
            int f = t - rollout + k;
            Var slotLoss = next.sub(target.get(f).value()).square().mean();
            Var pix = decode(next)[0].sub(clips.slice(1, f, f + 1).reshape(b, side * side)).square().mean();
            Var l = slotLoss.add(pix);
            loss = loss == null ? l : loss.add(l);
            past.add(next);
        }
        loss = loss.mul(1f / rollout);
        float v = loss.value().item();
        loss.backward();
        dynOptimizer.step(dynamicsParameters());
        for (Var p : autoencoderParameters()) {
            p.zeroGrad();
        }
        return v;
    }

    /**
     * Rolls out from context frames.
     *
     * @param context {@code [b, T, side, side]}
     * @param horizon frames to predict
     * @return {@code {frames [b, horizon, side, side], masks [b, horizon, K, side^2]}}
     */
    public Tensor[] rollout(Tensor context, int horizon) {
        return Var.noGrad(() -> {
            int b = context.shape().dim(0);
            List<Var> past = encodeSequence(context);
            List<Var> frames = new ArrayList<>();
            List<Var> masks = new ArrayList<>();
            for (int k = 0; k < horizon; k++) {
                Var next = step(past);
                Var[] d = decode(next);
                frames.add(d[0].reshape(b, 1, side, side));
                masks.add(d[1].reshape(b, 1, slots, side * side));
                past.add(next);
            }
            return new Tensor[] {Var.concat(1, frames.toArray(new Var[0])).value(),
                Var.concat(1, masks.toArray(new Var[0])).value()};
        });
    }

    /** @return slots per frame */
    public int slots() {
        return slots;
    }
}
