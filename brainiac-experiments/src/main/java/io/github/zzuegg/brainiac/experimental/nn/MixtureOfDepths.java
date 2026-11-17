package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Mixture-of-Depths: a router lets only a fixed share of each sequence's tokens through a
 * block; the rest skip it on the residual path. With capacity {@code c = ceil(capacity *
 * steps)}, the {@code c} tokens with the highest router score {@code r = x w} are gathered
 * (keeping their order), processed, multiplied by {@code sigmoid(r)} (so the router gets a
 * gradient) and added back:
 *
 * <pre>
 * y_i = x_i + sigmoid(r_i) f(X_selected)_i    if i is among the top c
 * y_i = x_i                                     otherwise
 * </pre>
 *
 * <p>Shapes stay static (always {@code c} tokens), so the compute saving is real. Top-c over
 * the sequence looks at future tokens, so for autoregressive sampling a small causal
 * predictor learns, from {@link #predictorLoss()}, whether a token would be in the top c;
 * {@code forward(x, false)} with {@link #useCausalPredictor(boolean)} routes with it.
 *
 * <p>The wrapped block maps {@code [b, c, features]} to its residual update of the same shape
 * (for example {@link GatedAttention}).
 *
 * <p>Paper: Raposo et al., "Mixture-of-Depths: Dynamically allocating compute in
 * transformer-based language models", <a href="https://arxiv.org/abs/2404.02258">arXiv:2404.02258</a>.
 */
public final class MixtureOfDepths implements Layer {

    private final Layer block;
    private final int features;
    private final float capacity;
    private final Var router;
    private final Var predictor;
    private final Var predictorBias;
    private boolean causalPredictor;
    private Var lastPredictorLoss;
    private int[][] lastSelected;

    /**
     * Wraps a block.
     *
     * @param block the block, {@code [b, c, features] -> [b, c, features]} residual update
     * @param features the width
     * @param capacity the share of tokens processed, in {@code (0, 1]}
     * @param random the random source
     */
    public MixtureOfDepths(Layer block, int features, float capacity, RandomGenerator random) {
        if (!(capacity > 0 && capacity <= 1)) {
            throw new IllegalArgumentException("capacity must be in (0, 1], got " + capacity);
        }
        this.block = block;
        this.features = features;
        this.capacity = capacity;
        this.router = MixerOps.weight(random, features, 1);
        this.predictor = MixerOps.weight(random, features, 1);
        this.predictorBias = MixerOps.filled(0f, 1);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param block the spec of the wrapped block, which sees {@code [c, features]}
     * @param capacity the share of tokens processed
     * @return the spec
     */
    public static LayerSpec of(LayerSpec block, float capacity) {
        return MixerOps.sameShape((in, r) -> new MixtureOfDepths(block.create(in, r), in.dim(-1), capacity, r));
    }

    /**
     * Routes with the causal predictor ({@code predictor > 0.5}, a variable number of
     * tokens) instead of top-c in inference mode.
     *
     * @param on whether to use the predictor at inference
     * @return this layer
     */
    public MixtureOfDepths useCausalPredictor(boolean on) {
        this.causalPredictor = on;
        return this;
    }

    /** @return the token positions each sequence processed in the last forward pass */
    public int[][] lastSelected() {
        return lastSelected;
    }

    /**
     * The binary cross-entropy of the causal predictor against the top-c choices of the
     * last training forward pass; its input is detached, so it does not disturb the model.
     *
     * @return a scalar, or {@code null} if none was computed yet
     */
    public Var predictorLoss() {
        return lastPredictorLoss;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        MixerOps.requireSequence(input, features, "MixtureOfDepths");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int c = Math.max(1, (int) Math.ceil(capacity * t));
        Var scores = input.matmul(router).reshape(b, t);
        Var predicted = input.detach().matmul(predictor).add(predictorBias).reshape(b, t);
        float[] r = scores.value().toArray();
        int[][] selected = new int[b][];
        float[] label = new float[b * t];
        boolean usePredictor = !training && causalPredictor;
        float[] p = usePredictor ? predicted.value().toArray() : null;
        for (int n = 0; n < b; n++) {
            int[] idx;
            if (usePredictor) {
                int count = 0;
                int[] tmp = new int[t];
                for (int i = 0; i < t; i++) {
                    if (p[n * t + i] > 0) {
                        tmp[count++] = i;
                    }
                }
                idx = Arrays.copyOf(tmp, count);
            } else {
                idx = MixerOps.topK(r, n * t, t, c);
                Arrays.sort(idx);
            }
            for (int i : idx) {
                label[n * t + i] = 1;
            }
            selected[n] = idx;
        }
        lastSelected = selected;
        if (training) {
            Tensor y = Tensor.of(Shape.of(b, t), label);
            // BCE with logits: softplus(z) - y z
            lastPredictorLoss = predicted.softplus().sub(predicted.mul(y)).mean();
        }
        Var flat = input.reshape(b * t, features);
        Var weight = scores.sigmoid().reshape(b * t, 1);
        Var update = null;
        if (!usePredictor) {
            int[] flatIdx = new int[b * c];
            for (int n = 0; n < b; n++) {
                for (int j = 0; j < c; j++) {
                    flatIdx[n * c + j] = n * t + selected[n][j];
                }
            }
            Tensor ids = MixerOps.ids(flatIdx);
            Var out = block.forward(flat.gather(0, ids).reshape(b, c, features), training).reshape(b * c, features);
            update = out.mul(weight.gather(0, ids).reshape(b * c, 1)).scatterAdd(0, ids, b * t);
        } else {
            List<Var> parts = new ArrayList<>();
            List<Integer> all = new ArrayList<>();
            for (int n = 0; n < b; n++) {
                if (selected[n].length == 0) {
                    continue;
                }
                int[] flatIdx = Arrays.stream(selected[n]).map(i -> i + 0).toArray();
                for (int j = 0; j < flatIdx.length; j++) {
                    flatIdx[j] += n * t;
                    all.add(flatIdx[j]);
                }
                Tensor ids = MixerOps.ids(flatIdx);
                Var out = block.forward(flat.gather(0, ids).reshape(1, flatIdx.length, features), false)
                        .reshape(flatIdx.length, features);
                parts.add(out.mul(weight.gather(0, ids).reshape(flatIdx.length, 1)));
            }
            if (!parts.isEmpty()) {
                Var cat = parts.size() == 1 ? parts.getFirst() : Var.concat(0, parts.toArray(Var[]::new));
                update = cat.scatterAdd(0, MixerOps.ids(all.stream().mapToInt(Integer::intValue).toArray()),
                        b * t);
            }
        }
        return update == null ? input : flat.add(update).reshape(b, t, features);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(router, predictor, predictorBias));
        p.addAll(block.parameters());
        return p;
    }

    @Override
    public String toString() {
        return "MixtureOfDepths(" + block + ", capacity=" + capacity + ")";
    }
}
