package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * The Feedback Transformer (experiment X12): a causal transformer in which every layer,
 * even the lowest, reads the highest-level representations of all earlier steps. Instead of
 * each layer attending to its own layer's past states, all layers attend to one shared
 * memory, where the memory vector of step {@code t} is a learned softmax-weighted sum of the
 * states of every layer (embedding included) at step {@code t}:
 *
 * <pre>
 * m_t = sum_l softmax(w)_l x_t^l
 * x_t^{l+1} = block_l(x_t^l, attend to [m_1 .. m_{t-1}, x_t^l])
 * </pre>
 *
 * <p>Abstract features computed at step {@code t-1} are therefore available to the
 * computation at step {@code t} from the first layer on, which gives the model recurrent
 * depth (it can track state, such as a running parity, that a fixed-depth transformer
 * cannot). The price: steps must be computed one after another, as in an RNN. Keys and
 * values of the memory are shared across layers, as the paper recommends, and every head
 * has a learned ALiBi-style recency bias in place of the paper's relative positions.
 *
 * <p>Input and output are {@code [batch, steps, features]}; the output is the top layer.
 *
 * <p>Paper: Fan et al., "Addressing Some Limitations of Transformers with Feedback Memory",
 * <a href="https://arxiv.org/abs/2002.09402">arXiv:2002.09402</a>.
 */
public final class FeedbackTransformer implements Layer {

    private final int features;
    private final int layers;
    private final int heads;
    private final Var wk;
    private final Var wv;
    private final Var layerWeights;
    private final Var recency;
    private final List<Var[]> blocks = new ArrayList<>();

    /**
     * Creates the model.
     *
     * @param features the width, a multiple of {@code heads}
     * @param layers the number of layers
     * @param heads attention heads
     * @param random the random source
     */
    public FeedbackTransformer(int features, int layers, int heads, RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("features " + features + " not divisible by heads " + heads);
        }
        this.features = features;
        this.layers = layers;
        this.heads = heads;
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.layerWeights = MixerOps.filled(0f, layers + 1);
        this.recency = MixerOps.filled(0.5f, heads, 1, 1);
        for (int l = 0; l < layers; l++) {
            blocks.add(new Var[] {MixerOps.weight(random, features, features), MixerOps.weight(random, features, features),
                MixerOps.weight(random, features, 2 * features), MixerOps.weight(random, 2 * features, features)});
        }
    }

    /**
     * Describes the model for {@code Model.sequential().add(...)}.
     *
     * @param layers the number of layers
     * @param heads attention heads
     * @return the spec
     */
    public static LayerSpec of(int layers, int heads) {
        return MixerOps.sameShape((in, r) -> new FeedbackTransformer(in.dim(-1), layers, heads, r));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "FeedbackTransformer");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int d = features / heads;
        float scale = (float) (1 / Math.sqrt(d));
        Var mix = MixerOps.softmax(layerWeights);
        List<Var> memK = new ArrayList<>();   // each [b, h, 1, d]
        List<Var> memV = new ArrayList<>();
        List<Var> outputs = new ArrayList<>();
        for (int step = 0; step < t; step++) {
            Var x = input.slice(1, step, step + 1);   // [b, 1, f]
            Var memory = x.mul(mix.slice(0, 0, 1));
            for (int l = 0; l < layers; l++) {
                Var[] w = blocks.get(l);
                Var n = MixerOps.rmsNorm(x);
                Var q = MixerOps.split(n.matmul(w[0]), heads);
                List<Var> ks = new ArrayList<>(memK);
                List<Var> vs = new ArrayList<>(memV);
                ks.add(MixerOps.split(n.matmul(wk), heads));
                vs.add(MixerOps.split(n.matmul(wv), heads));
                Var k = ks.size() == 1 ? ks.getFirst() : Var.concat(2, ks.toArray(Var[]::new));
                Var v = vs.size() == 1 ? vs.getFirst() : Var.concat(2, vs.toArray(Var[]::new));
                Var scores = q.matmul(k.transpose()).mul(scale);
                if (step > 0) {
                    // ALiBi-style recency bias with a learned slope per head
                    float[] dist = new float[step + 1];
                    for (int j = 0; j <= step; j++) {
                        dist[j] = step - j;
                    }
                    scores = scores.sub(recency.softplus().mul(Tensor.of(Shape.of(1, step + 1), dist)));
                }
                Var att = MixerOps.softmax(scores).matmul(v);
                x = x.add(MixerOps.merge(att).matmul(w[1]));
                x = x.add(MixerOps.mlp(MixerOps.rmsNorm(x), w[2], w[3]));
                memory = memory.add(x.mul(mix.slice(0, l + 1, l + 2)));
            }
            outputs.add(x);
            Var mn = MixerOps.rmsNorm(memory);
            memK.add(MixerOps.split(mn.matmul(wk), heads));
            memV.add(MixerOps.split(mn.matmul(wv), heads));
        }
        return outputs.size() == 1 ? outputs.getFirst() : Var.concat(1, outputs.toArray(Var[]::new));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wk, wv, layerWeights, recency));
        blocks.forEach(b -> p.addAll(List.of(b)));
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    /**
     * Returns the softmax weights that mix the layers into the memory.
     *
     * @return {@code [layers + 1]}, embedding first
     */
    public Tensor memoryMix() {
        return MixerOps.softmax(layerWeights).value();
    }

    @Override
    public String toString() {
        return "FeedbackTransformer(" + features + ", layers=" + layers + ", heads=" + heads + ")";
    }
}
