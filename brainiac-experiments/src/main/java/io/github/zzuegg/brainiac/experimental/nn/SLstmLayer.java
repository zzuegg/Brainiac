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
 * The sLSTM of xLSTM: a scalar-memory LSTM with exponential input and forget gates, a
 * normalizer state and a log-space stabilizer, and recurrent connections that are
 * block-diagonal per head (memory mixing within a head):
 *
 * <pre>
 * z = tanh(.)   i~, f~ = pre-activations   o = sigmoid(.)      (each from W x_t + R h_{t-1} + b)
 * m_t = max(f~ + m_{t-1}, i~)
 * c_t = exp(f~ + m_{t-1} - m_t) c_{t-1} + exp(i~ - m_t) z
 * n_t = exp(f~ + m_{t-1} - m_t) n_{t-1} + exp(i~ - m_t)
 * h_t = o . c_t / n_t
 * </pre>
 *
 * <p>The exponential gates let a unit overwrite its memory decisively, and the true
 * recurrence (through {@code h_{t-1}}) gives state tracking abilities (parity, counting)
 * that parallelizable linear RNNs lack; the price is sequential computation.
 *
 * <p>Input {@code [batch, steps, features]}, output {@code [batch, steps, hidden]}.
 *
 * <p>Paper: Beck et al., "xLSTM: Extended Long Short-Term Memory",
 * <a href="https://arxiv.org/abs/2405.04517">arXiv:2405.04517</a>.
 */
public final class SLstmLayer implements Layer {

    private final int features;
    private final int hidden;
    private final int heads;
    private final int headSize;
    private final Var w;
    private final Var r;
    private final Var bias;

    /**
     * Creates the layer.
     *
     * @param features the input width
     * @param hidden the hidden width, a multiple of {@code heads}
     * @param heads the number of heads (blocks of the recurrent matrix)
     * @param random the random source
     */
    public SLstmLayer(int features, int hidden, int heads, RandomGenerator random) {
        if (hidden % heads != 0) {
            throw new IllegalArgumentException("hidden " + hidden + " not divisible by heads " + heads);
        }
        this.features = features;
        this.hidden = hidden;
        this.heads = heads;
        this.headSize = hidden / heads;
        this.w = MixerOps.weight(random, features, 4 * hidden);
        this.r = MixerOps.uniform(random, (float) (1 / Math.sqrt(headSize)), heads, headSize, 4 * headSize);
        // forget gate bias 3 (remember), input gate bias 0
        float[] b = new float[4 * hidden];
        for (int h = 0; h < heads; h++) {
            for (int j = 0; j < headSize; j++) {
                b[h * 4 * headSize + 2 * headSize + j] = 3f;
            }
        }
        this.bias = Var.param(Tensor.of(Shape.of(4 * hidden), b));
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param hidden the hidden width
     * @param heads the number of heads
     * @return the spec
     */
    public static LayerSpec of(int hidden, int heads) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), hidden);
            }

            @Override
            public Layer create(Shape input, java.util.SplittableRandom random) {
                return new SLstmLayer(input.dim(-1), hidden, heads, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != features) {
            throw new IllegalArgumentException("SLstmLayer expects [batch, steps, " + features + "], got " + s);
        }
        int b = s.dim(0);
        int t = s.dim(1);
        int d = headSize;
        // layout per head: [z, i, f, o]
        Var pre = input.matmul(w).add(bias).reshape(b, t, heads, 4 * d).permute(0, 2, 1, 3);   // [b, H, t, 4d]
        Var h = Var.of(Tensor.zeros(Shape.of(b, heads, 1, d)));
        Var c = h;
        Var n = h;
        Var m = Var.of(Tensor.full(Shape.of(b, heads, 1, d), -1e30f));
        List<Var> out = new ArrayList<>();
        for (int step = 0; step < t; step++) {
            Var g = pre.slice(2, step, step + 1).add(h.matmul(r));   // [b, H, 1, 4d]
            Var z = g.slice(3, 0, d).tanh();
            Var iLog = g.slice(3, d, 2 * d);
            Var fLog = g.slice(3, 2 * d, 3 * d);
            Var o = g.slice(3, 3 * d, 4 * d).sigmoid();
            Var mNew = fLog.add(m).maximum(iLog);
            Var ig = iLog.sub(mNew).exp();
            Var fg = fLog.add(m).sub(mNew).exp();
            c = c.mul(fg).add(ig.mul(z));
            n = n.mul(fg).add(ig);
            m = mNew;
            h = o.mul(c.div(n));
            out.add(h);
        }
        return MixerOps.merge(Var.concat(2, out.toArray(Var[]::new)));
    }

    @Override
    public List<Var> parameters() {
        return List.of(w, r, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return hidden;
    }

    @Override
    public String toString() {
        return "SLstmLayer(" + features + " -> " + hidden + ", heads=" + heads + ")";
    }
}
