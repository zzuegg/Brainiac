package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Self-attention with relative positions (Shaw et al., 2018): each head learns an
 * embedding per offset {@code -window..window}, added to the keys and to the values of
 * keys that close to the query; farther keys get none. Translation invariant and
 * unbounded in length, it is the attention of the VITS text encoder. Input and output
 * are {@code [steps, features]} per sample.
 */
public final class RelativeAttention implements Layer {

    private final int features;
    private final int heads;
    private final int window;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var bq;
    private final Var bk;
    private final Var bv;
    private final Var bo;
    private final Var relK;
    private final Var relV;

    /**
     * A new layer with random weights.
     *
     * @param features the width of the input and output
     * @param heads the number of heads, dividing {@code features}
     * @param window the largest offset with its own embedding
     * @param init how to initialize the projections
     * @param random the random source
     */
    public RelativeAttention(int features, int heads, int window, Init init, SplittableRandom random) {
        this(heads, window, proj(features, init, random), zeros(features), proj(features, init, random),
                zeros(features), proj(features, init, random), zeros(features), proj(features, init, random),
                zeros(features), relative(features / heads, window, random), relative(features / heads, window, random));
    }

    /**
     * A layer from given weights, such as a pretrained model's.
     *
     * @param heads the number of heads
     * @param window the largest offset
     * @param wq query projection {@code [features, features]} (in, out)
     * @param bq its bias, or {@code null}
     * @param wk key projection
     * @param bk its bias, or {@code null}
     * @param wv value projection
     * @param bv its bias, or {@code null}
     * @param wo output projection
     * @param bo its bias, or {@code null}
     * @param relK key embeddings per offset {@code [2 window + 1, headSize]}
     * @param relV value embeddings per offset {@code [2 window + 1, headSize]}
     */
    public RelativeAttention(int heads, int window, Var wq, Var bq, Var wk, Var bk, Var wv, Var bv, Var wo, Var bo,
            Var relK, Var relV) {
        this.features = wq.shape().dim(0);
        if (heads < 1 || features % heads != 0) {
            throw new IllegalArgumentException("heads (" + heads + ") must divide the features (" + features + ")");
        }
        if (window < 1 || relK.shape().dim(0) != 2 * window + 1 || relK.shape().dim(1) != features / heads) {
            throw new IllegalArgumentException("relative embeddings must be [2 window + 1, headSize] = ["
                    + (2 * window + 1) + ", " + features / heads + "], got " + relK.shape());
        }
        this.heads = heads;
        this.window = window;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.bq = bq;
        this.bk = bk;
        this.bv = bv;
        this.bo = bo;
        this.relK = relK;
        this.relV = relV;
    }

    private static Var proj(int f, Init init, SplittableRandom random) {
        return Var.param(init.create(Shape.of(f, f), f, f, random));
    }

    private static Var zeros(int f) {
        return Var.param(Tensor.zeros(Shape.of(f)));
    }

    private static Var relative(int headSize, int window, SplittableRandom random) {
        return Var.param(Tensor.normal(Shape.of(2 * window + 1, headSize), 0, (float) (1 / Math.sqrt(headSize)),
                random));
    }

    @Override
    public Var forward(Var input) {
        boolean unbatched = input.shape().rank() == 2;
        Var x = unbatched ? input.reshape(1, input.shape().dim(0), input.shape().dim(1)) : input;
        int b = x.shape().dim(0);
        int s = x.shape().dim(1);
        if (x.shape().dim(2) != features) {
            throw new IllegalArgumentException(this + " expects " + features + " features, got " + x.shape());
        }
        int hd = features / heads;
        float scale = (float) (1 / Math.sqrt(hd));
        Var q = split(project(x, wq, bq).mul(scale), b, s, hd);
        Var k = split(project(x, wk, bk), b, s, hd);
        Var v = split(project(x, wv, bv), b, s, hd);
        Device device = x.value().device();
        int r = 2 * window + 1;
        // offsets j - i within the window, as indices into the embeddings, and which apply
        float[] keyIdx = new float[s * s];
        float[] keyOn = new float[s * s];
        float[] valIdx = new float[s * r];
        float[] valOn = new float[s * r];
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                int o = j - i + window;
                boolean in = o >= 0 && o < r;
                keyIdx[i * s + j] = in ? o : 0;
                keyOn[i * s + j] = in ? 1 : 0;
            }
            for (int o = 0; o < r; o++) {
                int j = i + o - window;
                boolean in = j >= 0 && j < s;
                valIdx[i * r + o] = in ? j : 0;
                valOn[i * r + o] = in ? 1 : 0;
            }
        }
        Var scores = q.matmul(k.permute(0, 1, 3, 2));
        Var relLogits = q.matmul(relK.transpose());   // [b, h, s, r]
        Tensor ki = Tensor.of(Shape.of(1, 1, s, s), keyIdx, device).expand(Shape.of(b, heads, s, s));
        Tensor kOn = Tensor.of(Shape.of(s, s), keyOn, device).to(x.dtype());
        scores = scores.add(relLogits.takeAlong(ki).mul(kOn));
        Var p = scores.sub(scores.logsumexp(-1, true)).exp();
        Var out = p.matmul(v);
        Tensor vi = Tensor.of(Shape.of(1, 1, s, r), valIdx, device).expand(Shape.of(b, heads, s, r));
        Tensor vOn = Tensor.of(Shape.of(s, r), valOn, device).to(x.dtype());
        out = out.add(p.takeAlong(vi).mul(vOn).matmul(relV));
        Var y = project(out.permute(0, 2, 1, 3).reshape(b, s, features), wo, bo);
        return unbatched ? y.reshape(s, features) : y;
    }

    private static Var project(Var x, Var w, Var bias) {
        Var y = x.matmul(w);
        return bias == null ? y : y.add(bias);
    }

    private Var split(Var x, int b, int s, int hd) {
        return x.reshape(b, s, heads, hd).permute(0, 2, 1, 3);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Var v : new Var[] {wq, bq, wk, bk, wv, bv, wo, bo, relK, relV}) {
            if (v != null) {
                p.add(v);
            }
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "RelativeAttention[" + features + ", " + heads + " heads, window " + window + "]";
    }

    static Config write(RelativeAttention a) {
        return Config.builder().put("features", a.features).put("heads", a.heads).put("window", a.window)
                .put("bias", a.bq != null).build();
    }

    static RelativeAttention read(Config c) {
        RelativeAttention a = new RelativeAttention(c.getInt("features"), c.getInt("heads"), c.getInt("window"),
                Init.ZEROS, new SplittableRandom());
        if (c.getBoolean("bias")) {
            return a;
        }
        return new RelativeAttention(a.heads, a.window, a.wq, null, a.wk, null, a.wv, null, a.wo, null, a.relK,
                a.relV);
    }
}
