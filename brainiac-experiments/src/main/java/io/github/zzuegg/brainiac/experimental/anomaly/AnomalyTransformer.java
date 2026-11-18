package io.github.zzuegg.brainiac.experimental.anomaly;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The Anomaly Transformer (Xu, Wu, Wang and Long, ICLR 2022,
 * <a href="https://arxiv.org/abs/2110.02642">arXiv:2110.02642</a>), a reconstruction
 * transformer over windows whose attention layers have two branches: the usual <b>series
 * association</b> {@code S = softmax(Q K^T / sqrt d)} and a <b>prior association</b>
 * {@code P_ij ~ exp(-|i - j|^2 / 2 sigma_i^2)}, a Gaussian kernel over the distance with a
 * learned per-step width. Normal steps attend widely across the window (they recur), anomalies
 * mostly to their neighbours, so the <b>association discrepancy</b> (the symmetric KL between
 * the rows of {@code P} and {@code S}, summed over layers) is small at anomalies. Training is
 * minimax: the prior is pulled towards the stopped-gradient series association (minimise
 * phase) while the series association is pushed away from the stopped-gradient prior (maximise
 * phase), both next to the reconstruction loss. The step score is
 * {@code softmax(-discrepancy) * |x - x_hat|^2} over the window.
 *
 * <p>Input {@code [batch, window, features]}; {@link #forward} returns the reconstruction.
 */
public final class AnomalyTransformer implements Layer {

    private final int window;
    private final int width;
    private final Var embed;
    private final Tensor positions;
    private final List<Var[]> layers = new ArrayList<>();
    private final List<LayerNorm[]> norms = new ArrayList<>();
    private final Var out;
    private final Tensor distance2;
    private List<Var[]> lastAssociations = List.of();

    /**
     * Creates the model.
     *
     * @param features input features per step
     * @param window the window length
     * @param width the model width
     * @param depth the number of anomaly-attention layers
     * @param random the initialisation source
     */
    public AnomalyTransformer(int features, int window, int width, int depth, SplittableRandom random) {
        this.window = window;
        this.width = width;
        float s = (float) Math.sqrt(1.0 / width);
        embed = Var.param(Tensor.normal(Shape.of(features, width), 0, (float) Math.sqrt(1.0 / features), random));
        float[] pe = new float[window * width];
        for (int t = 0; t < window; t++) {
            for (int i = 0; i < width; i++) {
                double f = Math.pow(10000, -2.0 * (i / 2) / width);
                pe[t * width + i] = (float) (i % 2 == 0 ? Math.sin(t * f) : Math.cos(t * f));
            }
        }
        positions = Tensor.of(Shape.of(1, window, width), pe);
        for (int l = 0; l < depth; l++) {
            layers.add(new Var[] {
                Var.param(Tensor.normal(Shape.of(width, width), 0, s, random)),     // q
                Var.param(Tensor.normal(Shape.of(width, width), 0, s, random)),     // k
                Var.param(Tensor.normal(Shape.of(width, width), 0, s, random)),     // v
                Var.param(Tensor.normal(Shape.of(width, 1), 0, s, random)),         // sigma
                Var.param(Tensor.normal(Shape.of(width, 2 * width), 0, s, random)), // ffn in
                Var.param(Tensor.normal(Shape.of(2 * width, width), 0, (float) Math.sqrt(0.5 / width), random))
            });
            norms.add(new LayerNorm[] {LayerNorm.of(width), LayerNorm.of(width)});
        }
        out = Var.param(Tensor.normal(Shape.of(width, features), 0, s, random));
        float[] d2 = new float[window * window];
        for (int i = 0; i < window; i++) {
            for (int j = 0; j < window; j++) {
                d2[i * window + j] = (i - j) * (i - j);
            }
        }
        distance2 = Tensor.of(Shape.of(1, window, window), d2);
    }

    private static Var softmax(Var x) {
        return x.sub(x.logsumexp(-1, true)).exp();
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var h = input.matmul(embed).add(positions);
        List<Var[]> assoc = new ArrayList<>();
        for (int l = 0; l < layers.size(); l++) {
            Var[] w = layers.get(l);
            Var q = h.matmul(w[0]);
            Var k = h.matmul(w[1]);
            Var v = h.matmul(w[2]);
            Var series = softmax(q.matmul(k.permute(0, 2, 1)).mul((float) (1 / Math.sqrt(width))));
            // sigma in [~0, 3^1 - 1] steps of width, following the paper's 3^(sigmoid(5x)) - 1
            Var sigma = h.matmul(w[3]).mul(5f).sigmoid().add(1e-5f).mul((float) Math.log(3)).exp().sub(1f)
                    .mul(window / 4f).add(0.5f);
            Var logits = Var.of(distance2).div(sigma.square().mul(-2f));
            Var prior = softmax(logits);
            assoc.add(new Var[] {prior, series});
            h = norms.get(l)[0].forward(h.add(series.matmul(v)));
            h = norms.get(l)[1].forward(h.add(h.matmul(w[4]).relu().matmul(w[5])));
        }
        lastAssociations = assoc;
        return h.matmul(out);
    }

    /** Symmetric KL between rows, {@code [batch, window]}. */
    private static Var discrepancy(Var p, Var s) {
        Var lp = p.add(1e-6f).log();
        Var ls = s.add(1e-6f).log();
        return p.mul(lp.sub(ls)).sum(-1, false).add(s.mul(ls.sub(lp)).sum(-1, false));
    }

    /**
     * The minimax training loss of one batch.
     *
     * @param x {@code [batch, window, features]}
     * @param lambda the weight of the association discrepancy, for example 3
     * @return the combined loss (backward it once)
     */
    public Var loss(Tensor x, float lambda) {
        Var rec = forward(Var.of(x), true).sub(x).square().mean();
        Var minimise = null;
        Var maximise = null;
        for (Var[] a : lastAssociations) {
            Var dp = discrepancy(a[0], a[1].detach()).mean();   // pulls the prior to the series
            Var ds = discrepancy(a[0].detach(), a[1]).mean();   // pushes the series from the prior
            minimise = minimise == null ? dp : minimise.add(dp);
            maximise = maximise == null ? ds : maximise.add(ds);
        }
        return rec.mul(2f).add(minimise.mul(lambda)).sub(maximise.mul(lambda));
    }

    /**
     * Trains on windows of a (mostly normal) series.
     *
     * @param series the training series
     * @param steps Adam steps on batches of 16 windows
     * @param lambda the discrepancy weight
     * @param seed the batching seed
     */
    public void fit(float[] series, int steps, float lambda, long seed) {
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int b = 16;
        for (int s = 0; s < steps; s++) {
            float[] x = new float[b * window];
            for (int i = 0; i < b; i++) {
                System.arraycopy(series, r.nextInt(series.length - window + 1), x, i * window, window);
            }
            loss(Tensor.of(Shape.of(b, window, 1), x), lambda).backward();
            opt.step(parameters());
        }
    }

    /**
     * Per-step anomaly scores over non-overlapping windows of a univariate series (the tail is
     * scored by a last window aligned to the end).
     *
     * @param series the series
     * @return one score per step
     */
    public double[] score(float[] series) {
        int n = series.length;
        double[] out = new double[n];
        List<Integer> starts = new ArrayList<>();
        for (int s = 0; s + window <= n; s += window) {
            starts.add(s);
        }
        if (n % window != 0) {
            starts.add(n - window);
        }
        float[] x = new float[starts.size() * window];
        for (int i = 0; i < starts.size(); i++) {
            System.arraycopy(series, starts.get(i), x, i * window, window);
        }
        Tensor tx = Tensor.of(Shape.of(starts.size(), window, 1), x);
        float[] score = Var.noGrad(() -> {
            Var err = forward(Var.of(tx), false).sub(tx).square().sum(-1, false);
            Var dis = null;
            for (Var[] a : lastAssociations) {
                Var d = discrepancy(a[0], a[1]);
                dis = dis == null ? d : dis.add(d);
            }
            return softmax(dis.neg()).mul(err);
        }).value().toArray();
        for (int i = 0; i < starts.size(); i++) {
            for (int t = 0; t < window; t++) {
                out[starts.get(i) + t] = score[i * window + t];
            }
        }
        return out;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, out));
        for (Var[] w : layers) {
            p.addAll(List.of(w));
        }
        for (LayerNorm[] n : norms) {
            p.addAll(n[0].parameters());
            p.addAll(n[1].parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "AnomalyTransformer(window=" + window + ", width=" + width + ", layers=" + layers.size() + ")";
    }
}
