package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GluFeedForward;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.ResidualMixerBlock;
import io.github.zzuegg.brainiac.experimental.nn.SliceCompressor;
import io.github.zzuegg.brainiac.experimental.nn.WeightPruning;
import io.github.zzuegg.brainiac.experiments.attention.ArchitectureBench;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * {@link SliceCompressor} (SliceGPT) against the two other ways of removing a share of a
 * transformer's weights in one shot, on a four-block model trained on the
 * {@link ArchitectureBench} toy language, from one calibration batch and without fine-tuning:
 * <ul>
 * <li>slicing: the residual stream rotated to its principal components and cut to {@code d'}
 *     (dense and smaller);</li>
 * <li>Wanda ({@link WeightPruning#wanda}) at the same share of every block matrix (sparse);</li>
 * <li>block removal: dropping the blocks that change the stream least (ShortGPT's block
 *     influence, one minus the cosine between a block's input and output).</li>
 * </ul>
 * Reported as validation loss against the share of block-matrix weights kept, one-shot and
 * (slicing and block removal, which give dense models) after a short recovery fine-tune.
 */
public final class SliceCompression {

    /**
     * One compressed model.
     *
     * @param method the method
     * @param kept the share of block-matrix weights kept
     * @param validationLoss the validation loss
     */
    public record Row(String method, double kept, double validationLoss) {
    }

    private static final int WIDTH = 32;
    private static final int DEPTH = 4;

    private SliceCompression() {
    }

    /**
     * Runs the comparison.
     *
     * @param steps training steps
     * @param keeps the sliced widths
     * @param recovery recovery fine-tuning steps for the sliced and the block-removed models
     * @param seed the seed
     * @return the dense model first, then per width: slicing, Wanda, block removal, and
     *     slicing and block removal after recovery fine-tuning
     */
    public static List<Row> run(int steps, int[] keeps, int recovery, long seed) {
        MultiHeadAttention.Spec mha = MultiHeadAttention.of(4).causal(true).rotary(10000f).bias(false);
        ArchitectureBench.LmResult base = ArchitectureBench.languageModel("MHA", ArchitectureBench.stack(mha, DEPTH),
                WIDTH, steps, seed);
        ArchitectureBench.TinyModel m = base.model();
        int[][] val = base.validation();
        LayerChain body = (LayerChain) m.body();
        Tensor input = embedded(m, ArchitectureBench.corpus(new SplittableRandom(seed + 3), 32));
        List<Row> rows = new ArrayList<>();
        rows.add(new Row("dense", 1, loss(m, val)));
        for (int keep : keeps) {
            double share = keep / (double) WIDTH;
            SliceCompressor.Sliced s = SliceCompressor.slice(List.of(m.embed(), m.previous()), body, m.readout(), input,
                    keep);
            ArchitectureBench.TinyModel sliced = new ArchitectureBench.TinyModel(Var.param(s.embeddings().get(0)),
                    Var.param(s.embeddings().get(1)), s.body(), Var.param(s.readout()));
            rows.add(new Row("slicegpt d'=" + keep, share, loss(sliced, val)));
            rows.add(new Row("wanda", share, wanda(m, body, input, 1 - share, val)));
            int drop = (int) Math.round(DEPTH * (1 - share));
            ArchitectureBench.TinyModel dropped = dropBlocks(m, body, input, drop);
            rows.add(new Row("drop " + drop + " blocks", 1 - drop / (double) DEPTH, loss(dropped, val)));
            fineTune(sliced, recovery, seed);
            rows.add(new Row("slicegpt+" + recovery, share, loss(sliced, val)));
            List<Var> params = m.parameters();
            List<Tensor> saved = new ArrayList<>();
            for (Var p : params) {
                saved.add(p.value());
            }
            fineTune(dropped, recovery, seed);
            rows.add(new Row("drop+" + recovery, 1 - drop / (double) DEPTH, loss(dropped, val)));
            for (int i = 0; i < params.size(); i++) {   // the kept blocks are the original's
                params.get(i).assign(saved.get(i));
            }
        }
        return rows;
    }

    private static double wanda(ArchitectureBench.TinyModel m, LayerChain body, Tensor input, double sparsity,
            int[][] val) {
        List<Var> params = body.parameters();
        List<Tensor> saved = new ArrayList<>();
        for (Var p : params) {
            saved.add(p.value());
        }
        Var x = Var.of(input);
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        for (Layer l : body.layers()) {
            ResidualMixerBlock block = (ResidualMixerBlock) l;
            MultiHeadAttention att = (MultiHeadAttention) block.mixer();
            GluFeedForward ffn = (GluFeedForward) block.ffn();
            List<Var> gains = block.parameters();
            Var a = rms(x).mul(gains.get(0));
            MultiHeadAttention.Settings s = att.settings();
            List<Var> w = att.parameters();
            Tensor z = heads(a, w, s, b, t);
            Tensor flatA = a.value().reshape(b * t, WIDTH);
            for (int i = 0; i < 3; i++) {
                w.get(i).assign(WeightPruning.wanda(w.get(i).value(), flatA, sparsity));
            }
            w.get(3).assign(WeightPruning.wanda(w.get(3).value(), z, sparsity));
            Var h = x.add(att.forward(a));
            Var c = rms(h).mul(gains.get(1));
            List<Var> f = ffn.parameters();   // up, down, gate
            Tensor flatC = c.value().reshape(b * t, WIDTH);
            Tensor hidden = ffn.kind().activation().forward(c.matmul(f.get(2))).mul(c.matmul(f.get(0))).value();
            hidden = hidden.reshape(b * t, hidden.shape().dim(2));
            f.get(0).assign(WeightPruning.wanda(f.get(0).value(), flatC, sparsity));
            f.get(2).assign(WeightPruning.wanda(f.get(2).value(), flatC, sparsity));
            f.get(1).assign(WeightPruning.wanda(f.get(1).value(), hidden, sparsity));
            x = block.forward(x);   // later blocks calibrate on the pruned earlier ones
        }
        double out = loss(m, val);
        for (int i = 0; i < params.size(); i++) {
            params.get(i).assign(saved.get(i));
        }
        return out;
    }

    /** The concatenated head outputs, the input of {@code wo}. */
    private static Tensor heads(Var a, List<Var> w, MultiHeadAttention.Settings s, int b, int t) {
        Var q = a.matmul(w.get(0)).reshape(b, t, s.heads(), s.headSize()).permute(0, 2, 1, 3);
        Var k = a.matmul(w.get(1)).reshape(b, t, s.kvHeads(), s.headSize()).permute(0, 2, 1, 3);
        Var v = a.matmul(w.get(2)).reshape(b, t, s.kvHeads(), s.valueSize()).permute(0, 2, 1, 3);
        q = Functional.rotary(q, s.rotaryBase(), 0);
        k = Functional.rotary(k, s.rotaryBase(), 0);
        float scale = s.scale() != 0 ? s.scale() : (float) (1 / Math.sqrt(s.headSize()));
        Var o = Functional.attention(q, k, v, true, scale);
        return o.permute(0, 2, 1, 3).reshape(b * t, s.heads() * s.valueSize()).value();
    }

    private static void fineTune(ArchitectureBench.TinyModel m, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 21);
        Optimizer opt = Optimizer.adam(3e-3f);
        List<Var> params = m.parameters();
        for (int s = 0; s < steps; s++) {
            ArchitectureBench.nextTokenLoss(m, ArchitectureBench.corpus(r, 16), true).backward();
            opt.step(params);
        }
    }

    private static ArchitectureBench.TinyModel dropBlocks(ArchitectureBench.TinyModel m, LayerChain body,
            Tensor input, int drop) {
        double[] influence = new double[DEPTH];
        Var x = Var.of(input);
        int rows = input.shape().dim(0) * input.shape().dim(1);
        for (int i = 0; i < DEPTH; i++) {
            Var y = body.layers().get(i).forward(x);
            float[] a = x.value().reshape(rows, WIDTH).toArray();
            float[] c = y.value().reshape(rows, WIDTH).toArray();
            double sum = 0;
            for (int r = 0; r < rows; r++) {
                double dot = 0;
                double na = 0;
                double nc = 0;
                for (int j = 0; j < WIDTH; j++) {
                    dot += a[r * WIDTH + j] * c[r * WIDTH + j];
                    na += a[r * WIDTH + j] * a[r * WIDTH + j];
                    nc += c[r * WIDTH + j] * c[r * WIDTH + j];
                }
                sum += 1 - dot / Math.sqrt(na * nc + 1e-12);
            }
            influence[i] = sum / rows;
            x = y;
        }
        Integer[] order = new Integer[DEPTH];
        for (int i = 0; i < DEPTH; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (p, q) -> Double.compare(influence[p], influence[q]));
        boolean[] dropped = new boolean[DEPTH];
        for (int i = 0; i < drop; i++) {
            dropped[order[i]] = true;
        }
        List<Layer> kept = new ArrayList<>();
        for (int i = 0; i < DEPTH; i++) {
            if (!dropped[i]) {
                kept.add(body.layers().get(i));
            }
        }
        return new ArchitectureBench.TinyModel(m.embed(), m.previous(), new LayerChain(kept), m.readout());
    }

    private static Var rms(Var x) {
        return x.div(x.square().mean(-1, true).add(1e-6f).sqrt());
    }

    private static double loss(ArchitectureBench.TinyModel m, int[][] tokens) {
        return ArchitectureBench.nextTokenLoss(m, tokens, false).value().item();
    }

    private static Tensor embedded(ArchitectureBench.TinyModel m, int[][] tokens) {
        int b = tokens.length;
        int t = tokens[0].length;
        float[] ids = new float[b * t];
        float[] prev = new float[b * t];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                ids[i * t + j] = tokens[i][j];
                prev[i * t + j] = tokens[i][Math.max(0, j - 1)];
            }
        }
        Var x = m.embed().gather(0, Tensor.of(Shape.of(b * t), ids))
                .add(m.previous().gather(0, Tensor.of(Shape.of(b * t), prev)));
        return x.value().reshape(b, t, WIDTH);
    }
}
