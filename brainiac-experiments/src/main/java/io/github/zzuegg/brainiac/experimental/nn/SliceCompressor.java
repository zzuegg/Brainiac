package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * SliceGPT: makes a pre-norm transformer narrower by deleting whole rows and columns, using
 * <em>computational invariance</em>. With the RMS norm gains folded into the weights that read
 * the normalized stream, any orthogonal {@code Q} can rotate the residual stream without
 * changing the function ({@code rms(x Q) = rms(x)}): embeddings and every weight that writes
 * to the stream are multiplied by {@code Q} on the right, every weight that reads it by
 * {@code Q^T} on the left. Choosing {@code Q} as the principal components of the stream's
 * states on calibration data puts most of the signal into the first coordinates, and keeping
 * only the first {@code d'} of them gives a dense model of width {@code d'}: smaller matrices,
 * not sparse ones, so the savings need no sparse kernels.
 *
 * <p>This version uses one global rotation (the paper's per-block rotations with shortcut
 * matrices squeeze out a little more) and works on chains of {@link ResidualMixerBlock}s
 * around core {@link MultiHeadAttention} and {@link GluFeedForward} (attention's inner widths
 * stay as they are; only the stream is sliced). The new blocks' norm gains are
 * {@code sqrt(d'/d)}, which corrects the RMS of the shorter vector, and the readout, which is
 * assumed to follow a gain-free RMS norm, gets the same factor.
 *
 * <p>Ashkboos et al., "SliceGPT: Compress Large Language Models by Deleting Rows and Columns"
 * (ICLR 2024), <a href="https://arxiv.org/abs/2401.15024">arXiv:2401.15024</a>.
 */
public final class SliceCompressor {

    /**
     * A sliced model.
     *
     * @param embeddings the sliced input embeddings {@code [vocab, d']}, in the order given
     * @param body the sliced blocks
     * @param readout the sliced readout {@code [d', out]}
     * @param rotation the kept principal directions {@code [d, d']}
     */
    public record Sliced(List<Tensor> embeddings, LayerChain body, Tensor readout, Tensor rotation) {
    }

    private SliceCompressor() {
    }

    /**
     * Slices a model.
     *
     * @param embeddings input embeddings {@code [vocab, d]} whose rows are summed into the stream
     * @param body the blocks
     * @param readout the readout {@code [d, out]} after a gain-free RMS norm
     * @param input the body's input on calibration data {@code [batch, steps, d]}
     * @param keep the new width {@code d'}
     * @return the sliced model
     */
    public static Sliced slice(List<Var> embeddings, LayerChain body, Var readout, Tensor input, int keep) {
        int d = input.shape().dim(2);
        if (keep < 1 || keep > d) {
            throw new IllegalArgumentException("keep must be in [1, " + d + "]");
        }
        Tensor q = rotation(body, input, keep);                                    // [d, d']
        Tensor qt = q.transpose();
        float factor = (float) Math.sqrt(keep / (double) d);
        List<Tensor> emb = new ArrayList<>();
        for (Var e : embeddings) {
            emb.add(e.value().matmul(q));
        }
        List<Layer> blocks = new ArrayList<>();
        SplittableRandom random = new SplittableRandom(0);
        for (Layer l : body.layers()) {
            ResidualMixerBlock b = block(l);
            MultiHeadAttention mha = (MultiHeadAttention) b.mixer();
            GluFeedForward ffn = (GluFeedForward) b.ffn();
            List<Var> bp = b.parameters();
            Tensor g1 = bp.get(0).value().reshape(d, 1);
            Tensor g2 = bp.get(1).value().reshape(d, 1);
            MultiHeadAttention.Settings s = mha.settings();
            float scale = s.scale() != 0 ? s.scale() : (float) (1 / Math.sqrt(s.headSize()));
            MultiHeadAttention.Settings ns = new MultiHeadAttention.Settings(keep, keep, s.heads(), s.kvHeads(),
                    s.headSize(), s.valueSize(), keep, false, s.causal(), s.rotaryBase(), s.alibi(), 0, scale, false);
            MultiHeadAttention nm = new MultiHeadAttention(ns, Init.XAVIER, random);
            List<Var> mp = mha.parameters();
            List<Var> np = nm.parameters();
            for (int i = 0; i < 3; i++) {   // wq, wk, wv read the stream
                np.get(i).assign(qt.matmul(mp.get(i).value().mul(g1)));
            }
            np.get(3).assign(mp.get(3).value().matmul(q));                         // wo writes it
            List<Var> fp = ffn.parameters();   // up, down, gate
            int hidden = fp.get(0).value().shape().dim(1);
            GluFeedForward nf = new GluFeedForward(ffn.kind(), keep, hidden, random);
            List<Var> nfp = nf.parameters();
            nfp.get(0).assign(qt.matmul(fp.get(0).value().mul(g2)));
            nfp.get(1).assign(fp.get(1).value().matmul(q));
            if (fp.size() > 2) {
                nfp.get(2).assign(qt.matmul(fp.get(2).value().mul(g2)));
            }
            ResidualMixerBlock nb = new ResidualMixerBlock(keep, nm, nf, null, null);
            List<Var> gains = nb.parameters();
            gains.get(0).assign(Tensor.full(Shape.of(keep), factor));
            gains.get(1).assign(Tensor.full(Shape.of(keep), factor));
            blocks.add(nb);
        }
        Tensor r = qt.matmul(readout.value()).mul(factor);
        return new Sliced(emb, new LayerChain(blocks), r, q);
    }

    /**
     * The principal directions of the residual stream: eigenvectors of the summed
     * second-moment matrix of every stream state (each block's input, the state between its
     * mixer and feed-forward layer, and the output).
     *
     * @param body the blocks
     * @param input the body's input {@code [batch, steps, d]}
     * @param keep the directions kept
     * @return {@code [d, keep]}, largest first
     */
    public static Tensor rotation(LayerChain body, Tensor input, int keep) {
        int d = input.shape().dim(2);
        int rows = input.shape().dim(0) * input.shape().dim(1);
        Var x = Var.of(input);
        Tensor cov = Tensor.zeros(Shape.of(d, d));
        for (Layer l : body.layers()) {
            ResidualMixerBlock b = block(l);
            Var a = x.div(x.square().mean(-1, true).add(1e-6f).sqrt()).mul(b.parameters().get(0));
            Var h = x.add(b.mixer().forward(a));
            for (Var state : new Var[] {x, h}) {
                Tensor m = state.value().reshape(rows, d);
                cov = cov.add(m.transpose().matmul(m));
            }
            x = b.forward(x);
        }
        Tensor m = x.value().reshape(rows, d);
        cov = cov.add(m.transpose().matmul(m));
        float[] v = Linalg.eigh(cov)[1].toArray();
        float[] q = new float[d * keep];
        for (int k = 0; k < keep; k++) {
            int col = d - 1 - k;
            for (int i = 0; i < d; i++) {
                q[i * keep + k] = v[i * d + col];
            }
        }
        return Tensor.of(Shape.of(d, keep), q);
    }

    private static ResidualMixerBlock block(Layer l) {
        if (l instanceof ResidualMixerBlock b && b.mixer() instanceof MultiHeadAttention
                && b.ffn() instanceof GluFeedForward) {
            return b;
        }
        throw new IllegalArgumentException("expected ResidualMixerBlocks around MultiHeadAttention and GluFeedForward, got "
                + l);
    }
}
