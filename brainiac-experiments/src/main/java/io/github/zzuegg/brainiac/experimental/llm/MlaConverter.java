package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.LatentAttention;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.ResidualMixerBlock;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Converts a trained grouped-query attention layer (the core {@link MultiHeadAttention} with
 * one KV head and RoPE) into multi-head latent attention ({@link LatentAttention}), as TransMLA
 * and MHA2MLA do, so a model trained with GQA can switch to MLA's compressed cache and then
 * only needs a short fine-tune instead of pretraining from scratch:
 * <ol>
 * <li><em>Partial RoPE.</em> MLA rotates only a small decoupled part of every query and key.
 *     Of the head's {@code headSize / 2} rotary frequency pairs the converter keeps
 *     {@code ropeDim / 2}, uniformly subsampled so that their frequencies are exactly the
 *     ones {@code LatentAttention} uses for its {@code ropeDim}-wide rotary part; their query
 *     and key columns become {@code W_qr} and the shared {@code W_kr}. The other pairs lose
 *     their rotation (the error MHA2MLA's fine-tune repairs).</li>
 * <li><em>Joint low-rank KV.</em> The remaining (non-rotary) key columns and the value
 *     columns are stacked into {@code W = [W_k,nope | W_v]} and factored with the principal
 *     directions of {@code X W} on calibration inputs {@code X} (an activation-aware SVD):
 *     {@code W ~ (W V_r) V_r^T}, {@code W_dkv = W V_r}, and {@code V_r^T} split into
 *     {@code W_uk} and {@code W_uv}, copied to every head (one KV head shared by all). With
 *     {@code latent = 2 headSize - ropeDim} the factorization is exact and the cache is as
 *     large as GQA's; smaller latents compress it.</li>
 * <li>Query columns are rescaled so that the scores keep GQA's {@code 1 / sqrt(headSize)}
 *     under MLA's {@code 1 / sqrt(headSize + ropeDim)}.</li>
 * </ol>
 * {@code LatentAttention} RMS-normalizes the latent, which no linear map can undo, so the
 * latent gain is set to the mean latent RMS on the calibration inputs: tokens with a typical
 * latent norm are converted exactly, others are rescaled. That and the dropped rotations are
 * what the fine-tune has to absorb.
 *
 * <p>Meng et al., "TransMLA: Multi-Head Latent Attention Is All You Need" (2025),
 * <a href="https://arxiv.org/abs/2502.07864">arXiv:2502.07864</a>; Ji et al., "Towards Economical
 * Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs"
 * (2025), <a href="https://arxiv.org/abs/2502.14837">arXiv:2502.14837</a>.
 */
public final class MlaConverter {

    private MlaConverter() {
    }

    /**
     * Converts one attention layer.
     *
     * @param mha causal GQA with one KV head, RoPE, no biases, value size = head size = features / heads
     * @param latent the latent width
     * @param ropeDim the decoupled rotary width, even, dividing the head size
     * @param calibration the layer's inputs {@code [n, features]}
     * @return the latent attention
     */
    public static LatentAttention convert(MultiHeadAttention mha, int latent, int ropeDim, Tensor calibration) {
        MultiHeadAttention.Settings s = mha.settings();
        int d = s.queryFeatures();
        int heads = s.heads();
        int hs = s.headSize();
        if (s.kvHeads() != 1 || s.rotaryBase() <= 0 || s.bias() || s.valueSize() != hs || heads * hs != d
                || s.outputSize() != d) {
            throw new IllegalArgumentException("need one KV head, RoPE, no bias and headSize = valueSize = features / heads");
        }
        if (ropeDim < 2 || ropeDim % 2 != 0 || hs % ropeDim != 0) {
            throw new IllegalArgumentException("ropeDim must be even and divide the head size " + hs);
        }
        List<Var> p = mha.parameters();
        float[] wq = p.get(0).value().toArray();
        float[] wk = p.get(1).value().toArray();
        float[] wv = p.get(2).value().toArray();
        float[] wo = p.get(3).value().toArray();
        int half = hs / 2;
        int rh = ropeDim / 2;
        int[] rope = new int[ropeDim];         // head dims that keep their rotation, in MLA's half-split order
        boolean[] rotary = new boolean[hs];
        for (int j = 0; j < rh; j++) {
            int pair = j * hs / ropeDim;
            rope[j] = pair;
            rope[rh + j] = half + pair;
            rotary[pair] = true;
            rotary[half + pair] = true;
        }
        int nope = hs - ropeDim;
        int[] plain = new int[nope];
        for (int i = 0, k = 0; i < hs; i++) {
            if (!rotary[i]) {
                plain[k++] = i;
            }
        }
        float origScale = s.scale() != 0 ? s.scale() : (float) (1 / Math.sqrt(hs));
        float qScale = (float) (origScale * Math.sqrt(hs + ropeDim));

        // queries: non-rotary columns first in each head (padded with zeros), rotary ones to W_qr
        float[] nq = new float[d * d];
        float[] nqr = new float[d * heads * ropeDim];
        for (int r = 0; r < d; r++) {
            for (int h = 0; h < heads; h++) {
                for (int m = 0; m < nope; m++) {
                    nq[r * d + h * hs + m] = qScale * wq[r * d + h * hs + plain[m]];
                }
                for (int j = 0; j < ropeDim; j++) {
                    nqr[r * heads * ropeDim + h * ropeDim + j] = qScale * wq[r * d + h * hs + rope[j]];
                }
            }
        }
        float[] nkr = new float[d * ropeDim];
        int joint = nope + hs;
        double[][] w = new double[d][joint];   // [W_k,nope | W_v]
        for (int r = 0; r < d; r++) {
            for (int j = 0; j < ropeDim; j++) {
                nkr[r * ropeDim + j] = wk[r * hs + rope[j]];
            }
            for (int m = 0; m < nope; m++) {
                w[r][m] = wk[r * hs + plain[m]];
            }
            for (int m = 0; m < hs; m++) {
                w[r][nope + m] = wv[r * hs + m];
            }
        }
        // principal output directions of X W
        float[] x = calibration.toArray();
        int n = calibration.shape().dim(0);
        double[] xw = new double[n * joint];
        for (int i = 0; i < n; i++) {
            for (int r = 0; r < d; r++) {
                double xv = x[i * d + r];
                if (xv != 0) {
                    for (int c = 0; c < joint; c++) {
                        xw[i * joint + c] += xv * w[r][c];
                    }
                }
            }
        }
        double[] gram = new double[joint * joint];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < joint; a++) {
                double va = xw[i * joint + a];
                for (int b = 0; b < joint; b++) {
                    gram[a * joint + b] += va * xw[i * joint + b];
                }
            }
        }
        float[] vectors = Linalg.eigh(Tensor.of(Shape.of(joint, joint), toFloat(gram)))[1].toArray();
        double[][] basis = new double[latent][joint];       // V_r^T, largest first
        for (int k = 0; k < Math.min(latent, joint); k++) {
            int col = joint - 1 - k;
            for (int c = 0; c < joint; c++) {
                basis[k][c] = vectors[c * joint + col];
            }
        }
        float[] dkv = new float[d * latent];
        for (int r = 0; r < d; r++) {
            for (int k = 0; k < latent; k++) {
                double sum = 0;
                for (int c = 0; c < joint; c++) {
                    sum += w[r][c] * basis[k][c];
                }
                dkv[r * latent + k] = (float) sum;
            }
        }
        // latent RMS on the calibration inputs, for the gain
        double rms = 0;
        for (int i = 0; i < n; i++) {
            double sq = 0;
            for (int k = 0; k < latent; k++) {
                double z = 0;
                for (int r = 0; r < d; r++) {
                    z += x[i * d + r] * dkv[r * latent + k];
                }
                sq += z * z;
            }
            rms += Math.sqrt(sq / latent + 1e-6) / n;
        }
        float[] uk = new float[latent * d];
        float[] uv = new float[latent * d];
        for (int k = 0; k < latent; k++) {
            for (int h = 0; h < heads; h++) {
                for (int m = 0; m < nope; m++) {
                    uk[k * d + h * hs + m] = (float) basis[k][m];
                }
                for (int m = 0; m < hs; m++) {
                    uv[k * d + h * hs + m] = (float) basis[k][nope + m];
                }
            }
        }
        LatentAttention mla = new LatentAttention(d, heads, latent, ropeDim, s.rotaryBase(), new SplittableRandom(0));
        List<Var> q = mla.parameters();   // wq, wqr, wdkv, latentGain, wuk, wuv, wkr, wo
        q.get(0).assign(Tensor.of(Shape.of(d, d), nq));
        q.get(1).assign(Tensor.of(Shape.of(d, heads * ropeDim), nqr));
        q.get(2).assign(Tensor.of(Shape.of(d, latent), dkv));
        q.get(3).assign(Tensor.full(Shape.of(latent), (float) rms));
        q.get(4).assign(Tensor.of(Shape.of(latent, d), uk));
        q.get(5).assign(Tensor.of(Shape.of(latent, d), uv));
        q.get(6).assign(Tensor.of(Shape.of(d, ropeDim), nkr));
        q.get(7).assign(Tensor.of(Shape.of(d, d), wo));
        return mla;
    }

    /**
     * Converts every block of a chain of {@link ResidualMixerBlock}s whose mixer is a GQA
     * {@link MultiHeadAttention}, calibrating each on the original model's normalized inputs
     * to that block. The new blocks keep the norm gains (copied) and share the feed-forward
     * layers with the old ones (canon layers are not carried over).
     *
     * @param chain the trained body
     * @param latent the latent width
     * @param ropeDim the decoupled rotary width
     * @param input the body's input on calibration data {@code [batch, steps, features]}
     * @return the converted body
     */
    public static LayerChain convert(LayerChain chain, int latent, int ropeDim, Tensor input) {
        List<Layer> out = new ArrayList<>();
        Var x = Var.of(input);
        for (Layer l : chain.layers()) {
            if (!(l instanceof ResidualMixerBlock block) || !(block.mixer() instanceof MultiHeadAttention mha)) {
                throw new IllegalArgumentException("expected ResidualMixerBlocks around MultiHeadAttention, got " + l);
            }
            int features = input.shape().dim(2);
            List<Var> gains = block.parameters();
            Var a = x.div(x.square().mean(-1, true).add(1e-6f).sqrt()).mul(gains.get(0));
            int rows = input.shape().dim(0) * input.shape().dim(1);
            LatentAttention mla = convert(mha, latent, ropeDim, a.value().reshape(rows, features));
            ResidualMixerBlock converted = new ResidualMixerBlock(features, mla, block.ffn(), null, null);
            List<Var> newGains = converted.parameters();
            newGains.get(0).assign(gains.get(0).value());
            newGains.get(1).assign(gains.get(1).value());
            out.add(converted);
            x = block.forward(x);
        }
        return new LayerChain(out);
    }

    private static float[] toFloat(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return f;
    }
}
