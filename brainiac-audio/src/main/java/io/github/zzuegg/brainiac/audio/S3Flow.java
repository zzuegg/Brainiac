package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.audio.ChatterboxNets.Conv;
import io.github.zzuegg.brainiac.audio.ChatterboxNets.Lin;
import io.github.zzuegg.brainiac.audio.ChatterboxNets.Norm;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

import static io.github.zzuegg.brainiac.audio.ChatterboxNets.channelNorm;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.conv;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.layerNorm;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.lin;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.norm;

/**
 * The token-to-mel half of Chatterbox's S3Gen (CosyVoice 2's causal flow matching): speech
 * tokens, after the reference's, go through an upsampling conformer to a 50 Hz mel prior, and
 * a conditional U-Net estimator integrates noise to an 80-band mel in ten cosine-spaced Euler
 * steps with classifier-free guidance, continuing the reference's mel.
 */
final class S3Flow {

    static final int MELS = 80;
    private static final int DECODER_HEAD_DIM = 64;
    private static final float CFG_RATE = 0.7f;

    private record Attention(Lin q, Lin k, Lin v, Lin out, Lin pos, Var biasU, Var biasV, int heads) {
    }

    private record Conformer(Attention attn, Lin ff1, Lin ff2, Norm normMha, Norm normFf) {
    }

    private record Embed(Lin lin, Norm norm) {
    }

    private record Resnet(Lin mlp, Conv conv1, Norm norm1, Conv conv2, Norm norm2, Conv res) {
    }

    private record Block(Norm norm1, Lin q, Lin k, Lin v, Lin out, Norm norm3, Lin ff1, Lin ff2) {
    }

    private record Stage(Resnet resnet, List<Block> blocks, Conv sample) {
    }

    private final Device device;
    private final Var inputEmbedding;
    private final Lin spkAffine;
    private final Embed embed;
    private final Conv look1;
    private final Conv look2;
    private final List<Conformer> encoders = new ArrayList<>();
    private final Conv up;
    private final Embed upEmbed;
    private final List<Conformer> upEncoders = new ArrayList<>();
    private final Norm afterNorm;
    private final Lin encoderProj;
    private final Lin time1;
    private final Lin time2;
    private final int timeChannels;
    private final Stage down;
    private final List<Stage> mids = new ArrayList<>();
    private final Stage upStage;
    private final Conv finalConv;
    private final Norm finalNorm;
    private final Conv finalProj;

    S3Flow(Weights w, String p) {
        device = w.device();
        inputEmbedding = w.read(p + "input_embedding.weight");
        spkAffine = lin(w, p + "spk_embed_affine_layer");
        String e = p + "encoder.";
        embed = new Embed(lin(w, e + "embed.out.0"), norm(w, e + "embed.out.1"));
        look1 = conv(w, e + "pre_lookahead_layer.conv1");
        look2 = conv(w, e + "pre_lookahead_layer.conv2");
        for (int i = 0; w.contains(e + "encoders." + i + ".norm_mha.weight"); i++) {
            encoders.add(conformer(w, e + "encoders." + i + "."));
        }
        up = conv(w, e + "up_layer.conv");
        upEmbed = new Embed(lin(w, e + "up_embed.out.0"), norm(w, e + "up_embed.out.1"));
        for (int i = 0; w.contains(e + "up_encoders." + i + ".norm_mha.weight"); i++) {
            upEncoders.add(conformer(w, e + "up_encoders." + i + "."));
        }
        afterNorm = norm(w, e + "after_norm");
        encoderProj = lin(w, p + "encoder_proj");
        String d = p + "decoder.estimator.";
        time1 = lin(w, d + "time_mlp.linear_1");
        time2 = lin(w, d + "time_mlp.linear_2");
        timeChannels = w.shape(d + "time_mlp.linear_1.weight").dim(1);
        if (w.contains(d + "down_blocks.1.0.res_conv.weight")) {
            throw new IllegalArgumentException("only a one-level flow estimator (CosyVoice 2's) is supported");
        }
        down = stage(w, d + "down_blocks.0.", true);
        for (int i = 0; w.contains(d + "mid_blocks." + i + ".0.res_conv.weight"); i++) {
            mids.add(stage(w, d + "mid_blocks." + i + ".", false));
        }
        upStage = stage(w, d + "up_blocks.0.", true);
        finalConv = conv(w, d + "final_block.block.0");
        finalNorm = norm(w, d + "final_block.block.2");
        finalProj = conv(w, d + "final_proj");
    }

    private static Conformer conformer(Weights w, String p) {
        String a = p + "self_attn.";
        Var u = w.read(a + "pos_bias_u");
        Attention attn = new Attention(lin(w, a + "linear_q"), lin(w, a + "linear_k"), lin(w, a + "linear_v"),
                lin(w, a + "linear_out"), lin(w, a + "linear_pos"), u, w.read(a + "pos_bias_v"), u.shape().dim(0));
        return new Conformer(attn, lin(w, p + "feed_forward.w_1"), lin(w, p + "feed_forward.w_2"),
                norm(w, p + "norm_mha"), norm(w, p + "norm_ff"));
    }

    private static Stage stage(Weights w, String p, boolean sample) {
        String r = p + "0.";
        Resnet res = new Resnet(lin(w, r + "mlp.1"), conv(w, r + "block1.block.0"), norm(w, r + "block1.block.2"),
                conv(w, r + "block2.block.0"), norm(w, r + "block2.block.2"), conv(w, r + "res_conv"));
        List<Block> blocks = new ArrayList<>();
        for (int i = 0; w.contains(p + "1." + i + ".norm1.weight"); i++) {
            String b = p + "1." + i + ".";
            blocks.add(new Block(norm(w, b + "norm1"), lin(w, b + "attn1.to_q"), lin(w, b + "attn1.to_k"),
                    lin(w, b + "attn1.to_v"), lin(w, b + "attn1.to_out.0"), norm(w, b + "norm3"),
                    lin(w, b + "ff.net.0.proj"), lin(w, b + "ff.net.2")));
        }
        return new Stage(res, blocks, sample ? conv(w, p + "2") : null);
    }

    // ---------------------------------------------------------------- encoder

    /**
     * The mel prior of tokens: {@code [1, 2 * tokens, 80]}.
     *
     * @param tokens the reference's tokens then the new ones
     */
    Var encode(int[] tokens) {
        Var x = ChatterboxNets.rows(inputEmbedding, tokens).reshape(1, tokens.length, inputEmbedding.shape().dim(1));
        x = embed(x, embed);
        Var pos = position(x.shape().dim(1), x.shape().dim(2), x.dtype());
        // pre-lookahead: three future tokens, then a causal kernel
        Var t = x.permute(0, 2, 1);
        Var h = look1.apply(t, 1, 0, look1.kernel() - 1, 1).leakyRelu(0.01f);
        h = look2.apply(h, 1, look2.kernel() - 1, 0, 1);
        x = Vocos.settle(h.permute(0, 2, 1).add(x));
        for (Conformer c : encoders) {
            x = Vocos.settle(conformer(x, pos, c));
        }
        // nearest-neighbour upsampling by 2 and a causal kernel
        Var u = x.permute(0, 2, 1).repeatInterleave(2, 2);
        int k = up.kernel();
        u = up.apply(u, 1, k - 1, 0, 1);
        x = embed(u.permute(0, 2, 1), upEmbed);
        pos = position(x.shape().dim(1), x.shape().dim(2), x.dtype());
        for (Conformer c : upEncoders) {
            x = Vocos.settle(conformer(x, pos, c));
        }
        x = layerNorm(x, afterNorm, 1e-5f);
        return encoderProj.apply(x);
    }

    private static Var embed(Var x, Embed e) {
        int d = e.lin().out();
        return layerNorm(e.lin().apply(x), e.norm(), 1e-5f).mul((float) Math.sqrt(d));
    }

    /** ESPnet's relative positions: {@code [2t - 1, d]}, from {@code t - 1} down to {@code -(t - 1)}. */
    private Var position(int t, int d, io.github.zzuegg.brainiac.tensor.DType dtype) {
        float[] pe = new float[(2 * t - 1) * d];
        for (int r = 0; r < 2 * t - 1; r++) {
            int rel = t - 1 - r;
            for (int i = 0; i < d; i += 2) {
                double div = Math.exp(i * -(Math.log(10000.0) / d));
                float a = rel * (float) div;
                pe[r * d + i] = (float) Math.sin(a);
                pe[r * d + i + 1] = (float) Math.cos(a);
            }
        }
        return Var.of(Tensor.of(Shape.of(2 * t - 1, d), pe, device).to(dtype));
    }

    private Var conformer(Var x, Var pos, Conformer c) {
        x = x.add(relAttention(layerNorm(x, c.normMha(), 1e-12f), pos, c.attn()));
        Var h = layerNorm(x, c.normFf(), 1e-12f);
        return x.add(c.ff2().apply(c.ff1().apply(h).silu()));
    }

    /** Transformer-XL attention with ESPnet's relative positions. */
    private Var relAttention(Var x, Var pos, Attention a) {
        int t = x.shape().dim(1);
        int d = x.shape().dim(2);
        int h = a.heads();
        int dk = d / h;
        Var q = a.q().apply(x).reshape(t, h, dk);                              // [t, h, dk]
        Var k = a.k().apply(x).reshape(t, h, dk).permute(1, 2, 0);              // [h, dk, t]
        Var v = a.v().apply(x).reshape(t, h, dk).permute(1, 0, 2);              // [h, t, dk]
        Var p = a.pos().apply(pos).reshape(2 * t - 1, h, dk).permute(1, 2, 0);  // [h, dk, 2t-1]
        Var ac = q.add(a.biasU()).permute(1, 0, 2).matmul(k);                   // [h, t, t]
        Var bd = q.add(a.biasV()).permute(1, 0, 2).matmul(p);                   // [h, t, 2t-1]
        // relative shift: query i, key j reads relative position i - j, column t - 1 - i + j
        float[] idx = new float[h * t * t];
        for (int hh = 0; hh < h; hh++) {
            for (int i = 0; i < t; i++) {
                for (int j = 0; j < t; j++) {
                    idx[(hh * t + i) * t + j] = t - 1 - i + j;
                }
            }
        }
        bd = bd.takeAlong(Tensor.of(Shape.of(h, t, t), idx, device));
        Var w = ac.add(bd).mul((float) (1 / Math.sqrt(dk))).softmax(2);
        Var y = w.matmul(v).permute(1, 0, 2).reshape(1, t, d);
        return a.out().apply(y);
    }

    // ---------------------------------------------------------------- estimator

    /**
     * The mel after the reference's: Euler steps from {@code noise} with guidance.
     *
     * @param mu the prior {@code [1, frames, 80]}
     * @param speaker the speaker embedding (CAMPPlus), not yet normalized
     * @param promptFeat the reference's mel {@code [promptFrames, 80]}
     * @param noise {@code [80, frames]} standard normal values
     * @param steps the Euler steps, 10 in the reference
     * @return {@code [80, frames - promptFrames]}
     */
    Var decode(Var mu, float[] speaker, float[] promptFeat, int promptFrames, float[] noise, int steps) {
        int frames = mu.shape().dim(1);
        var dtype = mu.dtype();
        double norm = 0;
        for (float s : speaker) {
            norm += (double) s * s;
        }
        float[] sp = new float[speaker.length];
        for (int i = 0; i < sp.length; i++) {
            sp[i] = (float) (speaker[i] / Math.max(Math.sqrt(norm), 1e-12));
        }
        Var spk = spkAffine.apply(Var.of(Tensor.of(Shape.of(1, sp.length), sp, device).to(dtype)));   // [1, 80]
        float[] cond = new float[MELS * frames];
        for (int f = 0; f < promptFrames; f++) {
            for (int m = 0; m < MELS; m++) {
                cond[m * frames + f] = promptFeat[f * MELS + m];
            }
        }
        Var muT = mu.permute(0, 2, 1);                                              // [1, 80, frames]
        Var zeros = Var.of(Tensor.zeros(Shape.of(1, MELS, frames), device).to(dtype));
        Var condV = Var.of(Tensor.of(Shape.of(1, MELS, frames), cond, device).to(dtype));
        // the guided and unguided rows go through the estimator together
        Var muIn = Var.concat(0, muT, zeros);
        Var spkIn = Var.concat(0, spk, spk.mul(0f)).reshape(2, MELS, 1).expand(Shape.of(2, MELS, frames));
        Var condIn = Var.concat(0, condV, zeros);
        Var context = Vocos.settle(Var.concat(1, muIn, spkIn, condIn));             // [2, 240, frames]
        Var x = Var.of(Tensor.of(Shape.of(1, MELS, frames), noise, device).to(dtype));
        double[] span = new double[steps + 1];
        for (int i = 0; i <= steps; i++) {
            double t = (float) i / steps;
            span[i] = 1 - Math.cos(t * 0.5 * Math.PI);
        }
        for (int i = 0; i < steps; i++) {
            float t = (float) span[i];
            float dt = (float) span[i + 1] - t;
            Var in = Var.concat(1, Var.concat(0, x, x), context);
            Var d = estimate(in, t);
            Var guided = d.slice(0, 0, 1).mul(1 + CFG_RATE).sub(d.slice(0, 1, 2).mul(CFG_RATE));
            x = Vocos.settle(x.add(guided.mul(dt)));
        }
        return x.reshape(MELS, frames).slice(1, promptFrames, frames);
    }

    /** The estimator's velocity for {@code [2, 320, frames]} at time {@code t}. */
    private Var estimate(Var x, float t) {
        var dtype = x.dtype();
        int half = timeChannels / 2;
        float[] emb = new float[timeChannels];
        double step = Math.log(10000) / (half - 1);
        for (int i = 0; i < half; i++) {
            float f = (float) Math.exp(i * -step);
            float a = 1000f * t * f;
            emb[i] = (float) Math.sin(a);
            emb[half + i] = (float) Math.cos(a);
        }
        Var te = Var.of(Tensor.of(Shape.of(1, timeChannels), emb, device).to(dtype));
        te = time2.apply(time1.apply(te).silu());                                   // [1, temb]
        Var temb = Var.concat(0, te, te);
        Var h = stage(x, down, temb);
        Var skip = h;
        h = down.sample().causal(h);
        for (Stage m : mids) {
            h = stage(h, m, temb);
        }
        h = Var.concat(1, h.slice(2, 0, skip.shape().dim(2)), skip);
        h = stage(h, upStage, temb);
        h = upStage.sample().causal(h);
        h = channelNorm(finalConv.causal(h), finalNorm, 1e-5f).mish();
        return Vocos.settle(finalProj.apply(h, 1, 0, 0, 1));
    }

    private Var stage(Var x, Stage s, Var temb) {
        Resnet r = s.resnet();
        Var h = channelNorm(r.conv1().causal(x), r.norm1(), 1e-5f).mish();
        Var tt = r.mlp().apply(temb.mish());
        h = h.add(tt.reshape(tt.shape().dim(0), tt.shape().dim(1), 1));
        h = channelNorm(r.conv2().causal(h), r.norm2(), 1e-5f).mish();
        h = Vocos.settle(h.add(r.res().apply(x, 1, 0, 0, 1)));
        Var y = h.permute(0, 2, 1);
        for (Block b : s.blocks()) {
            Var n = layerNorm(y, b.norm1(), 1e-5f);
            int heads = b.q().out() / DECODER_HEAD_DIM;
            y = y.add(b.out().apply(ChatterboxNets.attend(b.q().apply(n), b.k().apply(n), b.v().apply(n), heads)));
            n = layerNorm(y, b.norm3(), 1e-5f);
            y = Vocos.settle(y.add(b.ff2().apply(b.ff1().apply(n).gelu())));
        }
        return y.permute(0, 2, 1);
    }
}
