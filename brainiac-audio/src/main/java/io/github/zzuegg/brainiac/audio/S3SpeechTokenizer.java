package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.audio.ChatterboxNets.Conv;
import io.github.zzuegg.brainiac.audio.ChatterboxNets.Lin;
import io.github.zzuegg.brainiac.audio.ChatterboxNets.Norm;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

import static io.github.zzuegg.brainiac.audio.ChatterboxNets.conv;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.layerNorm;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.lin;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.norm;

/**
 * CosyVoice 2's S3 speech tokenizer (v2, 25 Hz), as Chatterbox carries it: a 128-band log-mel
 * of 16 kHz speech, two strided convolutions, transformer blocks with rotary positions and an
 * FSMN memory, then finite scalar quantization of 8 values to 3 levels: one of 6561 codes per
 * 40 ms.
 */
final class S3SpeechTokenizer {

    static final int SAMPLE_RATE = 16_000;
    private static final int N_FFT = 400;
    private static final int HOP = 160;
    private static final int HEAD_DIM = 64;

    private record Block(Norm attnLn, Lin q, Lin k, Lin v, Lin out, Conv fsmn, Norm mlpLn, Lin mlp1, Lin mlp2) {
    }

    private final Device device;
    private final DType dtype;
    private final Tensor melBasis;
    private final Conv conv1;
    private final Conv conv2;
    private final List<Block> blocks = new ArrayList<>();
    private final Lin down;

    S3SpeechTokenizer(Weights w, String p) {
        device = w.device();
        dtype = w.dtype();
        String e = p + "encoder.";
        conv1 = conv(w, e + "conv1");
        conv2 = conv(w, e + "conv2");
        for (int i = 0; w.contains(e + "blocks." + i + ".attn_ln.weight"); i++) {
            String b = e + "blocks." + i + ".";
            blocks.add(new Block(norm(w, b + "attn_ln"), lin(w, b + "attn.query"), lin(w, b + "attn.key"),
                    lin(w, b + "attn.value"), lin(w, b + "attn.out"), conv(w, b + "attn.fsmn_block"),
                    norm(w, b + "mlp_ln"), lin(w, b + "mlp.0"), lin(w, b + "mlp.2")));
        }
        down = lin(w, p + "quantizer._codebook.project_down");
        int mels = conv1.w().shape().dim(1);
        melBasis = Signal.melFilterbank(SAMPLE_RATE, N_FFT, mels, 0, SAMPLE_RATE / 2.0, Signal.MelScale.SLANEY, true)
                .to(device);
    }

    /**
     * The log-mel the tokenizer reads: Whisper's, {@code [mels, samples / 160]}.
     *
     * @param audio 16 kHz samples
     */
    Var mel(float[] audio) {
        Var x = Var.of(Tensor.of(Shape.of(audio.length), audio, device));
        Var spec = Signal.stft(x, N_FFT, HOP);                                // [frames + 1, 201, 2]
        int frames = spec.shape().dim(0) - 1;
        Var power = spec.slice(0, 0, frames).square().sum(2, false);         // [frames, 201]
        Var m = power.matmul(Var.of(melBasis)).clamp(1e-10f, Float.MAX_VALUE).log().mul((float) (1 / Math.log(10)));
        float max = m.max(1, false).max(0, false).value().toArray()[0];
        return m.maximum(max - 8f).add(4f).mul(0.25f).permute(1, 0);
    }

    /**
     * The codes of some speech.
     *
     * @param audio 16 kHz samples
     * @param maxTokens at most this many codes (the mel is cut to 4 frames per code), or 0
     * @return one code in {@code [0, 6561)} per 40 ms
     */
    int[] encode(float[] audio, int maxTokens) {
        return Var.noGrad(() -> {
            Var mel = mel(audio);
            if (maxTokens > 0 && mel.shape().dim(1) > 4 * maxTokens) {
                mel = mel.slice(1, 0, 4 * maxTokens);
            }
            Var x = mel.reshape(1, mel.shape().dim(0), mel.shape().dim(1)).to(dtype);
            x = conv1.apply(x, 2, 1, 1, 1).gelu();
            x = conv2.apply(x, 2, 1, 1, 1).gelu().permute(0, 2, 1);           // [1, t, state]
            for (Block b : blocks) {
                x = Vocos.settle(x.add(attention(layerNorm(x, b.attnLn(), 1e-5f), b)));
                Var h = layerNorm(x, b.mlpLn(), 1e-5f);
                x = Vocos.settle(x.add(b.mlp2().apply(b.mlp1().apply(h).gelu())));
            }
            float[] h = down.apply(x).tanh().to(DType.FLOAT32).value().toArray();
            int t = x.shape().dim(1);
            int levels = down.out();
            int[] codes = new int[t];
            for (int i = 0; i < t; i++) {
                int code = 0;
                int pow = 1;
                for (int j = 0; j < levels; j++) {
                    code += (int) (Math.rint(h[i * levels + j] * 0.9990000128746033f) + 1) * pow;
                    pow *= 3;
                }
                codes[i] = code;
            }
            return codes;
        });
    }

    private Var attention(Var x, Block b) {
        int t = x.shape().dim(1);
        int d = x.shape().dim(2);
        int heads = d / HEAD_DIM;
        Var v = b.v().apply(x);
        // the FSMN memory: a depthwise kernel over the values, plus the values
        Var vt = v.permute(0, 2, 1);
        int k = b.fsmn().kernel();
        int left = (k - 1) / 2;
        Var mem = b.fsmn().apply(vt, 1, left, k - 1 - left, 1).add(vt).permute(0, 2, 1);
        Var q = Functional.rotary(ChatterboxNets.heads(b.q().apply(x), heads), 10000f, 0);
        Var kk = Functional.rotary(ChatterboxNets.heads(b.k().apply(x), heads), 10000f, 0);
        Var o = Functional.attention(q, kk, ChatterboxNets.heads(v, heads), false, (float) (1 / Math.sqrt(HEAD_DIM)));
        return b.out().apply(ChatterboxNets.merge(o)).add(mem);
    }
}
