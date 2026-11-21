package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.audio.ChatterboxNets.BatchNorm;
import io.github.zzuegg.brainiac.audio.ChatterboxNets.Conv;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

import static io.github.zzuegg.brainiac.audio.ChatterboxNets.batchNorm;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.conv;

/**
 * The CAM++ speaker verification network (3D-Speaker) that S3Gen reads a voice's timbre with:
 * Kaldi filterbanks of 16 kHz speech, a 2-D residual front end, densely connected TDNN blocks
 * with context-aware masking, and statistics pooling to one embedding.
 */
final class CamPlusEncoder {

    private static final int[] KERNELS = {3, 3, 3};
    private static final int[] DILATIONS = {1, 2, 2};
    private static final int MELS = 80;

    private record Res(Conv conv1, BatchNorm bn1, Conv conv2, BatchNorm bn2, Conv shortcut, BatchNorm shortcutBn,
            int stride) {
    }

    private record Dense(BatchNorm bn1, Conv lin1, BatchNorm bn2, Conv local, Conv cam1, Conv cam2, int dilation) {
    }

    private record Transit(BatchNorm bn, Conv lin) {
    }

    private final Device device;
    private final DType dtype;
    private final Conv headConv1;
    private final BatchNorm headBn1;
    private final List<Res> res = new ArrayList<>();
    private final Conv headConv2;
    private final BatchNorm headBn2;
    private final Conv tdnn;
    private final BatchNorm tdnnBn;
    private final List<List<Dense>> blocks = new ArrayList<>();
    private final List<Transit> transits = new ArrayList<>();
    private final BatchNorm outBn;
    private final Conv dense;
    private final BatchNorm denseBn;

    CamPlusEncoder(Weights w, String p) {
        device = w.device();
        dtype = w.dtype();
        String h = p + "head.";
        headConv1 = conv(w, h + "conv1");
        headBn1 = batchNorm(w, h + "bn1");
        for (String layer : new String[] {"layer1", "layer2"}) {
            for (int i = 0; w.contains(h + layer + "." + i + ".conv1.weight"); i++) {
                String b = h + layer + "." + i + ".";
                boolean sc = w.contains(b + "shortcut.0.weight");
                res.add(new Res(conv(w, b + "conv1"), batchNorm(w, b + "bn1"), conv(w, b + "conv2"),
                        batchNorm(w, b + "bn2"), sc ? conv(w, b + "shortcut.0") : null,
                        sc ? batchNorm(w, b + "shortcut.1") : null, i == 0 ? 2 : 1));
            }
        }
        headConv2 = conv(w, h + "conv2");
        headBn2 = batchNorm(w, h + "bn2");
        String x = p + "xvector.";
        tdnn = conv(w, x + "tdnn.linear");
        tdnnBn = batchNorm(w, x + "tdnn.nonlinear.batchnorm");
        for (int b = 1; w.contains(x + "block" + b + ".tdnnd1.linear1.weight"); b++) {
            List<Dense> layers = new ArrayList<>();
            for (int i = 1; w.contains(x + "block" + b + ".tdnnd" + i + ".linear1.weight"); i++) {
                String l = x + "block" + b + ".tdnnd" + i + ".";
                layers.add(new Dense(batchNorm(w, l + "nonlinear1.batchnorm"), conv(w, l + "linear1"),
                        batchNorm(w, l + "nonlinear2.batchnorm"), conv(w, l + "cam_layer.linear_local"),
                        conv(w, l + "cam_layer.linear1"), conv(w, l + "cam_layer.linear2"), DILATIONS[b - 1]));
            }
            blocks.add(layers);
            transits.add(new Transit(batchNorm(w, x + "transit" + b + ".nonlinear.batchnorm"),
                    conv(w, x + "transit" + b + ".linear")));
        }
        outBn = batchNorm(w, x + "out_nonlinear.batchnorm");
        dense = conv(w, x + "dense.linear");
        denseBn = batchNorm(w, x + "dense.nonlinear.batchnorm");
    }

    /**
     * The speaker embedding of some speech.
     *
     * @param audio 16 kHz samples, at least 25 ms
     * @return the embedding (192 values in the real model)
     */
    float[] embed(float[] audio) {
        float[][] fb = fbank(audio);
        int frames = fb.length;
        float[] flat = new float[MELS * frames];
        for (int m = 0; m < MELS; m++) {
            double mean = 0;
            for (float[] f : fb) {
                mean += f[m];
            }
            mean /= frames;
            for (int t = 0; t < frames; t++) {
                flat[m * frames + t] = (float) (fb[t][m] - mean);
            }
        }
        return Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(1, 1, MELS, frames), flat, device).to(dtype)))
                .to(DType.FLOAT32).value().toArray());
    }

    private Var forward(Var x) {
        x = headBn1.apply(headConv1.apply2d(x, 1, 1, 1)).relu();
        for (Res r : res) {
            Var o = r.bn1().apply(r.conv1().apply2d(x, r.stride(), 1, 1)).relu();
            o = r.bn2().apply(r.conv2().apply2d(o, 1, 1, 1));
            Var s = r.shortcut() == null ? x : r.shortcutBn().apply(r.shortcut().apply2d(x, r.stride(), 1, 0));
            x = Vocos.settle(o.add(s).relu());
        }
        x = headBn2.apply(headConv2.apply2d(x, 2, 1, 1)).relu();
        Shape s = x.shape();
        x = x.reshape(1, s.dim(1) * s.dim(2), s.dim(3));
        x = tdnnBn.apply(tdnn.apply(x, 2, 2, 2, 1)).relu();
        for (int b = 0; b < blocks.size(); b++) {
            for (Dense d : blocks.get(b)) {
                Var h = d.lin1().apply(d.bn1().apply(x).relu(), 1, 0, 0, 1);
                h = d.bn2().apply(h).relu();
                x = Vocos.settle(Var.concat(1, x, cam(h, d)));
            }
            Transit t = transits.get(b);
            x = t.lin().apply(t.bn().apply(x).relu(), 1, 0, 0, 1);
        }
        x = outBn.apply(x).relu();
        Var stats = Var.concat(1, x.mean(2, false), x.std(2, 1, false));       // [1, 2c]
        Var e = dense.apply(stats.reshape(1, stats.shape().dim(1), 1), 1, 0, 0, 1);
        return denseBn.apply(e).reshape(e.shape().dim(1));
    }

    /** Context-aware masking: a local kernel gated by the utterance and 100-frame segment means. */
    private Var cam(Var x, Dense d) {
        Var y = d.local().same(x, d.dilation());
        int t = x.shape().dim(2);
        Var context = x.mean(2, true);
        List<Var> segs = new ArrayList<>();
        for (int s = 0; s < t; s += 100) {
            int e = Math.min(t, s + 100);
            segs.add(x.slice(2, s, e).mean(2, true).expand(Shape.of(1, x.shape().dim(1), e - s)));
        }
        Var seg = segs.size() == 1 ? segs.get(0) : Var.concat(2, segs.toArray(Var[]::new));
        Var c = context.add(seg);
        c = d.cam1().apply(c, 1, 0, 0, 1).relu();
        Var m = d.cam2().apply(c, 1, 0, 0, 1).sigmoid();
        return y.mul(m);
    }

    /**
     * Kaldi's filterbank features as torchaudio computes them with its defaults: 25 ms Povey
     * windows every 10 ms, DC removed, pre-emphasis 0.97, a 512-point power spectrum, 80
     * triangular bands from 20 Hz on Kaldi's mel scale, natural log.
     *
     * @param wave 16 kHz samples
     * @return {@code [frames][80]}
     */
    static float[][] fbank(float[] wave) {
        int win = 400;
        int shift = 160;
        int padded = 512;
        int frames = 1 + (wave.length - win) / shift;
        if (frames < 1) {
            throw new IllegalArgumentException("the recording is too short");
        }
        double[] window = new double[win];
        for (int i = 0; i < win; i++) {
            window[i] = Math.pow(0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win - 1)), 0.85);
        }
        int bins = padded / 2;
        double low = kaldiMel(20);
        double high = kaldiMel(SAMPLE_RATE_16K / 2.0);
        double delta = (high - low) / (MELS + 1);
        double[][] bank = new double[MELS][bins];
        for (int m = 0; m < MELS; m++) {
            double left = low + m * delta;
            double centre = low + (m + 1) * delta;
            double right = low + (m + 2) * delta;
            for (int k = 0; k < bins; k++) {
                double f = kaldiMel(k * (double) SAMPLE_RATE_16K / padded);
                if (f > left && f < right) {
                    bank[m][k] = f <= centre ? (f - left) / (centre - left) : (right - f) / (right - centre);
                }
            }
        }
        double eps = Math.ulp(1.0f);
        float[][] out = new float[frames][MELS];
        double[] re = new double[padded];
        double[] im = new double[padded];
        double[] frame = new double[win];
        for (int t = 0; t < frames; t++) {
            double mean = 0;
            for (int i = 0; i < win; i++) {
                frame[i] = wave[t * shift + i];
                mean += frame[i];
            }
            mean /= win;
            for (int i = 0; i < win; i++) {
                frame[i] -= mean;
            }
            for (int i = win - 1; i >= 0; i--) {
                frame[i] -= 0.97 * frame[Math.max(0, i - 1)];
            }
            java.util.Arrays.fill(re, 0);
            java.util.Arrays.fill(im, 0);
            for (int i = 0; i < win; i++) {
                re[i] = frame[i] * window[i];
            }
            fft(re, im);
            for (int m = 0; m < MELS; m++) {
                double e = 0;
                for (int k = 0; k < bins; k++) {
                    if (bank[m][k] != 0) {
                        e += bank[m][k] * (re[k] * re[k] + im[k] * im[k]);
                    }
                }
                out[t][m] = (float) Math.log(Math.max(e, eps));
            }
        }
        return out;
    }

    private static final int SAMPLE_RATE_16K = 16_000;

    private static double kaldiMel(double hz) {
        return 1127.0 * Math.log(1.0 + hz / 700.0);
    }

    /** An in-place radix-2 FFT. */
    private static void fft(double[] re, double[] im) {
        int n = re.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            if (i < j) {
                double t = re[i];
                re[i] = re[j];
                re[j] = t;
                t = im[i];
                im[i] = im[j];
                im[j] = t;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2 * Math.PI / len;
            for (int i = 0; i < n; i += len) {
                for (int k = 0; k < len / 2; k++) {
                    double c = Math.cos(ang * k);
                    double s = Math.sin(ang * k);
                    double ur = re[i + k];
                    double ui = im[i + k];
                    double vr = re[i + k + len / 2] * c - im[i + k + len / 2] * s;
                    double vi = re[i + k + len / 2] * s + im[i + k + len / 2] * c;
                    re[i + k] = ur + vr;
                    im[i + k] = ui + vi;
                    re[i + k + len / 2] = ur - vr;
                    im[i + k + len / 2] = ui - vi;
                }
            }
        }
    }
}
