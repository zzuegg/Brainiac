package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.GlobalPool;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Padding;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A small 1-D convolutional binary classifier from core layers: strided ReLU convolutions,
 * global max pooling and a linear read-out to one logit; trained by logistic loss.
 */
final class Cnn1d {

    private final List<Layer> convs = new ArrayList<>();
    private final Layer pool = GlobalPool.max();
    private final Var head;
    private final Var bias;
    private final int channels;

    Cnn1d(int channels, int[] filters, int kernel, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.channels = channels;
        int c = channels;
        int len = 256;
        for (int f : filters) {
            convs.add(Conv.of(f, kernel).stride(2).padding(Padding.SAME).activation(Activation.RELU)
                    .create(Shape.of(c, len), r));
            c = f;
            len = (len + 1) / 2;
        }
        float[] w = new float[c];
        for (int i = 0; i < c; i++) {
            w[i] = (float) (0.1 * r.nextGaussian());
        }
        head = Var.param(Tensor.of(Shape.of(c, 1), w));
        bias = Var.param(Tensor.zeros(Shape.of(1)));
    }

    List<Var> featureParameters() {
        List<Var> p = new ArrayList<>();
        for (Layer l : convs) {
            p.addAll(l.parameters());
        }
        return p;
    }

    int width() {
        return head.shape().dim(0);
    }

    List<Var> parameters() {
        List<Var> p = featureParameters();
        p.add(head);
        p.add(bias);
        return p;
    }

    Var features(Var x) {
        Var h = x;
        for (Layer l : convs) {
            h = l.forward(h);
        }
        return pool.forward(h);
    }

    Var logits(Var x) {
        return features(x).matmul(head).add(bias);
    }

    static Tensor batch(float[][][] x, int[] index) {
        int c = x[0].length;
        int l = x[0][0].length;
        float[] v = new float[index.length * c * l];
        for (int b = 0; b < index.length; b++) {
            for (int k = 0; k < c; k++) {
                System.arraycopy(x[index[b]][k], 0, v, (b * c + k) * l, l);
            }
        }
        return Tensor.of(Shape.of(index.length, c, l), v);
    }

    Cnn1d train(float[][][] x, int[] y, int steps, int batch, double lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam((float) lr);
        List<Var> params = parameters();
        for (int s = 0; s < steps; s++) {
            int[] idx = new int[batch];
            float[] t = new float[batch];
            for (int b = 0; b < batch; b++) {
                idx[b] = r.nextInt(x.length);
                t[b] = y[idx[b]];
            }
            Tensor xt = batch(x, idx);
            Tensor yt = Tensor.of(Shape.of(batch, 1), t);
            opt.minimize(params, () -> {
                Var z = logits(Var.of(xt));
                return z.softplus().sub(z.mul(Var.of(yt))).mean();
            });
        }
        return this;
    }

    double[] scores(float[][][] x) {
        double[] out = new double[x.length];
        for (int from = 0; from < x.length; from += 64) {
            int[] idx = new int[Math.min(64, x.length - from)];
            for (int i = 0; i < idx.length; i++) {
                idx[i] = from + i;
            }
            Tensor xt = batch(x, idx);
            float[] z = Var.noGrad(() -> logits(Var.of(xt))).value().toArray();
            for (int i = 0; i < idx.length; i++) {
                out[from + i] = z[i];
            }
        }
        return out;
    }

    int channels() {
        return channels;
    }
}
