package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A WaveNet stack (van den Oord et al., 2016) without causality, as VITS and Glow-TTS
 * use it: per layer a dilated convolution to twice the channels, a gate
 * {@code tanh(a) * sigmoid(b)}, and a 1x1 convolution split into a residual (added to the
 * input of the next layer) and a skip; the output is the sum of the skips. An optional
 * global conditioning vector (a speaker embedding) is added before each gate. Input and
 * output are {@code [channels, length]} per sample.
 */
public final class WaveNet implements Layer {

    private final int channels;
    private final int kernel;
    private final int dilationRate;
    private final Var[] inW;
    private final Var[] inB;
    private final Var[] rsW;
    private final Var[] rsB;
    private final Var condW;
    private final Var condB;

    /**
     * A new stack with random weights.
     *
     * @param channels the channels
     * @param kernel the kernel length (odd)
     * @param dilationRate layer {@code i} dilates by {@code dilationRate^i}
     * @param layers the number of layers
     * @param conditioning the width of the global conditioning, or 0 for none
     * @param random the random source
     */
    public WaveNet(int channels, int kernel, int dilationRate, int layers, int conditioning, SplittableRandom random) {
        this(kernel, dilationRate, weights(layers, i -> Shape.of(2 * channels, channels, kernel), random),
                zeros(layers, i -> 2 * channels),
                weights(layers, i -> Shape.of(i < layers - 1 ? 2 * channels : channels, channels, 1), random),
                zeros(layers, i -> i < layers - 1 ? 2 * channels : channels),
                conditioning > 0 ? Var.param(Tensor.normal(Shape.of(2 * channels * layers, conditioning, 1), 0, 0.02f,
                        random)) : null,
                conditioning > 0 ? Var.param(Tensor.zeros(Shape.of(2 * channels * layers))) : null);
    }

    /**
     * A stack from given weights.
     *
     * @param kernel the kernel length
     * @param dilationRate the dilation growth
     * @param inW the dilated convolutions, each {@code [2 channels, channels, kernel]}
     * @param inB their biases
     * @param rsW the residual-and-skip convolutions, each {@code [2 channels, channels, 1]}
     *     (the last {@code [channels, channels, 1]}: skip only)
     * @param rsB their biases
     * @param condW the conditioning projection {@code [2 channels layers, conditioning, 1]}, or {@code null}
     * @param condB its bias, or {@code null}
     */
    public WaveNet(int kernel, int dilationRate, Var[] inW, Var[] inB, Var[] rsW, Var[] rsB, Var condW, Var condB) {
        if (kernel % 2 == 0) {
            throw new IllegalArgumentException("WaveNet layers need an odd kernel, got " + kernel);
        }
        this.channels = inW[0].shape().dim(1);
        this.kernel = kernel;
        this.dilationRate = dilationRate;
        this.inW = inW.clone();
        this.inB = inB.clone();
        this.rsW = rsW.clone();
        this.rsB = rsB.clone();
        this.condW = condW;
        this.condB = condB;
    }

    private static Var[] weights(int n, java.util.function.IntFunction<Shape> shape, SplittableRandom random) {
        Var[] w = new Var[n];
        for (int i = 0; i < n; i++) {
            Shape s = shape.apply(i);
            w[i] = Var.param(Init.HE.create(s, s.dim(1) * s.dim(2), s.dim(0), random));
        }
        return w;
    }

    private static Var[] zeros(int n, java.util.function.IntUnaryOperator size) {
        Var[] b = new Var[n];
        for (int i = 0; i < n; i++) {
            b[i] = Var.param(Tensor.zeros(Shape.of(size.applyAsInt(i))));
        }
        return b;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, (Var) null);
    }

    /**
     * Runs the stack with global conditioning.
     *
     * @param input {@code [batch, channels, length]}
     * @param conditioning {@code [batch, conditioning, 1]}, or {@code null}
     * @return the summed skips {@code [batch, channels, length]}
     */
    public Var forward(Var input, Var conditioning) {
        boolean unbatched = input.shape().rank() == 2;
        Var x = unbatched ? input.reshape(1, input.shape().dim(0), input.shape().dim(1)) : input;
        Var cond = conditioning != null && condW != null ? Functional.conv(conditioning, condW, condB, new int[] {1},
                new int[] {0}, new int[] {0}, new int[] {1}, 1) : null;
        Var out = null;
        int c = channels;
        int dilation = 1;
        for (int i = 0; i < inW.length; i++) {
            Var h = HifiGanBlock.conv(x, inW[i], inB[i], dilation);
            if (cond != null) {
                h = h.add(cond.slice(1, 2 * c * i, 2 * c * (i + 1)));
            }
            Var acts = h.slice(1, 0, c).tanh().mul(h.slice(1, c, 2 * c).sigmoid());
            Var rs = HifiGanBlock.conv(acts, rsW[i], rsB[i], 1);
            if (i < inW.length - 1) {
                x = x.add(rs.slice(1, 0, c));
                Var skip = rs.slice(1, c, 2 * c);
                out = out == null ? skip : out.add(skip);
            } else {
                out = out == null ? rs : out.add(rs);
            }
            dilation *= dilationRate;
        }
        return unbatched ? out.reshape(input.shape().toArray()) : out;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int i = 0; i < inW.length; i++) {
            p.addAll(List.of(inW[i], inB[i], rsW[i], rsB[i]));
        }
        if (condW != null) {
            p.add(condW);
            p.add(condB);
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "WaveNet[" + channels + ", " + inW.length + " layers, kernel " + kernel + "]";
    }

    static Config write(WaveNet w) {
        return Config.builder().put("channels", w.channels).put("kernel", w.kernel).put("dilationRate", w.dilationRate)
                .put("layers", w.inW.length).put("conditioning", w.condW == null ? 0 : w.condW.shape().dim(1)).build();
    }

    static WaveNet read(Config c) {
        return new WaveNet(c.getInt("channels"), c.getInt("kernel"), c.getInt("dilationRate"), c.getInt("layers"),
                c.getInt("conditioning"), new SplittableRandom());
    }
}
