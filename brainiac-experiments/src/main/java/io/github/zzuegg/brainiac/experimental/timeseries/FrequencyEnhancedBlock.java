package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Complex;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * FEDformer's frequency-enhanced block with Fourier basis (Zhou et al., ICML 2022,
 * <a href="https://arxiv.org/abs/2201.12740">arXiv:2201.12740</a>): the input is transformed
 * along time ({@link Var#rfft}), a fixed random subset of {@code M} Fourier modes is kept
 * (chosen once at construction; the paper argues a random subset represents the series
 * better than the lowest modes alone), each kept mode is mixed across channels by its own
 * learned complex {@code width x width} matrix, every other mode is zeroed, and the result is
 * transformed back. Cost is linear in the length. Input and output {@code [batch, time, width]};
 * the length is fixed at construction because the modes are.
 */
public final class FrequencyEnhancedBlock implements Layer {

    private final int length;
    private final int width;
    private final int[] modes;
    private final Var[] real;
    private final Var[] imag;

    /**
     * Creates the block.
     *
     * @param length the series length
     * @param width the channels
     * @param modes the number of Fourier modes kept
     * @param random the source of the modes and weights
     */
    public FrequencyEnhancedBlock(int length, int width, int modes, SplittableRandom random) {
        this.length = length;
        this.width = width;
        int frequencies = length / 2 + 1;
        int m = Math.min(modes, frequencies);
        int[] all = new int[frequencies];
        for (int i = 0; i < frequencies; i++) {
            all[i] = i;
        }
        for (int i = frequencies - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = all[i];
            all[i] = all[j];
            all[j] = tmp;
        }
        this.modes = Arrays.copyOf(all, m);
        Arrays.sort(this.modes);
        real = new Var[m];
        imag = new Var[m];
        float s = (float) (1.0 / width);
        for (int i = 0; i < m; i++) {
            real[i] = Var.param(Tensor.uniform(Shape.of(width, width), 0, s, random));
            imag[i] = Var.param(Tensor.uniform(Shape.of(width, width), 0, s, random));
        }
    }

    /**
     * @param modes the number of Fourier modes kept
     * @return a spec for model builders
     */
    public static LayerSpec spec(int modes) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new FrequencyEnhancedBlock(input.dim(-2), input.dim(-1), modes, random);
            }
        };
    }

    /** @return the kept mode indices */
    public int[] modes() {
        return modes.clone();
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        if (input.shape().dim(1) != length) {
            throw new IllegalArgumentException("FrequencyEnhancedBlock expects length " + length + ", got "
                    + input.shape());
        }
        Var spec = input.permute(0, 2, 1).rfft();                       // [b, width, F, 2]
        Var re = Complex.real(spec);                                    // [b, width, F]
        Var im = Complex.imag(spec);
        Var[] outRe = new Var[modes.length];
        Var[] outIm = new Var[modes.length];
        for (int i = 0; i < modes.length; i++) {
            Var xr = re.slice(2, modes[i], modes[i] + 1).reshape(b, width);
            Var xi = im.slice(2, modes[i], modes[i] + 1).reshape(b, width);
            outRe[i] = xr.matmul(real[i]).sub(xi.matmul(imag[i])).reshape(b, width, 1);
            outIm[i] = xr.matmul(imag[i]).add(xi.matmul(real[i])).reshape(b, width, 1);
        }
        float[] idx = new float[modes.length];
        for (int i = 0; i < modes.length; i++) {
            idx[i] = modes[i];
        }
        Tensor where = Tensor.of(Shape.of(modes.length), idx);
        int frequencies = length / 2 + 1;
        Var mixed = Complex.of(Var.concat(2, outRe).scatterAdd(2, where, frequencies),
                Var.concat(2, outIm).scatterAdd(2, where, frequencies));
        return mixed.irfft(length).permute(0, 2, 1);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(real));
        p.addAll(List.of(imag));
        return p;
    }

    @Override
    public String toString() {
        return "FrequencyEnhancedBlock(length=" + length + ", width=" + width + ", modes=" + modes.length + ")";
    }
}
