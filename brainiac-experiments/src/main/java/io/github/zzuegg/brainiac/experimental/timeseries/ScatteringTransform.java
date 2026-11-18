package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Complex;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * The wavelet scattering transform (Mallat 2012, <a href="https://arxiv.org/abs/1101.2286">arXiv:1101.2286</a>;
 * Bruna and Mallat 2013, <a href="https://arxiv.org/abs/1203.1513">arXiv:1203.1513</a>; Andén and
 * Mallat 2014, <a href="https://arxiv.org/abs/1304.6763">arXiv:1304.6763</a>): a fixed cascade of
 * wavelet convolutions, complex moduli and low-pass averaging.
 * <pre>
 *   S0 x = x * phi
 *   S1 x(l1) = |x * psi_l1| * phi
 *   S2 x(l1, l2) = ||x * psi_l1| * psi_l2| * phi     (l2 at a coarser scale than l1)
 * </pre>
 * First-order coefficients resemble a mel spectrogram; second-order ones recover the
 * amplitude modulations the averaging removes. The representation is translation invariant
 * up to the averaging scale {@code 2^J} and Lipschitz-stable to small deformations, and has
 * no learned parameters, which suits small data. Filters are Morlet wavelets built in the
 * Fourier domain (a Gaussian bump at the centre frequency minus a Gaussian at zero so the
 * wavelet has zero mean); convolutions are products after core {@link Tensor#fft} (circular,
 * on inputs zero-padded to the transform size).
 *
 * <ul>
 *   <li>{@link #oneDimensional}: {@code J} octaves with {@code Q} wavelets per octave at the
 *       first order and one per octave at the second;</li>
 *   <li>{@link #twoDimensional}: {@code J} scales and {@code L} orientations, for images.</li>
 * </ul>
 * {@link #transform} returns the coefficients averaged over the whole signal (global
 * invariance) or sampled every {@code 2^J} steps.
 */
public final class ScatteringTransform {

    private final int dims;
    private final int size;
    private final Tensor first;           // [F1, N..., 1] Fourier responses
    private final Tensor second;          // [F2, N..., 1]
    private final Tensor lowPass;         // [N..., 1]
    private final int[] firstScale;
    private final int[] secondScale;
    private final int scales;

    private ScatteringTransform(int dims, int size, Tensor first, Tensor second, Tensor lowPass, int[] firstScale,
            int[] secondScale, int scales) {
        this.dims = dims;
        this.size = size;
        this.first = first;
        this.second = second;
        this.lowPass = lowPass;
        this.firstScale = firstScale;
        this.secondScale = secondScale;
        this.scales = scales;
    }

    /** The signed normalised frequency of bin {@code k} of an {@code n}-point transform, in (-pi, pi]. */
    private static double omega(int k, int n) {
        return 2 * Math.PI * (k <= n / 2 ? k : k - n) / n;
    }

    /**
     * A 1-D transform.
     *
     * @param size the transform length (signals are zero-padded or cut to it)
     * @param scales {@code J}, the averaging scale {@code 2^J}
     * @param perOctave {@code Q}, first-order wavelets per octave
     * @return the transform
     */
    public static ScatteringTransform oneDimensional(int size, int scales, int perOctave) {
        List<double[]> f1 = new ArrayList<>();
        List<Integer> s1 = new ArrayList<>();
        for (int j = 0; j < scales * perOctave; j++) {
            f1.add(morlet1d(size, Math.PI * 0.8 * Math.pow(2, -j / (double) perOctave), perOctave));
            s1.add(j / perOctave);
        }
        List<double[]> f2 = new ArrayList<>();
        List<Integer> s2 = new ArrayList<>();
        for (int j = 0; j < scales; j++) {
            f2.add(morlet1d(size, Math.PI * 0.8 * Math.pow(2, -j), 1));
            s2.add(j);
        }
        double sigmaPhi = 0.8 * Math.PI * Math.pow(2, -scales) / 2;
        float[] phi = new float[size];
        for (int k = 0; k < size; k++) {
            double w = omega(k, size);
            phi[k] = (float) Math.exp(-w * w / (2 * sigmaPhi * sigmaPhi));
        }
        return new ScatteringTransform(1, size, stack(f1, Shape.of(f1.size(), size, 1)),
                stack(f2, Shape.of(f2.size(), size, 1)), Tensor.of(Shape.of(size, 1), phi),
                s1.stream().mapToInt(Integer::intValue).toArray(), s2.stream().mapToInt(Integer::intValue).toArray(),
                scales);
    }

    private static double[] morlet1d(int n, double xi, int perOctave) {
        double sigma = xi * (1 - Math.pow(2, -1.0 / perOctave)) / 1.2;
        double kappa = Math.exp(-xi * xi / (2 * sigma * sigma));
        double[] f = new double[n];
        for (int k = 0; k < n; k++) {
            double w = omega(k, n);
            f[k] = w <= 0 ? 0 : Math.exp(-(w - xi) * (w - xi) / (2 * sigma * sigma))
                    - kappa * Math.exp(-w * w / (2 * sigma * sigma));
        }
        return f;
    }

    /**
     * A 2-D transform for square images.
     *
     * @param size the image side after zero-padding
     * @param scales {@code J}
     * @param orientations {@code L}, spread over half a turn
     * @return the transform
     */
    public static ScatteringTransform twoDimensional(int size, int scales, int orientations) {
        List<double[]> f = new ArrayList<>();
        List<Integer> s = new ArrayList<>();
        for (int j = 0; j < scales; j++) {
            double xi = 0.75 * Math.PI * Math.pow(2, -j);
            double sigma = 0.4 * xi;
            for (int l = 0; l < orientations; l++) {
                double theta = Math.PI * l / orientations;
                double cx = xi * Math.cos(theta);
                double cy = xi * Math.sin(theta);
                double kappa = Math.exp(-xi * xi / (2 * sigma * sigma));
                double[] g = new double[size * size];
                for (int u = 0; u < size; u++) {
                    for (int v = 0; v < size; v++) {
                        double wy = omega(u, size);
                        double wx = omega(v, size);
                        double d2 = (wx - cx) * (wx - cx) + (wy - cy) * (wy - cy);
                        g[u * size + v] = Math.exp(-d2 / (2 * sigma * sigma))
                                - kappa * Math.exp(-(wx * wx + wy * wy) / (2 * sigma * sigma));
                    }
                }
                f.add(g);
                s.add(j);
            }
        }
        double sigmaPhi = 0.75 * Math.PI * Math.pow(2, -scales) * 0.8;
        float[] phi = new float[size * size];
        for (int u = 0; u < size; u++) {
            for (int v = 0; v < size; v++) {
                double wy = omega(u, size);
                double wx = omega(v, size);
                phi[u * size + v] = (float) Math.exp(-(wx * wx + wy * wy) / (2 * sigmaPhi * sigmaPhi));
            }
        }
        Tensor bank = stack(f, Shape.of(f.size(), size, size, 1));
        int[] scale = s.stream().mapToInt(Integer::intValue).toArray();
        return new ScatteringTransform(2, size, bank, bank, Tensor.of(Shape.of(size, size, 1), phi), scale, scale,
                scales);
    }

    private static Tensor stack(List<double[]> filters, Shape shape) {
        int per = filters.get(0).length;
        float[] v = new float[filters.size() * per];
        for (int i = 0; i < filters.size(); i++) {
            for (int k = 0; k < per; k++) {
                v[i * per + k] = (float) filters.get(i)[k];
            }
        }
        return Tensor.of(shape, v);
    }

    /** The {@code d}-dimensional DFT over the trailing spatial axes of complex {@code [..., N..., 2]}. */
    private Var fft(Var z, boolean inverse) {
        if (dims == 1) {
            return z.fft(inverse);
        }
        int r = z.shape().rank();
        int[] swap = new int[r];
        for (int i = 0; i < r; i++) {
            swap[i] = i;
        }
        swap[r - 3] = r - 2;
        swap[r - 2] = r - 3;
        Var rows = z.fft(inverse);
        return rows.permute(swap).fft(inverse).permute(swap);
    }

    private Var complex(Var real) {
        return Complex.of(real, Var.of(Tensor.zeros(real.shape())));
    }

    /** {@code |ifft(Z * filters)|} with {@code Z [n, 1, N..., 2]} and filters {@code [F, N..., 1]}. */
    private Var modulus(Var spectrum, Tensor filters) {
        return Complex.abs(fft(spectrum.mul(filters), true));
    }

    /** The number of scattering paths (order 0, 1 and 2). */
    public int paths() {
        int n = 1 + firstScale.length;
        for (int a : firstScale) {
            for (int b : secondScale) {
                n += b > a ? 1 : 0;
            }
        }
        return n;
    }

    /**
     * Computes scattering coefficients.
     *
     * @param signals 1-D: {@code [n][length]}; 2-D: {@code [n][side * side]} row-major squares
     * @param side the image side for 2-D input (ignored for 1-D)
     * @param average whether to average over the whole signal (otherwise sample every {@code 2^J})
     * @return {@code [n][paths * positions]}, positions being 1 when averaging
     */
    public double[][] transform(float[][] signals, int side, boolean average) {
        int n = signals.length;
        int cells = dims == 1 ? size : size * size;
        float[] x = new float[n * cells];
        for (int i = 0; i < n; i++) {
            if (dims == 1) {
                System.arraycopy(signals[i], 0, x, i * size, Math.min(size, signals[i].length));
            } else {
                for (int u = 0; u < Math.min(side, size); u++) {
                    System.arraycopy(signals[i], u * side, x, i * cells + u * size, Math.min(side, size));
                }
            }
        }
        Shape spatial = dims == 1 ? Shape.of(n, 1, size) : Shape.of(n, 1, size, size);
        Var result = Var.noGrad(() -> {
            Var spec = fft(complex(Var.of(Tensor.of(spatial, x))), false);                // [n, 1, N..., 2]
            List<Var> orders = new ArrayList<>();
            orders.add(Complex.real(fft(spec, true)));                                     // [n, 1, N...] (x)
            Var u1 = modulus(spec, first);                                                 // [n, F1, N...]
            orders.add(u1);
            int[] u1Shape = u1.shape().toArray();
            Var spec1 = fft(complex(u1), false);
            int f1 = firstScale.length;
            for (int a = 0; a < f1; a++) {
                List<Integer> keep = new ArrayList<>();
                for (int b = 0; b < secondScale.length; b++) {
                    if (secondScale[b] > firstScale[a]) {
                        keep.add(b);
                    }
                }
                if (keep.isEmpty()) {
                    continue;
                }
                float[] idx = new float[keep.size()];
                for (int k = 0; k < idx.length; k++) {
                    idx[k] = keep.get(k);
                }
                Tensor filters = second.gather(0, Tensor.of(Shape.of(idx.length), idx));
                Var one = spec1.slice(1, a, a + 1);
                orders.add(modulus(one, filters));
            }
            Var all = Var.concat(1, orders.toArray(new Var[0]));                          // [n, P, N...]
            int p = all.shape().dim(1);
            Var pooled;
            if (average) {
                pooled = dims == 1 ? all.mean(2, false) : all.mean(3, false).mean(2, false);
            } else {
                Var low = Complex.real(fft(fft(complex(all), false).mul(lowPass), true));
                int step = 1 << scales;
                int m = size / step;
                if (dims == 1) {
                    pooled = low.reshape(n, p, m, step).slice(3, 0, 1).reshape(n, p * m);
                } else {
                    pooled = low.reshape(n, p, m, step, m, step).slice(5, 0, 1).slice(3, 0, 1).reshape(n, p * m * m);
                }
            }
            return pooled;
        });
        return toRows(result.value(), n);
    }

    private static double[][] toRows(Tensor t, int n) {
        float[] v = t.toArray();
        int d = v.length / n;
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = v[i * d + j];
            }
        }
        return out;
    }

    /**
     * Applies {@code log(1 + c s)}, the usual compression of scattering coefficients before a
     * linear classifier.
     *
     * @param s coefficients
     * @param c the scale, for example 10 to 1000
     * @return compressed coefficients (a new array)
     */
    public static double[][] logCompress(double[][] s, double c) {
        double[][] out = new double[s.length][];
        for (int i = 0; i < s.length; i++) {
            out[i] = new double[s[i].length];
            for (int j = 0; j < s[i].length; j++) {
                out[i][j] = Math.log1p(c * Math.max(0, s[i][j]));
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return "ScatteringTransform(" + dims + "-D, size " + size + ", J=" + scales + ", paths " + paths() + ")";
    }
}
