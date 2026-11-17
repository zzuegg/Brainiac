package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Views a model's parameters as one flat vector, the form black-box optimizers such as
 * {@link EvolutionStrategy} and {@link CmaEs} work with, and regenerates Gaussian noise
 * from a seed so a perturbation never has to be stored or sent: only its seed.
 *
 * <p>Seed-reproducible perturbations are what let evolution strategies scale (Salimans et
 * al. 2017, <a href="https://arxiv.org/abs/1703.03864">arXiv:1703.03864</a>; Qiu et al. 2025,
 * <a href="https://arxiv.org/abs/2509.24372">arXiv:2509.24372</a>) and what makes
 * zeroth-order fine-tuning run at inference memory (MeZO,
 * <a href="https://arxiv.org/abs/2305.17333">arXiv:2305.17333</a>).
 */
public final class ParameterVector {

    private ParameterVector() {
    }

    /**
     * The number of values in all parameters.
     *
     * @param parameters the parameters
     * @return their total size
     */
    public static int size(List<Var> parameters) {
        long n = 0;
        for (Var p : parameters) {
            n += p.value().size();
        }
        return Math.toIntExact(n);
    }

    /**
     * Copies all parameters into one vector, parameter after parameter, row by row.
     *
     * @param parameters the parameters
     * @return their values
     */
    public static double[] flatten(List<Var> parameters) {
        double[] out = new double[size(parameters)];
        int at = 0;
        for (Var p : parameters) {
            float[] v = p.value().to(DType.FLOAT32).toArray();
            for (float f : v) {
                out[at++] = f;
            }
        }
        return out;
    }

    /**
     * Writes a vector back into the parameters, in the order of {@link #flatten}.
     *
     * @param parameters the parameters to overwrite
     * @param values the values, exactly {@link #size} of them
     */
    public static void assign(List<Var> parameters, double[] values) {
        if (values.length != size(parameters)) {
            throw new IllegalArgumentException("expected " + size(parameters) + " values, got " + values.length);
        }
        int at = 0;
        for (Var p : parameters) {
            Tensor old = p.value();
            float[] v = new float[Math.toIntExact(old.size())];
            for (int i = 0; i < v.length; i++) {
                v[i] = (float) values[at++];
            }
            Tensor t = Tensor.of(old.shape(), v, old.device());
            p.assign(old.dtype() == DType.FLOAT32 ? t : t.to(old.dtype()));
        }
    }

    /**
     * Regenerates standard Gaussian noise from a seed: the same seed always gives the same
     * vector.
     *
     * @param seed the seed
     * @param size the vector length
     * @return the noise
     */
    public static double[] noise(long seed, int size) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = r.nextGaussian();
        }
        return out;
    }

    /**
     * Adds {@code scale} times the noise of a seed to a vector, without allocating the noise.
     *
     * @param target the vector to change in place
     * @param seed the noise seed
     * @param scale the factor
     */
    public static void addNoise(double[] target, long seed, double scale) {
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < target.length; i++) {
            target[i] += scale * r.nextGaussian();
        }
    }

    /**
     * Mixes several numbers into one well-spread seed.
     *
     * @param parts the numbers
     * @return the seed
     */
    public static long seed(long... parts) {
        long h = 0x9E3779B97F4A7C15L;
        for (long p : parts) {
            h ^= p + 0x9E3779B97F4A7C15L + (h << 6) + (h >>> 2);
            h = mix(h);
        }
        return h;
    }

    private static long mix(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }
}
