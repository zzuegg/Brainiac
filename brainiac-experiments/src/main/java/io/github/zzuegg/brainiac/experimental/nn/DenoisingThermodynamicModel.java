package io.github.zzuegg.brainiac.experimental.nn;

import java.util.SplittableRandom;

/**
 * A denoising thermodynamic model (DTM; Jelincic et al., "An efficient probabilistic hardware
 * architecture for diffusion-like models", <a href="https://arxiv.org/abs/2510.23972">arXiv:2510.23972</a>):
 * a chain of small Boltzmann machines run on {@link PBitIsingMachine p-bits}, each of which
 * undoes one step of a binary forward noising process, as diffusion models do with neural
 * networks (Ho et al., <a href="https://arxiv.org/abs/2006.11239">arXiv:2006.11239</a>).
 *
 * <p>The forward process flips each bit of a {@code +-1} vector with probability {@code q} per
 * stage, chosen so that after all stages only 2% of the original correlation is left. Stage
 * {@code t} is an Ising machine over the cleaner state {@code x_(t-1)} (visible p-bits, with
 * lateral couplings) and hidden p-bits, whose visible fields carry the noisier state
 * {@code x_t} through a learned per-bit coupling (initialised to the exact flip likelihood
 * {@code ln((1 - q) / q) / 2}). The idea is that each stage only models a nearly unimodal
 * conditional, which Gibbs sampling mixes quickly, where a single machine (one stage) has to
 * model the whole multimodal data distribution (at toy scale with hidden units a single
 * machine mixes well too). Stages train with contrastive divergence (Hinton,
 * "Training products of experts by minimizing contrastive divergence", 2002). Sampling runs
 * the chain from uniform noise on {@link PBitIsingMachine}s.
 */
public final class DenoisingThermodynamicModel {
    private final int bits;
    private final int hidden;
    private final int stages;
    private final double flip;
    private final double[][][] lateral;
    private final double[][][] weights;
    private final double[][] visibleBias;
    private final double[][] hiddenBias;
    private final double[][] noiseCoupling;

    private DenoisingThermodynamicModel(int bits, int hidden, int stages, long seed) {
        this.bits = bits;
        this.hidden = hidden;
        this.stages = stages;
        this.flip = (1 - Math.pow(0.02, 1.0 / stages)) / 2;
        SplittableRandom r = new SplittableRandom(seed);
        lateral = new double[stages][bits][bits];
        weights = new double[stages][bits][hidden];
        visibleBias = new double[stages][bits];
        hiddenBias = new double[stages][hidden];
        noiseCoupling = new double[stages][bits];
        for (int t = 0; t < stages; t++) {
            for (int i = 0; i < bits; i++) {
                for (int k = 0; k < hidden; k++) {
                    weights[t][i][k] = 0.01 * r.nextGaussian();
                }
                noiseCoupling[t][i] = 0.5 * Math.log((1 - flip) / flip);
            }
        }
    }

    /**
     * Creates an untrained model.
     *
     * @param bits the data size
     * @param hidden hidden p-bits per stage
     * @param stages denoising stages; 1 gives a single Boltzmann machine
     * @param seed the seed
     * @return the model
     */
    public static DenoisingThermodynamicModel of(int bits, int hidden, int stages, long seed) {
        return new DenoisingThermodynamicModel(bits, hidden, stages, seed);
    }

    /** @return the number of stages */
    public int stages() {
        return stages;
    }

    /** @return the per-stage flip probability of the forward process */
    public double flipProbability() {
        return flip;
    }

    /**
     * One forward noising step.
     *
     * @param x a {@code +-1} vector
     * @param random the random source
     * @return a copy with each bit flipped with the stage flip probability
     */
    public int[] noise(int[] x, SplittableRandom random) {
        int[] out = x.clone();
        for (int i = 0; i < out.length; i++) {
            if (random.nextDouble() < flip) {
                out[i] = -out[i];
            }
        }
        return out;
    }

    /**
     * The p-bit machine of a stage, conditioned on the noisier state. Bits {@code 0..bits-1}
     * are visible, the rest hidden.
     *
     * @param t the stage, from 0 (closest to the data)
     * @param noisier {@code x_(t+1)}, the state being denoised
     * @return the machine
     */
    public PBitIsingMachine machine(int t, int[] noisier) {
        int n = bits + hidden;
        double[][] j = new double[n][n];
        double[] h = new double[n];
        for (int i = 0; i < bits; i++) {
            for (int k = 0; k < bits; k++) {
                j[i][k] = lateral[t][i][k];
            }
            for (int k = 0; k < hidden; k++) {
                j[i][bits + k] = weights[t][i][k];
                j[bits + k][i] = weights[t][i][k];
            }
            h[i] = visibleBias[t][i] + noiseCoupling[t][i] * noisier[i];
        }
        for (int k = 0; k < hidden; k++) {
            h[bits + k] = hiddenBias[t][k];
        }
        return PBitIsingMachine.of(j, h);
    }

    /**
     * One contrastive-divergence update of every stage on a batch of data.
     *
     * @param data {@code +-1} data vectors
     * @param cdSteps Gibbs sweeps of the negative phase, started at the data
     * @param learningRate the step size
     * @param random the random source
     */
    public void train(int[][] data, int cdSteps, double learningRate, SplittableRandom random) {
        int[][] clean = new int[data.length][];
        for (int b = 0; b < data.length; b++) {
            clean[b] = data[b].clone();
        }
        for (int t = 0; t < stages; t++) {
            double[][] dLateral = new double[bits][bits];
            double[][] dWeights = new double[bits][hidden];
            double[] dVisible = new double[bits];
            double[] dHidden = new double[hidden];
            double[] dNoise = new double[bits];
            int[][] noisier = new int[data.length][];
            for (int b = 0; b < data.length; b++) {
                noisier[b] = noise(clean[b], random);
                PBitIsingMachine m = machine(t, noisier[b]);
                for (int i = 0; i < bits; i++) {
                    m.clamp(i, clean[b][i]);
                }
                m.sweep(1, random);
                accumulate(m.spins(), noisier[b], 1, dLateral, dWeights, dVisible, dHidden, dNoise);
                m.release();
                for (int s = 0; s < cdSteps; s++) {
                    m.sweep(1, random);
                }
                accumulate(m.spins(), noisier[b], -1, dLateral, dWeights, dVisible, dHidden, dNoise);
            }
            double step = learningRate / data.length;
            for (int i = 0; i < bits; i++) {
                for (int k = 0; k < bits; k++) {
                    if (k != i) {
                        lateral[t][i][k] += step * dLateral[i][k];
                    }
                }
                for (int k = 0; k < hidden; k++) {
                    weights[t][i][k] += step * dWeights[i][k];
                }
                visibleBias[t][i] += step * dVisible[i];
                noiseCoupling[t][i] += step * dNoise[i];
            }
            for (int k = 0; k < hidden; k++) {
                hiddenBias[t][k] += step * dHidden[k];
            }
            clean = noisier;
        }
    }

    private void accumulate(int[] s, int[] noisier, int sign, double[][] dLateral, double[][] dWeights,
            double[] dVisible, double[] dHidden, double[] dNoise) {
        for (int i = 0; i < bits; i++) {
            for (int k = 0; k < bits; k++) {
                dLateral[i][k] += sign * s[i] * s[k];
            }
            for (int k = 0; k < hidden; k++) {
                dWeights[i][k] += sign * s[i] * s[bits + k];
            }
            dVisible[i] += sign * s[i];
            dNoise[i] += sign * s[i] * noisier[i];
        }
        for (int k = 0; k < hidden; k++) {
            dHidden[k] += sign * s[bits + k];
        }
    }

    /**
     * Draws a sample by running the stages from uniform noise down to the data.
     *
     * @param sweeps Gibbs sweeps per stage
     * @param random the random source
     * @return a {@code +-1} vector
     */
    public int[] sample(int sweeps, SplittableRandom random) {
        int[] x = new int[bits];
        for (int i = 0; i < bits; i++) {
            x[i] = random.nextBoolean() ? 1 : -1;
        }
        for (int t = stages - 1; t >= 0; t--) {
            PBitIsingMachine m = machine(t, x);
            m.randomize(random);
            for (int i = 0; i < bits; i++) {
                m.clamp(i, x[i]);
            }
            m.release();
            for (int s = 0; s < sweeps; s++) {
                m.sweep(1, random);
            }
            int[] spins = m.spins();
            x = new int[bits];
            System.arraycopy(spins, 0, x, 0, bits);
        }
        return x;
    }
}
