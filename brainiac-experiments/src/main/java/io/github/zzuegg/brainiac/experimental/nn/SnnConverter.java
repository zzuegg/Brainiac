package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * ANN-to-SNN conversion of a ReLU (or {@link QuantClipFloorActivation}) MLP into
 * integrate-and-fire neurons (Rueckauer et al., Frontiers in Neuroscience 2017; Bu et al.,
 * ICLR 2022). Each hidden ReLU becomes an IF neuron with threshold 1 and reset by
 * subtraction (the residual charge above the threshold is kept), whose firing rate over
 * {@code T} steps approximates the activation divided by the layer's scale {@code λ_l}. The
 * weights are rescaled layer by layer, {@code W_l ← W_l · λ_(l-1) / λ_l} and
 * {@code b_l ← b_l / λ_l}, so neurons neither saturate nor stay silent:
 *
 * <ul>
 *   <li>data-based normalisation ({@link #percentileScales}): {@code λ_l} is a high
 *       percentile of layer {@code l}'s ReLU activations on calibration data;</li>
 *   <li>QCFS: {@code λ_l} is the trained threshold of the layer's
 *       {@link QuantClipFloorActivation}, with the initial membrane at {@code θ/2}.</li>
 * </ul>
 *
 * <p>The input is fed as a constant current every step; the output layer integrates without
 * spiking and the class is the largest accumulated potential.
 */
public final class SnnConverter {

    /**
     * The outcome of {@link #simulate}.
     *
     * @param steps the checkpoints {@code T}
     * @param accuracy accuracy at each checkpoint
     * @param spikes mean hidden spikes per example up to each checkpoint
     */
    public record Result(int[] steps, double[] accuracy, double[] spikes) {
    }

    private final List<Tensor> weights;
    private final List<Tensor> biases;

    private SnnConverter(List<Tensor> weights, List<Tensor> biases) {
        this.weights = weights;
        this.biases = biases;
    }

    /**
     * Converts an MLP.
     *
     * @param weights the dense weights {@code [in, out]}, input layer first
     * @param biases the biases {@code [out]}
     * @param scales {@code λ_l} for each hidden layer (one fewer than the layers)
     * @return the converter holding the rescaled weights
     */
    public static SnnConverter of(List<Tensor> weights, List<Tensor> biases, double[] scales) {
        if (weights.size() != biases.size() || scales.length != weights.size() - 1) {
            throw new IllegalArgumentException("need one bias per weight and one scale per hidden layer");
        }
        List<Tensor> w = new ArrayList<>();
        List<Tensor> b = new ArrayList<>();
        double previous = 1;
        for (int l = 0; l < weights.size(); l++) {
            double scale = l < scales.length ? scales[l] : 1;
            w.add(weights.get(l).mul((float) (previous / scale)));
            b.add(biases.get(l).mul((float) (1 / scale)));
            previous = scale;
        }
        return new SnnConverter(w, b);
    }

    /**
     * Data-based normalisation: the given percentile of every hidden layer's ReLU activations.
     *
     * @param weights the dense weights
     * @param biases the biases
     * @param calibration calibration inputs {@code [n, in]}
     * @param percentile e.g. 99.9; 100 is the maximum
     * @return one scale per hidden layer
     */
    public static double[] percentileScales(List<Tensor> weights, List<Tensor> biases, Tensor calibration,
            double percentile) {
        double[] out = new double[weights.size() - 1];
        Tensor h = calibration;
        for (int l = 0; l < out.length; l++) {
            h = h.matmul(weights.get(l)).add(biases.get(l)).relu();
            float[] v = h.toArray().clone();
            Arrays.sort(v);
            int k = (int) Math.min(v.length - 1, Math.floor(percentile / 100 * (v.length - 1)));
            out[l] = Math.max(v[k], 1e-6);
        }
        return out;
    }

    /**
     * Runs the spiking network.
     *
     * @param x inputs {@code [n, in]}, fed as a constant current
     * @param labels the classes
     * @param checkpoints the {@code T} at which to read the output, ascending
     * @param halfInitial start every membrane at half the threshold (QCFS) instead of 0
     * @return accuracy and spike counts at each checkpoint
     */
    public Result simulate(Tensor x, int[] labels, int[] checkpoints, boolean halfInitial) {
        int n = x.shape().dim(0);
        int layers = weights.size();
        Tensor[] v = new Tensor[layers];
        for (int l = 0; l < layers; l++) {
            int units = weights.get(l).shape().dim(1);
            v[l] = Tensor.zeros(Shape.of(n, units)).add(halfInitial && l < layers - 1 ? 0.5f : 0f);
        }
        double[] acc = new double[checkpoints.length];
        double[] spikes = new double[checkpoints.length];
        double total = 0;
        int next = 0;
        int maxT = checkpoints[checkpoints.length - 1];
        for (int t = 1; t <= maxT; t++) {
            Tensor s = x;
            for (int l = 0; l < layers; l++) {
                Tensor current = s.matmul(weights.get(l)).add(biases.get(l));
                v[l] = v[l].add(current);
                if (l < layers - 1) {
                    s = v[l].sub(1f - 1e-6f).step();
                    v[l] = v[l].sub(s);
                    total += s.sum().item();
                }
            }
            while (next < checkpoints.length && checkpoints[next] == t) {
                float[] o = v[layers - 1].toArray();
                int classes = weights.get(layers - 1).shape().dim(1);
                int right = 0;
                for (int i = 0; i < n; i++) {
                    int best = 0;
                    for (int c = 1; c < classes; c++) {
                        best = o[i * classes + c] > o[i * classes + best] ? c : best;
                    }
                    right += best == labels[i] ? 1 : 0;
                }
                acc[next] = right / (double) n;
                spikes[next] = total / n;
                next++;
            }
        }
        return new Result(checkpoints.clone(), acc, spikes);
    }
}
