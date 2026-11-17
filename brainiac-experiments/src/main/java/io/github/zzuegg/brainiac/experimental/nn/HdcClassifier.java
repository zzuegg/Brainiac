package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.experimental.tensor.Hypervector;

import java.util.ArrayList;
import java.util.List;

/**
 * A hyperdimensional classifier: each class is the bundle (majority) of the encoded
 * examples seen for it. Learning is one pass of counting, so a single example per class is
 * already a usable classifier (one-shot learning); {@link #retrain} adds the
 * perceptron-like correction step of "OnlineHD"-style training (subtract a misclassified
 * example from the wrong prototype, add it to the right one). Prediction is the most
 * similar prototype by Hamming distance. Inputs must already be hypervectors; how
 * examples are encoded (n-grams, records, levels) is up to the caller.
 */
public final class HdcClassifier {
    private final int dims;
    private final List<Hypervector.Accumulator> counters = new ArrayList<>();
    private final List<Hypervector> prototypes = new ArrayList<>();

    /**
     * Creates an empty classifier.
     *
     * @param classes the number of classes
     * @param dims the vector size
     */
    public HdcClassifier(int classes, int dims) {
        this.dims = dims;
        for (int c = 0; c < classes; c++) {
            counters.add(new Hypervector.Accumulator(dims));
            prototypes.add(null);
        }
    }

    /**
     * Learns one example.
     *
     * @param example the encoded example
     * @param label its class
     */
    public void add(Hypervector example, int label) {
        counters.get(label).add(example, 1);
        prototypes.set(label, null);
    }

    /**
     * Returns a class prototype.
     *
     * @param label the class
     * @return the bundle of its examples, or {@code null} if it has none
     */
    public Hypervector prototype(int label) {
        if (prototypes.get(label) == null && counters.get(label).total() > 0) {
            prototypes.set(label, counters.get(label).toVector());
        }
        return prototypes.get(label);
    }

    /**
     * Predicts a class.
     *
     * @param example the encoded example
     * @return the class with the most similar prototype
     */
    public int predict(Hypervector example) {
        int best = -1;
        double sim = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < counters.size(); c++) {
            Hypervector p = prototype(c);
            if (p != null) {
                double s = p.similarity(example);
                if (s > sim) {
                    sim = s;
                    best = c;
                }
            }
        }
        return best;
    }

    /**
     * One correction step: if the example is misclassified, move it from the predicted
     * prototype's counters to the true one's.
     *
     * @param example the encoded example
     * @param label its class
     * @return whether it was misclassified
     */
    public boolean retrain(Hypervector example, int label) {
        int p = predict(example);
        if (p == label) {
            return false;
        }
        counters.get(label).add(example, 1);
        if (p >= 0) {
            counters.get(p).add(example, -1);
            prototypes.set(p, null);
        }
        prototypes.set(label, null);
        return true;
    }

    /** @return the vector size */
    public int dims() {
        return dims;
    }
}
