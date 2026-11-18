package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;

import java.util.SplittableRandom;

/**
 * Adaptive random forest: online bagging over Hoeffding trees with random attribute subsets, each
 * tree watched by its own drift detectors.
 *
 * <p>Every example trains every tree with a weight drawn from Poisson(6) (online bagging with a
 * larger rate than Oza's 1, so trees see more data), and every leaf considers a random subset of
 * {@code floor(sqrt(d)) + 1} attributes. Each tree's 0/1 error stream feeds two {@link
 * DriftDetectors.Adwin} detectors: on a warning (the looser one) a background tree starts
 * learning, and on a drift (the stricter one) it replaces the tree (or a fresh tree does).
 * Trees vote with their class probabilities weighted by their accuracy since they were last
 * replaced.
 *
 * <p>Paper: Gomes et al., "Adaptive random forests for evolving data stream classification",
 * Machine Learning 106 (2017), <a href="https://doi.org/10.1007/s10994-017-5642-8">doi:10.1007/s10994-017-5642-8</a>.
 */
public final class AdaptiveRandomForest implements OnlineClassifier {

    private final int features;
    private final int classes;
    private final double lambda;
    private final double warningDelta;
    private final double driftDelta;
    private final SplittableRandom random;
    private final HoeffdingTree[] trees;
    private final HoeffdingTree[] background;
    private final DriftDetectors.Adwin[] warnings;
    private final DriftDetectors.Adwin[] drifts;
    private final double[] correct;
    private final double[] seen;
    private int replacements;
    private long nextSeed;

    /**
     * Creates the forest.
     *
     * @param features attributes
     * @param classes classes
     * @param trees ensemble size, such as 10
     * @param lambda the Poisson rate of online bagging, such as 6
     * @param warningDelta ADWIN confidence for warnings, such as 0.01
     * @param driftDelta ADWIN confidence for drifts, such as 0.001
     * @param seed the seed
     */
    public AdaptiveRandomForest(int features, int classes, int trees, double lambda, double warningDelta,
            double driftDelta, long seed) {
        this.features = features;
        this.classes = classes;
        this.lambda = lambda;
        this.warningDelta = warningDelta;
        this.driftDelta = driftDelta;
        this.random = new SplittableRandom(seed);
        this.nextSeed = seed * 31 + 7;
        this.trees = new HoeffdingTree[trees];
        this.background = new HoeffdingTree[trees];
        this.warnings = new DriftDetectors.Adwin[trees];
        this.drifts = new DriftDetectors.Adwin[trees];
        this.correct = new double[trees];
        this.seen = new double[trees];
        for (int i = 0; i < trees; i++) {
            this.trees[i] = newTree();
            this.warnings[i] = new DriftDetectors.Adwin(warningDelta, 5, 32);
            this.drifts[i] = new DriftDetectors.Adwin(driftDelta, 5, 32);
        }
    }

    private HoeffdingTree newTree() {
        int k = Math.min(features, (int) Math.floor(Math.sqrt(features)) + 1);
        return new HoeffdingTree(features, classes, 0.01, 0.05, 50, k, 20, nextSeed++);
    }

    private int poisson() {
        double l = Math.exp(-lambda);
        int k = 0;
        double p = 1;
        do {
            k++;
            p *= random.nextDouble();
        } while (p > l);
        return k - 1;
    }

    @Override
    public double[] predict(DriftingStreams.Sample sample) {
        double[] votes = new double[classes];
        double total = 0;
        for (int i = 0; i < trees.length; i++) {
            double w = seen[i] > 0 ? correct[i] / seen[i] : 0.5;
            double[] p = trees[i].predict(sample.x());
            for (int c = 0; c < classes; c++) {
                votes[c] += w * p[c];
            }
            total += w;
        }
        for (int c = 0; c < classes; c++) {
            votes[c] = total > 0 ? votes[c] / total : 1.0 / classes;
        }
        return votes;
    }

    @Override
    public void learn(DriftingStreams.Sample sample) {
        double[] x = sample.x();
        int y = sample.label();
        for (int i = 0; i < trees.length; i++) {
            boolean right = OnlineClassifier.argmax(trees[i].predict(x)) == y;
            correct[i] += right ? 1 : 0;
            seen[i]++;
            int k = poisson();
            trees[i].learn(x, y, k);
            if (background[i] != null) {
                background[i].learn(x, y, k);
            }
            double err = right ? 0 : 1;
            if (warnings[i].update(err) == DriftDetectors.Signal.DRIFT && background[i] == null) {
                background[i] = newTree();
            }
            if (drifts[i].update(err) == DriftDetectors.Signal.DRIFT) {
                trees[i] = background[i] != null ? background[i] : newTree();
                background[i] = null;
                warnings[i] = new DriftDetectors.Adwin(warningDelta, 5, 32);
                drifts[i] = new DriftDetectors.Adwin(driftDelta, 5, 32);
                correct[i] = 0;
                seen[i] = 0;
                replacements++;
            }
        }
    }

    /** @return how many trees drift detection has replaced */
    public int replacements() {
        return replacements;
    }

    @Override
    public long bytes() {
        long b = 0;
        for (int i = 0; i < trees.length; i++) {
            b += trees[i].bytes() + (background[i] == null ? 0 : background[i].bytes());
        }
        return b;
    }
}
