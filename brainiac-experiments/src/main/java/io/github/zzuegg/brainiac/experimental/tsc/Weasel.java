package io.github.zzuegg.brainiac.experimental.tsc;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Lbfgs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Word ExtrAction for time SEries cLassification (WEASEL; Schäfer and Leser, CIKM 2017,
 * <a href="https://arxiv.org/abs/1701.07681">arXiv:1701.07681</a>) and WEASEL 2.0 (Schäfer and
 * Leser, Machine Learning 2023).
 * <ul>
 *   <li>{@link Version#V1}: SFA words with supervised (information-gain) binning for many window
 *       lengths, unigrams plus bigrams of words one window apart, a chi-squared test that keeps
 *       only class-discriminative words, and a multinomial logistic regression on the counts.</li>
 *   <li>{@link Version#V2}: an ensemble of randomised SFA configurations in the style of ROCKET:
 *       each draws a window length, word length, dilation, binning and whether to use the first
 *       difference; dilated windows (every {@code d}-th value) see long patterns with short words;
 *       the word counts of every configuration are concatenated and read out by a ridge classifier
 *       with leave-one-out penalty selection ({@link WideRidgeClassifier}).</li>
 * </ul>
 */
public final class Weasel {

    /** Which WEASEL. */
    public enum Version {
        /** Unigrams and bigrams, chi-squared selection, logistic regression. */
        V1,
        /** Dilated randomised configurations with a ridge read-out. */
        V2
    }

    /**
     * Settings.
     *
     * @param version which WEASEL
     * @param windows V1: the number of window lengths; V2: the number of random configurations
     * @param alphabet SFA alphabet size
     * @param chiSquared V1: the chi-squared threshold of kept words
     * @param seed V2's configuration seed
     */
    public record Settings(Version version, int windows, int alphabet, double chiSquared, long seed) {

        /**
         * Defaults.
         *
         * @param version the version
         * @return 8 window lengths (V1) or 40 configurations (V2), alphabet 4
         */
        public static Settings defaults(Version version) {
            return new Settings(version, version == Version.V1 ? 8 : 40, 4, 2, 1);
        }
    }

    /** One word extractor: an SFA transform over the series (or its difference) at a dilation. */
    private record Config(SymbolicFourier sfa, int dilation, boolean difference, int offset) {
    }

    private final Settings settings;
    private final List<Config> configs;
    private final Map<Long, Integer> vocabulary;
    private final int features;
    private final double[][] logisticWeights;
    private final WideRidgeClassifier ridge;
    private final int classes;

    private Weasel(Settings settings, List<Config> configs, Map<Long, Integer> vocabulary, int features,
            double[][] logisticWeights, WideRidgeClassifier ridge, int classes) {
        this.settings = settings;
        this.configs = configs;
        this.vocabulary = vocabulary;
        this.features = features;
        this.logisticWeights = logisticWeights;
        this.ridge = ridge;
        this.classes = classes;
    }

    /**
     * Fits WEASEL.
     *
     * @param series training series (one length)
     * @param labels their classes
     * @param classes the number of classes
     * @param settings the settings
     * @return the classifier
     */
    public static Weasel fit(float[][] series, int[] labels, int classes, Settings settings) {
        int length = series[0].length;
        List<Config> configs = new ArrayList<>();
        if (settings.version() == Version.V1) {
            int lo = Math.min(8, length / 4);
            int hi = Math.max(lo + 1, length / 2);
            for (int i = 0; i < settings.windows(); i++) {
                int w = lo + (int) Math.round((hi - lo) * i / (double) Math.max(1, settings.windows() - 1));
                configs.add(new Config(SymbolicFourier.fit(series, labels, w, 4, settings.alphabet(), true,
                        SymbolicFourier.Binning.INFORMATION_GAIN, Math.max(1, w / 4)), 1, false, i));
            }
        } else {
            SplittableRandom r = new SplittableRandom(settings.seed());
            for (int i = 0; i < settings.windows(); i++) {
                boolean diff = r.nextInt(3) == 0;
                int len = diff ? length - 1 : length;
                int w = 6 + r.nextInt(Math.max(1, Math.min(32, len / 2) - 5));
                int maxExp = (int) Math.floor(Math.log((len - 1) / (double) (w - 1)) / Math.log(2));
                int d = 1 << r.nextInt(Math.max(1, Math.min(maxExp, 4) + 1));
                int wordLength = 2 * (2 + r.nextInt(2));
                SymbolicFourier.Binning b = r.nextBoolean() ? SymbolicFourier.Binning.EQUI_DEPTH
                        : SymbolicFourier.Binning.INFORMATION_GAIN;
                float[][] views = new float[series.length * d][];
                int[] viewLabels = new int[views.length];
                for (int s = 0; s < series.length; s++) {
                    float[] base = diff ? difference(series[s]) : series[s];
                    for (int o = 0; o < d; o++) {
                        views[s * d + o] = subsample(base, o, d);
                        viewLabels[s * d + o] = labels[s];
                    }
                }
                configs.add(new Config(SymbolicFourier.fit(views, viewLabels, w, wordLength, settings.alphabet(),
                        true, b, Math.max(1, w / 2)), d, diff, i));
            }
        }
        List<Map<Long, Integer>> bags = new ArrayList<>();
        for (float[] s : series) {
            bags.add(bag(s, configs, settings.version() == Version.V1));
        }
        Map<Long, Integer> vocabulary = new HashMap<>();
        if (settings.version() == Version.V1) {
            Map<Long, double[]> perClass = new HashMap<>();
            int[] classCount = new int[classes];
            for (int i = 0; i < series.length; i++) {
                classCount[labels[i]]++;
                for (Map.Entry<Long, Integer> e : bags.get(i).entrySet()) {
                    perClass.computeIfAbsent(e.getKey(), k -> new double[classes])[labels[i]] += e.getValue();
                }
            }
            for (Map.Entry<Long, double[]> e : perClass.entrySet()) {
                double total = 0;
                for (double v : e.getValue()) {
                    total += v;
                }
                double chi = 0;
                for (int c = 0; c < classes; c++) {
                    double expected = total * classCount[c] / series.length;
                    double diff = e.getValue()[c] - expected;
                    chi += diff * diff / Math.max(1e-9, expected);
                }
                if (chi >= settings.chiSquared()) {
                    vocabulary.put(e.getKey(), vocabulary.size());
                }
            }
        } else {
            for (Map<Long, Integer> b : bags) {
                for (Long k : b.keySet()) {
                    vocabulary.putIfAbsent(k, vocabulary.size());
                }
            }
        }
        int f = Math.max(1, vocabulary.size());
        double[][] x = new double[series.length][];
        for (int i = 0; i < series.length; i++) {
            x[i] = vector(bags.get(i), vocabulary, f);
        }
        if (settings.version() == Version.V1) {
            return new Weasel(settings, configs, vocabulary, f, logistic(x, labels, classes, 1e-3, 200), null,
                    classes);
        }
        return new Weasel(settings, configs, vocabulary, f, null,
                WideRidgeClassifier.fit(x, labels, classes, WideRidgeClassifier.defaultAlphas()), classes);
    }

    private static float[] difference(float[] s) {
        float[] d = new float[s.length - 1];
        for (int t = 0; t < d.length; t++) {
            d[t] = s[t + 1] - s[t];
        }
        return d;
    }

    private static float[] subsample(float[] s, int offset, int d) {
        int n = (s.length - offset + d - 1) / d;
        float[] out = new float[n];
        for (int k = 0; k < n; k++) {
            out[k] = s[offset + k * d];
        }
        return out;
    }

    /** The word counts of one series; keys pack the configuration, a bigram flag and the words. */
    private static Map<Long, Integer> bag(float[] s, List<Config> configs, boolean bigrams) {
        Map<Long, Integer> h = new HashMap<>();
        for (Config c : configs) {
            float[] base = c.difference() ? difference(s) : s;
            for (int o = 0; o < c.dilation(); o++) {
                float[] view = c.dilation() == 1 ? base : subsample(base, o, c.dilation());
                if (view.length < c.sfa().window()) {
                    continue;
                }
                int[] words = c.sfa().words(view);
                long prefix = (long) c.offset() << 48;
                for (int t = 0; t < words.length; t++) {
                    h.merge(prefix | words[t], 1, Integer::sum);
                    int prev = t - c.sfa().window();
                    if (bigrams && prev >= 0) {
                        long key = prefix | (1L << 47) | ((long) words[prev] << c.sfa().wordBits()) | words[t];
                        h.merge(key, 1, Integer::sum);
                    }
                }
            }
        }
        return h;
    }

    private static double[] vector(Map<Long, Integer> bag, Map<Long, Integer> vocabulary, int f) {
        double[] v = new double[f];
        for (Map.Entry<Long, Integer> e : bag.entrySet()) {
            Integer col = vocabulary.get(e.getKey());
            if (col != null) {
                v[col] = Math.sqrt(e.getValue());
            }
        }
        return v;
    }

    /**
     * Multinomial logistic regression with an L2 penalty, fitted by L-BFGS ({@link Lbfgs}).
     */
    private static double[][] logistic(double[][] x, int[] labels, int classes, double l2, int iterations) {
        int n = x.length;
        int d = x[0].length;
        float[] xv = new float[n * (d + 1)];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                xv[i * (d + 1) + j] = (float) x[i][j];
            }
            xv[i * (d + 1) + d] = 1;
        }
        float[] yv = new float[n];
        for (int i = 0; i < n; i++) {
            yv[i] = labels[i];
        }
        Tensor xt = Tensor.of(Shape.of(n, d + 1), xv);
        Tensor ids = Tensor.of(Shape.of(n), yv);
        Var w = Var.param(Tensor.zeros(Shape.of(d + 1, classes)));
        new Lbfgs(10, 1e-6).minimize(List.of(w), () -> Functional.crossEntropy(Var.of(xt).matmul(w), ids)
                .add(w.square().sum().mul((float) l2)), iterations);
        float[] wv = w.value().toArray();
        double[][] out = new double[classes][d + 1];
        for (int j = 0; j <= d; j++) {
            for (int c = 0; c < classes; c++) {
                out[c][j] = wv[j * classes + c];
            }
        }
        return out;
    }

    private static double[] softmax(double[][] w, double[] x) {
        int classes = w.length;
        int d = x.length;
        double[] z = new double[classes];
        double max = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < classes; c++) {
            double s = w[c][d];
            for (int j = 0; j < d; j++) {
                s += w[c][j] * x[j];
            }
            z[c] = s;
            max = Math.max(max, s);
        }
        double sum = 0;
        for (int c = 0; c < classes; c++) {
            z[c] = Math.exp(z[c] - max);
            sum += z[c];
        }
        for (int c = 0; c < classes; c++) {
            z[c] /= sum;
        }
        return z;
    }

    /**
     * The word-count features of series.
     *
     * @param series the series
     * @return {@code [n][features()]}
     */
    public double[][] transform(float[][] series) {
        double[][] x = new double[series.length][];
        for (int i = 0; i < series.length; i++) {
            x[i] = vector(bag(series[i], configs, settings.version() == Version.V1), vocabulary, features);
        }
        return x;
    }

    /**
     * Class scores (V1: probabilities; V2: ridge decision values).
     *
     * @param series the series
     * @return {@code [n][classes]}
     */
    public double[][] scores(float[][] series) {
        double[][] x = transform(series);
        if (ridge != null) {
            return ridge.scores(x);
        }
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = softmax(logisticWeights, x[i]);
        }
        return out;
    }

    /**
     * Predicts classes.
     *
     * @param series the series
     * @return the classes
     */
    public int[] predict(float[][] series) {
        return WideRidgeClassifier.argmax(scores(series));
    }

    /** @return the number of selected word features */
    public int features() {
        return features;
    }

    /** @return the number of classes */
    public int classes() {
        return classes;
    }
}
