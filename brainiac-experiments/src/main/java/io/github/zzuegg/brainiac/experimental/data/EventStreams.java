package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.events.EventSequence;
import io.github.zzuegg.brainiac.experimental.events.HawkesProcess;
import io.github.zzuegg.brainiac.experimental.events.TemporalPointProcess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Synthetic marked event streams with a known generating process, so every model's held-out
 * log-likelihood can be compared with the truth. All streams are simulated by Ogata thinning
 * ({@link TemporalPointProcess#simulate}):
 * <ul>
 *   <li>{@link Kind#EXPONENTIAL}: a multivariate Hawkes process with exponential kernels;</li>
 *   <li>{@link Kind#POWER_LAW}: Hawkes with heavy-tailed Omori kernels;</li>
 *   <li>{@link Kind#DELAYED}: Hawkes with a non-monotone kernel peaking two time units after the
 *       parent event;</li>
 *   <li>{@link Kind#INHIBITORY}: a non-linear Hawkes process,
 *       {@code lambda_k = max(0, mu_k + sum alpha g)}, whose self-effects are negative, so events
 *       suppress further events of their own type;</li>
 *   <li>{@link Kind#SELF_CORRECTING}: {@code lambda(t) = exp(eta t - gamma N(t))}, a non-Hawkes
 *       process that becomes more regular than Poisson, with independent random types.</li>
 * </ul>
 */
public final class EventStreams {

    /** A generating process. */
    public enum Kind {
        /** Hawkes with exponential kernels. */
        EXPONENTIAL,
        /** Hawkes with power-law kernels. */
        POWER_LAW,
        /** Hawkes with a delayed bump kernel. */
        DELAYED,
        /** Non-linear Hawkes with self-inhibition. */
        INHIBITORY,
        /** The self-correcting process. */
        SELF_CORRECTING
    }

    private EventStreams() {
    }

    /**
     * The true process of a kind.
     *
     * @param kind the kind
     * @param types event types
     * @param seed seeds the interaction weights
     * @return the process
     */
    public static TemporalPointProcess process(Kind kind, int types, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] mu = new double[types];
        double[][][] alpha = new double[types][types][1];
        for (int k = 0; k < types; k++) {
            mu[k] = 0.25 + 0.15 * r.nextDouble();
            for (int j = 0; j < types; j++) {
                alpha[k][j][0] = (k == j ? 0.35 : 0.25 / Math.max(1, types - 1)) * (0.6 + 0.8 * r.nextDouble());
            }
        }
        return switch (kind) {
            case EXPONENTIAL -> new HawkesProcess(mu, alpha, new HawkesProcess.Kernel[] {HawkesProcess.exponential(1.5)});
            case POWER_LAW -> new HawkesProcess(mu, alpha, new HawkesProcess.Kernel[] {HawkesProcess.powerLaw(0.3, 1.2)});
            case DELAYED -> new HawkesProcess(mu, alpha, new HawkesProcess.Kernel[] {HawkesProcess.delayed(2, 0.3)});
            case INHIBITORY -> {
                double[][] w = new double[types][types];
                for (int k = 0; k < types; k++) {
                    mu[k] = 1.0 + 0.4 * r.nextDouble();
                    for (int j = 0; j < types; j++) {
                        w[k][j] = k == j ? -0.9 : 0.4 / Math.max(1, types - 1);
                    }
                }
                yield new Inhibitory(mu, w, HawkesProcess.exponential(2));
            }
            case SELF_CORRECTING -> {
                double[] share = new double[types];
                for (int k = 0; k < types; k++) {
                    share[k] = 1 + r.nextDouble();
                }
                double sum = Arrays.stream(share).sum();
                for (int k = 0; k < types; k++) {
                    share[k] /= sum;
                }
                yield new SelfCorrecting(1.0, 1.0, share);
            }
        };
    }

    /**
     * Simulates sequences.
     *
     * @param process the generating process
     * @param count how many sequences
     * @param horizon each window's length
     * @param seed the seed
     * @return the sequences
     */
    public static List<EventSequence> sample(TemporalPointProcess process, int count, double horizon, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<EventSequence> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            out.add(process.simulate(horizon, r));
        }
        return out;
    }

    /** A non-linear Hawkes process with signed weights, rectified at zero. */
    static final class Inhibitory implements TemporalPointProcess {
        private final double[] mu;
        private final double[][] w;
        private final HawkesProcess.Kernel g;

        Inhibitory(double[] mu, double[][] w, HawkesProcess.Kernel g) {
            this.mu = mu;
            this.w = w;
            this.g = g;
        }

        @Override
        public int types() {
            return mu.length;
        }

        @Override
        public double[] intensities(double t, EventSequence h, int count) {
            double[] l = mu.clone();
            for (int i = count - 1; i >= 0; i--) {
                double lag = t - h.times()[i];
                if (lag > g.support()) {
                    break;
                }
                double v = g.value(lag);
                for (int k = 0; k < l.length; k++) {
                    l[k] += w[k][h.types()[i]] * v;
                }
            }
            for (int k = 0; k < l.length; k++) {
                l[k] = Math.max(0, l[k]);
            }
            return l;
        }

        @Override
        public double[] bound(double t, EventSequence h, int count) {
            // only the excitatory parts can raise the intensity, and the kernel decreases
            double total = 0;
            for (int k = 0; k < mu.length; k++) {
                double v = mu[k];
                for (int i = count - 1; i >= 0; i--) {
                    double lag = t - h.times()[i];
                    if (lag > g.support()) {
                        break;
                    }
                    v += Math.max(0, w[k][h.types()[i]]) * g.supAfter(lag);
                }
                total += v;
            }
            return new double[] {total, Double.POSITIVE_INFINITY};
        }
    }

    /** {@code lambda(t) = exp(eta t - gamma N(t))}, types drawn independently. */
    static final class SelfCorrecting implements TemporalPointProcess {
        private final double eta;
        private final double gamma;
        private final double[] share;

        SelfCorrecting(double eta, double gamma, double[] share) {
            this.eta = eta;
            this.gamma = gamma;
            this.share = share;
        }

        @Override
        public int types() {
            return share.length;
        }

        @Override
        public double[] intensities(double t, EventSequence h, int count) {
            double total = Math.exp(eta * t - gamma * count);
            double[] l = new double[share.length];
            for (int k = 0; k < l.length; k++) {
                l[k] = total * share[k];
            }
            return l;
        }

        @Override
        public double[] bound(double t, EventSequence h, int count) {
            // the intensity grows between events: bound it over a short window
            double until = t + 0.5;
            return new double[] {Math.exp(eta * until - gamma * count), until};
        }

        @Override
        public double compensator(double from, double to, EventSequence h, int count) {
            return (Math.exp(eta * to - gamma * count) - Math.exp(eta * from - gamma * count)) / eta;
        }
    }
}
