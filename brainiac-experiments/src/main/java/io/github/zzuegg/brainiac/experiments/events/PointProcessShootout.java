package io.github.zzuegg.brainiac.experiments.events;

import io.github.zzuegg.brainiac.experimental.data.EventStreams;
import io.github.zzuegg.brainiac.experimental.events.EventSequence;
import io.github.zzuegg.brainiac.experimental.events.HawkesProcess;
import io.github.zzuegg.brainiac.experimental.events.IntensityFreeTpp;
import io.github.zzuegg.brainiac.experimental.events.NeuralPointProcess;
import io.github.zzuegg.brainiac.experimental.events.TemporalPointProcess;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Temporal point processes on {@link EventStreams} (wave 10, items 10.5.7 to 10.5.9): every model
 * is scored by its held-out log-likelihood per event next to the generating process's own, so
 * the gap to the truth is visible. Models: a homogeneous Poisson process and a Weibull renewal
 * process (baselines); Hawkes processes with an exponential kernel, a sum of exponentials and a
 * spline kernel ({@link HawkesProcess}, EM); RMTPP, the continuous-time decaying intensity and the
 * transformer Hawkes process ({@link NeuralPointProcess}); and the intensity-free log-normal mixture
 * ({@link IntensityFreeTpp}), whose exact sampler is timed against Hawkes thinning.
 */
public final class PointProcessShootout {

    private PointProcessShootout() {
    }

    /**
     * Budget of one comparison.
     *
     * @param types event types
     * @param train training sequences
     * @param test test sequences
     * @param horizon window length
     * @param steps neural training steps
     * @param emIterations EM iterations
     */
    public record Budget(int types, int train, int test, double horizon, int steps, int emIterations) {
    }

    /**
     * Fits every model on one kind of stream.
     *
     * @param kind the generating process
     * @param b the budget
     * @param seed the seed
     * @return model to {@code {log-likelihood per event, next-type accuracy, time-rescaling KS (NaN
     *     if not computed), millis}}; the IFTPP row adds its next-time MAE, and
     *     {@code "sampling events/s"} holds {@code {IFTPP exact, spline-Hawkes thinning}}
     */
    public static Map<String, double[]> compare(EventStreams.Kind kind, Budget b, long seed) {
        TemporalPointProcess truth = EventStreams.process(kind, b.types(), seed);
        List<EventSequence> train = EventStreams.sample(truth, b.train(), b.horizon(), seed + 1);
        List<EventSequence> test = EventStreams.sample(truth, b.test(), b.horizon(), seed + 2);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("true process", classic(truth, test, 0));
        long t0 = System.currentTimeMillis();
        TemporalPointProcess poisson = poisson(train, b.types());
        out.put("Poisson", classic(poisson, test, System.currentTimeMillis() - t0));
        t0 = System.currentTimeMillis();
        TemporalPointProcess renewal = WeibullRenewal.fit(train, b.types());
        out.put("Weibull renewal", classic(renewal, test, System.currentTimeMillis() - t0));
        t0 = System.currentTimeMillis();
        HawkesProcess exp = HawkesProcess.fitExponential(train, b.types(), b.emIterations());
        out.put("Hawkes exponential (EM)", classic(exp, test, System.currentTimeMillis() - t0));
        t0 = System.currentTimeMillis();
        HawkesProcess soe = HawkesProcess.fitSumOfExponentials(train, b.types(), new double[] {0.3, 1, 3, 10},
                b.emIterations());
        out.put("Hawkes sum of exponentials", classic(soe, test, System.currentTimeMillis() - t0));
        t0 = System.currentTimeMillis();
        HawkesProcess spline = HawkesProcess.fitSpline(train, b.types(), 13, 6, 0.2, b.emIterations());
        out.put("Hawkes spline (EMS)", classic(spline, test, System.currentTimeMillis() - t0));
        for (NeuralPointProcess.Kind k : NeuralPointProcess.Kind.values()) {
            NeuralPointProcess m = new NeuralPointProcess(k, b.types(), 16, 8, seed);
            t0 = System.currentTimeMillis();
            m.train(train, b.steps(), 16, 1e-2f, seed);
            long ms = System.currentTimeMillis() - t0;
            double[] e = m.evaluate(test);
            out.put(switch (k) {
                case RMTPP -> "RMTPP";
                case CONTINUOUS_TIME -> "neural Hawkes (decaying cell)";
                case TRANSFORMER -> "transformer Hawkes";
            }, new double[] {e[0], e[1], Double.NaN, ms});
        }
        IntensityFreeTpp ifp = new IntensityFreeTpp(b.types(), 16, 4, seed);
        t0 = System.currentTimeMillis();
        ifp.train(train, b.steps(), 16, 1e-2f, seed);
        long ms = System.currentTimeMillis() - t0;
        double[] e = ifp.evaluate(test);
        out.put("log-normal mixture (intensity-free)", new double[] {e[0], e[1], Double.NaN, ms, e[2]});
        SplittableRandom r = new SplittableRandom(seed + 9);
        int cap = 3 * train.stream().mapToInt(EventSequence::size).max().orElse(10);
        t0 = System.nanoTime();
        int n = 0;
        for (int i = 0; i < 4; i++) {
            n += ifp.sample(b.horizon(), cap, r).size();
        }
        double ifRate = n / ((System.nanoTime() - t0) / 1e9);
        t0 = System.nanoTime();
        n = 0;
        for (int i = 0; i < 4; i++) {
            n += spline.simulate(b.horizon(), r).size();
        }
        double thinRate = n / ((System.nanoTime() - t0) / 1e9);
        out.put("sampling events/s", new double[] {ifRate, thinRate});
        return out;
    }

    private static double[] classic(TemporalPointProcess p, List<EventSequence> test, long millis) {
        return new double[] {p.logLikelihoodPerEvent(test), p.typeAccuracy(test), p.timeRescalingKs(test), millis};
    }

    /** A homogeneous Poisson process per type: a Hawkes process with no excitation. */
    static TemporalPointProcess poisson(List<EventSequence> train, int types) {
        double[] mu = new double[types];
        double total = 0;
        for (EventSequence s : train) {
            total += s.horizon();
            for (int k : s.types()) {
                mu[k]++;
            }
        }
        for (int k = 0; k < types; k++) {
            mu[k] = Math.max(mu[k], 1) / total;
        }
        return new HawkesProcess(mu, new double[types][types][1],
                new HawkesProcess.Kernel[] {HawkesProcess.exponential(1)});
    }

    /** A renewal process: Weibull inter-event times, types drawn from their frequencies. */
    static final class WeibullRenewal implements TemporalPointProcess {
        private final double shape;
        private final double scale;
        private final double[] share;

        private WeibullRenewal(double shape, double scale, double[] share) {
            this.shape = shape;
            this.scale = scale;
            this.share = share;
        }

        /** Maximum likelihood on the complete gaps (shape by Newton's method). */
        static WeibullRenewal fit(List<EventSequence> train, int types) {
            double[] share = new double[types];
            java.util.List<Double> gaps = new java.util.ArrayList<>();
            for (EventSequence s : train) {
                double prev = 0;
                for (int i = 0; i < s.size(); i++) {
                    gaps.add(Math.max(s.times()[i] - prev, 1e-9));
                    prev = s.times()[i];
                    share[s.types()[i]]++;
                }
            }
            double sum = Arrays.stream(share).sum();
            for (int k = 0; k < types; k++) {
                share[k] = Math.max(share[k], 1) / Math.max(sum, 1);
            }
            double[] x = gaps.stream().mapToDouble(Double::doubleValue).toArray();
            double k = 1;
            double meanLog = Arrays.stream(x).map(Math::log).average().orElse(0);
            for (int it = 0; it < 50; it++) {
                double a = 0;
                double bsum = 0;
                double c = 0;
                for (double v : x) {
                    double p = Math.pow(v, k);
                    double l = Math.log(v);
                    a += p * l;
                    bsum += p;
                    c += p * l * l;
                }
                double f = a / bsum - 1 / k - meanLog;
                double df = (c * bsum - a * a) / (bsum * bsum) + 1 / (k * k);
                k = Math.clamp(k - f / df, 0.05, 20);
            }
            double kk = k;
            double scale = Math.pow(Arrays.stream(x).map(v -> Math.pow(v, kk)).average().orElse(1), 1 / k);
            return new WeibullRenewal(k, scale, share);
        }

        @Override
        public int types() {
            return share.length;
        }

        private double hazard(double lag) {
            double u = Math.max(lag, 1e-9) / scale;
            return shape / scale * Math.pow(u, shape - 1);
        }

        @Override
        public double[] intensities(double t, EventSequence h, int count) {
            double last = count == 0 ? 0 : h.times()[count - 1];
            double z = hazard(t - last);
            double[] l = new double[share.length];
            for (int k = 0; k < l.length; k++) {
                l[k] = z * share[k];
            }
            return l;
        }

        @Override
        public double[] bound(double t, EventSequence h, int count) {
            double last = count == 0 ? 0 : h.times()[count - 1];
            double until = t + 0.5;
            double lag = shape >= 1 ? until - last : Math.max(t - last, 1e-3);
            return new double[] {hazard(lag), until};
        }

        @Override
        public double compensator(double from, double to, EventSequence h, int count) {
            double last = count == 0 ? 0 : h.times()[count - 1];
            return Math.pow((to - last) / scale, shape) - Math.pow(Math.max(from - last, 0) / scale, shape);
        }
    }

    /**
     * Prints the full comparison on every stream kind at a larger budget.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        Budget b = new Budget(3, 200, 100, 40, 600, 60);
        for (EventStreams.Kind kind : EventStreams.Kind.values()) {
            compare(kind, b, 1).forEach((k, v) -> System.out.printf("%s %s %s%n", kind, k, Arrays.toString(v)));
        }
    }
}
