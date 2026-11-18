package io.github.zzuegg.brainiac.experimental.events;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A marked temporal point process given by its conditional intensities
 * {@code lambda_k(t | history)}. The defaults give the exact log-likelihood (with a numerical
 * compensator), simulation by Ogata's thinning (Ogata, IEEE Trans. Inf. Theory 1981,
 * <a href="https://doi.org/10.1109/TIT.1981.1056305">doi:10.1109/TIT.1981.1056305</a>) and the
 * time-rescaling goodness-of-fit statistic (Brown et al., Neural Computation 2002): if the model is
 * right, the compensator increments between consecutive events are independent unit
 * exponentials.
 */
public interface TemporalPointProcess {

    /** @return the number of event types */
    int types();

    /**
     * The intensities just after the first {@code count} events.
     *
     * @param t a time after event {@code count - 1}
     * @param history the sequence
     * @param count how many of its events are history
     * @return {@code lambda_k(t)} per type
     */
    double[] intensities(double t, EventSequence history, int count);

    /**
     * An upper bound on the total intensity over {@code [t, until)} as long as no new event
     * happens, for thinning.
     *
     * @param t the current time
     * @param history the sequence
     * @param count how many of its events are history
     * @return {@code {bound, until}}; {@code until} may be infinite
     */
    double[] bound(double t, EventSequence history, int count);

    /**
     * The total compensator {@code int_from^to sum_k lambda_k(s) ds} with no event inside. The
     * default uses composite Simpson's rule with 32 panels.
     *
     * @param from the start
     * @param to the end
     * @param history the sequence
     * @param count how many of its events are history
     * @return the integral
     */
    default double compensator(double from, double to, EventSequence history, int count) {
        if (to <= from) {
            return 0;
        }
        int panels = 32;
        double h = (to - from) / panels;
        double sum = 0;
        for (int i = 0; i <= panels; i++) {
            double w = i == 0 || i == panels ? 1 : (i % 2 == 1 ? 4 : 2);
            double[] l = intensities(from + i * h, history, count);
            for (double v : l) {
                sum += w * v;
            }
        }
        return sum * h / 3;
    }

    /**
     * @param s a sequence
     * @return its log-likelihood: {@code sum_i log lambda_(k_i)(t_i) - int_0^T sum_k lambda_k}
     */
    default double logLikelihood(EventSequence s) {
        double ll = 0;
        double prev = 0;
        for (int i = 0; i < s.size(); i++) {
            ll -= compensator(prev, s.times()[i], s, i);
            ll += Math.log(Math.max(intensities(s.times()[i], s, i)[s.types()[i]], 1e-300));
            prev = s.times()[i];
        }
        return ll - compensator(prev, s.horizon(), s, s.size());
    }

    /**
     * Simulates one sequence by Ogata's thinning.
     *
     * @param horizon the window end
     * @param random the random source
     * @return the sequence
     */
    default EventSequence simulate(double horizon, SplittableRandom random) {
        List<double[]> events = new ArrayList<>();
        double t = 0;
        EventSequence current = new EventSequence(new double[0], new int[0], horizon);
        while (t < horizon) {
            double[] b = bound(t, current, current.size());
            if (b[0] <= 0) {
                if (Double.isInfinite(b[1])) {
                    break;
                }
                t = b[1];
                continue;
            }
            double w = -Math.log(1 - random.nextDouble()) / b[0];
            if (t + w >= b[1]) {
                t = b[1];
                continue;
            }
            t += w;
            if (t >= horizon) {
                break;
            }
            double[] l = intensities(t, current, current.size());
            double total = Arrays.stream(l).sum();
            double u = random.nextDouble() * b[0];
            if (u < total) {
                int k = 0;
                double acc = l[0];
                while (acc < u && k + 1 < l.length) {
                    acc += l[++k];
                }
                events.add(new double[] {t, k});
                double[] times = new double[events.size()];
                int[] types = new int[events.size()];
                for (int i = 0; i < events.size(); i++) {
                    times[i] = events.get(i)[0];
                    types[i] = (int) events.get(i)[1];
                }
                current = new EventSequence(times, types, horizon);
            }
        }
        return current;
    }

    /**
     * The time-rescaling Kolmogorov-Smirnov statistic: the largest gap between the empirical CDF
     * of the compensator increments and the unit exponential CDF (0 = perfect fit).
     *
     * @param sequences the data
     * @return the KS statistic
     */
    default double timeRescalingKs(List<EventSequence> sequences) {
        List<Double> taus = new ArrayList<>();
        for (EventSequence s : sequences) {
            double prev = 0;
            for (int i = 0; i < s.size(); i++) {
                taus.add(compensator(prev, s.times()[i], s, i));
                prev = s.times()[i];
            }
        }
        double[] v = taus.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        double ks = 0;
        for (int i = 0; i < v.length; i++) {
            double cdf = 1 - Math.exp(-v[i]);
            ks = Math.max(ks, Math.max(Math.abs(cdf - (double) i / v.length), Math.abs(cdf - (i + 1.0) / v.length)));
        }
        return ks;
    }

    /**
     * The share of events whose type has the largest intensity at the event's time.
     *
     * @param sequences the data
     * @return next-type accuracy given the time
     */
    default double typeAccuracy(List<EventSequence> sequences) {
        int right = 0;
        int total = 0;
        for (EventSequence s : sequences) {
            for (int i = 0; i < s.size(); i++) {
                double[] l = intensities(s.times()[i], s, i);
                int best = 0;
                for (int k = 1; k < l.length; k++) {
                    if (l[k] > l[best]) {
                        best = k;
                    }
                }
                right += best == s.types()[i] ? 1 : 0;
                total++;
            }
        }
        return total == 0 ? 0 : right / (double) total;
    }

    /**
     * @param sequences the data
     * @return the log-likelihood per event
     */
    default double logLikelihoodPerEvent(List<EventSequence> sequences) {
        double ll = 0;
        int events = 0;
        for (EventSequence s : sequences) {
            ll += logLikelihood(s);
            events += s.size();
        }
        return ll / Math.max(1, events);
    }
}
