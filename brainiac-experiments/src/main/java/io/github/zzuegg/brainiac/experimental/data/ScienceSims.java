package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Distributions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Simulators with known ground truth for the science-domain experiments: epidemics (renewal
 * equation outbreaks with known {@code R_t} paths and reporting delays, stochastic SEIR by the
 * Gillespie algorithm), ecology (closed and open capture–recapture histories, occupancy surveys
 * with known {@code psi} and {@code p}, presence-only species records from an inhomogeneous
 * Poisson process with optional road-like sampling bias), daily weather for rainfall–runoff
 * models, small DC power grids, three-component seismic traces with P and S arrivals,
 * inspiral-like chirps in coloured noise with glitches, light curves with planetary transits
 * and eclipsing-binary false positives, cosine-tuned neural populations driven by a 2-D
 * cursor, and multi-unit extracellular traces for spike sorting. Each generator is small
 * enough that every experiment runs in seconds.
 */
public final class ScienceSims {

    private ScienceSims() {
    }

    // ================================================================== random variates

    /**
     * A Poisson variate (Knuth's method below 30, a rounded normal with continuity
     * correction above).
     *
     * @param mean the mean
     * @param r the random source
     * @return the draw
     */
    public static int poisson(double mean, SplittableRandom r) {
        if (mean <= 0) {
            return 0;
        }
        if (mean < 30) {
            double l = Math.exp(-mean);
            int k = 0;
            double p = r.nextDouble();
            while (p > l) {
                k++;
                p *= r.nextDouble();
            }
            return k;
        }
        return (int) Math.max(0, Math.round(mean + Math.sqrt(mean) * r.nextGaussian()));
    }

    /**
     * A gamma variate by Marsaglia and Tsang's method.
     *
     * @param shape the shape
     * @param scale the scale
     * @param r the random source
     * @return the draw
     */
    public static double gamma(double shape, double scale, SplittableRandom r) {
        if (shape < 1) {
            return gamma(shape + 1, scale, r) * Math.pow(r.nextDouble(), 1 / shape);
        }
        double d = shape - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double x = r.nextGaussian();
            double v = 1 + c * x;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            double u = r.nextDouble();
            if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
                return d * v * scale;
            }
        }
    }

    /**
     * A negative-binomial variate with mean {@code mu} and dispersion {@code k}
     * (variance {@code mu + mu^2 / k}), as a gamma–Poisson mixture.
     *
     * @param mu the mean
     * @param k the dispersion
     * @param r the random source
     * @return the draw
     */
    public static int negativeBinomial(double mu, double k, SplittableRandom r) {
        return poisson(gamma(k, mu / k, r), r);
    }

    /**
     * A binomial variate.
     *
     * @param n trials
     * @param p the success probability
     * @param r the random source
     * @return the draw
     */
    public static int binomial(int n, double p, SplittableRandom r) {
        if (n > 200) {
            double m = n * p;
            return (int) Math.min(n, Math.max(0, Math.round(m + Math.sqrt(m * (1 - p)) * r.nextGaussian())));
        }
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (r.nextDouble() < p) {
                k++;
            }
        }
        return k;
    }

    // ================================================================== epidemics

    /**
     * A gamma distribution discretised to whole days 1..max (day 0 gets weight 0), normalised.
     *
     * @param mean the mean in days
     * @param sd the standard deviation in days
     * @param max the longest delay
     * @return the weights, index = delay
     */
    public static double[] discretizedGamma(double mean, double sd, int max) {
        double shape = mean * mean / (sd * sd);
        double rate = mean / (sd * sd);
        double[] w = new double[max + 1];
        double sum = 0;
        for (int s = 1; s <= max; s++) {
            w[s] = Distributions.chiSquaredCdf(2 * rate * (s + 0.5), 2 * shape)
                    - Distributions.chiSquaredCdf(2 * rate * (s - 0.5), 2 * shape);
            sum += w[s];
        }
        for (int s = 1; s <= max; s++) {
            w[s] /= sum;
        }
        return w;
    }

    /**
     * An {@code R_t} path with a step change (an intervention).
     *
     * @param days days
     * @param change the day of the change
     * @param before {@code R} before
     * @param after {@code R} after
     * @return the path
     */
    public static double[] stepRt(int days, int change, double before, double after) {
        double[] r = new double[days];
        for (int t = 0; t < days; t++) {
            r[t] = t < change ? before : after;
        }
        return r;
    }

    /**
     * A smoothly varying {@code R_t} path: a slow decline from {@code high} to {@code low}
     * and a partial rebound.
     *
     * @param days days
     * @param high the starting value
     * @param low the trough
     * @return the path
     */
    public static double[] smoothRt(int days, double high, double low) {
        double[] r = new double[days];
        for (int t = 0; t < days; t++) {
            double u = (double) t / days;
            r[t] = low + (high - low) * (0.5 + 0.5 * Math.cos(Math.PI * Math.min(1, 1.6 * u)))
                    + 0.25 * Math.max(0, u - 0.65) / 0.35 * (high - low);
        }
        return r;
    }

    /**
     * A renewal-equation outbreak with known {@code R_t}, under-ascertainment and reporting
     * delays.
     *
     * @param infections true daily infections {@code I_t ~ Poisson(R_t sum_s w_s I_(t-s))}
     * @param rt the true reproduction number
     * @param onsets ascertained cases by onset day (complete, eventually reported)
     * @param triangle {@code [day][delay]} counts reported {@code delay} days after onset
     * @param serialInterval the serial-interval weights {@code w}
     * @param delay the reporting-delay distribution
     */
    public record Outbreak(int[] infections, double[] rt, int[] onsets, int[][] triangle, double[] serialInterval,
            double[] delay) {

        /**
         * The counts known on a given day: onsets reported by then.
         *
         * @param now the current day
         * @return counts by onset day up to {@code now}
         */
        public int[] reportedBy(int now) {
            int[] c = new int[now + 1];
            for (int t = 0; t <= now; t++) {
                for (int d = 0; d < triangle[t].length && t + d <= now; d++) {
                    c[t] += triangle[t][d];
                }
            }
            return c;
        }
    }

    /**
     * Simulates a renewal-equation outbreak.
     *
     * @param rt the {@code R_t} path (its length sets the days)
     * @param seedCases infections on each of the first 7 days
     * @param ascertainment the probability a case is ever reported
     * @param delayMean the mean reporting delay in days
     * @param maxDelay the longest delay
     * @param seed the seed
     * @return the outbreak
     */
    public static Outbreak renewalOutbreak(double[] rt, int seedCases, double ascertainment, double delayMean,
            int maxDelay, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int days = rt.length;
        double[] w = discretizedGamma(5.2, 2.5, 20);
        double[] delay = discretizedGamma(delayMean, delayMean * 0.6, maxDelay);
        delay[0] = 0.05;
        double s = 0;
        for (double v : delay) {
            s += v;
        }
        for (int d = 0; d < delay.length; d++) {
            delay[d] /= s;
        }
        int[] inf = new int[days];
        int[] onsets = new int[days];
        int[][] tri = new int[days][maxDelay + 1];
        for (int t = 0; t < days; t++) {
            if (t < 7) {
                inf[t] = seedCases;
            } else {
                double lambda = 0;
                for (int k = 1; k < w.length && k <= t; k++) {
                    lambda += w[k] * inf[t - k];
                }
                inf[t] = poisson(rt[t] * lambda, r);
            }
            int obs = binomial(inf[t], ascertainment, r);
            onsets[t] = obs;
            for (int i = 0; i < obs; i++) {
                double u = r.nextDouble();
                int d = 0;
                double acc = delay[0];
                while (u > acc && d < maxDelay) {
                    d++;
                    acc += delay[d];
                }
                tri[t][d]++;
            }
        }
        return new Outbreak(inf, rt, onsets, tri, w, delay);
    }

    /**
     * A stochastic SEIR epidemic path.
     *
     * @param newInfections daily new infections (S to E)
     * @param newCases daily new infectious cases (E to I)
     * @param rt the effective reproduction number {@code R(t) S(t) / N} at the start of each day
     * @param population the population size
     */
    public record SeirPath(int[] newInfections, int[] newCases, double[] rt, int population) {
    }

    /**
     * Simulates an SEIR epidemic exactly by the Gillespie algorithm, with a time-varying basic
     * reproduction number {@code R0(t) = beta(t) / gamma}.
     *
     * @param population the population
     * @param r0 the basic reproduction number per day
     * @param sigma the E to I rate (1 / latent period)
     * @param gamma the recovery rate (1 / infectious period)
     * @param initialInfectious initially infectious individuals
     * @param seed the seed
     * @return the path (days = {@code r0.length})
     */
    public static SeirPath seirGillespie(int population, double[] r0, double sigma, double gamma,
            int initialInfectious, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int days = r0.length;
        int s = population - initialInfectious;
        int e = 0;
        int i = initialInfectious;
        int[] infections = new int[days];
        int[] cases = new int[days];
        double[] rt = new double[days];
        double t = 0;
        for (int d = 0; d < days; d++) {
            rt[d] = r0[d] * s / population;
            double beta = r0[d] * gamma;
            while (true) {
                double a1 = beta * s * i / population;
                double a2 = sigma * e;
                double a3 = gamma * i;
                double a0 = a1 + a2 + a3;
                if (a0 <= 0) {
                    t = d + 1;
                    break;
                }
                double dt = -Math.log(1 - r.nextDouble()) / a0;
                if (t + dt >= d + 1) {
                    t = d + 1;
                    break;
                }
                t += dt;
                double u = r.nextDouble() * a0;
                if (u < a1) {
                    s--;
                    e++;
                    infections[d]++;
                } else if (u < a1 + a2) {
                    e--;
                    i++;
                    cases[d]++;
                } else {
                    i--;
                }
            }
        }
        return new SeirPath(infections, cases, rt, population);
    }

    /**
     * Aggregates daily counts to weeks with under-reporting and negative-binomial noise.
     *
     * @param daily daily counts
     * @param ascertainment the reported fraction
     * @param dispersion the NB dispersion (large means Poisson-like)
     * @param seed the seed
     * @return weekly reported counts
     */
    public static int[] weeklyReports(int[] daily, double ascertainment, double dispersion, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] w = new int[daily.length / 7];
        for (int k = 0; k < w.length; k++) {
            int sum = 0;
            for (int d = 0; d < 7; d++) {
                sum += daily[7 * k + d];
            }
            w[k] = negativeBinomial(Math.max(1e-9, sum * ascertainment), dispersion, r);
        }
        return w;
    }

    // ================================================================== ecology

    /**
     * Capture histories: one row per animal caught at least once, one column per occasion.
     *
     * @param histories 0/1 capture indicators
     * @param population the true number of animals (closed) or released (open)
     */
    public record CaptureData(int[][] histories, int population) {
    }

    /**
     * A closed population sampled on several occasions with constant capture probability.
     *
     * @param population the true size
     * @param occasions occasions
     * @param p the capture probability
     * @param seed the seed
     * @return the histories of the animals ever caught
     */
    public static CaptureData closedCaptures(int population, int occasions, double p, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> rows = new ArrayList<>();
        for (int a = 0; a < population; a++) {
            int[] h = new int[occasions];
            boolean seen = false;
            for (int k = 0; k < occasions; k++) {
                if (r.nextDouble() < p) {
                    h[k] = 1;
                    seen = true;
                }
            }
            if (seen) {
                rows.add(h);
            }
        }
        return new CaptureData(rows.toArray(new int[0][]), population);
    }

    /**
     * An open population for the Cormack–Jolly–Seber model: {@code releases} marked animals are
     * released at each occasion except the last; each survives each interval with probability
     * {@code phi} and, if alive, is recaptured with probability {@code p}.
     *
     * @param releases newly marked animals per occasion
     * @param occasions occasions
     * @param phi survival per interval
     * @param p recapture probability
     * @param seed the seed
     * @return the histories (first 1 = release)
     */
    public static CaptureData cjsCaptures(int releases, int occasions, double phi, double p, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> rows = new ArrayList<>();
        for (int first = 0; first < occasions - 1; first++) {
            for (int a = 0; a < releases; a++) {
                int[] h = new int[occasions];
                h[first] = 1;
                boolean alive = true;
                for (int k = first + 1; k < occasions; k++) {
                    alive = alive && r.nextDouble() < phi;
                    if (alive && r.nextDouble() < p) {
                        h[k] = 1;
                    }
                }
                rows.add(h);
            }
        }
        return new CaptureData(rows.toArray(new int[0][]), releases * (occasions - 1));
    }

    /**
     * A single-season occupancy survey.
     *
     * @param x the site covariate driving occupancy
     * @param w the visit covariate driving detection {@code [site][visit]}
     * @param y detections {@code [site][visit]}
     * @param z true occupancy
     * @param psi the true occupancy probability per site
     * @param abundance the true abundance per site (Royle–Nichols variant), or null
     */
    public record OccupancySurvey(double[] x, double[][] w, int[][] y, int[] z, double[] psi, int[] abundance) {
    }

    /**
     * Simulates an occupancy survey with {@code logit psi = b0 + b1 x} and
     * {@code logit p = a0 + a1 w}; with {@code heterogeneity} the detection instead follows
     * Royle–Nichols, {@code N ~ Poisson(lambda)}, {@code p = 1 - (1 - r)^N}, with
     * {@code lambda = -log(1 - psi)}.
     *
     * @param sites sites
     * @param visits visits per site
     * @param psiCoef {b0, b1}
     * @param pCoef {a0, a1}
     * @param heterogeneity whether to use abundance-induced detection
     * @param seed the seed
     * @return the survey
     */
    public static OccupancySurvey occupancy(int sites, int visits, double[] psiCoef, double[] pCoef,
            boolean heterogeneity, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] x = new double[sites];
        double[][] w = new double[sites][visits];
        int[][] y = new int[sites][visits];
        int[] z = new int[sites];
        double[] psi = new double[sites];
        int[] n = heterogeneity ? new int[sites] : null;
        for (int i = 0; i < sites; i++) {
            x[i] = r.nextGaussian();
            psi[i] = sigmoid(psiCoef[0] + psiCoef[1] * x[i]);
            if (heterogeneity) {
                n[i] = poisson(-Math.log(1 - psi[i]), r);
                z[i] = n[i] > 0 ? 1 : 0;
            } else {
                z[i] = r.nextDouble() < psi[i] ? 1 : 0;
            }
            for (int j = 0; j < visits; j++) {
                w[i][j] = r.nextGaussian();
                double p = sigmoid(pCoef[0] + pCoef[1] * w[i][j]);
                if (heterogeneity) {
                    p = 1 - Math.pow(1 - p, n[i]);
                }
                y[i][j] = z[i] == 1 && r.nextDouble() < p ? 1 : 0;
            }
        }
        return new OccupancySurvey(x, w, y, z, psi, n);
    }

    /**
     * @param v a logit
     * @return the logistic function
     */
    public static double sigmoid(double v) {
        return 1 / (1 + Math.exp(-v));
    }

    /**
     * A presence-only species dataset on a raster.
     *
     * @param side the raster side (cells {@code side * side})
     * @param covariates environmental covariates {@code [cell][k]}, standardised
     * @param road the sampling-bias covariate (closeness to roads, in [0, 1])
     * @param logIntensity the true log intensity per cell
     * @param presences cell index of each presence record
     * @param surveyCells cells of a presence–absence test survey
     * @param surveyPresent the survey outcomes
     */
    public record SpeciesData(int side, double[][] covariates, double[] road, double[] logIntensity, int[] presences,
            int[] surveyCells, boolean[] surveyPresent) {
    }

    private static double[] smoothField(int side, int bumps, SplittableRandom r) {
        double[] f = new double[side * side];
        for (int b = 0; b < bumps; b++) {
            double cx = r.nextDouble() * side;
            double cy = r.nextDouble() * side;
            double s = side * (0.15 + 0.25 * r.nextDouble());
            double a = r.nextGaussian();
            for (int y = 0; y < side; y++) {
                for (int x = 0; x < side; x++) {
                    double d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                    f[y * side + x] += a * Math.exp(-d2 / (2 * s * s));
                }
            }
        }
        double m = 0;
        for (double v : f) {
            m += v;
        }
        m /= f.length;
        double sd = 0;
        for (double v : f) {
            sd += (v - m) * (v - m);
        }
        sd = Math.sqrt(sd / f.length);
        for (int i = 0; i < f.length; i++) {
            f[i] = (f[i] - m) / sd;
        }
        return f;
    }

    /**
     * Simulates presence-only records from an inhomogeneous Poisson process with
     * {@code log lambda = b0 + 0.8 c0 - 0.6 c0^2 + 0.7 c1 + 0.5 max(0, c2) } (covariates beyond
     * the third are noise), thinned by a sampling bias {@code exp(bias (road - 1))} when
     * {@code bias > 0}.
     *
     * @param side the raster side
     * @param covariateCount covariates (at least 3)
     * @param expectedPresences the expected number of records before bias
     * @param bias the strength of road-like sampling bias (0 for none)
     * @param surveySites cells in the presence–absence survey
     * @param seed the seed
     * @return the data
     */
    public static SpeciesData species(int side, int covariateCount, double expectedPresences, double bias,
            int surveySites, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int cells = side * side;
        double[][] cov = new double[cells][covariateCount];
        for (int k = 0; k < covariateCount; k++) {
            double[] f = smoothField(side, 6, r);
            for (int c = 0; c < cells; c++) {
                cov[c][k] = f[c];
            }
        }
        // roads: two random lines; closeness = exp(-distance / 4)
        double[] road = new double[cells];
        double[][] lines = new double[2][];
        for (int k = 0; k < 2; k++) {
            double ang = r.nextDouble() * Math.PI;
            lines[k] = new double[] {Math.cos(ang), Math.sin(ang), r.nextDouble() * side, r.nextDouble() * side};
        }
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                double best = Double.MAX_VALUE;
                for (double[] l : lines) {
                    double dx = x - l[2];
                    double dy = y - l[3];
                    best = Math.min(best, Math.abs(-l[1] * dx + l[0] * dy));
                }
                road[y * side + x] = Math.exp(-best / 4);
            }
        }
        double[] eta = new double[cells];
        double z = 0;
        for (int c = 0; c < cells; c++) {
            double[] v = cov[c];
            eta[c] = 0.8 * v[0] - 0.6 * v[0] * v[0] + 0.7 * v[1] + 0.5 * Math.max(0, v[2]);
            z += Math.exp(eta[c]);
        }
        double b0 = Math.log(expectedPresences / z);
        List<Integer> pres = new ArrayList<>();
        for (int c = 0; c < cells; c++) {
            eta[c] += b0;
            double thin = bias > 0 ? Math.exp(bias * (road[c] - 1)) : 1;
            int k = poisson(Math.exp(eta[c]) * thin, r);
            for (int j = 0; j < k; j++) {
                pres.add(c);
            }
        }
        int[] survey = new int[surveySites];
        boolean[] present = new boolean[surveySites];
        double scale = 30.0 / expectedPresences;   // survey effort: probability of presence 1 - exp(-scale lambda)
        for (int i = 0; i < surveySites; i++) {
            int c = r.nextInt(cells);
            survey[i] = c;
            present[i] = r.nextDouble() < 1 - Math.exp(-Math.exp(eta[c]) * scale * cells / 30.0);
        }
        return new SpeciesData(side, cov, road, eta, pres.stream().mapToInt(Integer::intValue).toArray(), survey,
                present);
    }

    // ================================================================== power grids

    /**
     * A small DC power grid.
     *
     * @param buses the number of buses (bus 0 is the slack)
     * @param from line start bus
     * @param to line end bus
     * @param reactance line reactance (p.u.)
     * @param limit line thermal limit (MW)
     * @param genBus the bus of each generator
     * @param genCost linear cost per MW of each generator
     * @param genMax capacity of each generator (MW)
     * @param baseLoad the nominal load per bus (MW)
     */
    public record PowerGrid(int buses, int[] from, int[] to, double[] reactance, double[] limit, int[] genBus,
            double[] genCost, double[] genMax, double[] baseLoad) {

        /** @return the number of lines */
        public int lines() {
            return from.length;
        }

        /** @return the number of generators */
        public int generators() {
            return genBus.length;
        }

        /**
         * @param factor a multiplier for every line limit
         * @return the same grid with scaled line limits
         */
        public PowerGrid withLimitsScaled(double factor) {
            double[] l = limit.clone();
            for (int k = 0; k < l.length; k++) {
                l[k] *= factor;
            }
            return new PowerGrid(buses, from, to, reactance, l, genBus, genCost, genMax, baseLoad);
        }
    }

    /**
     * A random meshed grid: a random spanning tree plus extra lines, about one generator per
     * three buses with distinct costs, loads on most buses, and line limits set so that some
     * load scenarios congest the network.
     *
     * @param buses buses
     * @param seed the seed
     * @return the grid
     */
    public static PowerGrid powerGrid(int buses, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> lines = new ArrayList<>();
        for (int b = 1; b < buses; b++) {
            lines.add(new int[] {r.nextInt(b), b});
        }
        int extra = buses / 2;
        for (int k = 0; k < extra; k++) {
            int a = r.nextInt(buses);
            int b = r.nextInt(buses);
            if (a != b) {
                lines.add(new int[] {Math.min(a, b), Math.max(a, b)});
            }
        }
        int m = lines.size();
        int[] from = new int[m];
        int[] to = new int[m];
        double[] x = new double[m];
        double[] lim = new double[m];
        for (int k = 0; k < m; k++) {
            from[k] = lines.get(k)[0];
            to[k] = lines.get(k)[1];
            x[k] = 0.05 + 0.15 * r.nextDouble();
        }
        int gens = Math.max(2, buses / 3);
        int[] gb = new int[gens];
        double[] cost = new double[gens];
        double[] gmax = new double[gens];
        double[] load = new double[buses];
        double total = 0;
        for (int b = 0; b < buses; b++) {
            load[b] = r.nextDouble() < 0.8 ? 20 + 60 * r.nextDouble() : 0;
            total += load[b];
        }
        for (int g = 0; g < gens; g++) {
            gb[g] = g == 0 ? 0 : r.nextInt(buses);
            cost[g] = 10 + 40 * r.nextDouble();
            gmax[g] = 1.6 * total / gens * (0.7 + 0.6 * r.nextDouble());
        }
        for (int k = 0; k < m; k++) {
            lim[k] = total * (0.15 + 0.25 * r.nextDouble());
        }
        return new PowerGrid(buses, from, to, x, lim, gb, cost, gmax, load);
    }

    /**
     * Load scenarios: every bus load scaled by an independent factor uniform in
     * {@code [1 - spread, 1 + spread]}, times a common factor uniform in
     * {@code [1 - common, 1 + common]}.
     *
     * @param grid the grid
     * @param count scenarios
     * @param spread per-bus spread, such as 0.3
     * @param common system-wide spread
     * @param seed the seed
     * @return {@code [scenario][bus]}
     */
    public static double[][] loadScenarios(PowerGrid grid, int count, double spread, double common, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] out = new double[count][grid.buses()];
        for (int s = 0; s < count; s++) {
            double c = 1 + common * (2 * r.nextDouble() - 1);
            for (int b = 0; b < grid.buses(); b++) {
                out[s][b] = grid.baseLoad()[b] * c * (1 + spread * (2 * r.nextDouble() - 1));
            }
        }
        return out;
    }

    // ================================================================== coloured noise

    /**
     * AR(1)-filtered Gaussian noise, unit variance.
     *
     * @param n samples
     * @param phi the AR coefficient (0 = white, near 1 = red)
     * @param r the random source
     * @return the noise
     */
    public static double[] colouredNoise(int n, double phi, SplittableRandom r) {
        double[] x = new double[n];
        double s = Math.sqrt(1 - phi * phi);
        x[0] = r.nextGaussian();
        for (int i = 1; i < n; i++) {
            x[i] = phi * x[i - 1] + s * r.nextGaussian();
        }
        return x;
    }

    // ================================================================== seismic

    /**
     * A three-component seismic trace.
     *
     * @param data {@code [3][samples]}: Z, N, E
     * @param pIndex the P arrival sample, or -1 for noise only
     * @param sIndex the S arrival sample, or -1
     * @param snrDb the signal-to-noise ratio in dB
     */
    public record SeismicTrace(double[][] data, int pIndex, int sIndex, double snrDb) {
    }

    /**
     * Simulates a trace: a P wavelet (higher frequency, mostly vertical) and a later, larger S
     * wavelet (lower frequency, mostly horizontal) with exponential codas, in coloured noise.
     *
     * @param samples samples
     * @param rate the sampling rate in Hz
     * @param snrDb the P-wave SNR in dB
     * @param event whether there is an event
     * @param seed the seed
     * @return the trace
     */
    public static SeismicTrace seismicTrace(int samples, double rate, double snrDb, boolean event, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] d = new double[3][];
        for (int c = 0; c < 3; c++) {
            d[c] = colouredNoise(samples, 0.7, r);
        }
        if (!event) {
            return new SeismicTrace(d, -1, -1, snrDb);
        }
        int p = (int) (samples * (0.15 + 0.3 * r.nextDouble()));
        int s = p + (int) (rate * (1.5 + 3.5 * r.nextDouble()));
        s = Math.min(s, samples - (int) rate);
        double amp = Math.pow(10, snrDb / 20);
        double fp = 6 + 4 * r.nextDouble();
        double fs = 2 + 2 * r.nextDouble();
        double[] pw = {1.0, 0.3, 0.3};
        double[] sw = {0.3, 1.0, 0.9};
        for (int i = p; i < samples; i++) {
            double t = (i - p) / rate;
            double v = amp * Math.exp(-t / 0.8) * Math.sin(2 * Math.PI * fp * t) * Math.min(1, t * 20);
            for (int c = 0; c < 3; c++) {
                d[c][i] += pw[c] * v;
            }
        }
        double phase = r.nextDouble() * Math.PI;
        for (int i = s; i < samples; i++) {
            double t = (i - s) / rate;
            double v = 1.6 * amp * Math.exp(-t / 1.5) * Math.sin(2 * Math.PI * fs * t + phase) * Math.min(1, t * 10);
            for (int c = 0; c < 3; c++) {
                d[c][i] += sw[c] * v;
            }
        }
        return new SeismicTrace(d, p, s, snrDb);
    }

    // ================================================================== chirps

    /**
     * A Newtonian inspiral-like chirp: frequency {@code f(t) = f0 (1 - t / tau)^(-3/8)} up to a
     * cut-off, amplitude growing as {@code f^(2/3)}, where the duration {@code tau} plays the
     * role of the chirp mass ({@code tau ~ M^(-5/3)}).
     *
     * @param samples samples
     * @param rate the sampling rate
     * @param tau the time to coalescence from the start, in seconds
     * @param f0 the starting frequency
     * @param fMax the cut-off frequency
     * @param start the sample where the chirp starts
     * @param phase the initial phase
     * @return the waveform (unnormalised)
     */
    public static double[] chirp(int samples, double rate, double tau, double f0, double fMax, int start,
            double phase) {
        double[] h = new double[samples];
        double ph = phase;
        for (int i = start; i < samples; i++) {
            double t = (i - start) / rate;
            if (t >= tau) {
                break;
            }
            double f = f0 * Math.pow(1 - t / tau, -3.0 / 8);
            if (f > fMax) {
                break;
            }
            ph += 2 * Math.PI * f / rate;
            h[i] = Math.pow(f / f0, 2.0 / 3) * Math.sin(ph);
        }
        return h;
    }

    /**
     * A sine-Gaussian glitch.
     *
     * @param samples samples
     * @param rate the sampling rate
     * @param centre the centre sample
     * @param frequency the frequency
     * @param width the width in seconds
     * @return the glitch (peak 1)
     */
    public static double[] glitch(int samples, double rate, int centre, double frequency, double width) {
        double[] g = new double[samples];
        for (int i = 0; i < samples; i++) {
            double t = (i - centre) / rate;
            g[i] = Math.exp(-t * t / (2 * width * width)) * Math.sin(2 * Math.PI * frequency * t);
        }
        return g;
    }

    // ================================================================== light curves

    /**
     * A light curve.
     *
     * @param time observation times in days (with gaps)
     * @param flux relative flux
     * @param period the injected period (days), or 0
     * @param depth the transit depth
     * @param duration the transit duration (days)
     * @param epoch the first mid-transit time
     * @param label 1 for a planet, 0 for a false positive or nothing
     * @param kind "planet", "grazing EB", "secondary EB" or "none"
     */
    public record LightCurve(double[] time, double[] flux, double period, double depth, double duration, double epoch,
            int label, String kind) {
    }

    /**
     * Simulates a light curve at a 30-minute cadence with two data gaps, red noise, stellar
     * rotation and optionally a periodic dip.
     *
     * @param days the span
     * @param kind "planet" (box transits), "grazing EB" (V-shaped eclipses), "secondary EB"
     *     (primary plus a shallower secondary at phase 0.5), or "none"
     * @param period the period in days
     * @param depth the dip depth
     * @param noise white-noise standard deviation per point
     * @param rotationAmplitude the rotation amplitude
     * @param seed the seed
     * @return the light curve
     */
    public static LightCurve lightCurve(double days, String kind, double period, double depth, double noise,
            double rotationAmplitude, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double cadence = 1.0 / 48;
        int n = (int) (days / cadence);
        double gap1 = days * (0.2 + 0.2 * r.nextDouble());
        double gap2 = days * (0.6 + 0.2 * r.nextDouble());
        double rotP = 3 + 10 * r.nextDouble();
        double epoch = period > 0 ? r.nextDouble() * period : 0;
        double duration = period > 0 ? 0.08 * Math.pow(period, 1.0 / 3) : 0;
        double[] red = colouredNoise(n, 0.98, r);
        List<double[]> pts = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double t = i * cadence;
            if (Math.abs(t - gap1) < 1.5 || Math.abs(t - gap2) < 1.0) {
                continue;
            }
            double f = 1 + rotationAmplitude * Math.sin(2 * Math.PI * t / rotP)
                    + 0.5 * rotationAmplitude * Math.sin(4 * Math.PI * t / rotP + 1)
                    + noise * 0.5 * red[i] + noise * r.nextGaussian();
            if (period > 0) {
                double ph = ((t - epoch) % period + period) % period;
                double dt = Math.min(ph, period - ph);
                switch (kind) {
                    case "planet" -> {
                        if (dt < duration / 2) {
                            f -= depth;
                        }
                    }
                    case "grazing EB" -> {
                        if (dt < duration / 2) {
                            f -= depth * (1 - 2 * dt / duration);
                        }
                    }
                    case "secondary EB" -> {
                        if (dt < duration / 2) {
                            f -= depth;
                        }
                        double ds = Math.abs(ph - period / 2);
                        if (ds < duration / 2) {
                            f -= 0.6 * depth;
                        }
                    }
                    default -> {
                    }
                }
            }
            pts.add(new double[] {t, f});
        }
        double[] time = new double[pts.size()];
        double[] flux = new double[pts.size()];
        for (int i = 0; i < time.length; i++) {
            time[i] = pts.get(i)[0];
            flux[i] = pts.get(i)[1];
        }
        return new LightCurve(time, flux, period, depth, duration, epoch, "planet".equals(kind) ? 1 : 0, kind);
    }

    // ================================================================== neural populations

    /**
     * A recorded neural population with the cursor that drove it.
     *
     * @param counts spike counts {@code [bin][neuron]}
     * @param position cursor position {@code [bin][2]}
     * @param velocity cursor velocity {@code [bin][2]}
     * @param trialOf the trial of each bin
     * @param preferred preferred direction of each neuron (radians)
     * @param binSeconds the bin width
     */
    public record NeuralRecording(double[][] counts, double[][] position, double[][] velocity, int[] trialOf,
            double[] preferred, double binSeconds) {
    }

    /**
     * Simulates centre-out reaches (bell-shaped speed profiles toward one of eight targets, and
     * back) driving cosine-tuned Poisson neurons, {@code rate = exp(b + k |v| cos(theta - theta_pref) + g |v|)}.
     * With {@code drift > 0} the preferred directions rotate linearly over the session by that
     * many radians.
     *
     * @param neurons neurons
     * @param trials reaches
     * @param binSeconds the bin width
     * @param baseRate the baseline rate in Hz
     * @param drift the total drift of preferred directions over the session (radians)
     * @param seed the seed
     * @return the recording
     */
    public static NeuralRecording neuralPopulation(int neurons, int trials, double binSeconds, double baseRate,
            double drift, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] pref = new double[neurons];
        double[] depth = new double[neurons];
        double[] base = new double[neurons];
        for (int j = 0; j < neurons; j++) {
            pref[j] = r.nextDouble() * 2 * Math.PI;
            depth[j] = 0.6 + 0.8 * r.nextDouble();
            base[j] = Math.log(baseRate * (0.5 + r.nextDouble()));
        }
        int perTrial = (int) Math.round(1.0 / binSeconds);
        int bins = trials * perTrial;
        double[][] c = new double[bins][neurons];
        double[][] pos = new double[bins][2];
        double[][] vel = new double[bins][2];
        int[] trial = new int[bins];
        double[] cur = new double[2];
        int b = 0;
        for (int k = 0; k < trials; k++) {
            double[] target;
            if (k % 2 == 0) {
                double a = r.nextInt(8) * Math.PI / 4;
                target = new double[] {Math.cos(a), Math.sin(a)};
            } else {
                target = new double[] {0, 0};
            }
            double[] start = cur.clone();
            for (int i = 0; i < perTrial; i++) {
                double u = (i + 1.0) / perTrial;
                double s = u * u * u * (10 - 15 * u + 6 * u * u);   // minimum-jerk profile
                double ds = 30 * u * u * (1 - u) * (1 - u) / perTrial;
                for (int d = 0; d < 2; d++) {
                    pos[b][d] = start[d] + (target[d] - start[d]) * s + 0.01 * r.nextGaussian();
                    vel[b][d] = (target[d] - start[d]) * ds / binSeconds;
                }
                double speed = Math.hypot(vel[b][0], vel[b][1]);
                double dir = Math.atan2(vel[b][1], vel[b][0]);
                double rot = drift * b / bins;
                for (int j = 0; j < neurons; j++) {
                    double rate = Math.exp(base[j] + depth[j] * speed * Math.cos(dir - pref[j] - rot) + 0.2 * speed);
                    c[b][j] = poisson(rate * binSeconds, r);
                }
                trial[b] = k;
                b++;
            }
            cur = target;
        }
        return new NeuralRecording(c, pos, vel, trial, pref, binSeconds);
    }

    // ================================================================== spike sorting

    /**
     * An extracellular recording with ground truth.
     *
     * @param trace {@code [channel][sample]}
     * @param templates {@code [unit][channel][width]}, peak at {@code width / 3}
     * @param spikeTimes per unit the sample of each spike's peak
     * @param rate the sampling rate
     * @param noise the noise standard deviation
     */
    public record SpikeRecording(double[][] trace, double[][][] templates, int[][] spikeTimes, double rate,
            double noise) {
    }

    /**
     * Simulates a recording: {@code units} neurons with distinct (or, with {@code similar},
     * pairwise similar) biphasic templates spread over the channels, Poisson firing with a
     * 2 ms refractory period and slow rate drift, plus a fraction of forced near-coincident
     * spikes, in coloured noise.
     *
     * @param channels channels
     * @param units units
     * @param seconds duration
     * @param rate sampling rate in Hz
     * @param noise noise standard deviation (templates peak at about 1)
     * @param overlapFraction fraction of spikes placed 0.2-0.8 ms after another unit's spike
     * @param similar whether templates come in similar pairs
     * @param seed the seed
     * @return the recording
     */
    public static SpikeRecording spikeTrace(int channels, int units, double seconds, double rate, double noise,
            double overlapFraction, boolean similar, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int width = (int) (rate * 0.0015);
        int peak = width / 3;
        double[][][] tpl = new double[units][channels][width];
        double[] base = null;
        for (int u = 0; u < units; u++) {
            double amp = 0.9 + 0.5 * r.nextDouble();
            double w1 = 0.12 + 0.12 * r.nextDouble();
            double w2 = 0.3 + 0.4 * r.nextDouble();
            double ratio = 0.25 + 0.35 * r.nextDouble();
            double[] shape = new double[width];
            for (int i = 0; i < width; i++) {
                double t = (i - peak) / (rate * 0.001);   // ms
                shape[i] = -amp * Math.exp(-t * t / (2 * w1 * w1))
                        + ratio * amp * Math.exp(-(t - 0.45) * (t - 0.45) / (2 * w2 * w2));
            }
            if (similar && u % 2 == 1 && base != null) {
                for (int i = 0; i < width; i++) {
                    shape[i] = base[i] * 0.8 + 0.2 * shape[i];
                }
            }
            base = shape;
            double centre = r.nextDouble() * Math.max(1, channels - 1);
            for (int c = 0; c < channels; c++) {
                double g = Math.exp(-(c - centre) * (c - centre) / 2.0);
                for (int i = 0; i < width; i++) {
                    tpl[u][c][i] = g * shape[i];
                }
            }
        }
        int n = (int) (seconds * rate);
        double[][] trace = new double[channels][];
        for (int c = 0; c < channels; c++) {
            double[] x = colouredNoise(n, 0.3, r);
            for (int i = 0; i < n; i++) {
                x[i] *= noise;
            }
            trace[c] = x;
        }
        int refractory = (int) (0.002 * rate);
        List<List<Integer>> times = new ArrayList<>();
        for (int u = 0; u < units; u++) {
            List<Integer> t = new ArrayList<>();
            double hz = 8 + 12 * r.nextDouble();
            int last = -refractory;
            for (int i = width; i < n - width; i++) {
                double drift = 1 + 0.5 * Math.sin(2 * Math.PI * i / n + u);
                if (i - last > refractory && r.nextDouble() < hz * drift / rate) {
                    t.add(i);
                    last = i;
                }
            }
            times.add(t);
        }
        // forced overlaps: copy some spikes of unit u to unit (u+1) shortly after
        for (int u = 0; u < units; u++) {
            List<Integer> src = times.get(u);
            List<Integer> dst = times.get((u + 1) % units);
            int k = (int) (overlapFraction * src.size());
            for (int j = 0; j < k; j++) {
                int t = src.get(r.nextInt(src.size())) + (int) (rate * (0.0002 + 0.0006 * r.nextDouble()));
                if (t < n - width) {
                    boolean ok = true;
                    for (int v : dst) {
                        if (Math.abs(v - t) <= refractory) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        dst.add(t);
                    }
                }
            }
        }
        int[][] st = new int[units][];
        for (int u = 0; u < units; u++) {
            st[u] = times.get(u).stream().mapToInt(Integer::intValue).sorted().toArray();
            for (int t : st[u]) {
                for (int c = 0; c < channels; c++) {
                    for (int i = 0; i < width; i++) {
                        int at = t - peak + i;
                        if (at >= 0 && at < n) {
                            trace[c][at] += tpl[u][c][i];
                        }
                    }
                }
            }
        }
        return new SpikeRecording(trace, tpl, st, rate, noise);
    }

    // ================================================================== weather

    /**
     * Daily weather.
     *
     * @param precipitation mm per day
     * @param temperature degrees C
     * @param potentialEvaporation mm per day (Hamon-style, from temperature and day length)
     */
    public record Weather(double[] precipitation, double[] temperature, double[] potentialEvaporation) {
    }

    /**
     * Synthetic daily weather for a snowy mid-latitude catchment: seasonal temperature with AR(1)
     * anomalies, a two-state Markov chain of wet days with gamma amounts, and potential
     * evaporation from temperature.
     *
     * @param days days
     * @param warming degrees added to every temperature
     * @param seed the seed
     * @return the weather
     */
    public static Weather weather(int days, double warming, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] p = new double[days];
        double[] t = new double[days];
        double[] pet = new double[days];
        double anomaly = 0;
        boolean wet = false;
        for (int d = 0; d < days; d++) {
            double season = Math.cos(2 * Math.PI * (d % 365 - 200) / 365.0);
            anomaly = 0.7 * anomaly + 2.0 * r.nextGaussian();
            t[d] = 4 + 11 * season + anomaly + warming;
            double pWet = wet ? 0.65 : 0.3;
            wet = r.nextDouble() < pWet;
            p[d] = wet ? gamma(0.8, 7, r) : 0;
            double dayLength = 12 + 4 * season;
            double esat = 0.6108 * Math.exp(17.27 * t[d] / (t[d] + 237.3));
            pet[d] = Math.max(0, 0.1651 * dayLength / 12 * 216.7 * esat / (t[d] + 273.3) * 1.1);
        }
        return new Weather(p, t, pet);
    }

    /**
     * Concatenates arrays.
     *
     * @param parts the parts
     * @return the concatenation
     */
    public static double[] concat(double[]... parts) {
        int n = 0;
        for (double[] p : parts) {
            n += p.length;
        }
        double[] out = new double[n];
        int at = 0;
        for (double[] p : parts) {
            System.arraycopy(p, 0, out, at, p.length);
            at += p.length;
        }
        return out;
    }

    /**
     * @param v values
     * @return a copy sorted ascending
     */
    static double[] sorted(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s;
    }
}
