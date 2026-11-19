package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.tensor.Optimize;

import java.util.function.ToDoubleFunction;

/**
 * The HBV conceptual rainfall–runoff model (Bergström, 1976; the HBV-light formulation of
 * Seibert): a degree-day snow routine, a soil-moisture routine with a non-linear recharge
 * {@code (SM / FC)^BETA} and evaporation limited below {@code LP FC}, an upper and a lower
 * groundwater store with three outflows, and a triangular routing filter of length
 * {@code MAXBAS}. Daily stepping in double precision for calibration, and a forward-mode
 * differentiable version ({@link #simulateDual}) whose parameters may vary day by day, for the
 * δ-HBV hybrids of {@link HybridHydrology}.
 *
 * <p>Parameters, in order: TT, CFMAX, SFCF, CFR, CWH, FC, LP, BETA, K0, K1, K2, UZL, PERC,
 * MAXBAS; {@link #LOWER} and {@link #UPPER} are the usual calibration bounds. An optional extra
 * slow store ({@code slowFraction} of percolation drained at rate {@code slowRate}) gives the
 * "true" catchment a structure the calibrated model lacks.
 */
public final class Hbv {

    /** Parameter names. */
    public static final String[] NAMES = {"TT", "CFMAX", "SFCF", "CFR", "CWH", "FC", "LP", "BETA", "K0", "K1", "K2",
        "UZL", "PERC", "MAXBAS"};
    /** Lower calibration bounds. */
    public static final double[] LOWER = {-2, 0.5, 0.5, 0, 0, 50, 0.3, 1, 0.05, 0.01, 0.001, 0, 0, 1};
    /** Upper calibration bounds. */
    public static final double[] UPPER = {2, 8, 1.3, 0.1, 0.2, 500, 1, 6, 0.5, 0.3, 0.1, 70, 6, 5};

    private Hbv() {
    }

    /**
     * Maps unit-interval values to parameters within the bounds.
     *
     * @param unit values in [0, 1]
     * @return parameters
     */
    public static double[] fromUnit(double[] unit) {
        double[] p = new double[LOWER.length];
        for (int i = 0; i < p.length; i++) {
            double u = Math.min(1, Math.max(0, unit[i]));
            p[i] = LOWER[i] + u * (UPPER[i] - LOWER[i]);
        }
        return p;
    }

    private static double[] routingWeights(double maxbas) {
        int n = (int) Math.ceil(maxbas);
        double[] w = new double[Math.max(1, n)];
        double s = 0;
        for (int i = 0; i < w.length; i++) {
            double a = i;
            double b = Math.min(i + 1, maxbas);
            // integral of the triangle 1 - |2t / maxbas - 1| over [a, b]
            int steps = 10;
            double v = 0;
            for (int k = 0; k < steps; k++) {
                double t = a + (b - a) * (k + 0.5) / steps;
                v += Math.max(0, 1 - Math.abs(2 * t / maxbas - 1)) * (b - a) / steps;
            }
            w[i] = v;
            s += v;
        }
        for (int i = 0; i < w.length; i++) {
            w[i] /= s;
        }
        return w;
    }

    /**
     * Simulates daily runoff.
     *
     * @param p the 14 parameters
     * @param w the forcings
     * @param slowFraction share of percolation diverted to an extra slow store (0 for standard HBV)
     * @param slowRate the slow store's outflow rate
     * @return runoff in mm/day
     */
    public static double[] simulate(double[] p, ScienceSims.Weather w, double slowFraction, double slowRate) {
        int n = w.precipitation().length;
        double[] q = new double[n];
        double snow = 0;
        double liquid = 0;
        double sm = p[5] * 0.5;
        double uz = 0;
        double lz = 10;
        double slow = slowRate > 0 ? slowFraction * 2 / slowRate : 0;
        double[] route = routingWeights(p[13]);
        double[] gen = new double[n];
        for (int t = 0; t < n; t++) {
            double prec = w.precipitation()[t];
            double temp = w.temperature()[t];
            double tt = p[0];
            double input;
            if (temp < tt) {
                snow += prec * p[2];
                double refreeze = Math.min(liquid, p[3] * p[1] * (tt - temp));
                liquid -= refreeze;
                snow += refreeze;
                input = 0;
            } else {
                double melt = Math.min(snow, p[1] * (temp - tt));
                snow -= melt;
                liquid += melt + prec;
                input = 0;
            }
            double hold = p[4] * snow;
            if (liquid > hold) {
                input += liquid - hold;
                liquid = hold;
            }
            double ratio = Math.min(1, sm / p[5]);
            double recharge = input * Math.pow(ratio, p[7]);
            sm += input - recharge;
            if (sm > p[5]) {
                recharge += sm - p[5];
                sm = p[5];
            }
            double et = w.potentialEvaporation()[t] * Math.min(1, sm / (p[6] * p[5]));
            sm = Math.max(0, sm - et);
            uz += recharge;
            double perc = Math.min(p[12], uz);
            uz -= perc;
            lz += perc * (1 - slowFraction);
            slow += perc * slowFraction;
            double q0 = p[8] * Math.max(0, uz - p[11]);
            double q1 = p[9] * uz;
            uz -= q0 + q1;
            double q2 = p[10] * lz;
            lz -= q2;
            double q3 = slowRate * slow;
            slow -= q3;
            gen[t] = q0 + q1 + q2 + q3;
        }
        for (int t = 0; t < n; t++) {
            for (int k = 0; k < route.length && k <= t; k++) {
                q[t] += route[k] * gen[t - k];
            }
        }
        return q;
    }

    /**
     * The Nash–Sutcliffe efficiency.
     *
     * @param sim simulated
     * @param obs observed
     * @param from the first day scored (after warm-up)
     * @return {@code 1 - SSE / SST}
     */
    public static double nse(double[] sim, double[] obs, int from) {
        double m = 0;
        for (int t = from; t < obs.length; t++) {
            m += obs[t];
        }
        m /= obs.length - from;
        double sse = 0;
        double sst = 0;
        for (int t = from; t < obs.length; t++) {
            sse += (sim[t] - obs[t]) * (sim[t] - obs[t]);
            sst += (obs[t] - m) * (obs[t] - m);
        }
        return 1 - sse / sst;
    }

    /**
     * The Kling–Gupta efficiency {@code 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)}.
     *
     * @param sim simulated
     * @param obs observed
     * @param from the first day scored
     * @return the KGE
     */
    public static double kge(double[] sim, double[] obs, int from) {
        int n = obs.length - from;
        double ms = 0;
        double mo = 0;
        for (int t = from; t < obs.length; t++) {
            ms += sim[t] / n;
            mo += obs[t] / n;
        }
        double vs = 0;
        double vo = 0;
        double c = 0;
        for (int t = from; t < obs.length; t++) {
            vs += (sim[t] - ms) * (sim[t] - ms);
            vo += (obs[t] - mo) * (obs[t] - mo);
            c += (sim[t] - ms) * (obs[t] - mo);
        }
        double r = c / Math.sqrt(vs * vo);
        double alpha = Math.sqrt(vs / vo);
        double beta = ms / mo;
        return 1 - Math.sqrt((r - 1) * (r - 1) + (alpha - 1) * (alpha - 1) + (beta - 1) * (beta - 1));
    }

    /** The calibration optimiser. */
    public enum Calibrator {
        /** Core CMA-ES. */
        CMA_ES,
        /** Core Nelder–Mead. */
        NELDER_MEAD
    }

    /**
     * Calibrates the standard model by maximising NSE in the unit cube.
     *
     * @param obs observed runoff
     * @param w forcings
     * @param warmup days excluded from the score
     * @param calibrator the optimiser
     * @param budget generations (CMA-ES) or iterations (Nelder–Mead)
     * @param seed the seed
     * @return the parameters
     */
    public static double[] calibrate(double[] obs, ScienceSims.Weather w, int warmup, Calibrator calibrator,
            int budget, long seed) {
        ToDoubleFunction<double[]> f = u -> {
            double pen = 0;
            for (double v : u) {
                pen += Math.max(0, v - 1) + Math.max(0, -v);
            }
            return -nse(simulate(fromUnit(u), w, 0, 0), obs, warmup) + 10 * pen;
        };
        double[] x0 = new double[LOWER.length];
        java.util.Arrays.fill(x0, 0.5);
        Optimize.Minimum m = calibrator == Calibrator.CMA_ES
                ? Optimize.cmaEs(f, x0, 0.3, budget, 1e-8, seed)
                : Optimize.nelderMead(f, x0, 0.2, budget, 1e-10);
        return fromUnit(m.x());
    }

    // ------------------------------------------------------------------ forward-mode differentiable version

    /**
     * A dual number with a gradient over {@code k} directions, for forward-mode
     * differentiation through the daily recursion.
     */
    public static final class Dual {
        /** The value. */
        public final double v;
        /** The derivatives. */
        public final double[] d;

        /**
         * @param v the value
         * @param d the derivatives (not copied)
         */
        public Dual(double v, double[] d) {
            this.v = v;
            this.d = d;
        }

        /**
         * @param v a constant
         * @param k directions
         * @return the constant
         */
        public static Dual constant(double v, int k) {
            return new Dual(v, new double[k]);
        }

        Dual add(Dual o) {
            double[] g = new double[d.length];
            for (int i = 0; i < g.length; i++) {
                g[i] = d[i] + o.d[i];
            }
            return new Dual(v + o.v, g);
        }

        Dual add(double c) {
            return new Dual(v + c, d);
        }

        Dual sub(Dual o) {
            double[] g = new double[d.length];
            for (int i = 0; i < g.length; i++) {
                g[i] = d[i] - o.d[i];
            }
            return new Dual(v - o.v, g);
        }

        Dual mul(Dual o) {
            double[] g = new double[d.length];
            for (int i = 0; i < g.length; i++) {
                g[i] = d[i] * o.v + v * o.d[i];
            }
            return new Dual(v * o.v, g);
        }

        Dual mul(double c) {
            double[] g = new double[d.length];
            for (int i = 0; i < g.length; i++) {
                g[i] = d[i] * c;
            }
            return new Dual(v * c, g);
        }

        Dual div(Dual o) {
            double[] g = new double[d.length];
            for (int i = 0; i < g.length; i++) {
                g[i] = (d[i] * o.v - v * o.d[i]) / (o.v * o.v);
            }
            return new Dual(v / o.v, g);
        }

        Dual min(Dual o) {
            return v <= o.v ? this : o;
        }

        Dual max(double c) {
            return v >= c ? this : constant(c, d.length);
        }

        Dual min(double c) {
            return v <= c ? this : constant(c, d.length);
        }

        Dual pow(Dual e) {
            if (v <= 0) {
                return constant(0, d.length);
            }
            double val = Math.pow(v, e.v);
            double lv = Math.log(v);
            double[] g = new double[d.length];
            for (int i = 0; i < g.length; i++) {
                g[i] = val * (e.d[i] * lv + e.v * d[i] / v);
            }
            return new Dual(val, g);
        }
    }

    /**
     * Simulates with day-varying, differentiable parameters (MAXBAS is fixed and not
     * differentiated). Returns the generated runoff after routing, as dual numbers.
     *
     * @param params {@code params[t][i]} the parameters on day {@code t}
     * @param w the forcings
     * @param maxbas the routing length
     * @param k gradient directions
     * @return runoff per day with derivatives
     */
    public static Dual[] simulateDual(java.util.function.IntFunction<Dual[]> params, ScienceSims.Weather w,
            double maxbas, int k) {
        int n = w.precipitation().length;
        Dual snow = Dual.constant(0, k);
        Dual liquid = Dual.constant(0, k);
        Dual sm = null;
        Dual uz = Dual.constant(0, k);
        Dual lz = Dual.constant(10, k);
        Dual[] gen = new Dual[n];
        for (int t = 0; t < n; t++) {
            Dual[] p = params.apply(t);
            if (sm == null) {
                sm = p[5].mul(0.5);
            }
            double prec = w.precipitation()[t];
            double temp = w.temperature()[t];
            Dual input = Dual.constant(0, k);
            if (temp < p[0].v) {
                snow = snow.add(p[2].mul(prec));
                Dual refreeze = liquid.min(p[3].mul(p[1]).mul(p[0].add(-temp)));
                liquid = liquid.sub(refreeze);
                snow = snow.add(refreeze);
            } else {
                Dual melt = snow.min(p[1].mul(p[0].mul(-1).add(temp)));
                snow = snow.sub(melt);
                liquid = liquid.add(melt).add(prec);
            }
            Dual hold = p[4].mul(snow);
            if (liquid.v > hold.v) {
                input = input.add(liquid.sub(hold));
                liquid = hold;
            }
            Dual ratio = sm.div(p[5]).min(1);
            Dual recharge = input.mul(ratio.pow(p[7]));
            sm = sm.add(input).sub(recharge);
            if (sm.v > p[5].v) {
                recharge = recharge.add(sm.sub(p[5]));
                sm = p[5];
            }
            Dual et = sm.div(p[6].mul(p[5])).min(1).mul(w.potentialEvaporation()[t]);
            sm = sm.sub(et).max(0);
            uz = uz.add(recharge);
            Dual perc = p[12].min(uz);
            uz = uz.sub(perc);
            lz = lz.add(perc);
            Dual q0 = p[8].mul(uz.sub(p[11]).max(0));
            Dual q1 = p[9].mul(uz);
            uz = uz.sub(q0).sub(q1);
            Dual q2 = p[10].mul(lz);
            lz = lz.sub(q2);
            gen[t] = q0.add(q1).add(q2);
        }
        double[] route = routingWeights(maxbas);
        Dual[] q = new Dual[n];
        for (int t = 0; t < n; t++) {
            Dual s = Dual.constant(0, k);
            for (int j = 0; j < route.length && j <= t; j++) {
                s = s.add(gen[t - j].mul(route[j]));
            }
            q[t] = s;
        }
        return q;
    }
}
