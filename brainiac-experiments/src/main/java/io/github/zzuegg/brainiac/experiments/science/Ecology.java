package io.github.zzuegg.brainiac.experiments.science;

import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experimental.science.CaptureRecapture;
import io.github.zzuegg.brainiac.experimental.science.OccupancyModel;
import io.github.zzuegg.brainiac.experimental.science.SpeciesDistribution;
import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Ecology with imperfect detection and presence-only data (wave 13, items 13.3.9 and 13.3.10).
 *
 * <ul>
 *   <li>{@link #abundance}: relative bias and 95% coverage of Lincoln–Petersen, Chapman and
 *       Schnabel population sizes against the naive count of distinct animals, and CJS survival
 *       against the naive return rate, over replicate simulated studies.</li>
 *   <li>{@link #occupancy}: bias of mean occupancy and coverage of the coefficients for the
 *       occupancy model against the naive proportion of sites with a detection and a naive
 *       logistic regression (which absorbs the detection covariate), plus the Royle–Nichols
 *       model under abundance-induced heterogeneity.</li>
 *   <li>{@link #distributions}: test AUROC on a presence–absence survey and correlation with
 *       the true log intensity for MaxEnt, the Poisson process, presence–background logistic
 *       regression at several background sizes, a random forest and a kernel density estimate,
 *       with and without road-like sampling bias and its correction.</li>
 * </ul>
 */
public final class Ecology {

    private Ecology() {
    }

    /**
     * Capture–recapture table.
     *
     * @param population true population size
     * @param p capture probability
     * @param replicates replicate studies
     * @param seed the seed
     * @return metric to {relative bias, coverage}
     */
    public static Map<String, double[]> abundance(int population, double p, int replicates, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] acc = new double[5][2];
        for (int rep = 0; rep < replicates; rep++) {
            ScienceSims.CaptureData two = ScienceSims.closedCaptures(population, 2, p, seed + rep);
            ScienceSims.CaptureData five = ScienceSims.closedCaptures(population, 5, p, seed + 1000 + rep);
            CaptureRecapture.Estimate[] e = {
                CaptureRecapture.estimate(CaptureRecapture.Estimator.LINCOLN_PETERSEN, two.histories()),
                CaptureRecapture.estimate(CaptureRecapture.Estimator.CHAPMAN, two.histories()),
                CaptureRecapture.estimate(CaptureRecapture.Estimator.SCHNABEL, five.histories())};
            for (int k = 0; k < 3; k++) {
                acc[k][0] += (e[k].value() - population) / population / replicates;
                acc[k][1] += e[k].covers(population) ? 1.0 / replicates : 0;
            }
            acc[3][0] += (two.histories().length - population) / (double) population / replicates;
            acc[4][0] += (five.histories().length - population) / (double) population / replicates;
        }
        out.put("Lincoln-Petersen (2 occasions) {rel bias, coverage}", acc[0]);
        out.put("Chapman (2 occasions) {rel bias, coverage}", acc[1]);
        out.put("Schnabel (5 occasions) {rel bias, coverage}", acc[2]);
        out.put("naive distinct count {rel bias 2 occ, rel bias 5 occ}", new double[] {acc[3][0], acc[4][0]});
        // CJS survival
        double phi = 0.75;
        double[] cjs = new double[3];
        for (int rep = 0; rep < replicates; rep++) {
            ScienceSims.CaptureData d = ScienceSims.cjsCaptures(40, 6, phi, 0.4, seed + 2000 + rep);
            CaptureRecapture.Estimate e = CaptureRecapture.estimate(CaptureRecapture.Estimator.CJS, d.histories());
            cjs[0] += (e.value() - phi) / phi / replicates;
            cjs[1] += e.covers(phi) ? 1.0 / replicates : 0;
            // naive: fraction of animals seen on occasion t that are seen again on t + 1
            double seen = 0;
            double again = 0;
            for (int[] h : d.histories()) {
                for (int t = 0; t + 1 < h.length; t++) {
                    if (h[t] == 1) {
                        seen++;
                        again += h[t + 1];
                    }
                }
            }
            cjs[2] += (again / seen - phi) / phi / replicates;
        }
        out.put("CJS phi=0.75 p=0.4 {rel bias, coverage, naive return-rate rel bias}", cjs);
        return out;
    }

    /**
     * Occupancy table.
     *
     * @param sites sites
     * @param visits visits per site
     * @param replicates replicate surveys
     * @param seed the seed
     * @return metric to values
     */
    public static Map<String, double[]> occupancy(int sites, int visits, int replicates, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] psiCoef = {0.0, 1.0};
        double[] pCoef = {-0.5, 1.0};
        double[] model = new double[4];
        double[] naive = new double[3];
        for (int rep = 0; rep < replicates; rep++) {
            ScienceSims.OccupancySurvey s = ScienceSims.occupancy(sites, visits, psiCoef, pCoef, false, seed + rep);
            double truePsi = 0;
            for (double v : s.psi()) {
                truePsi += v / sites;
            }
            OccupancyModel.Fit f = OccupancyModel.fit(s);
            model[0] += (f.meanPsi() - truePsi) / truePsi / replicates;
            model[1] += f.covers(0, psiCoef[0]) ? 1.0 / replicates : 0;
            model[2] += f.covers(1, psiCoef[1]) ? 1.0 / replicates : 0;
            model[3] += f.covers(3, pCoef[1]) ? 1.0 / replicates : 0;
            // naive: share of sites with a detection; logistic of "detected" on x and mean visit covariate
            double[][] x = new double[sites][2];
            double[] y = new double[sites];
            double share = 0;
            for (int i = 0; i < sites; i++) {
                double mw = 0;
                for (int j = 0; j < visits; j++) {
                    mw += s.w()[i][j] / visits;
                    y[i] = Math.max(y[i], s.y()[i][j]);
                }
                x[i][0] = s.x()[i];
                x[i][1] = mw;
                share += y[i] / sites;
            }
            PenalizedRegression.Fit lf = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO)
                    .family(PenalizedRegression.Family.LOGISTIC).fit(x, y, 1e-4);
            naive[0] += (share - truePsi) / truePsi / replicates;
            naive[1] += lf.beta()[0] / replicates;
            naive[2] += lf.beta()[1] / replicates;
        }
        out.put("occupancy model {mean-psi rel bias, cover b0, cover b1, cover a1}", model);
        out.put("naive {share-detected rel bias, logistic slope on x (truth 1), slope on detection covariate (truth 0)}",
                naive);
        // abundance-induced heterogeneity
        double[] rn = new double[2];
        for (int rep = 0; rep < Math.max(1, replicates / 3); rep++) {
            ScienceSims.OccupancySurvey s = ScienceSims.occupancy(sites, visits, psiCoef, pCoef, true, seed + 500 + rep);
            double truePsi = 0;
            for (double v : s.psi()) {
                truePsi += v / sites;
            }
            int k = Math.max(1, replicates / 3);
            rn[0] += (OccupancyModel.fit(s).meanPsi() - truePsi) / truePsi / k;
            rn[1] += (OccupancyModel.royleNichols(s, 15).meanPsi() - truePsi) / truePsi / k;
        }
        out.put("heterogeneous detection mean-psi rel bias {standard model, Royle-Nichols}", rn);
        return out;
    }

    /**
     * Species distribution table.
     *
     * @param side raster side
     * @param presences expected presence records
     * @param seed the seed
     * @return metric to {AUROC, map correlation}
     */
    public static Map<String, double[]> distributions(int side, int presences, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int biased = 0; biased <= 1; biased++) {
            ScienceSims.SpeciesData d = ScienceSims.species(side, 5, presences, biased == 1 ? 3.0 : 0, 400, seed);
            String tag = biased == 1 ? "road bias: " : "unbiased: ";
            int l1Background = 4000;
            out.put(tag + "IPP", score(d, SpeciesDistribution.fit(SpeciesDistribution.Method.IPP, d.covariates(),
                    d.presences(), null, 0, 0, seed)));
            out.put(tag + "MaxEnt (l1, hinge/product)", score(d, SpeciesDistribution.fit(
                    SpeciesDistribution.Method.MAXENT, d.covariates(), d.presences(), null, 0.01, 0, seed)));
            if (biased == 1) {
                out.put(tag + "MaxEnt + bias covariate", score(d, SpeciesDistribution.fit(
                        SpeciesDistribution.Method.MAXENT, d.covariates(), d.presences(), d.road(), 0.01, 0, seed)));
                out.put(tag + "IPP + bias covariate", score(d, SpeciesDistribution.fit(
                        SpeciesDistribution.Method.IPP, d.covariates(), d.presences(), d.road(), 0, 0, seed)));
            }
            for (int bg : new int[] {100, 1000, l1Background}) {
                out.put(tag + "presence-background logistic B=" + bg, score(d, SpeciesDistribution.fit(
                        SpeciesDistribution.Method.PRESENCE_BACKGROUND, d.covariates(), d.presences(), null, 1e-4,
                        bg, seed)));
            }
            // random forest presence vs background
            SplittableRandom r = new SplittableRandom(seed);
            int n = d.presences().length + 1000;
            double[][] x = new double[n][];
            int[] y = new int[n];
            for (int i = 0; i < d.presences().length; i++) {
                x[i] = d.covariates()[d.presences()[i]];
                y[i] = 1;
            }
            for (int i = d.presences().length; i < n; i++) {
                x[i] = d.covariates()[r.nextInt(d.covariates().length)];
            }
            RandomForest rf = RandomForest.fit(x, y, 2,
                    new RandomForest.Settings(RandomForest.Mode.RANDOM_FOREST, 60, 10, 3, 0, seed));
            double[] rfScore = new double[d.covariates().length];
            for (int c = 0; c < rfScore.length; c++) {
                double pr = Math.min(0.999, Math.max(0.001, rf.predictProbability(d.covariates()[c])[1]));
                rfScore[c] = Math.log(pr / (1 - pr));
            }
            out.put(tag + "random forest (presence vs background)", metrics(d, rfScore));
            // kernel density of presence locations
            double[] kde = new double[d.covariates().length];
            double bw = side / 12.0;
            for (int c = 0; c < kde.length; c++) {
                int cx = c % side;
                int cy = c / side;
                double s = 1e-12;
                for (int pc : d.presences()) {
                    double dx = cx - pc % side;
                    double dy = cy - pc / side;
                    s += Math.exp(-(dx * dx + dy * dy) / (2 * bw * bw));
                }
                kde[c] = Math.log(s);
            }
            out.put(tag + "KDE of locations", metrics(d, kde));
        }
        return out;
    }

    private static double[] score(ScienceSims.SpeciesData d, SpeciesDistribution m) {
        double[] s = new double[d.covariates().length];
        for (int c = 0; c < s.length; c++) {
            s[c] = m.score(d.covariates()[c]);
        }
        return metrics(d, s);
    }

    private static double[] metrics(ScienceSims.SpeciesData d, double[] score) {
        double[] sv = new double[d.surveyCells().length];
        for (int i = 0; i < sv.length; i++) {
            sv[i] = score[d.surveyCells()[i]];
        }
        return new double[] {MlMetrics.auroc(sv, d.surveyPresent()), correlation(score, d.logIntensity())};
    }

    private static double correlation(double[] a, double[] b) {
        double ma = 0;
        double mb = 0;
        for (int i = 0; i < a.length; i++) {
            ma += a[i] / a.length;
            mb += b[i] / b.length;
        }
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < a.length; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return sab / Math.sqrt(saa * sbb);
    }
}
