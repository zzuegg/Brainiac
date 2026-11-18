package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The CLEAR-MOT and identity metrics of multi-object tracking (Bernardin and Stiefelhagen,
 * 2008; Ristani et al., 2016), over one sequence:
 *
 * <ul>
 *   <li>MOTA {@code = 1 - (misses + false positives + ID switches) / ground-truth boxes},
 *       with tracks matched to ground truth per frame at IoU 0.5 or more (keeping last
 *       frame's pairs when they still overlap, then by the Hungarian algorithm);</li>
 *   <li>ID switches: the times a ground-truth object's matched track id changes;</li>
 *   <li>IDF1: the F1 score of the best one-to-one mapping between ground-truth and track
 *       identities over the whole sequence, {@code 2 IDTP / (2 IDTP + IDFP + IDFN)}, which
 *       rewards keeping one identity per object.</li>
 * </ul>
 */
public final class TrackingMetrics {

    private TrackingMetrics() {
    }

    /**
     * The result.
     *
     * @param mota multi-object tracking accuracy
     * @param idf1 identity F1
     * @param idSwitches identity switches
     * @param misses missed ground-truth boxes
     * @param falsePositives unmatched track boxes
     */
    public record Result(double mota, double idf1, int idSwitches, int misses, int falsePositives) {
    }

    /**
     * Scores one sequence.
     *
     * @param truth per frame the ground-truth outputs (id and box)
     * @param tracks per frame the tracker outputs
     * @return the metrics
     */
    public static Result evaluate(List<List<TrackingByDetection.Output>> truth,
            List<List<TrackingByDetection.Output>> tracks) {
        Map<Integer, Integer> lastMatch = new HashMap<>();
        int gtTotal = 0;
        int misses = 0;
        int fp = 0;
        int switches = 0;
        Map<Long, Integer> pairCounts = new HashMap<>();
        Map<Integer, Integer> gtCounts = new HashMap<>();
        Map<Integer, Integer> trCounts = new HashMap<>();
        for (int f = 0; f < truth.size(); f++) {
            List<TrackingByDetection.Output> g = truth.get(f);
            List<TrackingByDetection.Output> t = tracks.get(f);
            gtTotal += g.size();
            for (TrackingByDetection.Output o : g) {
                gtCounts.merge(o.id(), 1, Integer::sum);
            }
            for (TrackingByDetection.Output o : t) {
                trCounts.merge(o.id(), 1, Integer::sum);
            }
            int[] match = new int[g.size()];
            java.util.Arrays.fill(match, -1);
            boolean[] trackUsed = new boolean[t.size()];
            for (int i = 0; i < g.size(); i++) {
                Integer prev = lastMatch.get(g.get(i).id());
                for (int j = 0; prev != null && j < t.size(); j++) {
                    if (!trackUsed[j] && t.get(j).id() == prev
                            && DetectionMetrics.iou(g.get(i).box(), t.get(j).box()) >= 0.5) {
                        match[i] = j;
                        trackUsed[j] = true;
                    }
                }
            }
            int n = Math.max(g.size(), t.size());
            if (n > 0) {
                double[][] cost = new double[n][n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        boolean valid = i < g.size() && j < t.size() && match[i] < 0 && !trackUsed[j];
                        double o = valid ? DetectionMetrics.iou(g.get(i).box(), t.get(j).box()) : 0;
                        cost[i][j] = o >= 0.5 ? 1 - o : 2;
                    }
                }
                int[] assign = MinibatchOptimalTransport.assignCost(cost);
                for (int i = 0; i < g.size(); i++) {
                    if (match[i] < 0 && assign[i] < t.size() && cost[i][assign[i]] <= 1) {
                        match[i] = assign[i];
                        trackUsed[assign[i]] = true;
                    }
                }
            }
            for (int i = 0; i < g.size(); i++) {
                if (match[i] < 0) {
                    misses++;
                    continue;
                }
                int tid = t.get(match[i]).id();
                Integer prev = lastMatch.put(g.get(i).id(), tid);
                if (prev != null && prev != tid) {
                    switches++;
                }
            }
            for (boolean u : trackUsed) {
                fp += u ? 0 : 1;
            }
            for (int i = 0; i < g.size(); i++) {
                for (TrackingByDetection.Output o : t) {
                    if (DetectionMetrics.iou(g.get(i).box(), o.box()) >= 0.5) {
                        pairCounts.merge(((long) g.get(i).id() << 32) | (o.id() & 0xffffffffL), 1, Integer::sum);
                    }
                }
            }
        }
        double mota = gtTotal == 0 ? 1 : 1 - (double) (misses + fp + switches) / gtTotal;
        Integer[] gtIds = gtCounts.keySet().toArray(new Integer[0]);
        Integer[] trIds = trCounts.keySet().toArray(new Integer[0]);
        int n = Math.max(gtIds.length, trIds.length);
        int idtp = 0;
        if (n > 0) {
            double[][] cost = new double[n][n];
            for (int i = 0; i < gtIds.length; i++) {
                for (int j = 0; j < trIds.length; j++) {
                    cost[i][j] = -pairCounts.getOrDefault(((long) gtIds[i] << 32) | (trIds[j] & 0xffffffffL), 0);
                }
            }
            int[] assign = MinibatchOptimalTransport.assignCost(cost);
            for (int i = 0; i < gtIds.length; i++) {
                idtp += (int) -cost[i][assign[i]];
            }
        }
        int gtBoxes = gtCounts.values().stream().mapToInt(Integer::intValue).sum();
        int trBoxes = trCounts.values().stream().mapToInt(Integer::intValue).sum();
        double idf1 = gtBoxes + trBoxes == 0 ? 1 : 2.0 * idtp / (gtBoxes + trBoxes);
        return new Result(mota, idf1, switches, misses, fp);
    }
}
