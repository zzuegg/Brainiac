package io.github.zzuegg.brainiac.experimental.events;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A padded batch of event sequences for the neural point processes. Position 0 of every row is
 * a beginning-of-sequence token at time 0; position {@code j} holds event {@code j} (1-based).
 * The interval after position {@code j} ends either at the next event or at the horizon, and
 * every quantity is per interval: its length, whether it ends with an event, and that event's
 * type.
 */
final class EventBatch {

    final int batch;
    final int positions;
    final int types;
    final Tensor inputs;      // [B, P, types + 2]: one-hot type (BOS = types), log(1 + gap)
    final Tensor typeIds;     // [B, P] type id (BOS = types)
    final Tensor times;       // [B, P, 1] absolute time of the position's event
    final Tensor gaps;        // [B, P, 1] interval length to the next event or the horizon
    final Tensor ends;        // [B, P, 1] 1 if the interval ends with an event
    final Tensor nextType;    // [B, P, types] one-hot type of the event ending the interval
    final Tensor mask;        // [B, P, 1] 1 for real intervals
    final int events;

    EventBatch(List<EventSequence> sequences, int types) {
        this.batch = sequences.size();
        this.types = types;
        int longest = 0;
        int count = 0;
        for (EventSequence s : sequences) {
            longest = Math.max(longest, s.size());
            count += s.size();
        }
        this.events = count;
        this.positions = longest + 1;
        int p = positions;
        int f = types + 2;
        float[] in = new float[batch * p * f];
        float[] ids = new float[batch * p];
        float[] tm = new float[batch * p];
        float[] gp = new float[batch * p];
        float[] en = new float[batch * p];
        float[] nt = new float[batch * p * types];
        float[] mk = new float[batch * p];
        java.util.Arrays.fill(gp, 1f);   // padded intervals: a harmless length, masked out
        for (int b = 0; b < batch; b++) {
            EventSequence s = sequences.get(b);
            for (int j = 0; j <= s.size(); j++) {
                int at = b * p + j;
                double t = j == 0 ? 0 : s.times()[j - 1];
                double prev = j <= 1 ? 0 : s.times()[j - 2];
                int type = j == 0 ? types : s.types()[j - 1];
                in[at * f + type] = 1;
                in[at * f + types + 1] = (float) Math.log1p(t - prev);
                ids[at] = type;
                tm[at] = (float) t;
                double next = j < s.size() ? s.times()[j] : s.horizon();
                gp[at] = (float) Math.max(next - t, 1e-6);
                mk[at] = 1;
                if (j < s.size()) {
                    en[at] = 1;
                    nt[at * types + s.types()[j]] = 1;
                }
            }
        }
        this.inputs = Tensor.of(Shape.of(batch, p, f), in);
        this.typeIds = Tensor.of(Shape.of(batch, p), ids);
        this.times = Tensor.of(Shape.of(batch, p, 1), tm);
        this.gaps = Tensor.of(Shape.of(batch, p, 1), gp);
        this.ends = Tensor.of(Shape.of(batch, p, 1), en);
        this.nextType = Tensor.of(Shape.of(batch, p, types), nt);
        this.mask = Tensor.of(Shape.of(batch, p, 1), mk);
    }

    /**
     * Sample points inside each interval for a Monte Carlo compensator.
     *
     * @param samples points per interval
     * @param random the source, or null for the stratified midpoints {@code (m + 0.5) / M}
     * @return {@code [B, P, M, 1]} lags {@code u * gap}
     */
    Tensor lags(int samples, RandomGenerator random) {
        float[] g = gaps.toArray();
        float[] out = new float[g.length * samples];
        for (int i = 0; i < g.length; i++) {
            for (int m = 0; m < samples; m++) {
                double u = random == null ? (m + 0.5) / samples : (m + random.nextDouble()) / samples;
                out[i * samples + m] = (float) (u * g[i]);
            }
        }
        return Tensor.of(Shape.of(batch, positions, samples, 1), out);
    }

    /** A weight {@code [in, out]} with the Glorot range. */
    static Var weight(RandomGenerator random, int in, int out) {
        float limit = (float) Math.sqrt(6.0 / (in + out));
        return Var.param(Tensor.uniform(Shape.of(in, out), -limit, limit, random));
    }

    /** A zero bias. */
    static Var bias(int... dims) {
        return Var.param(Tensor.zeros(Shape.of(dims)));
    }

    /** Log-softmax over the last axis. */
    static Var logSoftmax(Var x) {
        return x.sub(x.logsumexp(-1, true));
    }

    /**
     * Picks the masked per-interval log-likelihood terms and returns minus their sum per event.
     */
    Var negativeMeanPerEvent(Var perInterval) {
        return perInterval.mul(mask).sum().neg().div(Math.max(1, events));
    }
}
