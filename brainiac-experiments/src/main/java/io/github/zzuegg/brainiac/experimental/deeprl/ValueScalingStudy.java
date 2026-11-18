package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.eval.CurveExtrapolator;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Value-based scaling studies (Rybkin et al., "Value-Based Deep RL Scales Predictably", ICML
 * 2025, <a href="https://arxiv.org/abs/2502.04327">arXiv:2502.04327</a>; Fu et al., 2025,
 * <a href="https://arxiv.org/abs/2508.14881">arXiv:2508.14881</a>): grids over critic width and
 * batch size whose outcome is the held-out TD error, so the interaction the papers report can be
 * measured cheaply and without the noise of returns.
 *
 * <p>A {@link Dataset} of a fixed behaviour policy's transitions is split into training and
 * held-out episodes; a critic is trained by TD policy evaluation ({@code y = r + gamma
 * Q_target(s', a')} with a Polyak target network) for a fixed number of gradient steps, and is
 * scored by its TD error and by its error against the Monte Carlo return on held-out episodes.
 * TD-overfitting (Fu et al.) shows as held-out TD error rising with the batch size for narrow
 * critics while wide critics benefit. {@link #passive} trains a critic on another (wide)
 * critic's targets instead of its own, the experiment that separates target quality from
 * capacity. {@link #fitBestBatch} fits the best batch size as a power law of the width and
 * {@link #fitWidthLaw} the held-out error against width with {@link CurveExtrapolator}.
 */
public final class ValueScalingStudy {

    private ValueScalingStudy() {
    }

    /**
     * Transitions of a behaviour policy with their discounted Monte Carlo returns.
     *
     * @param observations observations
     * @param actions actions taken
     * @param rewards rewards
     * @param next next observations
     * @param nextActions the behaviour's next actions (0 at episode ends)
     * @param notDone 0 where the next state is terminal
     * @param returns discounted returns from each step to the episode end
     */
    public record Dataset(float[][] observations, int[] actions, float[] rewards, float[][] next, int[] nextActions,
            float[] notDone, float[] returns) {
        /** @return the number of transitions */
        public int size() {
            return rewards.length;
        }
    }

    /**
     * Logs episodes of a discrete-action behaviour policy.
     *
     * @param env the environment
     * @param behaviour the policy (asked with {@code explore = true})
     * @param episodes episodes
     * @param gamma the discount of the returns
     * @param seed the seed
     * @return the data
     */
    public static Dataset collect(Environment env, Agent behaviour, int episodes, float gamma, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<float[]> o = new ArrayList<>();
        List<Integer> a = new ArrayList<>();
        List<Float> r = new ArrayList<>();
        List<float[]> n = new ArrayList<>();
        List<Integer> na = new ArrayList<>();
        List<Float> nd = new ArrayList<>();
        List<Float> g = new ArrayList<>();
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset(random.split());
            int act = Math.round(behaviour.act(obs, true)[0]);
            int start = r.size();
            Environment.Step s;
            do {
                s = env.step(act);
                int nextAct = s.done() ? 0 : Math.round(behaviour.act(s.observation(), true)[0]);
                o.add(obs);
                a.add(act);
                r.add(s.reward());
                n.add(s.observation());
                na.add(nextAct);
                nd.add(s.terminated() ? 0f : 1f);
                obs = s.observation();
                act = nextAct;
            } while (!s.done());
            float ret = 0;
            float[] tail = new float[r.size() - start];
            for (int i = r.size() - 1; i >= start; i--) {
                ret = r.get(i) + gamma * ret;
                tail[i - start] = ret;
            }
            for (float t : tail) {
                g.add(t);
            }
        }
        int size = r.size();
        int[] ai = new int[size];
        int[] nai = new int[size];
        float[] rf = new float[size];
        float[] ndf = new float[size];
        float[] gf = new float[size];
        for (int i = 0; i < size; i++) {
            ai[i] = a.get(i);
            nai[i] = na.get(i);
            rf[i] = r.get(i);
            ndf[i] = nd.get(i);
            gf[i] = g.get(i);
        }
        return new Dataset(o.toArray(new float[0][]), ai, rf, n.toArray(new float[0][]), nai, ndf, gf);
    }

    /**
     * One grid point.
     *
     * @param width critic width
     * @param batch batch size
     * @param trainTd mean absolute TD error on the training data
     * @param heldOutTd mean absolute TD error on held-out episodes
     * @param heldOutMc mean absolute error against held-out Monte Carlo returns
     */
    public record Point(int width, int batch, double trainTd, double heldOutTd, double heldOutMc) {
    }

    /** A trained critic and its target network. */
    private record Critic(Model online, Model target) {
    }

    private static Critic critic(int obs, int actions, int width, long seed) {
        Model q = Nets.mlp(obs, actions, Activation.RELU, seed, width, width);
        Model t = Nets.mlp(obs, actions, Activation.RELU, seed, width, width);
        RlTensors.copy(q.parameters(), t.parameters());
        return new Critic(q, t);
    }

    private static Tensor pick(Model m, float[][] obs, int[] actions, int[] rows, int actionCount) {
        float[][] x = new float[rows.length][];
        float[] a = new float[rows.length];
        for (int i = 0; i < rows.length; i++) {
            x[i] = obs[rows[i]];
            a[i] = actions[rows[i]];
        }
        Tensor in = RlTensors.rows(x, x[0].length, m.device());
        Tensor hot = RlTensors.oneHot(a, actionCount, m.device());
        return Var.noGrad(() -> m.forward(Var.of(in), false)).value().mul(hot).sum(1, true);
    }

    private static float[] targets(Model target, Dataset d, int[] rows, int actions, float gamma) {
        float[] q = pick(target, d.next(), d.nextActions(), rows, actions).toArray();
        float[] y = new float[rows.length];
        for (int i = 0; i < rows.length; i++) {
            y[i] = d.rewards()[rows[i]] + gamma * d.notDone()[rows[i]] * q[i];
        }
        return y;
    }

    /** Trains a critic by TD policy evaluation; targets come from {@code targetSource} or its own target network. */
    private static Critic train(Dataset d, int actions, int width, int batch, int steps, float lr, float gamma,
            Critic targetSource, long seed) {
        int obs = d.observations()[0].length;
        Critic c = critic(obs, actions, width, seed);
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom random = new SplittableRandom(seed);
        int[] rows = new int[batch];
        for (int s = 0; s < steps; s++) {
            for (int i = 0; i < batch; i++) {
                rows[i] = random.nextInt(d.size());
            }
            Model tgt = targetSource == null ? c.target() : targetSource.target();
            float[] y = targets(tgt, d, rows, actions, gamma);
            float[][] x = new float[batch][];
            float[] a = new float[batch];
            for (int i = 0; i < batch; i++) {
                x[i] = d.observations()[rows[i]];
                a[i] = d.actions()[rows[i]];
            }
            Var q = c.online().forward(Var.of(RlTensors.rows(x, obs, c.online().device())), true)
                    .mul(RlTensors.oneHot(a, actions, c.online().device())).sum(1, true);
            Var loss = q.sub(RlTensors.column(y, c.online().device())).square().mean();
            loss.backward();
            opt.step(c.online().parameters());
            RlTensors.blend(c.online().parameters(), c.target().parameters(), 0.01f);
        }
        return c;
    }

    private static double tdError(Critic c, Dataset d, int actions, float gamma) {
        int[] rows = new int[d.size()];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = i;
        }
        float[] q = pick(c.online(), d.observations(), d.actions(), rows, actions).toArray();
        float[] y = targets(c.target(), d, rows, actions, gamma);
        double sum = 0;
        for (int i = 0; i < rows.length; i++) {
            sum += Math.abs(q[i] - y[i]);
        }
        return sum / rows.length;
    }

    private static double mcError(Critic c, Dataset d, int actions) {
        int[] rows = new int[d.size()];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = i;
        }
        float[] q = pick(c.online(), d.observations(), d.actions(), rows, actions).toArray();
        double sum = 0;
        for (int i = 0; i < rows.length; i++) {
            sum += Math.abs(q[i] - d.returns()[i]);
        }
        return sum / rows.length;
    }

    /**
     * Trains one critic per (width, batch) cell for the same number of gradient steps.
     *
     * @param train training data
     * @param heldOut held-out episodes
     * @param actions discrete actions
     * @param widths critic widths
     * @param batches batch sizes
     * @param steps gradient steps per cell
     * @param lr the learning rate
     * @param gamma the discount
     * @param seed the seed
     * @return the grid, width-major
     */
    public static List<Point> grid(Dataset train, Dataset heldOut, int actions, int[] widths, int[] batches, int steps,
            float lr, float gamma, long seed) {
        List<Point> out = new ArrayList<>();
        for (int w : widths) {
            for (int b : batches) {
                Critic c = train(train, actions, w, b, steps, lr, gamma, null, seed);
                out.add(new Point(w, b, tdError(c, train, actions, gamma), tdError(c, heldOut, actions, gamma),
                        mcError(c, heldOut, actions)));
            }
        }
        return out;
    }

    /**
     * The passive-critic experiment: a narrow critic trained on its own targets, and the same
     * narrow critic trained on targets from a wide critic (trained first on the same data).
     *
     * @param train training data
     * @param heldOut held-out episodes
     * @param actions discrete actions
     * @param narrow the narrow width
     * @param wide the wide width
     * @param batch the batch size
     * @param steps gradient steps
     * @param lr the learning rate
     * @param gamma the discount
     * @param seed the seed
     * @return {own targets, wide critic's targets, the wide critic itself}
     */
    public static Point[] passive(Dataset train, Dataset heldOut, int actions, int narrow, int wide, int batch,
            int steps, float lr, float gamma, long seed) {
        Critic big = train(train, actions, wide, batch, steps, lr, gamma, null, seed);
        Critic own = train(train, actions, narrow, batch, steps, lr, gamma, null, seed);
        Critic pas = train(train, actions, narrow, batch, steps, lr, gamma, big, seed);
        return new Point[] {
            new Point(narrow, batch, tdError(own, train, actions, gamma), tdError(own, heldOut, actions, gamma),
                    mcError(own, heldOut, actions)),
            new Point(narrow, batch, tdError(pas, train, actions, gamma), tdError(pas, heldOut, actions, gamma),
                    mcError(pas, heldOut, actions)),
            new Point(wide, batch, tdError(big, train, actions, gamma), tdError(big, heldOut, actions, gamma),
                    mcError(big, heldOut, actions))
        };
    }

    /**
     * The best batch size per width (lowest held-out TD error) and a power-law fit
     * {@code batch* = a width^b} by least squares in log-log space.
     *
     * @param grid the grid
     * @return {a, b}
     */
    public static double[] fitBestBatch(List<Point> grid) {
        List<double[]> best = new ArrayList<>();
        grid.stream().mapToInt(Point::width).distinct().forEach(w -> {
            Point p = grid.stream().filter(q -> q.width() == w).min((x, y) -> Double.compare(x.heldOutTd(),
                    y.heldOutTd())).orElseThrow();
            best.add(new double[] {Math.log(w), Math.log(p.batch())});
        });
        double mx = 0;
        double my = 0;
        for (double[] p : best) {
            mx += p[0];
            my += p[1];
        }
        mx /= best.size();
        my /= best.size();
        double sxy = 0;
        double sxx = 0;
        for (double[] p : best) {
            sxy += (p[0] - mx) * (p[1] - my);
            sxx += (p[0] - mx) * (p[0] - mx);
        }
        double b = sxx == 0 ? 0 : sxy / sxx;
        return new double[] {Math.exp(my - b * mx), b};
    }

    /**
     * Fits held-out TD error against width at one batch size with a power law
     * {@code c + a width^-alpha} ({@link CurveExtrapolator#powerLawLeastSquares}).
     *
     * @param grid the grid
     * @param batch the batch size
     * @return the fit
     */
    public static CurveExtrapolator.Fit fitWidthLaw(List<Point> grid, int batch) {
        List<Point> rows = grid.stream().filter(p -> p.batch() == batch).toList();
        double[] x = rows.stream().mapToDouble(Point::width).toArray();
        double[] y = rows.stream().mapToDouble(Point::heldOutTd).toArray();
        return CurveExtrapolator.powerLawLeastSquares(x, y);
    }
}
