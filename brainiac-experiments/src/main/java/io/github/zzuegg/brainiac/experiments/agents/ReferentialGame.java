package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GumbelChannel;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Emergent communication in a Lewis referential game (Lazaridou et al.,
 * <a href="https://arxiv.org/abs/1612.07182">1612.07182</a>; Havrylov and Titov,
 * <a href="https://arxiv.org/abs/1705.11192">1705.11192</a>). Objects have three attributes of
 * four values. A sender sees the target and sends a message of three symbols through a
 * {@link GumbelChannel}; a receiver sees the message and four candidates and must point at
 * the target. Both are trained together end to end, and they invent a code. The code is
 * judged on objects never seen in training (does it generalize?) and by its topographic
 * similarity: the correlation between how different two objects are and how different their
 * messages are, high when the language is compositional.
 */
public final class ReferentialGame {

    private static final int ATTRIBUTES = 3;
    private static final int VALUES = 4;
    private static final int FEATURES = ATTRIBUTES * VALUES;
    private static final int CANDIDATES = 4;

    /**
     * @param trainAccuracy accuracy with greedy messages on the training objects
     * @param testAccuracy on held-out objects
     * @param chance the accuracy of guessing
     * @param distinctMessages distinct messages over all objects
     * @param topographicSimilarity Pearson correlation of meaning and message Hamming distances
     */
    public record Result(double trainAccuracy, double testAccuracy, double chance, int distinctMessages,
            double topographicSimilarity) {
    }

    private ReferentialGame() {
    }

    static int[] attributes(int object) {
        int[] a = new int[ATTRIBUTES];
        for (int i = 0; i < ATTRIBUTES; i++) {
            a[i] = object % VALUES;
            object /= VALUES;
        }
        return a;
    }

    private static float[] features(int object) {
        float[] x = new float[FEATURES];
        int[] a = attributes(object);
        for (int i = 0; i < ATTRIBUTES; i++) {
            x[i * VALUES + a[i]] = 1;
        }
        return x;
    }

    private static Tensor rows(int[] objects) {
        float[] x = new float[objects.length * FEATURES];
        for (int i = 0; i < objects.length; i++) {
            System.arraycopy(features(objects[i]), 0, x, i * FEATURES, FEATURES);
        }
        return Tensor.of(Shape.of(objects.length, FEATURES), x);
    }

    /**
     * @param steps training steps
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(int steps, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int objects = (int) Math.pow(VALUES, ATTRIBUTES);
        List<Integer> all = new ArrayList<>();
        for (int i = 0; i < objects; i++) {
            all.add(i);
        }
        for (int i = all.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            all.set(i, all.set(j, all.get(i)));
        }
        List<Integer> train = all.subList(0, objects - 12);
        List<Integer> test = all.subList(objects - 12, objects);
        GumbelChannel channel = GumbelChannel.of(3, 6, 1f, seed);
        Model sender = Model.sequential().input(FEATURES).dense(64, Activation.TANH).dense(channel.width())
                .seed(seed).build();
        Model listener = Model.sequential().input(channel.width()).dense(64, Activation.TANH).dense(32)
                .seed(seed + 1).build();
        Model viewer = Model.sequential().input(FEATURES).dense(64, Activation.TANH).dense(32).seed(seed + 2)
                .build();
        List<Var> params = new ArrayList<>(sender.parameters());
        params.addAll(listener.parameters());
        params.addAll(viewer.parameters());
        Optimizer opt = Optimizer.adam(5e-3f);
        int batch = 32;
        for (int step = 0; step < steps; step++) {
            int[][] game = games(train, batch, random);
            Var message = channel.sample(sender.forward(Var.of(rows(game[0])), true), true);
            Var loss = Functional.crossEntropy(scores(listener, viewer, message, game[1], batch, true),
                    targets(game[2]));
            loss.backward();
            opt.step(params);
        }
        return new Result(accuracy(sender, listener, viewer, channel, train, random),
                accuracy(sender, listener, viewer, channel, test, random), 1.0 / CANDIDATES,
                distinct(sender, channel, objects), topographic(sender, channel, objects));
    }

    /** Games: targets, the candidates (row-major, target among them), the target's slot. */
    private static int[][] games(List<Integer> pool, int batch, SplittableRandom random) {
        int[] targets = new int[batch];
        int[] candidates = new int[batch * CANDIDATES];
        int[] slots = new int[batch];
        for (int b = 0; b < batch; b++) {
            targets[b] = pool.get(random.nextInt(pool.size()));
            slots[b] = random.nextInt(CANDIDATES);
            for (int k = 0; k < CANDIDATES; k++) {
                int c;
                do {
                    c = pool.get(random.nextInt(pool.size()));
                } while (c == targets[b] && k != slots[b]);
                candidates[b * CANDIDATES + k] = k == slots[b] ? targets[b] : c;
            }
        }
        return new int[][] {targets, candidates, slots};
    }

    private static Tensor targets(int[] slots) {
        float[] t = new float[slots.length];
        for (int i = 0; i < slots.length; i++) {
            t[i] = slots[i];
        }
        return Tensor.of(Shape.of(slots.length), t);
    }

    private static Var scores(Model listener, Model viewer, Var message, int[] candidates, int batch,
            boolean training) {
        Var m = listener.forward(message, training).reshape(batch, 32, 1);
        Var c = viewer.forward(Var.of(rows(candidates)), training).reshape(batch, CANDIDATES, 32);
        return c.matmul(m).reshape(batch, CANDIDATES);
    }

    private static double accuracy(Model sender, Model listener, Model viewer, GumbelChannel channel,
            List<Integer> targetsFrom, SplittableRandom random) {
        List<Integer> pool = new ArrayList<>();
        for (int i = 0; i < (int) Math.pow(VALUES, ATTRIBUTES); i++) {
            pool.add(i);
        }
        int n = 400;
        int[][] game = games(pool, n, random);
        for (int b = 0; b < n; b++) {
            int target = targetsFrom.get(random.nextInt(targetsFrom.size()));
            game[0][b] = target;
            for (int k = 0; k < CANDIDATES; k++) {
                if (game[1][b * CANDIDATES + k] == target) {
                    game[1][b * CANDIDATES + k] = (target + 1) % pool.size();
                }
            }
            game[1][b * CANDIDATES + game[2][b]] = target;
        }
        Tensor message = channel.greedy(sender.forward(Var.of(rows(game[0])), false).value());
        float[] s = scores(listener, viewer, Var.of(message), game[1], n, false).value().toArray();
        int right = 0;
        for (int b = 0; b < n; b++) {
            int best = 0;
            for (int k = 1; k < CANDIDATES; k++) {
                if (s[b * CANDIDATES + k] > s[b * CANDIDATES + best]) {
                    best = k;
                }
            }
            right += best == game[2][b] ? 1 : 0;
        }
        return (double) right / n;
    }

    private static int[][] messages(Model sender, GumbelChannel channel, int objects) {
        int[] all = new int[objects];
        for (int i = 0; i < objects; i++) {
            all[i] = i;
        }
        float[] m = channel.greedy(sender.forward(Var.of(rows(all)), false).value()).toArray();
        int[][] out = new int[objects][];
        int w = channel.width();
        for (int i = 0; i < objects; i++) {
            out[i] = channel.symbols(Arrays.copyOfRange(m, i * w, (i + 1) * w));
        }
        return out;
    }

    private static int distinct(Model sender, GumbelChannel channel, int objects) {
        Set<String> set = new HashSet<>();
        for (int[] m : messages(sender, channel, objects)) {
            set.add(Arrays.toString(m));
        }
        return set.size();
    }

    private static double topographic(Model sender, GumbelChannel channel, int objects) {
        int[][] msg = messages(sender, channel, objects);
        List<double[]> pairs = new ArrayList<>();
        for (int i = 0; i < objects; i++) {
            for (int j = i + 1; j < objects; j++) {
                pairs.add(new double[] {hamming(attributes(i), attributes(j)), hamming(msg[i], msg[j])});
            }
        }
        double mx = 0;
        double my = 0;
        for (double[] p : pairs) {
            mx += p[0];
            my += p[1];
        }
        mx /= pairs.size();
        my /= pairs.size();
        double sxy = 0;
        double sxx = 0;
        double syy = 0;
        for (double[] p : pairs) {
            sxy += (p[0] - mx) * (p[1] - my);
            sxx += (p[0] - mx) * (p[0] - mx);
            syy += (p[1] - my) * (p[1] - my);
        }
        return syy == 0 ? 0 : sxy / Math.sqrt(sxx * syy);
    }

    private static int hamming(int[] a, int[] b) {
        int d = 0;
        for (int i = 0; i < a.length; i++) {
            d += a[i] != b[i] ? 1 : 0;
        }
        return d;
    }
}
