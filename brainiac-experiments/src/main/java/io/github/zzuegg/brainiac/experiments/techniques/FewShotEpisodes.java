package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.experimental.data.FewShotEpisode;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Few-shot episode sources over 8x8 images (64 features). Meta-training draws classes from
 * the training classes of a domain; testing draws from classes never seen in training.
 *
 * <ul>
 *   <li>{@link #shapes(long)}: {@link ShapeImages}, train on shapes 0-3, test on shapes 4 and 5.</li>
 *   <li>{@link #mixed(long)}: shapes plus the MNIST-like {@link ToyTasks#digits}; an episode
 *       comes from one domain chosen at random (digits train on 0-6, test on 7-9), the
 *       mixed-domain setting CNAPs targets.</li>
 * </ul>
 */
@SuppressWarnings("exports")
public final class FewShotEpisodes {

    /** The pixels per image. */
    public static final int FEATURES = 64;

    private record Domain(List<List<float[]>> byClass, int[] train, int[] test) {
    }

    private final List<Domain> domains;

    private FewShotEpisodes(List<Domain> domains) {
        this.domains = domains;
    }

    /**
     * The single-domain shapes source.
     *
     * @param seed the image seed
     * @return the source
     */
    public static FewShotEpisodes shapes(long seed) {
        return new FewShotEpisodes(List.of(shapeDomain(seed)));
    }

    /**
     * The two-domain source: shapes and digits.
     *
     * @param seed the image seed
     * @return the source
     */
    public static FewShotEpisodes mixed(long seed) {
        return new FewShotEpisodes(List.of(shapeDomain(seed), digitDomain(seed)));
    }

    private static Domain shapeDomain(long seed) {
        Tensor[] pool = new ShapeImages(8, 0.1f).generate(3000, seed + 5);
        List<List<float[]>> out = new ArrayList<>();
        for (int c = 0; c < ShapeImages.CLASSES; c++) {
            out.add(new ArrayList<>());
        }
        float[] x = pool[0].toArray();
        float[] y = pool[1].toArray();
        for (int i = 0; i < y.length; i++) {
            float[] img = new float[FEATURES];
            System.arraycopy(x, i * FEATURES, img, 0, FEATURES);
            out.get((int) y[i]).add(img);
        }
        return new Domain(out, new int[] {0, 1, 2, 3}, new int[] {4, 5});
    }

    private static Domain digitDomain(long seed) {
        ToyTasks.Data d = ToyTasks.digits(10, 200, 8, 0.05, seed + 11);
        List<List<float[]>> out = new ArrayList<>();
        for (int c = 0; c < 10; c++) {
            out.add(new ArrayList<>());
        }
        for (int i = 0; i < d.size(); i++) {
            out.get(d.y()[i]).add(d.x()[i]);
        }
        return new Domain(out, new int[] {0, 1, 2, 3, 4, 5, 6}, new int[] {7, 8, 9});
    }

    /**
     * A meta-training episode.
     *
     * @param ways classes per episode
     * @param shots support examples per class
     * @param queries query examples per class
     * @param random the draw
     * @return the episode, support rows in random order
     */
    public FewShotEpisode train(int ways, int shots, int queries, SplittableRandom random) {
        return episode(true, ways, shots, queries, random);
    }

    /**
     * A test episode from held-out classes.
     *
     * @param ways classes per episode
     * @param shots support examples per class
     * @param queries query examples per class
     * @param random the draw
     * @return the episode
     */
    public FewShotEpisode test(int ways, int shots, int queries, SplittableRandom random) {
        return episode(false, ways, shots, queries, random);
    }

    /**
     * Pretraining data from the training classes of every domain (for extractors that are
     * trained by plain classification first, as LEO's are).
     *
     * @param perClass images per class
     * @param random the draw
     * @return the data, one class id per training class across domains
     */
    public ToyTasks.Data pretraining(int perClass, SplittableRandom random) {
        List<float[]> xs = new ArrayList<>();
        List<Integer> ys = new ArrayList<>();
        int id = 0;
        for (Domain d : domains) {
            for (int c : d.train()) {
                List<float[]> imgs = d.byClass().get(c);
                for (int i = 0; i < perClass; i++) {
                    xs.add(imgs.get(random.nextInt(imgs.size())));
                    ys.add(id);
                }
                id++;
            }
        }
        int[] y = new int[ys.size()];
        for (int i = 0; i < y.length; i++) {
            y[i] = ys.get(i);
        }
        return new ToyTasks.Data(xs.toArray(new float[0][]), y, id);
    }

    private FewShotEpisode episode(boolean train, int ways, int shots, int queries, SplittableRandom r) {
        Domain d = domains.get(r.nextInt(domains.size()));
        int[] pool = (train ? d.train() : d.test()).clone();
        for (int i = pool.length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = pool[i];
            pool[i] = pool[j];
            pool[j] = t;
        }
        int n = ways * shots;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        float[] s = new float[n * FEATURES];
        int[] sl = new int[n];
        float[] q = new float[ways * queries * FEATURES];
        int[] ql = new int[ways * queries];
        for (int k = 0; k < ways; k++) {
            List<float[]> imgs = d.byClass().get(pool[k]);
            for (int i = 0; i < shots; i++) {
                int row = order[k * shots + i];
                System.arraycopy(imgs.get(r.nextInt(imgs.size())), 0, s, row * FEATURES, FEATURES);
                sl[row] = k;
            }
            for (int i = 0; i < queries; i++) {
                System.arraycopy(imgs.get(r.nextInt(imgs.size())), 0, q, (k * queries + i) * FEATURES, FEATURES);
                ql[k * queries + i] = k;
            }
        }
        return new FewShotEpisode(Tensor.of(Shape.of(n, FEATURES), s), sl,
                Tensor.of(Shape.of(ways * queries, FEATURES), q), ql, ways);
    }
}
