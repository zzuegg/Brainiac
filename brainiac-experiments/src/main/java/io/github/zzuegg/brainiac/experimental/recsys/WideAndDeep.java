package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FactorizationMachine;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Two classic click-through-rate architectures over one-hot categorical fields, as a
 * {@link Layer} from {@code [batch, fields * categories]} one-hot rows to one click logit.
 * <ul>
 *   <li>{@link #wideDeep} (Cheng et al., 2016, <a href="https://arxiv.org/abs/1606.07792">
 *       arXiv:1606.07792</a>): a <em>wide</em> linear model over the raw one-hots plus all crossed
 *       features (the outer product of every pair of fields: memorisation), trained jointly with a
 *       <em>deep</em> MLP over learned per-field embeddings (generalisation);</li>
 *   <li>{@link #deepFm} (Guo et al., 2017, <a href="https://arxiv.org/abs/1703.04247">
 *       arXiv:1703.04247</a>): the wide part is replaced by a factorization machine
 *       ({@link FactorizationMachine}) whose factor vectors are also the deep part's embeddings,
 *       so no manual crosses are needed and the two parts share what they learn.</li>
 * </ul>
 */
public final class WideAndDeep implements Layer {

    private final int fields;
    private final int categories;
    private final boolean fm;
    private final Var wide;
    private final Var wideBias;
    private final FactorizationMachine machine;
    private final Var embeddings;
    private final Var h1;
    private final Var b1;
    private final Var h2;
    private final Var b2;
    private final Var out;

    private WideAndDeep(int fields, int categories, int embed, int hidden, boolean fm, SplittableRandom r) {
        this.fields = fields;
        this.categories = categories;
        this.fm = fm;
        int width = fields * categories;
        if (fm) {
            machine = new FactorizationMachine(width, embed, r);
            embeddings = machine.factors();
            wide = null;
            wideBias = null;
        } else {
            machine = null;
            embeddings = RecsysOps.normal(r, 0.1f, width, embed);
            int crosses = fields * (fields - 1) / 2 * categories * categories;
            wide = RecsysOps.zeros(width + crosses, 1);
            wideBias = RecsysOps.zeros(1);
        }
        h1 = RecsysOps.matrix(r, fields * embed, hidden);
        b1 = RecsysOps.zeros(hidden);
        h2 = RecsysOps.matrix(r, hidden, hidden);
        b2 = RecsysOps.zeros(hidden);
        out = RecsysOps.matrix(r, hidden, 1);
    }

    /**
     * Wide &amp; Deep.
     *
     * @param fields categorical fields
     * @param categories categories per field
     * @param embed the embedding size per field
     * @param hidden the deep part's hidden width
     * @param seed the seed
     * @return the layer
     */
    public static WideAndDeep wideDeep(int fields, int categories, int embed, int hidden, long seed) {
        return new WideAndDeep(fields, categories, embed, hidden, false, new SplittableRandom(seed));
    }

    /**
     * DeepFM.
     *
     * @param fields categorical fields
     * @param categories categories per field
     * @param embed the shared factor / embedding size
     * @param hidden the deep part's hidden width
     * @param seed the seed
     * @return the layer
     */
    public static WideAndDeep deepFm(int fields, int categories, int embed, int hidden, long seed) {
        return new WideAndDeep(fields, categories, embed, hidden, true, new SplittableRandom(seed));
    }

    /**
     * @param fields categorical fields
     * @param categories categories per field
     * @param embed the embedding size
     * @param hidden the hidden width
     * @param deepFm whether to build DeepFM rather than Wide &amp; Deep
     * @return a spec for model builders
     */
    public static LayerSpec spec(int fields, int categories, int embed, int hidden, boolean deepFm) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                if (input.dim(-1) != fields * categories) {
                    throw new IllegalArgumentException("expected " + fields * categories + " one-hot columns");
                }
                return Shape.of(1);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new WideAndDeep(fields, categories, embed, hidden, deepFm, random);
            }
        };
    }

    private Var field(Var x, int f) {
        return x.slice(1, f * categories, (f + 1) * categories);
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        List<Var> embedded = new ArrayList<>();
        for (int f = 0; f < fields; f++) {
            embedded.add(field(input, f).matmul(embeddings.slice(0, f * categories, (f + 1) * categories)));
        }
        Var deep = Var.concat(1, embedded.toArray(Var[]::new)).matmul(h1).add(b1).relu().matmul(h2).add(b2).relu()
                .matmul(out);
        if (fm) {
            return deep.add(machine.forward(input));
        }
        List<Var> parts = new ArrayList<>(List.of(input));
        for (int f = 0; f < fields; f++) {
            for (int g = f + 1; g < fields; g++) {
                parts.add(field(input, f).reshape(b, categories, 1).mul(field(input, g).reshape(b, 1, categories))
                        .reshape(b, categories * categories));
            }
        }
        return deep.add(Var.concat(1, parts.toArray(Var[]::new)).matmul(wide).add(wideBias));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(fm ? machine.parameters() : List.of(embeddings, wide, wideBias));
        p.addAll(List.of(h1, b1, h2, b2, out));
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return 1;
    }

    @Override
    public String toString() {
        return (fm ? "DeepFM" : "WideAndDeep") + "(" + fields + "x" + categories + ")";
    }
}
