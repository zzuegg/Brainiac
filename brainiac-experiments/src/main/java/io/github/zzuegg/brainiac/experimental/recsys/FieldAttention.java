package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * AutoInt (Song et al., CIKM 2019, <a href="https://arxiv.org/abs/1810.11921">arXiv:1810.11921</a>)
 * as a {@link Layer} from {@code [batch, fields * categories]} one-hot rows to one click logit.
 * Every field is embedded, and multi-head self-attention runs <em>across fields</em> with a
 * residual projection and ReLU, {@code h' = relu(attention(h) + h W_res)}; each layer raises the
 * order of the feature interactions it can express by one, and its attention weights show which
 * field crosses matter. The final field representations are flattened into a linear output (plus a
 * logistic term over the raw one-hots).
 */
public final class FieldAttention implements Layer {

    private final int fields;
    private final int categories;
    private final int embed;
    private final int heads;
    private final int headDim;
    private final Var table;
    private final List<Var[]> layers = new ArrayList<>();
    private final Var out;
    private final Var linear;
    private final Var bias;

    /**
     * Creates the layer.
     *
     * @param fields categorical fields
     * @param categories categories per field
     * @param embed the field embedding width
     * @param heads attention heads
     * @param headDim the width of each head
     * @param depth attention layers
     * @param random the initializer
     */
    public FieldAttention(int fields, int categories, int embed, int heads, int headDim, int depth,
            SplittableRandom random) {
        this.fields = fields;
        this.categories = categories;
        this.embed = embed;
        this.heads = heads;
        this.headDim = headDim;
        this.table = RecsysOps.normal(random, 0.1f, fields * categories, embed);
        int in = embed;
        for (int l = 0; l < depth; l++) {
            int w = heads * headDim;
            layers.add(new Var[] {RecsysOps.matrix(random, in, w), RecsysOps.matrix(random, in, w),
                RecsysOps.matrix(random, in, w), RecsysOps.matrix(random, in, w)});
            in = w;
        }
        this.out = RecsysOps.matrix(random, fields * in, 1);
        this.linear = RecsysOps.zeros(fields * categories, 1);
        this.bias = RecsysOps.zeros(1);
    }

    /**
     * @param fields fields
     * @param categories categories per field
     * @param embed embedding width
     * @param heads heads
     * @param headDim head width
     * @param depth layers
     * @return a spec for model builders
     */
    public static LayerSpec spec(int fields, int categories, int embed, int heads, int headDim, int depth) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(1);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new FieldAttention(fields, categories, embed, heads, headDim, depth, random);
            }
        };
    }

    /**
     * Embeds one-hot fields.
     *
     * @param input {@code [batch, fields * categories]}
     * @param table the {@code [fields * categories, embed]} table
     * @param fields fields
     * @param categories categories per field
     * @return {@code [batch, fields, embed]}
     */
    static Var embedFields(Var input, Var table, int fields, int categories) {
        int b = input.shape().dim(0);
        int e = table.shape().dim(1);
        List<Var> parts = new ArrayList<>();
        for (int f = 0; f < fields; f++) {
            parts.add(input.slice(1, f * categories, (f + 1) * categories)
                    .matmul(table.slice(0, f * categories, (f + 1) * categories)).reshape(b, 1, e));
        }
        return Var.concat(1, parts.toArray(Var[]::new));
    }

    /**
     * The field representations after the attention layers.
     *
     * @param input one-hot rows
     * @return {@code [batch, fields, heads * headDim]}
     */
    public Var fieldRepresentations(Var input) {
        int b = input.shape().dim(0);
        Var h = embedFields(input, table, fields, categories);
        for (Var[] w : layers) {
            Var q = h.matmul(w[0]).reshape(b, fields, heads, headDim).permute(0, 2, 1, 3);
            Var k = h.matmul(w[1]).reshape(b, fields, heads, headDim).permute(0, 2, 1, 3);
            Var v = h.matmul(w[2]).reshape(b, fields, heads, headDim).permute(0, 2, 1, 3);
            Var att = Functional.attention(q, k, v, false, (float) (1 / Math.sqrt(headDim))).permute(0, 2, 1, 3)
                    .reshape(b, fields, heads * headDim);
            h = att.add(h.matmul(w[3])).relu();
        }
        return h;
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        Var h = fieldRepresentations(input);
        return h.reshape(b, -1).matmul(out).add(input.matmul(linear)).add(bias);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(table));
        for (Var[] w : layers) {
            p.addAll(List.of(w));
        }
        p.addAll(List.of(out, linear, bias));
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return 1;
    }

    @Override
    public String toString() {
        return "FieldAttention(" + fields + " fields, " + layers.size() + " layers, " + heads + " heads)";
    }

    /** @return the embedding width */
    public int embed() {
        return embed;
    }
}
