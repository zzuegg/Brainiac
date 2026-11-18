package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * xDeepFM (Lian et al., KDD 2018, <a href="https://arxiv.org/abs/1803.05170">arXiv:1803.05170</a>)
 * as a {@link Layer} from {@code [batch, fields * categories]} one-hot rows to one click logit: a
 * linear term, a plain DNN over the field embeddings, and the <em>compressed interaction
 * network</em> (CIN). The CIN learns explicit, vector-wise feature crosses: layer {@code k} takes
 * the outer products of the previous layer's {@code H_{k-1}} feature maps with the {@code m} field
 * embeddings, per embedding dimension, and compresses the {@code H_{k-1} * m} products into
 * {@code H_k} maps with a 1x1 convolution (a learned linear map); each layer's maps are sum-pooled
 * over the embedding dimension and concatenated into the output. Layer {@code k} therefore holds
 * interactions of order {@code k + 1}.
 */
public final class CompressedInteraction implements Layer {

    private final int fields;
    private final int categories;
    private final int embed;
    private final Var table;
    private final List<Var> cin = new ArrayList<>();
    private final int[] maps;
    private final Var cinOut;
    private final Var h1;
    private final Var b1;
    private final Var h2;
    private final Var linear;
    private final Var bias;

    /**
     * Creates the layer.
     *
     * @param fields categorical fields
     * @param categories categories per field
     * @param embed the field embedding width
     * @param maps feature maps per CIN layer, such as {@code {16, 16, 16}}
     * @param hidden the DNN hidden width
     * @param random the initializer
     */
    public CompressedInteraction(int fields, int categories, int embed, int[] maps, int hidden,
            SplittableRandom random) {
        this.fields = fields;
        this.categories = categories;
        this.embed = embed;
        this.maps = maps.clone();
        this.table = RecsysOps.normal(random, 0.1f, fields * categories, embed);
        int prev = fields;
        int pooled = 0;
        for (int h : maps) {
            cin.add(RecsysOps.matrix(random, prev * fields, h));
            prev = h;
            pooled += h;
        }
        this.cinOut = RecsysOps.matrix(random, pooled, 1);
        this.h1 = RecsysOps.matrix(random, fields * embed, hidden);
        this.b1 = RecsysOps.zeros(hidden);
        this.h2 = RecsysOps.matrix(random, hidden, 1);
        this.linear = RecsysOps.zeros(fields * categories, 1);
        this.bias = RecsysOps.zeros(1);
    }

    /**
     * @param fields fields
     * @param categories categories per field
     * @param embed embedding width
     * @param maps CIN feature maps per layer
     * @param hidden DNN width
     * @return a spec for model builders
     */
    public static LayerSpec spec(int fields, int categories, int embed, int[] maps, int hidden) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(1);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new CompressedInteraction(fields, categories, embed, maps, hidden, random);
            }
        };
    }

    /**
     * The sum-pooled CIN feature maps of every layer.
     *
     * @param input one-hot rows
     * @return {@code [batch, sum of maps]}
     */
    public Var cin(Var input) {
        int b = input.shape().dim(0);
        // x0 as [batch, embed, fields]: one row of field values per embedding dimension
        Var x0 = FieldAttention.embedFields(input, table, fields, categories).permute(0, 2, 1);
        Var xk = x0;
        int prev = fields;
        List<Var> pooled = new ArrayList<>();
        for (int l = 0; l < cin.size(); l++) {
            Var z = xk.reshape(b, embed, prev, 1).mul(x0.reshape(b, embed, 1, fields)).reshape(b, embed,
                    prev * fields);
            xk = z.matmul(cin.get(l)).relu();
            prev = maps[l];
            pooled.add(xk.sum(1, false));
        }
        return Var.concat(1, pooled.toArray(Var[]::new));
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        Var dnn = FieldAttention.embedFields(input, table, fields, categories).reshape(b, fields * embed).matmul(h1)
                .add(b1).relu().matmul(h2);
        return cin(input).matmul(cinOut).add(dnn).add(input.matmul(linear)).add(bias);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(table));
        p.addAll(cin);
        p.addAll(List.of(cinOut, h1, b1, h2, linear, bias));
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return 1;
    }

    @Override
    public String toString() {
        return "CompressedInteraction(" + fields + " fields, CIN " + java.util.Arrays.toString(maps) + ")";
    }
}
