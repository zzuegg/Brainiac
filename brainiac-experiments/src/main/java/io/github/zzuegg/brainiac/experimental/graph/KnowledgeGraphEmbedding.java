package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Complex;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Knowledge-graph embeddings for link prediction over {@code (head, relation, tail)}
 * triples, four scoring functions in one class. Entities are {@code d} complex numbers
 * (stored {@code [entities, d, 2]}, read as {@code 2d} reals by the real models):
 * <ul>
 *   <li>{@link Model#TRANSE} (Bordes et al., 2013): {@code gamma - |h + r - t|_1}, a
 *       relation is a translation. It cannot model symmetric relations ({@code r} and
 *       {@code -r} would both have to be 0);</li>
 *   <li>{@link Model#DISTMULT} (Yang et al., 2015, <a href="https://arxiv.org/abs/1412.6575">arXiv:1412.6575</a>):
 *       the trilinear {@code <h, r, t>}, symmetric in {@code h} and {@code t}, so it cannot
 *       tell {@code parent(a, b)} from {@code parent(b, a)};</li>
 *   <li>{@link Model#COMPLEX} (Trouillon et al., 2016, <a href="https://arxiv.org/abs/1606.06357">arXiv:1606.06357</a>):
 *       {@code Re <h, r, conj(t)>}; the imaginary part of {@code r} makes it antisymmetric;</li>
 *   <li>{@link Model#ROTATE} (Sun et al., 2019, <a href="https://arxiv.org/abs/1902.10197">arXiv:1902.10197</a>):
 *       {@code gamma - |h o r - t|} with {@code |r_k| = 1}: a relation rotates every
 *       coordinate, which models symmetry (rotation by pi), inversion (the opposite
 *       rotation) and composition (the sum of angles) at once.</li>
 * </ul>
 * Training uses self-adversarial negative sampling (from the RotatE paper): corrupted
 * tails are weighted by {@code softmax(alpha * score)} (detached), so hard negatives count
 * most. Complex products go through the autograd {@link Complex}.
 */
public final class KnowledgeGraphEmbedding {

    /** The scoring function. */
    public enum Model {
        /** translation */
        TRANSE,
        /** real trilinear product */
        DISTMULT,
        /** complex trilinear product */
        COMPLEX,
        /** complex rotation */
        ROTATE
    }

    private final Model model;
    private final int entities;
    private final int dim;
    private final float margin;
    private final Var entity;
    private final Var relation;

    /**
     * Creates the embedding.
     *
     * @param model the scoring function
     * @param entities the entity count
     * @param relations the relation count
     * @param dim complex dimensions per entity ({@code 2 * dim} reals)
     * @param margin {@code gamma} of the distance models
     * @param random the initialization source
     */
    public KnowledgeGraphEmbedding(Model model, int entities, int relations, int dim, float margin,
            SplittableRandom random) {
        this.model = Objects.requireNonNull(model);
        this.entities = entities;
        this.dim = dim;
        this.margin = margin;
        float range = model == Model.TRANSE || model == Model.ROTATE ? (margin + 2f) / dim : 0.5f;
        this.entity = Var.param(Tensor.uniform(Shape.of(entities, dim, 2), -range, range, random));
        this.relation = model == Model.ROTATE
                ? Var.param(Tensor.uniform(Shape.of(relations, dim), (float) -Math.PI, (float) Math.PI, random))
                : Var.param(Tensor.uniform(Shape.of(relations, dim, 2), -range, range, random));
    }

    /** @return the scoring function */
    public Model model() {
        return model;
    }

    /** @return the entity and relation tables */
    public List<Var> parameters() {
        return List.of(entity, relation);
    }

    /**
     * Scores broadcastable batches of heads, relations and tails.
     *
     * @param h entities {@code [..., d, 2]}
     * @param r relations {@code [..., d, 2]} ({@code [..., d]} phases for RotatE)
     * @param t entities {@code [..., d, 2]}
     * @return scores {@code [...]}, higher is more plausible
     */
    public Var score(Var h, Var r, Var t) {
        int rank = h.shape().rank();
        return switch (model) {
            case TRANSE -> h.add(r).sub(t).abs().sum(rank - 1, false).sum(rank - 2, false).neg().add(margin);
            case DISTMULT -> h.mul(r).mul(t).sum(rank - 1, false).sum(rank - 2, false);
            case COMPLEX -> Complex.real(Complex.mul(Complex.mul(h, r), Complex.conj(t))).sum(rank - 2, false);
            case ROTATE -> {
                Var rot = Complex.polar(r.mul(0f).add(1f), r);
                Var diff = Complex.mul(h, rot).sub(t);
                yield Complex.abs(diff).sum(rank - 2, false).neg().add(margin);
            }
        };
    }

    private Var entities(Tensor ids) {
        return entity.gather(0, ids);
    }

    private Var relations(Tensor ids) {
        return relation.gather(0, ids);
    }

    /**
     * The self-adversarial negative-sampling loss of a batch, corrupting tails and heads.
     *
     * @param batch triples {@code {head, relation, tail}}
     * @param negatives corrupted entities per triple and side
     * @param temperature the self-adversarial {@code alpha} (0 = uniform weights)
     * @param random the corruption source
     * @return the loss
     */
    public Var loss(List<int[]> batch, int negatives, float temperature, RandomGenerator random) {
        int b = batch.size();
        float[] hs = new float[b];
        float[] rs = new float[b];
        float[] ts = new float[b];
        float[] negT = new float[b * negatives];
        float[] negH = new float[b * negatives];
        for (int i = 0; i < b; i++) {
            hs[i] = batch.get(i)[0];
            rs[i] = batch.get(i)[1];
            ts[i] = batch.get(i)[2];
            for (int k = 0; k < negatives; k++) {
                negT[i * negatives + k] = random.nextInt(entities);
                negH[i * negatives + k] = random.nextInt(entities);
            }
        }
        Tensor bShape = Tensor.of(Shape.of(b), hs);
        Var h = entities(bShape);
        Var r = relations(Tensor.of(Shape.of(b), rs));
        Var t = entities(Tensor.of(Shape.of(b), ts));
        Var pos = score(h, r, t);
        int[] tail = model == Model.ROTATE ? new int[] {b, 1, dim} : new int[] {b, 1, dim, 2};
        Var rb = r.reshape(tail);
        Var nt = entities(Tensor.of(Shape.of(b * negatives), negT)).reshape(b, negatives, dim, 2);
        Var nh = entities(Tensor.of(Shape.of(b * negatives), negH)).reshape(b, negatives, dim, 2);
        Var negTail = score(h.reshape(b, 1, dim, 2), rb, nt);
        Var negHead = score(nh, rb, t.reshape(b, 1, dim, 2));
        Var loss = pos.neg().softplus().mean();
        for (Var neg : List.of(negTail, negHead)) {
            Tensor w = neg.value().mul(temperature);
            w = w.sub(w.max(1, true)).exp();
            w = w.div(w.sum(1, true));
            loss = loss.add(neg.softplus().mul(w).sum(1, false).mean().mul(0.5f));
        }
        if (model == Model.DISTMULT || model == Model.COMPLEX) {
            loss = loss.add(h.square().mean().add(t.square().mean()).add(r.square().mean()).mul(1e-3f));
        }
        return loss;
    }

    /**
     * Scores every entity as the tail of {@code (head, relation, ?)}.
     *
     * @param head the head entity
     * @param rel the relation
     * @return one score per entity
     */
    public float[] scoreTails(int head, int rel) {
        return Var.noGrad(() -> {
            Var h = entities(Tensor.of(Shape.of(1), (float) head));
            Var r = relations(Tensor.of(Shape.of(1), (float) rel));
            return score(h, r, entity).value().toArray();
        });
    }

    /**
     * The filtered rank of the true tail: 1 + the number of entities that score higher and
     * do not form a known true triple.
     *
     * @param triple {@code {head, relation, tail}}
     * @param known every true triple, as {@link #key}
     * @return the rank, 1 = best
     */
    public int filteredRank(int[] triple, Set<Long> known) {
        float[] s = scoreTails(triple[0], triple[1]);
        int rank = 1;
        for (int e = 0; e < entities; e++) {
            if (e != triple[2] && s[e] > s[triple[2]] && !known.contains(key(triple[0], triple[1], e))) {
                rank++;
            }
        }
        return rank;
    }

    /**
     * A set key for a triple.
     *
     * @param head the head
     * @param rel the relation
     * @param tail the tail
     * @return a unique key
     */
    public static long key(int head, int rel, int tail) {
        return ((long) head * 4096 + rel) * 1_000_000L + tail;
    }

    @Override
    public String toString() {
        return "KnowledgeGraphEmbedding(" + model + ", d=" + dim + ")";
    }
}
