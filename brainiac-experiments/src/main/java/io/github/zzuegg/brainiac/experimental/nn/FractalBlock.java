package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A FractalNet block (Larsson, Maire and Shakhnarovich, <a href="https://arxiv.org/abs/1605.07648">1605.07648</a>):
 * the fractal expansion
 *
 * <pre>
 * f_1(z) = unit(z);   f_(C+1)(z) = join(f_C(f_C(z)), unit(z))
 * </pre>
 *
 * gives {@code C} columns of depths 1, 2, 4, … 2^(C-1) units, joined by element-wise means
 * wherever they meet. The unit is any shape-preserving {@link LayerSpec} after its first
 * application (dense, convolution, …). Without residual connections, depth is trained by
 * <em>drop-path</em>: while training, each batch either drops join inputs at random (local:
 * each input with the given probability, keeping at least one) or, with the given share of
 * batches, keeps a single column for the whole block (global). Global drop-path trains every
 * column to work alone, so {@link #useColumn} can then run one column as a standalone,
 * shallower or deeper network ("anytime" prediction).
 */
public final class FractalBlock implements Layer {
    private final Node root;
    private final int columns;
    private final double localDrop;
    private final double globalShare;
    private final SplittableRandom random;
    private int column;

    /** A level of the fractal: a shallow unit, and for {@code C > 1} two copies of the level below. */
    private static final class Node {
        final Layer unit;
        final Node first;
        final Node second;

        Node(Layer unit, Node first, Node second) {
            this.unit = unit;
            this.first = first;
            this.second = second;
        }
    }

    private FractalBlock(int columns, LayerSpec unit, double localDrop, double globalShare, Shape input,
            SplittableRandom random) {
        this.columns = columns;
        this.localDrop = localDrop;
        this.globalShare = globalShare;
        this.random = random.split();
        this.root = build(columns, unit, input, random);
    }

    private static Node build(int c, LayerSpec unit, Shape input, SplittableRandom random) {
        Layer shallow = unit.create(input, random);
        if (c == 1) {
            return new Node(shallow, null, null);
        }
        Node first = build(c - 1, unit, input, random);
        Node second = build(c - 1, unit, unit.outputShape(input), random);
        return new Node(shallow, first, second);
    }

    /**
     * Creates a block.
     *
     * @param columns the number of columns C
     * @param unit the building unit
     * @param input the input shape, without batch
     * @param localDrop the probability of dropping each join input under local drop-path
     * @param globalShare the share of training batches that use global drop-path
     * @param seed the seed
     * @return the block
     */
    public static FractalBlock of(int columns, LayerSpec unit, Shape input, double localDrop, double globalShare,
            long seed) {
        if (columns < 1) {
            throw new IllegalArgumentException("need at least one column");
        }
        return new FractalBlock(columns, unit, localDrop, globalShare, input, new SplittableRandom(seed));
    }

    /**
     * Describes a block for model builders.
     *
     * @param columns the number of columns
     * @param unit the building unit
     * @param localDrop the local drop probability
     * @param globalShare the share of global drop-path batches
     * @return the spec
     */
    public static LayerSpec spec(int columns, LayerSpec unit, double localDrop, double globalShare) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return unit.outputShape(input);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new FractalBlock(columns, unit, localDrop, globalShare, input, random);
            }
        };
    }

    /** @return the number of columns */
    public int columns() {
        return columns;
    }

    /**
     * Restricts evaluation to one column, or lifts the restriction.
     *
     * @param column 1 (shallowest, one unit) to {@link #columns()} (deepest, 2^(C-1) units), or 0 for all
     */
    public void useColumn(int column) {
        if (column < 0 || column > columns) {
            throw new IllegalArgumentException("column must be in 0.." + columns);
        }
        this.column = column;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (column > 0) {
            return single(root, columns, column, input, training);
        }
        if (training && random.nextDouble() < globalShare) {
            return single(root, columns, 1 + random.nextInt(columns), input, true);
        }
        return join(branches(root, input, training), training);
    }

    /** Column {@code c} of a level with {@code levels} columns: column 1 is the shallow unit here. */
    private static Var single(Node node, int levels, int c, Var x, boolean training) {
        if (c == 1 || levels == 1) {
            return node.unit.forward(x, training);
        }
        return single(node.second, levels - 1, c - 1, single(node.first, levels - 1, c - 1, x, training), training);
    }

    /** The column outputs that meet at this level's final join (the joins of nested levels merge into it). */
    private List<Var> branches(Node node, Var x, boolean training) {
        List<Var> out = new ArrayList<>();
        if (node.first != null) {
            out.addAll(branches(node.second, join(branches(node.first, x, training), training), training));
        }
        out.add(node.unit.forward(x, training));
        return out;
    }

    private Var join(List<Var> inputs, boolean training) {
        List<Var> kept = inputs;
        if (training && localDrop > 0 && inputs.size() > 1) {
            kept = new ArrayList<>();
            for (Var v : inputs) {
                if (random.nextDouble() >= localDrop) {
                    kept.add(v);
                }
            }
            if (kept.isEmpty()) {
                kept.add(inputs.get(random.nextInt(inputs.size())));
            }
        }
        Var sum = kept.get(0);
        for (int i = 1; i < kept.size(); i++) {
            sum = sum.add(kept.get(i));
        }
        return kept.size() == 1 ? sum : sum.mul(1f / kept.size());
    }

    @Override
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>();
        collect(root, out);
        return out;
    }

    private static void collect(Node node, List<Var> out) {
        if (node == null) {
            return;
        }
        out.addAll(node.unit.parameters());
        collect(node.first, out);
        collect(node.second, out);
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        return root.unit.outputShape(inputShape);
    }

    @Override
    public int outputSize(int inputSize) {
        return root.unit.outputSize(inputSize);
    }
}
