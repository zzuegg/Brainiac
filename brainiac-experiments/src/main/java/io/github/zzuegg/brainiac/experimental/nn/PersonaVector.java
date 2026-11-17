package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A persona (trait) vector: a direction in one layer's activations that tracks a character
 * trait, extracted from contrastive prompts that do and do not show it. On top of
 * {@link SteeringVector}'s activation addition it gives the three uses of the paper and the
 * refusal-direction work:
 *
 * <ul>
 *   <li><b>monitoring</b> ({@link #projection}): how far activations lie along the trait
 *       direction, which predicts the trait before it shows in the output and measures the
 *       shift a fine-tuning data set causes;</li>
 *   <li><b>steering</b> ({@link #steer}): add or subtract the vector at inference (CAA);</li>
 *   <li><b>preventative steering</b>: add the vector <em>during fine-tuning</em> (again
 *       {@link #steer}, in the training forward pass) so the model no longer needs to move
 *       its own weights toward the trait to fit trait-inducing data; at inference, without
 *       the vector, the trait stays low and capabilities are kept better than with
 *       inference-time subtraction;</li>
 *   <li><b>directional ablation</b> ({@link #ablate}): project the direction out of the
 *       activations, {@code h - (h . u) u}, which removes the trait (or, for a refusal
 *       direction, refusal) at every input.</li>
 * </ul>
 *
 * <p>Chen et al. 2025, "Persona Vectors: Monitoring and Controlling Character Traits in
 * Language Models", <a href="https://arxiv.org/abs/2507.21509">arXiv:2507.21509</a>; Arditi et
 * al. 2024, "Refusal in Language Models Is Mediated by a Single Direction",
 * <a href="https://arxiv.org/abs/2406.11717">arXiv:2406.11717</a>.
 */
public final class PersonaVector {

    private final SteeringVector vector;
    private final float[] unit;

    private PersonaVector(SteeringVector vector) {
        this.vector = vector;
        float[] d = vector.direction();
        double n = vector.norm();
        this.unit = new float[d.length];
        for (int i = 0; i < d.length; i++) {
            unit[i] = (float) (d[i] / Math.max(n, 1e-12));
        }
    }

    /**
     * The mean activation difference of a layer between trait and non-trait inputs.
     *
     * @param model a sequential model
     * @param layer the layer
     * @param withTrait inputs that show the trait
     * @param without inputs that do not
     * @return the persona vector
     */
    public static PersonaVector fromContrast(Model model, int layer, Tensor withTrait, Tensor without) {
        return new PersonaVector(SteeringVector.fromContrast(model, layer, withTrait, without));
    }

    /**
     * A persona vector from recorded activations.
     *
     * @param layer the layer they come from
     * @param withTrait activations of trait inputs, one row each
     * @param without activations of non-trait inputs
     * @return the persona vector
     */
    public static PersonaVector fromActivations(int layer, float[][] withTrait, float[][] without) {
        int d = withTrait[0].length;
        float[] dir = new float[d];
        for (float[] row : withTrait) {
            for (int i = 0; i < d; i++) {
                dir[i] += row[i] / withTrait.length;
            }
        }
        for (float[] row : without) {
            for (int i = 0; i < d; i++) {
                dir[i] -= row[i] / without.length;
            }
        }
        return new PersonaVector(new SteeringVector(layer, dir));
    }

    /** @return the layer */
    public int layer() {
        return vector.layer();
    }

    /** @return the raw (unnormalised) vector */
    public SteeringVector vector() {
        return vector;
    }

    /** @return the unit direction */
    public float[] unit() {
        return unit.clone();
    }

    /**
     * The projection of each input's layer activation on the unit direction.
     *
     * @param model the model
     * @param inputs a batch {@code [rows, ...]}
     * @return one value per row (per row and position for sequence activations)
     */
    public double[] projection(Model model, Tensor inputs) {
        Tensor a = LayerHooks.trace(model, Var.of(inputs)).get(vector.layer()).value();
        float[] v = a.to(DType.FLOAT32).toArray();
        int d = unit.length;
        double[] out = new double[v.length / d];
        for (int r = 0; r < out.length; r++) {
            for (int i = 0; i < d; i++) {
                out[r] += v[r * d + i] * unit[i];
            }
        }
        return out;
    }

    /**
     * A hook that adds {@code scale} times the raw vector to the layer's output, for
     * inference-time or preventative steering.
     *
     * @param scale the strength; negative steers away from the trait
     * @return the hook
     */
    public Model.Hook steer(float scale) {
        return vector.hook(scale);
    }

    /**
     * A hook that removes the direction from the layer's output, {@code h - (h . u) u}.
     *
     * @return the hook
     */
    public Model.Hook ablate() {
        Tensor u = Tensor.of(Shape.of(unit.length, 1), unit);
        Tensor ut = Tensor.of(Shape.of(1, unit.length), unit);
        return LayerHooks.at(vector.layer(), out -> {
            Shape s = out.shape();
            Var flat = out.reshape(Math.toIntExact(s.size() / unit.length), unit.length);
            return flat.sub(flat.matmul(u).matmul(ut)).reshape(s.toArray());
        });
    }

    @Override
    public String toString() {
        return "PersonaVector(layer " + vector.layer() + ", norm " + String.format("%.3f", vector.norm()) + ")";
    }
}
