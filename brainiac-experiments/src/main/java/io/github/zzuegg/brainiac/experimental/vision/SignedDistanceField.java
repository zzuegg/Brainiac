package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FourierFeatures;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A neural signed distance field over the plane {@code [-1, 1]^2}, shared by a family of
 * shapes through per-shape latent codes (the DeepSDF auto-decoder, Park et al., 2019): the
 * network sees {@code (x, z_shape)}, and the codes are free parameters learned together with
 * the weights, with no encoder. The fit uses only points on each outline and their normals:
 * {@code |f| -> 0} on the outline, {@code grad f} aligned with the normal there, the eikonal
 * term {@code | |grad f| - 1 |} at all points (Gropp et al., 2020, which makes f a true
 * distance), and {@code exp(-alpha |f|)} off the outline (Sitzmann et al., 2020), which keeps
 * spurious zero sets away. The input gradient comes from autograd with a recorded graph, so
 * the eikonal and normal terms are trained through it.
 *
 * <p>The body is a {@link SineLayer} network (SIREN), a ReLU MLP, or a ReLU MLP behind random
 * {@link FourierFeatures}, so the three can be compared under the same losses.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2006.09661">arXiv:2006.09661</a>,
 * <a href="https://arxiv.org/abs/1901.05103">arXiv:1901.05103</a>,
 * <a href="https://arxiv.org/abs/2002.10099">arXiv:2002.10099</a>.
 */
public final class SignedDistanceField {

    /** The network body. */
    public enum Kind {
        /** sine layers with the SIREN initialisation */
        SIREN,
        /** a ReLU MLP on raw coordinates */
        RELU,
        /** a ReLU MLP behind random Fourier features */
        FOURIER
    }

    /**
     * The terms of one loss evaluation.
     *
     * @param total the weighted sum, to differentiate
     * @param surface the mean |f| on the outlines
     * @param eikonal the mean eikonal residual
     */
    public record Terms(Var total, double surface, double eikonal) {
    }

    private static final float SIREN_OMEGA = 10f;

    private final Kind kind;
    private final Var latents;
    private final Layer encoder;
    private final List<Layer> body = new ArrayList<>();

    /**
     * Creates a field.
     *
     * @param kind the body
     * @param shapes the number of shapes (latent codes)
     * @param latent the code width
     * @param width the hidden width
     * @param depth the hidden layers
     * @param random the source of initial weights
     */
    public SignedDistanceField(Kind kind, int shapes, int latent, int width, int depth, RandomGenerator random) {
        this.kind = kind;
        float[] z = new float[shapes * latent];
        for (int i = 0; i < z.length; i++) {
            z[i] = (float) (0.01 * random.nextGaussian());
        }
        this.latents = Var.param(Tensor.of(Shape.of(shapes, latent), z));
        int in = 2;
        if (kind == Kind.FOURIER) {
            this.encoder = FourierFeatures.gaussian(2, 32, 0.75f, random.nextLong(), true);
            in = 66;
        } else {
            this.encoder = null;
        }
        in += latent;
        for (int l = 0; l < depth; l++) {
            body.add(kind == Kind.SIREN ? new SineLayer(in, width, SIREN_OMEGA, l == 0, random)
                    : new Dense(in, width, Activation.RELU, Init.HE, random));
            in = width;
        }
        body.add(new Dense(in, 1, Activation.IDENTITY, Init.XAVIER, random));
    }

    /** @return the body kind */
    public Kind kind() {
        return kind;
    }

    /**
     * Evaluates the field.
     *
     * @param points {@code [n, 2]}
     * @param shapeIds {@code [n]} whole-number shape indices
     * @return {@code [n, 1]} signed distances (negative inside)
     */
    public Var value(Var points, Tensor shapeIds) {
        Var z = latents.gather(0, shapeIds);
        Var h = Var.concat(1, encoder == null ? points : encoder.forward(points), z);
        for (Layer l : body) {
            h = l.forward(h);
        }
        return h;
    }

    /**
     * The fitting loss.
     *
     * @param points {@code [n, 2]}: the outline points first, then free points
     * @param shapeIds {@code [n]} shape indices
     * @param normals {@code [m, 2]} outward unit normals of the first {@code m} points
     * @return the loss terms
     */
    public Terms loss(Tensor points, Tensor shapeIds, Tensor normals) {
        int m = normals.shape().dim(0);
        int n = points.shape().dim(0);
        Var p = Var.param(points);
        Var f = value(p, shapeIds);
        Var g = Var.gradients(f.sum(), List.of(p), true).get(0);
        Var norm = g.square().sum(1, true).add(1e-8f).sqrt();
        Var surface = f.slice(0, 0, m).abs().mean();
        Var cosine = g.slice(0, 0, m).mul(normals).sum(1, true).div(norm.slice(0, 0, m));
        Var normal = cosine.neg().add(1f).mean();
        Var eikonal = norm.sub(1f).abs().mean();
        Var off = f.slice(0, m, n).abs().mul(-50f).exp().mean();
        Var total = surface.mul(30f).add(normal).add(eikonal.mul(0.5f)).add(off.mul(0.5f))
                .add(latents.square().mean().mul(1e-2f));
        return new Terms(total, surface.value().item(), eikonal.value().item());
    }

    /**
     * Samples the field on a pixel grid over {@code [-1, 1]^2}.
     *
     * @param shape the shape index
     * @param res pixels per side
     * @return {@code {values, gradientNorms}}, each {@code res * res} row-major
     */
    public float[][] grid(int shape, int res) {
        float[] pts = new float[res * res * 2];
        for (int y = 0; y < res; y++) {
            for (int x = 0; x < res; x++) {
                pts[(y * res + x) * 2] = -1 + (x + 0.5f) * 2 / res;
                pts[(y * res + x) * 2 + 1] = -1 + (y + 0.5f) * 2 / res;
            }
        }
        float[] ids = new float[res * res];
        java.util.Arrays.fill(ids, shape);
        Var p = Var.param(Tensor.of(Shape.of(res * res, 2), pts));
        Var f = value(p, Tensor.of(Shape.of(res * res), ids));
        Var g = Var.gradients(f.sum(), List.of(p), false).get(0);
        return new float[][] {f.value().toArray(), g.square().sum(1, false).sqrt().value().toArray()};
    }

    /**
     * The zero crossings of a sampled field, linearly interpolated along grid edges (the
     * vertices marching squares would place).
     *
     * @param values {@code res * res} samples over {@code [-1, 1]^2}
     * @param res pixels per side
     * @return the crossing points, {@code [x0, y0, x1, y1, ...]}
     */
    public static float[] zeroCrossings(float[] values, int res) {
        List<float[]> out = new ArrayList<>();
        for (int y = 0; y < res; y++) {
            for (int x = 0; x < res; x++) {
                float a = values[y * res + x];
                if (x + 1 < res) {
                    cross(out, a, values[y * res + x + 1], x, y, 1, 0, res);
                }
                if (y + 1 < res) {
                    cross(out, a, values[(y + 1) * res + x], x, y, 0, 1, res);
                }
            }
        }
        float[] r = new float[out.size() * 2];
        for (int i = 0; i < out.size(); i++) {
            r[2 * i] = out.get(i)[0];
            r[2 * i + 1] = out.get(i)[1];
        }
        return r;
    }

    private static void cross(List<float[]> out, float a, float b, int x, int y, int dx, int dy, int res) {
        if ((a < 0) == (b < 0)) {
            return;
        }
        float t = a / (a - b);
        out.add(new float[] {-1 + (x + 0.5f + t * dx) * 2 / res, -1 + (y + 0.5f + t * dy) * 2 / res});
    }

    /** @return the weights and the latent codes */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(latents));
        if (encoder != null) {
            p.addAll(encoder.parameters());
        }
        for (Layer l : body) {
            p.addAll(l.parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "SignedDistanceField(" + kind + ")";
    }
}
