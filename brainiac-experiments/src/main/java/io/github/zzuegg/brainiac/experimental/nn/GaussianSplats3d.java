package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A differentiable 3D Gaussian Splatting renderer (Kerbl et al., "3D Gaussian Splatting for
 * Real-Time Radiance Field Rendering", <a href="https://arxiv.org/abs/2308.04079">arXiv:2308.04079</a>),
 * the 3-D extension of {@link GaussianSplats2d}. A scene is {@code n} anisotropic 3-D Gaussians,
 * each with a centre, three scales, a rotation quaternion, a colour and an opacity. Rendering
 * a pinhole {@link Camera} view:
 *
 * <ol>
 *   <li>the covariance {@code Sigma = R S S^T R^T} is moved into camera space and projected
 *       with the local affine (EWA) approximation {@code Sigma' = J W Sigma W^T J^T}, where
 *       {@code J} is the Jacobian of the perspective division at the centre (Zwicker et al.,
 *       "EWA volume splatting", 2001), plus a small low-pass term;</li>
 *   <li>the splats are sorted front to back by view depth; the sort itself is not
 *       differentiable, so it is applied as a permutation matrix, through which gradients
 *       flow to the reordered values;</li>
 *   <li>pixels alpha-composite the sorted 2-D Gaussians exactly as in
 *       {@link GaussianSplats2d} (transmittance by an exclusive cumulative sum of
 *       {@code log(1 - a)}), over a black background.</li>
 * </ol>
 *
 * <p>Colours are view-independent (spherical-harmonic degree 0), and there is no tiling,
 * densification or pruning: every pixel sees every splat, which is fine at toy sizes.
 */
public final class GaussianSplats3d {
    private static final float LOW_PASS = 1e-3f;

    private final int count;
    private final Var centre;    // [3, n]
    private final Var logScale;  // [3, n]
    private final Var rotation;  // [4, n] unnormalised quaternion (w, x, y, z)
    private final Var colour;    // [n, channels]
    private final Var opacity;   // [1, n] logits
    private final Tensor upper;  // [n, n], 1 where row < column

    /**
     * A pinhole camera: image coordinates in [0, 1] are {@code focal * x / z + 0.5} and
     * {@code focal * y / z + 0.5} of the camera-space point {@code rotation * (p - eye)}.
     *
     * @param eye the camera position
     * @param rotation the world-to-camera rotation, row-major 3x3 (rows: image x, image y, view axis)
     * @param focal the focal length in image widths
     */
    public record Camera(float[] eye, float[] rotation, float focal) {

        /**
         * A camera at {@code eye} looking at {@code target}, with world {@code +y} as up.
         *
         * @param eye the position
         * @param target the point at the image centre
         * @param focal the focal length in image widths
         * @return the camera
         */
        public static Camera lookAt(float[] eye, float[] target, float focal) {
            double[] f = normalize(target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]);
            double[] right = normalize(f[2], 0, -f[0]);   // f x (0, 1, 0), negated so x points right
            double[] down = {f[1] * right[2] - f[2] * right[1], f[2] * right[0] - f[0] * right[2],
                f[0] * right[1] - f[1] * right[0]};
            float[] r = new float[9];
            for (int k = 0; k < 3; k++) {
                r[k] = (float) right[k];
                r[3 + k] = (float) down[k];
                r[6 + k] = (float) f[k];
            }
            return new Camera(eye.clone(), r, focal);
        }

        private static double[] normalize(double x, double y, double z) {
            double n = Math.sqrt(x * x + y * y + z * z);
            return new double[] {x / n, y / n, z / n};
        }

        /**
         * The unit world-space direction of the ray through an image point.
         *
         * @param u the image x in [0, 1]
         * @param v the image y in [0, 1]
         * @return the direction
         */
        public float[] ray(float u, float v) {
            double cx = (u - 0.5) / focal;
            double cy = (v - 0.5) / focal;
            double[] d = new double[3];
            for (int k = 0; k < 3; k++) {
                d[k] = rotation[k] * cx + rotation[3 + k] * cy + rotation[6 + k];
            }
            double n = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            return new float[] {(float) (d[0] / n), (float) (d[1] / n), (float) (d[2] / n)};
        }
    }

    private GaussianSplats3d(int count, int channels, float extent, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.count = count;
        centre = Var.param(Tensor.uniform(Shape.of(3, count), -extent, extent, r));
        logScale = Var.param(Tensor.full(Shape.of(3, count), (float) Math.log(extent / Math.cbrt(count))));
        rotation = Var.param(Tensor.normal(Shape.of(4, count), 0, 1, r));
        colour = Var.param(Tensor.uniform(Shape.of(count, channels), 0.3f, 0.7f, r));
        opacity = Var.param(Tensor.zeros(Shape.of(1, count)));
        float[] u = new float[count * count];
        for (int i = 0; i < count; i++) {
            for (int j = i + 1; j < count; j++) {
                u[i * count + j] = 1;
            }
        }
        upper = Tensor.of(Shape.of(count, count), u);
    }

    /**
     * Creates randomly placed splats.
     *
     * @param count the number of Gaussians
     * @param channels colour channels
     * @param extent the half-width of the cube the centres start in
     * @param seed the seed
     * @return the splats
     */
    public static GaussianSplats3d of(int count, int channels, float extent, long seed) {
        return new GaussianSplats3d(count, channels, extent, seed);
    }

    /** @return the number of Gaussians */
    public int count() {
        return count;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.of(centre, logScale, rotation, colour, opacity);
    }

    private static Var row(Var v, int i) {
        return v.slice(0, i, i + 1);
    }

    /**
     * Renders pixels of a view.
     *
     * @param camera the camera
     * @param pixels {@code [pixels, 2]} image coordinates in [0, 1], as from {@link GaussianSplats2d#grid}
     * @return {@code [pixels, channels]} colours
     */
    public Var render(Camera camera, Tensor pixels) {
        float[] w = camera.rotation();
        float[] e = camera.eye();
        float f = camera.focal();
        Var[] rel = new Var[3];
        for (int k = 0; k < 3; k++) {
            rel[k] = row(centre, k).sub(e[k]);
        }
        Var[] cam = new Var[3];
        for (int i = 0; i < 3; i++) {
            cam[i] = rel[0].mul(w[3 * i]).add(rel[1].mul(w[3 * i + 1])).add(rel[2].mul(w[3 * i + 2]));
        }
        Var z = cam[2].maximum(0.2f);
        Var u = cam[0].div(z).mul(f).add(0.5f);
        Var v = cam[1].div(z).mul(f).add(0.5f);

        Var qw = row(rotation, 0);
        Var qx = row(rotation, 1);
        Var qy = row(rotation, 2);
        Var qz = row(rotation, 3);
        Var norm = qw.square().add(qx.square()).add(qy.square()).add(qz.square()).add(1e-8f);
        Var s2 = norm.pow(-1f).mul(2f);   // 2 / |q|^2, so the matrix is a rotation for any q
        Var[][] rot = {
            {qy.square().add(qz.square()).mul(s2).neg().add(1f), qx.mul(qy).sub(qw.mul(qz)).mul(s2),
                qx.mul(qz).add(qw.mul(qy)).mul(s2)},
            {qx.mul(qy).add(qw.mul(qz)).mul(s2), qx.square().add(qz.square()).mul(s2).neg().add(1f),
                qy.mul(qz).sub(qw.mul(qx)).mul(s2)},
            {qx.mul(qz).sub(qw.mul(qy)).mul(s2), qy.mul(qz).add(qw.mul(qx)).mul(s2),
                qx.square().add(qy.square()).mul(s2).neg().add(1f)}};
        Var[] scale = {row(logScale, 0).exp(), row(logScale, 1).exp(), row(logScale, 2).exp()};
        // camera-space columns of R S: mc[i][j] = sum_k W_ik R_kj s_j
        Var[][] mc = new Var[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Var m = rot[0][j].mul(w[3 * i]).add(rot[1][j].mul(w[3 * i + 1])).add(rot[2][j].mul(w[3 * i + 2]));
                mc[i][j] = m.mul(scale[j]);
            }
        }
        Var fz = z.pow(-1f).mul(f);
        Var jx = cam[0].mul(fz).div(z).neg();
        Var jy = cam[1].mul(fz).div(z).neg();
        Var sxx = null;
        Var syy = null;
        Var sxy = null;
        for (int j = 0; j < 3; j++) {
            Var a0 = mc[0][j].mul(fz).add(mc[2][j].mul(jx));
            Var a1 = mc[1][j].mul(fz).add(mc[2][j].mul(jy));
            sxx = sxx == null ? a0.square() : sxx.add(a0.square());
            syy = syy == null ? a1.square() : syy.add(a1.square());
            sxy = sxy == null ? a0.mul(a1) : sxy.add(a0.mul(a1));
        }
        sxx = sxx.add(LOW_PASS);
        syy = syy.add(LOW_PASS);
        Var det = sxx.mul(syy).sub(sxy.square());

        Tensor permutation = depthOrder(cam[2].value().toArray());
        Var p = Var.of(permutation);
        Var dx = Var.of(pixels.slice(1, 0, 1)).sub(u.matmul(p));   // [pixels, n], sorted
        Var dy = Var.of(pixels.slice(1, 1, 2)).sub(v.matmul(p));
        Var q = dx.square().mul(syy.matmul(p)).sub(dx.mul(dy).mul(sxy.matmul(p)).mul(2f))
                .add(dy.square().mul(sxx.matmul(p))).div(det.matmul(p));
        Var alpha = q.mul(-0.5f).exp().mul(opacity.sigmoid().matmul(p)).mul(0.99f);
        Var logT = alpha.neg().add(1f).log().matmul(Var.of(upper));
        return alpha.mul(logT.exp()).matmul(Var.of(permutation.transpose()).matmul(colour));
    }

    /** The permutation matrix whose column {@code k} picks the {@code k}-th nearest splat. */
    private Tensor depthOrder(float[] depth) {
        Integer[] order = new Integer[count];
        for (int i = 0; i < count; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Float.compare(depth[a], depth[b]));
        float[] m = new float[count * count];
        for (int k = 0; k < count; k++) {
            m[order[k] * count + k] = 1;
        }
        return Tensor.of(Shape.of(count, count), m);
    }
}
