package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Differentiable volume rendering of a radiance field along camera rays (NeRF, Mildenhall et
 * al., 2020). Each ray is sampled between {@code near} and {@code far}; the field's density
 * sigma and colour c at the samples are composited as
 * {@code C = sum_i T_i (1 - exp(-sigma_i delta_i)) c_i} with the transmittance
 * {@code T_i = exp(-sum_{j<i} sigma_j delta_j)}, over a black background.
 *
 * <p>Hierarchical sampling: a first, stratified pass gives the compositing weights; they
 * form a piecewise-constant density along the ray, and inverse-transform sampling draws
 * {@code fine} more samples where the weights are large. The field is evaluated again on
 * the union of both sets (one network serves both passes, as in mip-NeRF), and the result is
 * returned with the coarse colour so both passes can be supervised.
 *
 * <p>The field takes {@code [n, 6]} rows (point, unit direction) and returns
 * {@code [n, 4]} rows (raw density, raw colour), like {@link RadianceField}; density goes
 * through softplus and colour through a sigmoid.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2003.08934">arXiv:2003.08934</a>.
 */
public final class VolumeRenderer {

    private final float near;
    private final float far;
    private final int coarse;
    private final int fine;

    /**
     * The colours of a batch of rays.
     *
     * @param colour the final colour {@code [rays, 3]}
     * @param coarse the colour of the coarse pass alone (the same as {@code colour} without
     *     hierarchical sampling)
     */
    public record Rendering(Var colour, Var coarse) {
    }

    /**
     * Creates a renderer.
     *
     * @param near the start of every ray
     * @param far its end
     * @param coarse stratified samples per ray
     * @param fine extra importance samples per ray, 0 for none
     */
    public VolumeRenderer(float near, float far, int coarse, int fine) {
        this.near = near;
        this.far = far;
        this.coarse = coarse;
        this.fine = fine;
    }

    /** @return field evaluations per ray */
    public int evaluationsPerRay() {
        return fine == 0 ? coarse : 2 * coarse + fine;
    }

    /**
     * Renders rays from one eye.
     *
     * @param field the radiance field
     * @param eye the ray origin
     * @param directions unit directions, {@code 3 * rays} values
     * @param random the jitter of the stratified samples, or {@code null} for bin centres
     * @return the colours
     */
    public Rendering render(Layer field, float[] eye, float[] directions, RandomGenerator random) {
        int n = directions.length / 3;
        float bin = (far - near) / coarse;
        float[] t = new float[n * coarse];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < coarse; k++) {
                t[i * coarse + k] = near + bin * (k + (random == null ? 0.5f : random.nextFloat()));
            }
        }
        Var[] first = composite(field, eye, directions, t, coarse);
        if (fine == 0) {
            return new Rendering(first[0], first[0]);
        }
        float[] w = first[1].value().toArray();
        int total = coarse + fine;
        float[] all = new float[n * total];
        for (int i = 0; i < n; i++) {
            double[] cdf = new double[coarse + 1];
            for (int k = 0; k < coarse; k++) {
                cdf[k + 1] = cdf[k] + w[i * coarse + k] + 1e-4;
            }
            float[] row = new float[total];
            System.arraycopy(t, i * coarse, row, 0, coarse);
            for (int j = 0; j < fine; j++) {
                double u = (random == null ? (j + 0.5) : j + random.nextDouble()) / fine * cdf[coarse];
                int k = 0;
                while (k < coarse - 1 && cdf[k + 1] < u) {
                    k++;
                }
                double frac = (u - cdf[k]) / (cdf[k + 1] - cdf[k]);
                row[coarse + j] = (float) (near + bin * (k + Math.min(1, Math.max(0, frac))));
            }
            Arrays.sort(row);
            System.arraycopy(row, 0, all, i * total, total);
        }
        return new Rendering(composite(field, eye, directions, all, total)[0], first[0]);
    }

    private Var[] composite(Layer field, float[] eye, float[] directions, float[] t, int samples) {
        int n = directions.length / 3;
        float[] rows = new float[n * samples * 6];
        float[] delta = new float[n * samples];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < samples; k++) {
                int s = i * samples + k;
                float tk = t[s];
                delta[s] = Math.max(1e-3f, (k + 1 < samples ? t[s + 1] : far) - tk);
                for (int c = 0; c < 3; c++) {
                    rows[s * 6 + c] = eye[c] + tk * directions[3 * i + c];
                    rows[s * 6 + 3 + c] = directions[3 * i + c];
                }
            }
        }
        Var out = field.forward(Var.of(Tensor.of(Shape.of(n * samples, 6), rows)));
        Var tau = out.slice(1, 0, 1).softplus().reshape(n, samples).mul(Tensor.of(Shape.of(n, samples), delta));
        Var alpha = tau.neg().exp().neg().add(1f);
        Var weight = alpha.mul(tau.cumsum(1).sub(tau).neg().exp());
        Var rgb = out.slice(1, 1, 4).sigmoid().reshape(n, samples, 3);
        Var colour = rgb.mul(weight.reshape(n, samples, 1)).sum(1, false);
        return new Var[] {colour, weight};
    }
}
