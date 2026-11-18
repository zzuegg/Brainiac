package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Frame averaging (Puny et al., ICLR 2022): makes any backbone exactly invariant or
 * equivariant to E(3) acting on point clouds. For each input a small set of frames is computed
 * from the data itself: the centroid and the principal axes (eigenvectors of the covariance,
 * by the core {@link Linalg#eigh}), with every sign choice, eight frames for O(3) (four with
 * determinant +1 for SO(3)). The input is canonicalised in each frame, passed through the
 * backbone, and the outputs are averaged; for vector outputs each one is first rotated back
 * by its frame. Rotating, reflecting or translating the input permutes the frames, so the
 * average does not change (or rotates with the input).
 *
 * <p>As a {@link Layer} the input is a flattened cloud {@code [b, points * 3]}, the backbone
 * maps {@code [b * frames, points * 3]} to {@code [b * frames, out]}, and the output is
 * {@code [b, out]}. Frames are computed without gradient; gradients still flow to the input
 * through the canonicalisation, so forces from an invariant energy work.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2110.03336">arXiv:2110.03336</a>.
 */
public final class FrameAveraging implements Layer {

    private final Layer backbone;
    private final int points;
    private final boolean vectorOutput;
    private final boolean properRotations;

    private FrameAveraging(Layer backbone, int points, boolean vectorOutput, boolean properRotations) {
        this.backbone = backbone;
        this.points = points;
        this.vectorOutput = vectorOutput;
        this.properRotations = properRotations;
    }

    /**
     * An invariant model.
     *
     * @param backbone {@code [m, points * 3] -> [m, out]}, any network
     * @param points the points per cloud
     * @param properRotations {@code true} for SO(3) (four frames), {@code false} for O(3)
     *     (eight)
     * @return the averaged model
     */
    public static FrameAveraging invariant(Layer backbone, int points, boolean properRotations) {
        return new FrameAveraging(backbone, points, false, properRotations);
    }

    /**
     * An equivariant model whose output is a list of 3-vectors, rotated back by each frame.
     *
     * @param backbone {@code [m, points * 3] -> [m, k * 3]}
     * @param points the points per cloud
     * @param properRotations SO(3) rather than O(3)
     * @return the averaged model
     */
    public static FrameAveraging equivariant(Layer backbone, int points, boolean properRotations) {
        return new FrameAveraging(backbone, points, true, properRotations);
    }

    /**
     * The frames of point clouds: principal axes with all sign choices.
     *
     * @param clouds {@code [b, points * 3]}
     * @param properRotations whether to keep only determinant +1
     * @return frame matrices {@code [b][frames][3][3]}, one axis per column
     */
    public static double[][][][] frames(Tensor clouds, boolean properRotations) {
        int b = clouds.shape().dim(0);
        int n = clouds.shape().dim(1) / 3;
        double[] x = clouds.toDoubleArray();
        double[] cov = new double[b * 9];
        for (int s = 0; s < b; s++) {
            double[] c = new double[3];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < 3; k++) {
                    c[k] += x[s * n * 3 + i * 3 + k] / n;
                }
            }
            for (int i = 0; i < n; i++) {
                for (int p = 0; p < 3; p++) {
                    for (int q = 0; q < 3; q++) {
                        cov[s * 9 + p * 3 + q] += (x[s * n * 3 + i * 3 + p] - c[p]) * (x[s * n * 3 + i * 3 + q] - c[q]);
                    }
                }
            }
        }
        double[] vec = Linalg.eigh(Tensor.of(Shape.of(b, 3, 3), cov, DType.FLOAT64, Device.cpu()))[1].toDoubleArray();
        int count = properRotations ? 4 : 8;
        double[][][][] out = new double[b][count][3][3];
        for (int s = 0; s < b; s++) {
            int f = 0;
            for (int signs = 0; signs < 8; signs++) {
                double[][] m = new double[3][3];
                for (int col = 0; col < 3; col++) {
                    double sign = ((signs >> col) & 1) == 0 ? 1 : -1;
                    for (int row = 0; row < 3; row++) {
                        m[row][col] = sign * vec[s * 9 + row * 3 + col];
                    }
                }
                if (properRotations && determinant(m) < 0) {
                    continue;
                }
                out[s][f++] = m;
            }
        }
        return out;
    }

    private static double determinant(double[][] m) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        double[][][][] fr = frames(input.value(), properRotations);
        int count = fr[0].length;
        float[] f = new float[b * count * 9];
        for (int s = 0; s < b; s++) {
            for (int k = 0; k < count; k++) {
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        f[((s * count + k) * 3 + r) * 3 + c] = (float) fr[s][k][r][c];
                    }
                }
            }
        }
        Tensor frameTensor = Tensor.of(Shape.of(b * count, 3, 3), f);
        Var cloud = input.reshape(b, points, 3);
        Var centred = cloud.sub(cloud.mean(1, true));
        Var expanded = centred.reshape(b, 1, points, 3).expand(Shape.of(b, count, points, 3))
                .reshape(b * count, points, 3);
        Var canonical = expanded.matmul(Var.of(frameTensor)).reshape(b * count, points * 3);
        Var y = backbone.forward(canonical);
        int out = y.shape().dim(1);
        if (vectorOutput) {
            Var v = y.reshape(b * count, out / 3, 3).matmul(Var.of(frameTensor.transpose()));
            y = v.reshape(b * count, out);
        }
        return y.reshape(b, count, out).mean(1, false);
    }

    @Override
    public List<Var> parameters() {
        return backbone.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return backbone.outputSize(inputSize);
    }

    @Override
    public String toString() {
        return "FrameAveraging(" + backbone + (properRotations ? ", SO(3))" : ", O(3))");
    }
}
