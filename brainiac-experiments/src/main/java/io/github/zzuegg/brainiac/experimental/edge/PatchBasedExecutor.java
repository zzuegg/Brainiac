package io.github.zzuegg.brainiac.experimental.edge;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * MCUNetV2's patch-based inference: the first, high-resolution convolutions of an
 * {@link EdgeNet} run patch by patch. The output of the patch stage (unit {@code split - 1}) is
 * cut into a {@code grid x grid} of tiles; for each tile the receptive field is traced back
 * through the stage (overlapping input patches, clipped to the image), and the stage is computed
 * on that patch alone, with zero padding only where the patch touches the image border, so the
 * result equals layer-by-layer execution. Only one patch's activations and the stage's output
 * buffer are alive at a time, at the price of recomputing the overlaps.
 *
 * <p>Not done: receptive-field redistribution (MCUNetV2 shrinks the kernels of the patch stage and
 * enlarges later ones in its architecture search).
 *
 * <p>Paper: Lin et al., "MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep
 * Learning", NeurIPS 2021, <a href="https://arxiv.org/abs/2110.15352">arXiv:2110.15352</a>.
 */
public final class PatchBasedExecutor {

    private final EdgeNet net;
    private final int split;
    private final int grid;
    /** {@code regions[patch][layer]} = {y0, y1, x0, x1} of layer's output (layer -1 = input at index 0). */
    private final int[][][] regions;
    private final List<int[]> shapes;

    /**
     * Plans patch execution.
     *
     * @param net the network (evaluation mode; batch norm is used folded)
     * @param split the number of leading units run per patch, all convolutions
     * @param grid patches per side
     */
    public PatchBasedExecutor(EdgeNet net, int split, int grid) {
        for (int i = 0; i < split; i++) {
            if (!(net.units().get(i) instanceof EdgeNet.Conv)) {
                throw new IllegalArgumentException("the patch stage must be convolutions");
            }
        }
        this.net = net;
        this.split = split;
        this.grid = grid;
        this.shapes = net.shapes();
        int[] last = shapes.get(split - 1);
        regions = new int[grid * grid][split + 1][];
        for (int py = 0; py < grid; py++) {
            for (int px = 0; px < grid; px++) {
                int[][] r = regions[py * grid + px];
                r[split] = new int[] {py * last[1] / grid, (py + 1) * last[1] / grid - 1,
                    px * last[2] / grid, (px + 1) * last[2] / grid - 1};
                for (int l = split - 1; l >= 0; l--) {
                    EdgeNet.Conv c = (EdgeNet.Conv) net.units().get(l);
                    int[] in = l == 0 ? net.inputShape() : shapes.get(l - 1);
                    int[] o = r[l + 1];
                    r[l] = new int[] {
                        Math.max(0, o[0] * c.stride() - c.pad()),
                        Math.min(in[1] - 1, o[1] * c.stride() - c.pad() + c.kernel() - 1),
                        Math.max(0, o[2] * c.stride() - c.pad()),
                        Math.min(in[2] - 1, o[3] * c.stride() - c.pad() + c.kernel() - 1)};
                }
            }
        }
    }

    /**
     * Runs the network with the patch stage executed patch by patch.
     *
     * @param x inputs {@code [n, C, H, W]}
     * @return the outputs
     */
    public Tensor forward(Tensor x) {
        return Var.noGrad(() -> {
            int n = x.shape().dim(0);
            int[] last = shapes.get(split - 1);
            float[] buffer = new float[n * last[0] * last[1] * last[2]];
            for (int[][] r : regions) {
                Var h = Var.of(x.slice(2, r[0][0], r[0][1] + 1).slice(3, r[0][2], r[0][3] + 1));
                for (int l = 0; l < split; l++) {
                    EdgeNet.Conv c = (EdgeNet.Conv) net.units().get(l);
                    int[] in = r[l];
                    int[] o = r[l + 1];
                    int s = c.stride();
                    int p = c.pad();
                    int k = c.kernel();
                    int[] before = {in[0] - (o[0] * s - p), in[2] - (o[2] * s - p)};
                    int[] after = {o[1] * s - p + k - 1 - in[1], o[3] * s - p + k - 1 - in[3]};
                    Tensor[] f = c.folded();
                    h = Functional.conv(h, Var.of(f[0]), Var.of(f[1]), new int[] {s, s}, before, after,
                            new int[] {1, 1}, c.groups());
                    if (c.relu()) {
                        h = h.relu();
                    }
                }
                float[] tile = h.value().toArray();
                int th = r[split][1] - r[split][0] + 1;
                int tw = r[split][3] - r[split][2] + 1;
                for (int b = 0; b < n; b++) {
                    for (int ch = 0; ch < last[0]; ch++) {
                        for (int y = 0; y < th; y++) {
                            System.arraycopy(tile, ((b * last[0] + ch) * th + y) * tw, buffer,
                                    ((b * last[0] + ch) * last[1] + r[split][0] + y) * last[2] + r[split][2], tw);
                        }
                    }
                }
            }
            Var stage = Var.of(Tensor.of(Shape.of(n, last[0], last[1], last[2]), buffer));
            return net.forwardUnits(stage, split, net.units().size(), false).value();
        });
    }

    /**
     * The largest absolute difference from layer-by-layer execution.
     *
     * @param x inputs
     * @return the difference
     */
    public double maxDifference(Tensor x) {
        float[] a = forward(x).toArray();
        float[] b = Var.noGrad(() -> net.forward(Var.of(x), false)).value().toArray();
        double d = 0;
        for (int i = 0; i < a.length; i++) {
            d = Math.max(d, Math.abs(a[i] - b[i]));
        }
        return d;
    }

    private long macsPerPixel(int l) {
        EdgeNet.Conv c = (EdgeNet.Conv) net.units().get(l);
        return (long) c.outChannels() * (c.inChannels() / c.groups()) * c.kernel() * c.kernel();
    }

    private static long area(int[] r) {
        return (long) (r[1] - r[0] + 1) * (r[3] - r[2] + 1);
    }

    /** @return multiply-adds per sample of the patch stage computed patch by patch */
    public long patchStageMacs() {
        long m = 0;
        for (int[][] r : regions) {
            for (int l = 0; l < split; l++) {
                m += area(r[l + 1]) * macsPerPixel(l);
            }
        }
        return m;
    }

    /** @return multiply-adds per sample of the network with patch-based execution */
    public long macs() {
        long[] per = net.macsPerUnit();
        long m = patchStageMacs();
        for (int i = split; i < per.length; i++) {
            m += per[i];
        }
        return m;
    }

    /** @return extra multiply-adds over layer-by-layer execution, as a share of the network's */
    public double extraMacShare() {
        return macs() / (double) net.macs() - 1;
    }

    /**
     * Peak activation memory: during the patch stage, the stage's output buffer plus the
     * largest input and output of one layer on one patch; afterwards, layer by layer.
     *
     * @param bytesPerElement 1 for int8
     * @return the peak in bytes
     */
    public long peakBytes(int bytesPerElement) {
        long buffer = EdgeNet.size(shapes.get(split - 1));
        long patchPeak = 0;
        for (int[][] r : regions) {
            for (int l = 0; l < split; l++) {
                int inC = l == 0 ? net.inputShape()[0] : shapes.get(l - 1)[0];
                int outC = shapes.get(l)[0];
                patchPeak = Math.max(patchPeak, area(r[l]) * inC + area(r[l + 1]) * outC);
            }
        }
        long peak = buffer + patchPeak;
        for (int i = split; i < net.units().size(); i++) {
            peak = Math.max(peak, EdgeNet.size(shapes.get(i - 1)) + EdgeNet.size(shapes.get(i)));
        }
        return peak * bytesPerElement;
    }

    /**
     * Peak activation memory of layer-by-layer execution.
     *
     * @param net the network
     * @param bytesPerElement 1 for int8
     * @return the peak in bytes
     */
    public static long layerByLayerPeak(EdgeNet net, int bytesPerElement) {
        return MemoryPlanner.chain(net, bytesPerElement).peak(MemoryPlanner.chain(net, bytesPerElement).defaultOrder());
    }

    /** @return patches per side */
    public int grid() {
        return grid;
    }
}
