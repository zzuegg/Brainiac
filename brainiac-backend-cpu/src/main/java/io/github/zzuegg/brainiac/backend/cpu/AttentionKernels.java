package io.github.zzuegg.brainiac.backend.cpu;

/**
 * Fused attention on the CPU, one query row (or key row, for the key and value gradients)
 * at a time, with an online softmax: the {@code [tq, tk]} weights are never stored.
 * Arrays are {@code [batch, heads, steps, size]}; see {@code Backend#attention}.
 */
final class AttentionKernels {

    private AttentionKernels() {
    }

    /** {@code shape}: batch, heads, kvHeads, tq, tk, dk, dv. */
    static void forward(double[] q, double[] k, double[] v, int[] shape, double scale, boolean causal,
                        double[] out, double[] lse) {
        int heads = shape[1];
        int kvHeads = shape[2];
        int tq = shape[3];
        int tk = shape[4];
        int dk = shape[5];
        int dv = shape[6];
        int rows = shape[0] * heads * tq;
        Parallel.forRange(rows, (long) tk * (dk + dv) * 4, (from, to) -> {
            double[] acc = new double[dv];
            for (int g = from; g < to; g++) {
                int i = g % tq;
                int bh = g / tq;
                int kv = (bh / heads) * kvHeads + (bh % heads) / (heads / kvHeads);
                int kOff = kv * tk * dk;
                int vOff = kv * tk * dv;
                int qOff = g * dk;
                java.util.Arrays.fill(acc, 0);
                int last = causal ? Math.min(tk - 1, i + tk - tq) : tk - 1;
                double m = Double.NEGATIVE_INFINITY;
                double l = 0;
                for (int j = 0; j <= last; j++) {
                    double s = 0;
                    for (int d = 0; d < dk; d++) {
                        s += q[qOff + d] * k[kOff + j * dk + d];
                    }
                    s *= scale;
                    double mn = Math.max(m, s);
                    double corr = Math.exp(m - mn);
                    double p = Math.exp(s - mn);
                    l = l * corr + p;
                    for (int d = 0; d < dv; d++) {
                        acc[d] = acc[d] * corr + p * v[vOff + j * dv + d];
                    }
                    m = mn;
                }
                for (int d = 0; d < dv; d++) {
                    out[g * dv + d] = acc[d] / l;
                }
                lse[g] = m + Math.log(l);
            }
        });
    }

    /**
     * Latent attention (DeepSeek): every query head reads the one head of latents {@code c},
     * which are both keys (with the rotary keys {@code r}) and values; see
     * {@code Backend#latentAttention}. {@code shape}: batch, heads, tq, tk, latent, rope.
     */
    static void latent(double[] qc, double[] qr, double[] c, double[] r, int[] shape, double scale, double[] bias,
                       long[] strides, double[] out) {
        int heads = shape[1];
        int tq = shape[2];
        int tk = shape[3];
        int dc = shape[4];
        int dr = shape[5];
        int rows = shape[0] * heads * tq;
        Parallel.forRange(rows, (long) tk * (2 * dc + dr) * 4, (from, to) -> {
            double[] acc = new double[dc];
            for (int g = from; g < to; g++) {
                int i = g % tq;
                int bh = g / tq;
                int h = bh % heads;
                int b = bh / heads;
                int cOff = b * tk * dc;
                int rOff = b * tk * dr;
                long bOff = b * strides[0] + h * strides[1] + i * strides[2];
                java.util.Arrays.fill(acc, 0);
                double m = Double.NEGATIVE_INFINITY;
                double l = 0;
                for (int j = 0; j < tk; j++) {
                    double s = 0;
                    for (int d = 0; d < dc; d++) {
                        s += qc[g * dc + d] * c[cOff + j * dc + d];
                    }
                    for (int d = 0; d < dr; d++) {
                        s += qr[g * dr + d] * r[rOff + j * dr + d];
                    }
                    s = s * scale + bias[(int) (bOff + j * strides[3])];
                    double mn = Math.max(m, s);
                    double corr = Math.exp(m - mn);
                    double p = Math.exp(s - mn);
                    l = l * corr + p;
                    for (int d = 0; d < dc; d++) {
                        acc[d] = acc[d] * corr + p * c[cOff + j * dc + d];
                    }
                    m = mn;
                }
                for (int d = 0; d < dc; d++) {
                    out[g * dc + d] = acc[d] / l;
                }
            }
        });
    }

    static void backward(double[] q, double[] k, double[] v, double[] out, double[] lse, double[] grad, int[] shape,
                         double scale, boolean causal, double[] dq, double[] dk, double[] dv) {
        int heads = shape[1];
        int kvHeads = shape[2];
        int tq = shape[3];
        int tk = shape[4];
        int dK = shape[5];
        int dV = shape[6];
        int group = heads / kvHeads;
        int rows = shape[0] * heads * tq;
        double[] dsum = new double[rows];
        Parallel.forRange(rows, (long) tk * (2 * dK + dV) * 2, (from, to) -> {
            for (int g = from; g < to; g++) {
                int i = g % tq;
                int bh = g / tq;
                int kv = (bh / heads) * kvHeads + (bh % heads) / group;
                int kOff = kv * tk * dK;
                int vOff = kv * tk * dV;
                double dd = 0;
                for (int d = 0; d < dV; d++) {
                    dd += grad[g * dV + d] * out[g * dV + d];
                }
                dsum[g] = dd;
                for (int d = 0; d < dK; d++) {
                    dq[g * dK + d] = 0;
                }
                int last = causal ? Math.min(tk - 1, i + tk - tq) : tk - 1;
                for (int j = 0; j <= last; j++) {
                    double s = 0;
                    for (int d = 0; d < dK; d++) {
                        s += q[g * dK + d] * k[kOff + j * dK + d];
                    }
                    double p = Math.exp(s * scale - lse[g]);
                    double dp = 0;
                    for (int d = 0; d < dV; d++) {
                        dp += grad[g * dV + d] * v[vOff + j * dV + d];
                    }
                    double ds = p * (dp - dd) * scale;
                    for (int d = 0; d < dK; d++) {
                        dq[g * dK + d] += ds * k[kOff + j * dK + d];
                    }
                }
            }
        });
        int keys = shape[0] * kvHeads * tk;
        Parallel.forRange(keys, (long) tq * group * (2 * dK + dV) * 2, (from, to) -> {
            for (int g = from; g < to; g++) {
                int j = g % tk;
                int bkv = g / tk;
                int kh = bkv % kvHeads;
                int b = bkv / kvHeads;
                for (int d = 0; d < dK; d++) {
                    dk[g * dK + d] = 0;
                }
                for (int d = 0; d < dV; d++) {
                    dv[g * dV + d] = 0;
                }
                int first = causal ? Math.max(0, j - (tk - tq)) : 0;
                for (int hh = 0; hh < group; hh++) {
                    int qh = b * heads + kh * group + hh;
                    for (int i = first; i < tq; i++) {
                        int r = qh * tq + i;
                        double s = 0;
                        for (int d = 0; d < dK; d++) {
                            s += q[r * dK + d] * k[g * dK + d];
                        }
                        double p = Math.exp(s * scale - lse[r]);
                        double dp = 0;
                        for (int d = 0; d < dV; d++) {
                            double o = grad[r * dV + d];
                            dv[g * dV + d] += p * o;
                            dp += o * v[g * dV + d];
                        }
                        double ds = p * (dp - dsum[r]) * scale;
                        for (int d = 0; d < dK; d++) {
                            dk[g * dK + d] += ds * q[r * dK + d];
                        }
                    }
                }
            }
        });
    }
}
