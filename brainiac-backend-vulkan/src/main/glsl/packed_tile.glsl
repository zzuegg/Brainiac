// The values of a packed matrix four columns at a time, for products that stage it in
// shared memory; included after packed_layout.glsl.

// the values of row r, columns c .. c + 3: level times scale plus offset, 0 past column n
ACC4 packed4(uint r, uint c) {
    ACC4 v = ACC4(0);
    uint g = r / p.group;
    bool offsets = (p.flags & 2u) != 0;
    if ((p.n & 3u) == 0 && c + 3 < p.n) {
        // one word of levels
        ivec4 l = levels4At(r, c);
        for (uint e = 0; e < 4; e++) {
            float s = scaleAt(g, c + e);
            v[e] = ACC(l[e]) * ACC(s) + (offsets ? ACC(offsetAt(g, c + e, s)) : ACC(0));
        }
    } else {
        for (uint e = 0; e < 4; e++) {
            if (c + e < p.n) {
                float s = scaleAt(g, c + e);
                v[e] = ACC(levelAt(r, c + e)) * ACC(s) + (offsets ? ACC(offsetAt(g, c + e, s)) : ACC(0));
            }
        }
    }
    return v;
}

// Scales and offsets staged per tile step: with groups of a multiple of 16 rows and steps
// starting at multiples of 16, the rows t .. t + 31 of a step span at most two groups, so
// the step's scales and offsets are computed once per column instead of once per value.
// staged(from): whether this piece's steps can (uniform over the work group).
bool staged(uint from, uint step) {
    return (p.group & 15u) == 0 && (from % step) == 0 && (p.n & 3u) == 0;
}

// the scale and offset of the i-th group from row t's (i = 0, 1) in column c, 0 past k or n
vec2 stagedFactors(uint t, uint i, uint c) {
    uint g = t / p.group + i;
    if (c >= p.n || g * p.group >= p.k) return vec2(0);
    float s = scaleAt(g, c);
    return vec2(s, (p.flags & 2u) != 0 ? offsetAt(g, c, s) : 0.0);
}
