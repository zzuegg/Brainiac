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
