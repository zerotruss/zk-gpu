// Signed-digit decomposition of every scalar. Each point lands in one bucket per window
// (digit magnitude - 1), negated when the digit is negative. Run twice with the same code:
// first to count points per bucket, then (SCATTER) to write them into sorted order.
// Constants C, W, B and SCATTER are prepended; the point count N is sizes[0] (a run-time value, so
// the compiled shader is shared by every input size).

@group(0) @binding(0) var<storage, read> scalars: array<u32>;            // 8 words per point
@group(0) @binding(1) var<storage, read_write> counts: array<atomic<u32>>; // W * B
@group(0) @binding(2) var<storage, read> starts: array<u32>;              // W * B (scatter only)
@group(0) @binding(3) var<storage, read_write> sorted: array<u32>;        // W * N (scatter only)
@group(0) @binding(4) var<storage, read> sizes: array<u32>;               // [N, IPW, CH]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= sizes[0]) { return; }
  var carry = 0u;
  for (var w = 0u; w < W; w++) {
    let pos = w * C;
    let word = pos >> 5u;
    let off = pos & 31u;
    var v = scalars[i * 8u + word] >> off;
    if (off + C > 32u && word + 1u < 8u) { v |= scalars[i * 8u + word + 1u] << (32u - off); }
    v = (v & ((1u << C) - 1u)) + carry;
    var mag = v;
    var neg = 0u;
    carry = 0u;
    if (v >= B) { mag = (1u << C) - v; neg = 1u; carry = 1u; } // digit = v - 2^C
    if (mag > 0u) {
      let slot = w * B + mag - 1u;
      if (SCATTER) {
        sorted[w * sizes[0] + starts[slot] + atomicAdd(&counts[slot], 1u)] = i | (neg << 31u);
      } else {
        atomicAdd(&counts[slot], 1u);
      }
    }
  }
}
