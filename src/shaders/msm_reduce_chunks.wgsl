// Bucket reduction  S_w = sum_b (b + 1) * bucket[w][b], step 1. A thread takes L consecutive
// buckets starting at b0 = j * L and, with a running sum from the top, computes
//   tot = sum_k (k + 1) * bucket[b0 + k]   and   run = sum_k bucket[b0 + k].
// Since (b + 1) = (k + 1) + b0, the chunk contributes  tot + b0 * run.
// Constants C, W, B, L, T (= B / L) are prepended.

@group(0) @binding(0) var<storage, read> buckets: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<u32>>; // W * T Xyzz
@group(0) @binding(2) var<storage, read> slice: array<u32>; // [first thread of this dispatch]

fn load_bucket(i: u32) -> Xyzz {
  var a: array<vec4<u32>, XW>;
  for (var k = 0u; k < XW; k++) { a[k] = buckets[XW * i + k]; }
  return xyzz_unpack(a);
}

fn mul_small(p: Xyzz, m: u32) -> Xyzz {
  var r = xyzz_infinity();
  for (var k = C; k > 0u; k--) {
    if (!xyzz_is_infinity(r)) { r = xyzz_double(r); }
    if (((m >> (k - 1u)) & 1u) == 1u) { r = xyzz_add(r, p); }
  }
  return r;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let t = id.x + slice[0];
  if (t >= W * T) { return; }
  let b0 = (t % T) * L;
  let first = (t / T) * B + b0;
  var run = xyzz_infinity();
  var tot = xyzz_infinity();
  for (var k = L; k > 0u; k--) {
    run = xyzz_add(run, load_bucket(first + k - 1u));
    tot = xyzz_add(tot, run);
  }
  let a = xyzz_pack(xyzz_add(tot, mul_small(run, b0)));
  for (var k = 0u; k < XW; k++) { out[XW * t + k] = a[k]; }
}
