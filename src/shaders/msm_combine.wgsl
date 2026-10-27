// result = sum_w 2^(C*w) * S_w, by Horner's rule from the top window. Single thread.
// Constants C, W are prepended.

@group(0) @binding(0) var<storage, read> window_sums: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> result: array<vec4<u32>>; // one Xyzz

fn load_sum(i: u32) -> Xyzz {
  var a: array<vec4<u32>, XW>;
  for (var k = 0u; k < XW; k++) { a[k] = window_sums[XW * i + k]; }
  return xyzz_unpack(a);
}

@compute @workgroup_size(1)
fn main() {
  var acc = load_sum(W - 1u);
  for (var w = W - 1u; w > 0u; w--) {
    for (var k = 0u; k < C; k++) { acc = xyzz_double(acc); }
    acc = xyzz_add(acc, load_sum(w - 1u));
  }
  let a = xyzz_pack(acc);
  for (var k = 0u; k < XW; k++) { result[k] = a[k]; }
}
