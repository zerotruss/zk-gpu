// One thread per work item = a run of at most CH points of one bucket. Work items are
// numbered per window; a binary search over `istarts` finds the bucket an item belongs to.
// Constants C, W, B are prepended; sizes = [N, IPW (item slots per window), CH] is a run-time value.

@group(0) @binding(0) var<storage, read> bases: array<vec4<u32>>;     // AW vec4 per point: x, y
@group(0) @binding(1) var<storage, read> sorted: array<u32>;
@group(0) @binding(2) var<storage, read> starts: array<u32>;
@group(0) @binding(3) var<storage, read> counts: array<u32>;
@group(0) @binding(4) var<storage, read> istarts: array<u32>;
@group(0) @binding(5) var<storage, read> item_totals: array<u32>;
@group(0) @binding(6) var<storage, read_write> item_results: array<vec4<u32>>; // XW vec4 per Xyzz
@group(0) @binding(7) var<storage, read> sizes: array<u32>;
@group(0) @binding(8) var<storage, read_write> item_info: array<u32>; // per item: [k, items in its bucket]
@group(0) @binding(9) var<storage, read> slice: array<u32>; // [first window of this dispatch]

fn load_base(i: u32) -> Affine {
  var a: array<vec4<u32>, AW>;
  for (var k = 0u; k < AW; k++) { a[k] = bases[AW * i + k]; }
  return affine_unpack(a);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  let window = id.y + slice[0];
  if (item >= item_totals[window]) { return; }
  // Largest bucket whose first item is <= item.
  var lo = 0u;
  var hi = B;
  while (hi - lo > 1u) {
    let mid = (lo + hi) / 2u;
    if (istarts[window * B + mid] <= item) { lo = mid; } else { hi = mid; }
  }
  let slot = window * B + lo;
  let skip = (item - istarts[slot]) * sizes[2];
  let len = min(sizes[2], counts[slot] - skip);
  let first = window * sizes[0] + starts[slot] + skip;
  var acc = xyzz_infinity();
  for (var j = 0u; j < len; j++) {
    let entry = sorted[first + j];
    var p = load_base(entry & 0x7fffffffu);
    if ((entry >> 31u) == 1u) { p = affine_neg(p); }
    acc = xyzz_add_affine(acc, p);
  }
  let a = xyzz_pack(acc);
  let at = window * sizes[1] + item;
  for (var k = 0u; k < XW; k++) { item_results[XW * at + k] = a[k]; }
  item_info[2u * at] = item - istarts[slot];
  item_info[2u * at + 1u] = (counts[slot] + sizes[2] - 1u) / sizes[2];
}
