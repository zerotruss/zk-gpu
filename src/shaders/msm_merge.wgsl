// Merge round for heavy buckets. A bucket's work items sit next to each other (item k of bucket b
// is at istarts[b] + k). Round r has stride s = G^r. A bucket that still has more than G items at
// that stride (ceil(items / s) > G) is merged: every item whose k is a multiple of s * G adds the
// items k + s, k + 2s, ... (up to G in all) into itself, in place. Light buckets do nothing here;
// `finalize` sums their at most G remaining items. This keeps a huge bucket (a sparse top window,
// many equal scalars) from becoming one long serial sum in a single thread.
// The constant G is prepended; sizes = [N, IPW, CH] and prm = [s] are run-time values.

@group(0) @binding(0) var<storage, read_write> item_results: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> item_info: array<u32>; // per item: [k, items in its bucket]
@group(0) @binding(2) var<storage, read> item_totals: array<u32>;
@group(0) @binding(3) var<storage, read> sizes: array<u32>;
@group(0) @binding(4) var<storage, read> prm: array<u32>;

fn load_item(i: u32) -> Xyzz {
  var a: array<vec4<u32>, XW>;
  for (var k = 0u; k < XW; k++) { a[k] = item_results[XW * i + k]; }
  return xyzz_unpack(a);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let item = id.x;
  let window = id.y;
  if (item >= item_totals[window]) { return; }
  let base = window * sizes[1] + item;
  let k = item_info[2u * base];
  let total = item_info[2u * base + 1u];
  let s = prm[0];
  if ((total + s - 1u) / s <= G) { return; }
  if (k % (s * G) != 0u) { return; }
  var acc = load_item(base);
  for (var j = 1u; j < G; j++) {
    if (k + j * s >= total) { break; }
    acc = xyzz_add(acc, load_item(base + j * s));
  }
  let a = xyzz_pack(acc);
  for (var q = 0u; q < XW; q++) { item_results[XW * base + q] = a[q]; }
}
