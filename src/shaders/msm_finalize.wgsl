// One thread per bucket: sum the bucket's work items. Heavy buckets were reduced by the merge
// rounds, so their remaining items are spaced `stride` apart (stride = G^r, the first power of G
// with ceil(items / stride) <= G); a light bucket has stride 1. At most G items are added.
// Constants C, G are prepended; sizes = [N, IPW (item slots per window), CH] is a run-time value.

@group(0) @binding(0) var<storage, read> item_results: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> istarts: array<u32>;
@group(0) @binding(3) var<storage, read_write> buckets: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> dims: array<u32>; // [W * B]
@group(0) @binding(5) var<storage, read> sizes: array<u32>;

fn load_item(i: u32) -> Xyzz {
  var a: array<vec4<u32>, XW>;
  for (var k = 0u; k < XW; k++) { a[k] = item_results[XW * i + k]; }
  return xyzz_unpack(a);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x;
  if (slot >= dims[0]) { return; }
  let first = (slot >> (C - 1u)) * sizes[1] + istarts[slot];
  let total = (counts[slot] + sizes[2] - 1u) / sizes[2];
  var stride = 1u;
  while ((total + stride - 1u) / stride > G) { stride *= G; }
  var acc = xyzz_infinity();
  for (var k = 0u; k < total; k += stride) { acc = xyzz_add(acc, load_item(first + k)); }
  let a = xyzz_pack(acc);
  for (var k = 0u; k < XW; k++) { buckets[XW * slot + k] = a[k]; }
}
