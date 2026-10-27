// Per window (one workgroup each): exclusive prefix sums over the buckets of
//   starts[slot]  = first sorted position of the bucket's points
//   istarts[slot] = first work item of the bucket (a bucket of m points has ceil(m / CH) items, CH = sizes[2])
// plus item_totals[window]. Also resets `counts` to 0 so the scatter pass can reuse it as a
// cursor. The constant B is prepended (a multiple of 256); CH is sizes[2].

@group(0) @binding(0) var<storage, read_write> counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> starts: array<u32>;
@group(0) @binding(2) var<storage, read_write> istarts: array<u32>;
@group(0) @binding(3) var<storage, read_write> item_totals: array<u32>;
@group(0) @binding(4) var<storage, read> sizes: array<u32>; // [N, IPW, CH]
var<workgroup> point_sums: array<u32, 256>;
var<workgroup> item_sums: array<u32, 256>;

fn items(count: u32) -> u32 { return (count + sizes[2] - 1u) / sizes[2]; }

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let per = B / 256u;
  let base = wid.x * B + lid.x * per;
  var points = 0u;
  var its = 0u;
  for (var k = 0u; k < per; k++) {
    let c = counts[base + k];
    points += c;
    its += items(c);
  }
  point_sums[lid.x] = points;
  item_sums[lid.x] = its;
  workgroupBarrier();
  if (lid.x == 0u) {
    var run_p = 0u;
    var run_i = 0u;
    for (var t = 0u; t < 256u; t++) {
      let p = point_sums[t]; point_sums[t] = run_p; run_p += p;
      let i = item_sums[t]; item_sums[t] = run_i; run_i += i;
    }
    item_totals[wid.x] = run_i;
  }
  workgroupBarrier();
  var run_p = point_sums[lid.x];
  var run_i = item_sums[lid.x];
  for (var k = 0u; k < per; k++) {
    let c = counts[base + k];
    starts[base + k] = run_p;
    istarts[base + k] = run_i;
    run_p += c;
    run_i += items(c);
    counts[base + k] = 0u;
  }
}
