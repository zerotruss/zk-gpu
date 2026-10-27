//! BN254 prime fields as WGSL code.
//!
//! WGSL has no 32x32->64 multiply, so a 254-bit number is stored as 20 limbs of
//! 13 bits (one per u32). A limb product is then < 2^26, so ~40 of them can be summed
//! in a u32 without any carry handling. That lets Montgomery multiplication do only
//! multiply-adds in its inner loops and propagate carries once, at the end.
//!
//! The functions are generated fully unrolled with scalar variables: array
//! accumulators make GPU compilers spill to memory, scalars stay in registers.

use std::fmt::Write;

pub const LIMBS: usize = 20;
const BITS: u32 = 13;
const MASK: u32 = (1 << BITS) - 1;

/// Base field modulus (curve coordinates), little-endian u64 words.
pub const FQ: [u64; 4] = [0x3c208c16d87cfd47, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029];
/// Scalar field modulus (exponents, NTT domain), little-endian u64 words.
pub const FR: [u64; 4] = [0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];

/// Little-endian u64 words -> 20 limbs of 13 bits.
pub fn to_limbs(x: [u64; 4]) -> [u32; LIMBS] {
    let mut out = [0u32; LIMBS];
    for bit in 0..256 {
        if (x[bit / 64] >> (bit % 64)) & 1 == 1 {
            out[bit / BITS as usize] |= 1 << (bit % BITS as usize);
        }
    }
    out
}

pub fn from_limbs(l: &[u32]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for bit in 0..256 {
        if (l[bit / BITS as usize] >> (bit % BITS as usize)) & 1 == 1 {
            out[bit / 64] |= 1 << (bit % 64);
        }
    }
    out
}

/// -p^-1 mod 2^13, by Newton iteration (p is odd).
fn neg_inv(p0: u32) -> u32 {
    let mut inv = 1u32;
    for _ in 0..5 {
        inv = inv.wrapping_mul(2u32.wrapping_sub(p0.wrapping_mul(inv)));
    }
    inv.wrapping_neg() & MASK
}

/// WGSL for the field `name` ("fq" or "fr") with the given modulus: type `Fq`,
/// `fq_mul` (Montgomery, R = 2^260), `fq_add`, `fq_sub`, `fq_is_zero`, `FQ_ONE` (Montgomery
/// one), and `fq_load` / `fq_words` to read and write the 32-byte memory format.
pub fn wgsl(name: &str, modulus: [u64; 4]) -> String {
    let ty = format!("{}{}", name[..1].to_uppercase(), &name[1..]);
    let p = to_limbs(modulus);
    let np = neg_inv(p[0]);
    let n = LIMBS;
    let mut s = String::new();
    let w = &mut s;

    writeln!(w, "struct {ty} {{ l: array<u32, {n}> }}").unwrap();

    // Memory format is 8 x 32-bit words (compact, 32 bytes); registers use 20 x 13-bit limbs.
    writeln!(w, "fn {name}_unpack(w: array<u32, 8>) -> {ty} {{").unwrap();
    write!(w, "  return {ty}(array<u32, {n}>(").unwrap();
    for j in 0..n {
        let (word, off) = (13 * j / 32, 13 * j % 32);
        let mut e = if off == 0 { format!("w[{word}]") } else { format!("(w[{word}] >> {off}u)") };
        if off + 13 > 32 && word + 1 < 8 {
            e = format!("({e} | (w[{}] << {}u))", word + 1, 32 - off);
        }
        write!(w, "{e} & {MASK}u{}", if j + 1 < n { ", " } else { "" }).unwrap();
    }
    writeln!(w, "));\n}}").unwrap();

    writeln!(w, "fn {name}_pack(a: {ty}) -> array<u32, 8> {{").unwrap();
    write!(w, "  return array<u32, 8>(").unwrap();
    for word in 0..8 {
        let parts: Vec<String> = (0..n)
            .filter(|j| 13 * j < 32 * (word + 1) && 13 * j + 13 > 32 * word)
            .map(|j| {
                if 13 * j >= 32 * word {
                    format!("(a.l[{j}] << {}u)", 13 * j - 32 * word)
                } else {
                    format!("(a.l[{j}] >> {}u)", 32 * word - 13 * j)
                }
            })
            .collect();
        write!(w, "{}{}", parts.join(" | "), if word < 7 { ", " } else { "" }).unwrap();
    }
    writeln!(w, ");\n}}").unwrap();

    // Montgomery one (R mod p) and zero, plus small helpers shared by all kernels.
    let one = to_limbs(crate::host::montgomery_r(&modulus));
    let list = |l: &[u32]| l.iter().map(|x| format!("{x}u")).collect::<Vec<_>>().join(", ");
    writeln!(w, "const {}_ONE = {ty}(array<u32, {n}>({}));", name.to_uppercase(), list(&one)).unwrap();
    writeln!(w, "const {}_ZERO = {ty}(array<u32, {n}>({}));", name.to_uppercase(), list(&[0; LIMBS])).unwrap();
    let any = (0..n).map(|j| format!("a.l[{j}]")).collect::<Vec<_>>().join(" | ");
    writeln!(w, "fn {name}_is_zero(a: {ty}) -> bool {{ return ({any}) == 0u; }}").unwrap();
    writeln!(w, "struct {ty}Words {{ lo: vec4<u32>, hi: vec4<u32> }}").unwrap();
    writeln!(w, "fn {name}_load(lo: vec4<u32>, hi: vec4<u32>) -> {ty} {{ return {name}_unpack(array<u32, 8>(lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w)); }}").unwrap();
    writeln!(w, "fn {name}_words(a: {ty}) -> {ty}Words {{ let w = {name}_pack(a); return {ty}Words(vec4<u32>(w[0], w[1], w[2], w[3]), vec4<u32>(w[4], w[5], w[6], w[7])); }}").unwrap();

    // Montgomery product: a * b * R^-1 mod p. Row i adds a_i*b, then m_i*p, which zeroes
    // the low limb; the shift by one limb is just the index offset (t[i+j]).
    writeln!(w, "fn {name}_mul(a: {ty}, b: {ty}) -> {ty} {{").unwrap();
    for k in 0..2 * n {
        writeln!(w, "  var t{k}: u32 = 0u;").unwrap();
    }
    for i in 0..n {
        for j in 0..n {
            writeln!(w, "  t{} += a.l[{i}] * b.l[{j}];", i + j).unwrap();
        }
        writeln!(w, "  let m{i} = (t{i} * {np}u) & {MASK}u;").unwrap();
        for (j, pj) in p.iter().enumerate().filter(|(_, pj)| **pj != 0) {
            writeln!(w, "  t{} += m{i} * {pj}u;", i + j).unwrap();
        }
        writeln!(w, "  t{} += t{i} >> {BITS}u;", i + 1).unwrap();
    }
    // Upper half holds the result with deferred carries: normalise, then reduce below p.
    for j in 0..n {
        writeln!(w, "  let r{j} = t{} & {MASK}u;", n + j).unwrap();
        if j + 1 < n {
            writeln!(w, "  t{} += t{} >> {BITS}u;", n + j + 1, n + j).unwrap();
        }
    }
    emit_reduce(w, &ty, &p, "r");
    writeln!(w, "}}").unwrap();

    // Addition: limb-wise sum with carries, then reduce below p.
    writeln!(w, "fn {name}_add(a: {ty}, b: {ty}) -> {ty} {{").unwrap();
    writeln!(w, "  var c: u32 = 0u;").unwrap();
    for j in 0..n {
        writeln!(w, "  let x{j} = a.l[{j}] + b.l[{j}] + c; let r{j} = x{j} & {MASK}u; c = x{j} >> {BITS}u;").unwrap();
    }
    emit_reduce(w, &ty, &p, "r");
    writeln!(w, "}}").unwrap();

    // Subtraction: a - b, and add p back if it went negative.
    writeln!(w, "fn {name}_sub(a: {ty}, b: {ty}) -> {ty} {{").unwrap();
    writeln!(w, "  var br: u32 = 0u;").unwrap();
    for j in 0..n {
        writeln!(
            w,
            "  let x{j} = a.l[{j}] + {} - b.l[{j}] - br; let d{j} = x{j} & {MASK}u; br = 1u - (x{j} >> {BITS}u);",
            1 << BITS
        )
        .unwrap();
    }
    writeln!(w, "  var c: u32 = 0u;").unwrap();
    for (j, pj) in p.iter().enumerate() {
        writeln!(w, "  let y{j} = d{j} + {pj}u + c; let e{j} = y{j} & {MASK}u; c = y{j} >> {BITS}u;").unwrap();
    }
    write!(w, "  return {ty}(array<u32, {n}>(").unwrap();
    for j in 0..n {
        write!(w, "select(d{j}, e{j}, br == 1u){}", if j + 1 < n { ", " } else { "" }).unwrap();
    }
    writeln!(w, "));\n}}").unwrap();
    s
}

/// Emit "return (v < p) ? v : v - p" for limbs named `{v}0..{v}19` (value < 2p).
fn emit_reduce(w: &mut String, ty: &str, p: &[u32; LIMBS], v: &str) {
    let n = LIMBS;
    writeln!(w, "  var br: u32 = 0u;").unwrap();
    for (j, pj) in p.iter().enumerate() {
        writeln!(
            w,
            "  let z{j} = {v}{j} + {} - {pj}u - br; let s{j} = z{j} & {MASK}u; br = 1u - (z{j} >> {BITS}u);",
            1 << BITS
        )
        .unwrap();
    }
    write!(w, "  return {ty}(array<u32, {n}>(").unwrap();
    for j in 0..n {
        write!(w, "select(s{j}, {v}{j}, br == 1u){}", if j + 1 < n { ", " } else { "" }).unwrap();
    }
    writeln!(w, "));").unwrap();
}
