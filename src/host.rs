//! Tiny 256-bit modular arithmetic for setup work on the CPU (roots of unity, Montgomery
//! constants). Slow but simple: only used a few thousand times per plan.

pub type U256 = [u64; 4];

fn geq(a: &U256, b: &U256) -> bool {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

fn sub(a: &U256, b: &U256) -> U256 {
    let (mut out, mut borrow) = ([0u64; 4], false);
    for i in 0..4 {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow as u64);
        out[i] = d2;
        borrow = b1 || b2;
    }
    out
}

/// a - b mod m (both already reduced).
pub fn sub_mod(a: &U256, b: &U256, m: &U256) -> U256 {
    if geq(a, b) {
        sub(a, b)
    } else {
        add_mod(a, &sub(m, b), m)
    }
}

pub fn add_mod(a: &U256, b: &U256, m: &U256) -> U256 {
    let (mut out, mut carry) = ([0u64; 4], false);
    for i in 0..4 {
        let (s1, c1) = a[i].overflowing_add(b[i]);
        let (s2, c2) = s1.overflowing_add(carry as u64);
        out[i] = s2;
        carry = c1 || c2;
    }
    if carry || geq(&out, m) {
        sub(&out, m)
    } else {
        out
    }
}

/// a * b mod m by double-and-add (m < 2^255 so doubling never overflows).
pub fn mul_mod(a: &U256, b: &U256, m: &U256) -> U256 {
    let mut acc = [0u64; 4];
    for bit in (0..256).rev() {
        acc = add_mod(&acc, &acc, m);
        if (b[bit / 64] >> (bit % 64)) & 1 == 1 {
            acc = add_mod(&acc, a, m);
        }
    }
    acc
}

pub fn pow_mod(base: &U256, exp: &U256, m: &U256) -> U256 {
    let mut acc = [1, 0, 0, 0];
    for bit in (0..256).rev() {
        acc = mul_mod(&acc, &acc, m);
        if (exp[bit / 64] >> (bit % 64)) & 1 == 1 {
            acc = mul_mod(&acc, base, m);
        }
    }
    acc
}

pub fn shr(a: &U256, k: u32) -> U256 {
    let mut out = [0u64; 4];
    for bit in k as usize..256 {
        if (a[bit / 64] >> (bit % 64)) & 1 == 1 {
            let t = bit - k as usize;
            out[t / 64] |= 1 << (t % 64);
        }
    }
    out
}

/// R mod m for the GPU kernels' Montgomery radix R = 2^(20 limbs * 13 bits) = 2^260.
pub fn montgomery_r(m: &U256) -> U256 {
    let mut r = [1, 0, 0, 0];
    for _ in 0..260 {
        r = add_mod(&r, &r, m);
    }
    r
}

/// Primitive 2^log_n-th root of unity of the BN254 scalar field (generator 5).
pub fn root_of_unity(log_n: u32) -> U256 {
    let r = crate::field::FR;
    let exp = shr(&sub(&r, &[1, 0, 0, 0]), log_n);
    pow_mod(&[5, 0, 0, 0], &exp, &r)
}

pub fn inverse(x: &U256, m: &U256) -> U256 {
    pow_mod(x, &sub(m, &[2, 0, 0, 0]), m)
}

pub fn words32(x: &U256) -> [u32; 8] {
    std::array::from_fn(|i| (x[i / 2] >> (32 * (i % 2))) as u32)
}

pub fn from_words32(w: &[u32]) -> U256 {
    std::array::from_fn(|i| w[2 * i] as u64 | (w[2 * i + 1] as u64) << 32)
}

/// x * R^-1 mod m: converts a GPU Montgomery-form value back to its plain integer.
pub fn from_montgomery(x: &U256, m: &U256) -> U256 {
    mul_mod(x, &inverse(&montgomery_r(m), m), m)
}
