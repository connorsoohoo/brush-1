//! Minimal numpy `.npy` v1.0 writer for C-order little-endian f32 arrays —
//! the exact subset `brush-dataset/src/load_features.rs` reads back.

use std::io::Write;
use std::path::Path;

pub fn write_npy_f32(path: &Path, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
    assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "npy data length must match shape"
    );
    let shape_str = match shape {
        [n] => format!("({n},)"),
        dims => format!(
            "({})",
            dims.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}");
    // Pad so magic (8) + header-len field (2) + header is a multiple of 64,
    // with a trailing newline, per the npy v1.0 spec.
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat(unpadded.div_ceil(64) * 64 - unpadded));
    header.push('\n');

    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(
        &u16::try_from(header.len())
            .expect("npy header too long")
            .to_le_bytes(),
    )?;
    file.write_all(header.as_bytes())?;
    for v in data {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Minimal reader for the same subset — test-only, for golden fixtures.
#[cfg(test)]
pub fn read_npy_f32(path: &Path) -> std::io::Result<(Vec<f32>, Vec<usize>)> {
    let bytes = std::fs::read(path)?;
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00", "npy magic/version");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).expect("utf8 header");
    assert!(
        header.contains("'<f4'") && header.contains("'fortran_order': False"),
        "unsupported npy header: {header}"
    );
    let shape_part = header.split("'shape':").nth(1).expect("shape key");
    let open = shape_part.find('(').expect("shape open paren");
    let close = shape_part.find(')').expect("shape close paren");
    let shape: Vec<usize> = shape_part[open + 1..close]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data: Vec<f32> = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(data.len(), shape.iter().product::<usize>(), "npy length");
    Ok((data, shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_shape_and_data() {
        let dir = std::env::temp_dir().join("dino_extract_npy_test");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("t.npy");
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
        write_npy_f32(&path, &data, &[2, 3, 4]).expect("write npy");

        let bytes = std::fs::read(&path).expect("read back");
        assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0, "header padded to 64");
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).expect("utf8 header");
        assert!(header.contains("'shape': (2, 3, 4)"), "header: {header}");
        let payload = &bytes[10 + header_len..];
        let read: Vec<f32> = payload
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(read, data, "payload round-trip");
    }
}
