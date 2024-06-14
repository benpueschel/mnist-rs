use math::{Matrix, Vector};

pub trait Serialized {
    fn serialize_binary(&self) -> Vec<u8>;
    fn deserialize_binary(data: &[u8]) -> (Self, usize)
    where
        Self: Sized;
    fn tag() -> &'static str 
    where
        Self: Sized;
}

pub fn f64_from_bytes(bytes: &[u8]) -> f64 {
    let b = bytes;
    f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

pub fn u64_from_bytes(bytes: &[u8]) -> u64 {
    let b: &[u8] = bytes;
    u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

impl Serialized for Vector {
    fn serialize_binary(&self) -> Vec<u8> {
        let len = self.0.len();
        let mut data = Vec::with_capacity(8 + len * 8);
        data.extend(len.to_be_bytes());
        for i in 0..len {
            data.extend(self.at(i).to_be_bytes());
        }
        data
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        let len = u64_from_bytes(&data[0..]) as usize;
        let mut result = Vector::new(len);
        let mut offset = 8;
        for i in 0..len {
            result.set(i, f64_from_bytes(&data[offset..]));
            offset += 8;
        }
        (result, offset)
    }
    fn tag() -> &'static str {
        "Vector"
    }
}

impl Serialized for Matrix {
    fn serialize_binary(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(16 + self.cols() * self.rows() * 8);
        data.extend((self.rows() as u64).to_be_bytes());
        data.extend((self.cols() as u64).to_be_bytes());
        for i in 0..self.cols() {
            for j in 0..self.rows() {
                data.extend(self.at(i, j).to_be_bytes());
            }
        }
        data
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        let rows = u64_from_bytes(&data[0..]) as usize;
        let cols = u64_from_bytes(&data[8..]) as usize;
        let mut offset = 16;
        let mut result = Matrix::new(rows, cols);
        for i in 0..cols {
            for j in 0..rows {
                result.set(i, j, f64_from_bytes(&data[offset..]));
                offset += 8;
            }
        }
        (result, offset)
    }
    fn tag() -> &'static str {
        "Matrix"
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn test_deserialize_u64() {
        let num = rand::random::<u64>();
        assert_eq!(num, u64_from_bytes(&num.to_be_bytes()));
    }

    #[test]
    pub fn test_deserialize_f64() {
        let num = rand::random::<f64>();
        assert_eq!(num, f64_from_bytes(&num.to_be_bytes()));
    }
}
