use math::{Matrix, Vector};

pub mod literals;

pub trait Serialized {
    fn serialize_binary(&self) -> Vec<u8>;
    fn deserialize_binary(data: &[u8]) -> (Self, usize)
    where
        Self: Sized;
    fn tag() -> &'static str 
    where
        Self: Sized;
}

#[macro_export]
macro_rules! test_serialization {
    ($x: expr, $y: ident) => {
        let x = $x;
        let serialized = x.serialize_binary();
        assert_eq!(x, $y::deserialize_binary(&serialized).0);
    };
}
#[macro_export]
macro_rules! test_random_value {
    ($fn_name: ident, $x: ident) => {
        #[test]
        fn $fn_name() {
            $crate::test_serialization!(rand::random::<$x>(), $x);
        }
    };
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
        let len = u64::deserialize_binary(&data[0..]).0 as usize;
        let mut result = Vector::new(len);
        let mut offset = 8;
        for i in 0..len {
            result.set(i, f64::deserialize_binary(&data[offset..]).0);
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
        let rows = u64::deserialize_binary(&data[0..]).0 as usize;
        let cols = u64::deserialize_binary(&data[8..]).0 as usize;
        let mut offset = 16;
        let mut result = Matrix::new(rows, cols);
        for i in 0..cols {
            for j in 0..rows {
                result.set(i, j, f64::deserialize_binary(&data[offset..]).0);
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
    pub fn test_deserialize_vector() {
        let len = rand::random::<u8>() as usize;
        let v = Vector::new(len);
        test_serialization!(v, Vector);
    }

    #[test]
    pub fn test_deserialize_matrix() {
        let rows = rand::random::<u8>() as usize;
        let cols = rand::random::<u8>() as usize;
        let v = Matrix::new(rows, cols);
        test_serialization!(v, Matrix);
    }
    
}