use crate::Serialized;

pub fn f64_from_bytes(bytes: &[u8]) -> f64 {
    let b = bytes;
    f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

pub fn u64_from_bytes(bytes: &[u8]) -> u64 {
    let b: &[u8] = bytes;
    u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

pub fn i64_from_bytes(bytes: &[u8]) -> i64 {
    let b: &[u8] = bytes;
    i64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

impl<'a> Serialized for String {
    fn serialize_binary(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(self.len() + 8);
        data.extend(self.len().serialize_binary());
        data.extend(self.bytes());
        data
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        let len = usize::deserialize_binary(&data[0..]).0;
        (Self::from_utf8(data[8..8+len].to_vec()).unwrap(), len + 8)
    }

    fn tag() -> &'static str {
        "String"
    }
}

impl Serialized for f64 {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (f64_from_bytes(data), 8)
    }
    fn tag() -> &'static str {
        "f64"
    }
}

impl Serialized for f32 {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (f32::from_be_bytes([data[0], data[1], data[2], data[3]]), 4)
    }
    fn tag() -> &'static str {
        "f32"
    }
}

impl Serialized for u64 {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (u64_from_bytes(data), 8)
    }
    fn tag() -> &'static str {
        "u64"
    }
}

impl Serialized for u32 {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (u32::from_be_bytes([data[0], data[1], data[2], data[3]]), 4)
    }
    fn tag() -> &'static str {
        "u32"
    }
}

impl Serialized for u16 {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (u16::from_be_bytes([data[0], data[1]]), 2)
    }
    fn tag() -> &'static str {
        "u16"
    }
}

impl Serialized for u8 {
    fn serialize_binary(&self) -> Vec<u8> {
        vec![*self]
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (data[0], 1)
    }
    fn tag() -> &'static str {
        "u8"
    }
}

impl Serialized for usize {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        #[cfg(target_pointer_width = "64")]
        type Target = u64;
        #[cfg(target_pointer_width = "32")]
        type Target = u32;
        #[cfg(target_pointer_width = "16")]
        type Target = u16;
        let x = Target::deserialize_binary(data);
        (x.0 as usize, x.1)
    }
    fn tag() -> &'static str {
        "usize"
    }
}
impl Serialized for i64 {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (i64_from_bytes(data), 8)
    }
    fn tag() -> &'static str {
        "i64"
    }
}

impl Serialized for i32 {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (i32::from_be_bytes([data[0], data[1], data[2], data[3]]), 4)
    }
    fn tag() -> &'static str {
        "i32"
    }
}

impl Serialized for i16 {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (i16::from_be_bytes([data[0], data[1]]), 2)
    }
    fn tag() -> &'static str {
        "i16"
    }
}

impl Serialized for i8 {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        (Self::from_be_bytes([data[0]]), 1)
    }
    fn tag() -> &'static str {
        "i8"
    }
}

impl Serialized for isize {
    fn serialize_binary(&self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        #[cfg(target_pointer_width = "64")]
        type Target = i64;
        #[cfg(target_pointer_width = "32")]
        type Target = i32;
        #[cfg(target_pointer_width = "16")]
        type Target = i16;
        let x = Target::deserialize_binary(data);
        (x.0 as isize, x.1)
    }
    fn tag() -> &'static str {
        "isize"
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{test_random_value, test_serialization};

    #[test]
    fn test_deserialize_string() {
        test_serialization!(String::from("Hello, World!"), String);
    }

    test_random_value!(test_deserialize_usize, usize);
    test_random_value!(test_deserialize_u64, u64);
    test_random_value!(test_deserialize_u32, u32);
    test_random_value!(test_deserialize_u16, u16);
    test_random_value!(test_deserialize_u8, u8);

    test_random_value!(test_deserialize_isize, isize);
    test_random_value!(test_deserialize_i64, i64);
    test_random_value!(test_deserialize_i32, i32);
    test_random_value!(test_deserialize_i16, i16);
    test_random_value!(test_deserialize_i8, i8);

    test_random_value!(test_deserialize_f64, f64);
    test_random_value!(test_deserialize_f32, f32);
}
