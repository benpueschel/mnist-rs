use serialization::Serialized;

#[test]
fn test_serialize_macro() {
    use serialize_macro::Serialize;
    use serialization::Serialized;

    #[derive(Serialize)]
    pub enum TestEnum {
        OneTest,
        TwoTest(usize),
        ThreeTest
    }

}

use serialize_macro::Serialize;
use serialization::Serialized;
pub enum TestEnum {
    OneTest,
    TwoTest(usize),
    ThreeTest,
}
impl Serialized for TestEnum {
    fn serialize_binary(&self) -> Vec<u8> {
        let mut data = Vec::new();
        let tag = Self::tag().as_bytes();
        data.extend((tag.len() as u64).to_be_bytes());
        data.extend(tag);
        match self {
            TestEnum::OneTest => {
                let tag = ":: OneTest".as_bytes();
                data.extend((tag.len() as u64).to_be_bytes());
                data.extend(tag);
            }
            TestEnum::TwoTest(_) => {
                let tag = ":: TwoTest".as_bytes();
                data.extend((tag.len() as u64).to_be_bytes());
                data.extend(tag);
                let mut serialized = self.0.serialize_binary();
                data.extend_from_slice(&serialized);
            }
            TestEnum::ThreeTest => {
                let tag = ":: ThreeTest".as_bytes();
                data.extend((tag.len() as u64).to_be_bytes());
                data.extend(tag);
            }
        }
        data
    }
    fn deserialize_binary(data: &[u8]) -> (Self, usize) {
        let tag = {
            let len = u64::from_be_bytes([
                data[0],
                data[1],
                data[2],
                data[3],
                data[4],
                data[5],
                data[6],
                data[7],
            ]) as usize;
            String::from_utf8(data[8..8 + len].to_vec()).unwrap()
        };
        let mut offset = tag.len() + 8;
        let variant_name = {
            let len = u64::from_be_bytes([
                data[offset + 0],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]) as usize;
            String::from_utf8(data[offset + 8..offset + 8 + len].to_vec()).unwrap()
        };
        offset += 8 + variant_name.len();
        (
            match format!("{0}{1}", tag.to_string(), variant_name).as_str() {
                "TestEnum :: OneTest" => TestEnum::OneTest,
                "TestEnum :: TwoTest" => {
                    TestEnum::TwoTest({
                        let (field, read_bytes) = Serialized::deserialize_binary(
                            &data[offset..],
                        );
                        offset += read_bytes;
                        field
                    })
                }
                "TestEnum :: ThreeTest" => TestEnum::ThreeTest,
                x => {
                    ::core::panicking::panic_fmt(
                        format_args!("invalid enum variant {0}", x),
                    );
                }
            },
            offset,
        )
    }
    fn tag() -> &'static str {
        "TestEnum"
    }
}