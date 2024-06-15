use std::f32::consts::PI;

use serialize_macro::Serialize;
use serialization::{test_serialization, Serialized};

#[test]
fn test_serialize_union() {
    #[derive(Debug, PartialEq, Serialize)]
    pub struct TestUnion;
    let test = TestUnion;
    test_serialization!(test, TestUnion);
}

#[test]
fn test_serialize_tuple_struct() {
    #[derive(Debug, PartialEq, Serialize)]
    pub struct TestTupleStruct(usize, f32, String);
    let test = TestTupleStruct(42, PI, "Hello, World!".to_string());
    test_serialization!(test, TestTupleStruct);
}

#[test]
fn test_serialize_struct() {
    #[derive(Debug, PartialEq, Serialize)]
    pub struct TestStruct {
        x: usize,
        y: f32,
        z: String
    }
    let test = TestStruct { x: 42, y: PI, z: "Hello, World!".to_string() };
    test_serialization!(test, TestStruct);
}

#[test]
fn test_serialize_enum_tuple() {
    #[derive(Debug, PartialEq, Serialize)]
    pub enum TestEnumTuple {
        One,
        Two(usize),
        Three(usize, f32)
    }
    let one = TestEnumTuple::One;
    let two = TestEnumTuple::Two(42);
    let three = TestEnumTuple::Three(42, PI);
    test_serialization!(one, TestEnumTuple);
    test_serialization!(two, TestEnumTuple);
    test_serialization!(three, TestEnumTuple);
}

#[test]
fn test_serialize_enum_struct() {
    #[derive(Debug, PartialEq, Serialize)]
    pub enum TestEnumStruct {
        One,
        Two { x: usize, y: f32 },
        Three { a: u32 }
    }
    let one = TestEnumStruct::One;
    let two = TestEnumStruct::Two { x: 42, y: PI };
    let three = TestEnumStruct::Three { a: 42 };
    test_serialization!(one, TestEnumStruct);
    test_serialization!(two, TestEnumStruct);
    test_serialization!(three, TestEnumStruct);
}
