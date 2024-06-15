use quote::quote;
use syn::{DataUnion, Generics, Ident};

pub fn derive_union_serialization(
    _data: DataUnion,
    name: Ident,
    generics: Generics,
) -> proc_macro::TokenStream {
    quote! {
        // Generate the implementation for the `Serialized` trait
        impl #generics Serialized for #name #generics {
            fn serialize_binary(&self) -> Vec<u8> {
                Self::tag().serialize_binary()
            }
            fn deserialize_binary(data: &[u8]) -> (Self, usize) {
                let (tag, mut offset) = String::deserialize_binary(&data[0..]);
                assert_eq!(tag, Self::tag().to_string());
                (
                    Self,
                    offset
                )
            }
            fn tag() -> &'static str {
                stringify!(#name)
            }
        }
    }
    .into()
}
