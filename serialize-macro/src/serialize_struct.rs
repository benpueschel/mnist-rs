use proc_macro2::TokenStream;
use quote::quote;
use syn::{DataStruct, Fields, Generics, Ident};

pub fn derive_struct_serialization(
    data: DataStruct,
    name: Ident,
    generics: Generics,
) -> proc_macro::TokenStream {
    let serialize_code = derive_serialize_field(&data.fields);
    let deserialize_code = derive_deserialize_field(&data.fields);

    let invocation = if matches!(data.fields, Fields::Unnamed(_)) {
        quote! {
            Self(#(#deserialize_code),*)
        }
    } else {
        quote! {
            Self { #(#deserialize_code),* }
        }
    };

    // Generate the implementation for the `Serialized` trait
    quote! {
        impl #generics Serialized for #name #generics {
            fn serialize_binary(&self) -> Vec<u8> {
                let mut data = Vec::new();
                data.extend(String::from(Self::tag()).serialize_binary());
                #(#serialize_code)*
                data
            }

            fn deserialize_binary(data: &[u8]) -> (Self, usize) {
                // tag format: len(u64) -> tag data ([u8])
                let (tag, mut offset) = String::deserialize_binary(&data[0..]);
                assert_eq!(tag, Self::tag().to_string());
                (
                    #invocation,
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

fn derive_serialize_field(fields: &Fields) -> Vec<TokenStream> {
    match fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .map(|f| {
                let field_name = &f.ident;
                quote! {
                    let serialized = self.#field_name.serialize_binary();
                    data.extend_from_slice(&serialized);
                }
            })
            .collect::<Vec<_>>(),
        Fields::Unnamed(fields) => fields
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let index = syn::Index::from(i);
                quote! {
                    let serialized = self.#index.serialize_binary();
                    data.extend_from_slice(&serialized);
                }
            })
            .collect::<Vec<_>>(),
        Fields::Unit => vec![],
    }
}

fn derive_deserialize_field(fields: &Fields) -> Vec<TokenStream> {
    match fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .map(|f| {
                let field_name = &f.ident;
                let field_type = &f.ty;
                quote! {
                    #field_name: {
                        let (field, read_bytes) = #field_type::deserialize_binary(&data[offset..]);
                        offset += read_bytes;
                        field
                    }
                }
            })
            .collect::<Vec<_>>(),
        Fields::Unnamed(fields) => fields
            .unnamed
            .iter()
            .map(|f| {
                let field_type = &f.ty;
                quote! {
                    {
                        let (field, read_bytes) = #field_type::deserialize_binary(&data[offset..]);
                        offset += read_bytes;
                        field
                    }
                }
            })
            .collect::<Vec<_>>(),
        Fields::Unit => vec![],
    }
}
