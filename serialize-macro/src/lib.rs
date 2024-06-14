extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(Serialize)]
pub fn derive_deserialize_layer(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let generics = input.generics;

    let data = match input.data {
        Data::Struct(data) => data,
        _ => panic!("#[derive(Serialize)] can only be used with structs"),
    };

    let (serialize_code, deserialize_code) = match data.fields {
        Fields::Named(fields) => {
            let serialize_fields = fields.named.iter().map(|f| {
                let field_name = &f.ident;
                quote! {
                    let mut serialized = self.#field_name.serialize_binary();
                    data.extend_from_slice(&serialized);
                }
            });

            let deserialize_fields = fields.named.iter().map(|f| {
                let field_name = &f.ident;
                let field_type = &f.ty;
                quote! {
                    #field_name: {
                        let (field, read_bytes) = #field_type::deserialize_binary(&data[offset..]);
                        offset += read_bytes;
                        field
                    }
                }
            });

            (serialize_fields.collect::<Vec<_>>(), deserialize_fields.collect::<Vec<_>>())
        },
        Fields::Unnamed(fields) => {
            let serialize_fields = fields.unnamed.iter().enumerate().map(|(i, _)| {
                let index = syn::Index::from(i);
                quote! {
                    let mut serialized = self.#index.serialize_binary();
                    data.extend_from_slice(&serialized);
                }
            });

            let deserialize_fields = fields.unnamed.iter().enumerate().map(|_| {
                quote! {
                    {
                        let (field, read_bytes) = Serialized::deserialize_binary(&data[offset..]);
                        offset += read_bytes;
                        field
                    }
                }
            });

            (serialize_fields.collect::<Vec<_>>(), deserialize_fields.collect::<Vec<_>>())
        },
        Fields::Unit => (vec![], vec![]),
    };

    // Generate the implementation for the `Serialized` trait
    let expanded = quote! {
        impl #generics Serialized for #name #generics {
            fn serialize_binary(&self) -> Vec<u8> {
                let mut data = Vec::new();
                let tag = Self::tag().as_bytes();
                data.extend((tag.len() as u64).to_be_bytes());
                data.extend(tag);
                #(#serialize_code)*
                data
            }

            fn deserialize_binary(data: &[u8]) -> (Self, usize) {
                // tag format: len(u64) -> tag data ([u8])
                let tag = {
                    let len = u64::from_be_bytes([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]]) as usize;
                    String::from_utf8(data[8..8 + len].to_vec()).unwrap()
                };
                assert_eq!(tag, Self::tag().to_string());
                let mut offset = tag.len() + 8;
                (
                    Self {
                        #(#deserialize_code),*
                    },
                    offset
                )
            }
            fn tag() -> &'static str {
                stringify!(#name)
            }
        }

        /*
        // Register this implementation in the global registry
        inventory::submit! {
            LayerEntry {
                tag: #name::tag(),
                deserialize_fn: #name::deserialize,
            }
        }
        */
    };

    TokenStream::from(expanded)
}
