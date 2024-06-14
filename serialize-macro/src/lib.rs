extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DataEnum, DataStruct, DeriveInput, Fields, Generics, Ident};

#[proc_macro_derive(Serialize)]
pub fn derive_deserialize_layer(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let generics = input.generics;

    let expanded = match input.data {
        Data::Struct(data) => derive_struct_serialization(data, name, generics),
        Data::Enum(data) => derive_enum_serialization(data, name, generics),
        _ => panic!("#[derive(Serialize)] can only be used with structs"),
    };

        /*
        // Register this implementation in the global registry
        inventory::submit! {
            LayerEntry {
                tag: #name::tag(),
                deserialize_fn: #name::deserialize,
            }
        }
        */
    
    expanded
}

fn derive_enum_serialization(data: DataEnum, name: Ident, generics: Generics) -> TokenStream {
    let serialize_code = data.variants.iter().map(|v| {
        let variant_name = &v.ident;
        let (serialize_code, _) = derive_field_serialization(&v.fields);

        let enum_pattern = if v.fields.is_empty() {
            quote! { #name::#variant_name }
        } else {
            let fields = v.fields.iter().enumerate().map(|(i, f)| {
                let name = if let Some(name) = &f.ident {
                    quote! { #name }
                } else {
                     quote! { format!("f_{}", #i) }
                };
                quote! {
                    #name
                }
            });
            quote! { #name::#variant_name(#(#fields)*) }
        };

        quote! {
            #enum_pattern => {
                let tag = stringify!(::#variant_name).as_bytes();
                data.extend((tag.len() as u64).to_be_bytes());
                data.extend(tag);
                #(#serialize_code)*
            }
        }
    });
    let deserialize_code = data.variants.iter().map(|v| {
        let (_, deserialize_code) = derive_field_serialization(&v.fields);
        let variant_name = &v.ident;

        let enum_invocation = if v.fields.is_empty() {
            quote! { #name::#variant_name }
        } else {
            quote! { #name::#variant_name(#(#deserialize_code)*) }
        };

        quote! {
            stringify!(#name::#variant_name) => {
                #enum_invocation
            }
        }
    });

    quote! {
        impl #generics Serialized for #name #generics {
            fn serialize_binary(&self) -> Vec<u8> {
                let mut data = Vec::new();
                let tag = Self::tag().as_bytes();
                data.extend((tag.len() as u64).to_be_bytes());
                data.extend(tag);
                match self {
                    #(#serialize_code)*
                }
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
                
                // tag format: len(u64) -> tag data ([u8])
                let variant_name = {
                    let len = u64::from_be_bytes([
                        data[offset + 0], data[offset + 1], data[offset + 2], data[offset + 3], 
                        data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]
                    ]) as usize;
                    String::from_utf8(data[offset + 8..offset + 8 + len].to_vec()).unwrap()
                };
                offset += 8 + variant_name.len();
                (
                    match format!("{}{}",tag.to_string(), variant_name).as_str() {
                        #(#deserialize_code)*
                        x => panic!("invalid enum variant {}", x)
                    }, offset
                )
            }
            fn tag() -> &'static str {
                stringify!(#name)
            }
        }
    }.into()
}

fn derive_struct_serialization(data: DataStruct, name: Ident, generics: Generics) -> TokenStream {
    let (serialize_code, deserialize_code) = derive_field_serialization(&data.fields);

    // Generate the implementation for the `Serialized` trait
    quote! {
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

    }.into()
}

fn derive_field_serialization(fields: &Fields) -> (Vec<proc_macro2::TokenStream>, Vec<proc_macro2::TokenStream>) {
    match fields {
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
    }
}