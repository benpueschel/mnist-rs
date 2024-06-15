use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, ToTokens, TokenStreamExt};
use syn::{DataEnum, Field, Fields, Generics, Ident, Path, Variant};

pub fn derive_enum_serialization(
    data: DataEnum,
    name: Ident,
    generics: Generics,
) -> proc_macro::TokenStream {
    let serialize_code = data
        .variants
        .iter()
        .map(|v| derive_variant_serialize_code(&name, v));
    let deserialize_code = data
        .variants
        .iter()
        .map(|v| derive_variant_deserialize_code(&name, v));

    quote! {
        impl #generics Serialized for #name #generics {
            fn serialize_binary(&self) -> Vec<u8> {
                let mut data = Vec::new();
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
                let mut offset = tag.len() + 8;
                (
                    match tag.as_str() {
                        #(#deserialize_code)*
                        x => panic!("invalid enum variant {}", x)
                    },
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

fn variant_name(name: &Ident, variant: &Variant) -> TokenStream {
    let variant_ident = &variant.ident;
    quote! {#name::#variant_ident }
}

fn variant_name_str(name: &Ident, variant: &Variant) -> String {
    format!("{}::{}", name, variant.ident)
}

fn derive_variant_serialize_code(name: &Ident, variant: &Variant) -> TokenStream {
    let (serialize_code, _) = derive_field_serialization(&variant.fields);
    let tag = variant_name(name, variant);

    let enum_pattern = if variant.fields.is_empty() {
        quote! { #tag }
    } else {
        let fields = variant.fields.iter().enumerate().map(get_field_name);
        if is_tuple_variant(variant) {
            quote! { #tag(#(#fields,)*) }
        } else {
            quote! { #tag { #(#fields),* } }
        }
    };

    let tag_str = variant_name_str(name, variant);
    quote! {
        #enum_pattern => {
            let tag = #tag_str.as_bytes();
            data.extend((tag.len() as u64).to_be_bytes());
            data.extend(tag);
            #(#serialize_code)*
        }
    }
}

fn derive_variant_deserialize_code(name: &Ident, variant: &Variant) -> TokenStream {
    let (_, deserialize_code) = derive_field_serialization(&variant.fields);
    let tag = variant_name(name, variant);
    let tag_str = variant_name_str(name, variant);

    let enum_invocation = if variant.fields.is_empty() {
        quote! { #tag }
    } else if is_tuple_variant(variant) {
        quote! { #tag(#(#deserialize_code),*) }
    } else {
        quote! { #tag{ #(#deserialize_code),* } }
    };

    quote! {
        #tag_str => {
            #enum_invocation
        }
    }
}

fn is_tuple_variant(variant: &Variant) -> bool {
    matches!(&variant.fields, Fields::Unnamed(_))
}

fn get_field_name(args: (usize, &Field)) -> TokenStream {
    let (i, field) = args;
    if let Some(name) = &field.ident {
        quote! { #name }
    } else {
        let varname = format_ident!("f_{}", i);
        quote! { #varname }
    }
}

type FieldSerialization = (Vec<TokenStream>, Vec<TokenStream>);
fn derive_field_serialization(fields: &Fields) -> FieldSerialization {
    match fields {
        Fields::Named(fields) => {
            let serialize_fields = fields.named.iter().map(|f| {
                let field_name = &f.ident;
                quote! {
                    let mut serialized = #field_name.serialize_binary();
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

            (
                serialize_fields.collect::<Vec<_>>(),
                deserialize_fields.collect::<Vec<_>>(),
            )
        }
        Fields::Unnamed(fields) => {
            let serialize_fields = fields.unnamed.iter().enumerate().map(|(i, f)| {
                let name = get_field_name((i, f));
                quote! {
                    let mut serialized = #name.serialize_binary();
                    data.extend_from_slice(&serialized);
                }
            });

            let deserialize_fields = fields.unnamed.iter().map(|f| {
                let field_type = &f.ty;
                quote! {
                    {
                        let (field, read_bytes) = #field_type::deserialize_binary(&data[offset..]);
                        offset += read_bytes;
                        field
                    }
                }
            });

            (
                serialize_fields.collect::<Vec<_>>(),
                deserialize_fields.collect::<Vec<_>>(),
            )
        }
        Fields::Unit => (vec![], vec![]),
    }
}