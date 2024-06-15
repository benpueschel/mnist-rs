extern crate proc_macro;

mod util;
mod serialize_enum;
mod serialize_struct;
mod serialize_union;

use serialize_enum::derive_enum_serialization;
use serialize_struct::derive_struct_serialization;
use serialize_union::derive_union_serialization;

use proc_macro::TokenStream;
use syn::{parse_macro_input, Data, DeriveInput};

#[proc_macro_derive(Serialize)]
pub fn derive_deserialize_layer(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let generics = input.generics;

    let expanded = match input.data {
        Data::Struct(data) => derive_struct_serialization(data, name, generics),
        Data::Enum(data) => derive_enum_serialization(data, name, generics),
        Data::Union(data) => derive_union_serialization(data, name, generics)
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

