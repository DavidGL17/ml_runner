mod layer;

#[cfg(not(feature = "simd"))]
mod scalar;

#[cfg(feature = "simd")]
mod simd;


pub use layer::Conv2DLayer;
