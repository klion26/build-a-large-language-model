pub mod exercises;
pub mod listings;

/// Exercise trait
pub trait Exercise {
    fn name(&self) -> String;
    fn main(&self);
}
