pub mod examples;
pub mod exercises;
pub mod listings;

/// Exercise trait
pub trait Exercise: Send + Sync {
    fn name(&self) -> String;
    fn main(&self);
}

pub trait Example: Send + Sync {
    fn description(&self) -> String;
    fn page_source(&self) -> usize;
    fn main(&self);
}
