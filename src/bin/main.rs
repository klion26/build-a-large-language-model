use build_a_large_language_model::examples::ch02::EG01;
use build_a_large_language_model::exercises::ch02::{X2P1, X2P2};
use build_a_large_language_model::{Example, Exercise};
use std::collections::HashMap;
use std::sync::LazyLock;

static EXERCISE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Exercise>>> =
    LazyLock::new(|| {
        let mut m: HashMap<&'static str, Box<dyn Exercise + 'static>> = HashMap::new();
        m.insert("2.1", Box::new(X2P1 {}));
        m.insert("2.2", Box::new(X2P2 {}));
        m
    });

static EXAMPLE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Example>>> = LazyLock::new(|| {
    let mut m: HashMap<&'static str, Box<dyn Example + 'static>> = HashMap::new();
    m.insert("02.01", Box::new(EG01 {}));
    m
});
fn main() {
    let exercise_registry = &*EXERCISE_REGISTRY;
    let example_registry = &*EXAMPLE_REGISTRY;
    let ex = exercise_registry.get("2.2").unwrap();
    ex.main();

    let eg = example_registry.get("02.01").unwrap();
    eg.main();
}
