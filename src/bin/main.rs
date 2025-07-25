use build_a_large_language_model::examples::{ch02, ch03, ch04, ch05};
use build_a_large_language_model::exercises::ch02::{X2P1, X2P2};
use build_a_large_language_model::{exercises, Example, Exercise};
use std::collections::HashMap;
use std::sync::LazyLock;

static EXERCISE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Exercise>>> =
    LazyLock::new(|| {
        let mut m: HashMap<&'static str, Box<dyn Exercise + 'static>> = HashMap::new();
        // ch02
        m.insert("2.1", Box::new(X2P1));
        m.insert("2.2", Box::new(X2P2));
        // ch03
        m.insert("3.1", Box::new(exercises::ch03::X3P1));
        m.insert("3.2", Box::new(exercises::ch03::X3P2));
        m.insert("3.3", Box::new(exercises::ch03::X3P3));

        // ch04
        m.insert("4.1", Box::new(exercises::ch04::X4P1));
        m.insert("4.2", Box::new(exercises::ch04::X4P2));
        m.insert("4.3", Box::new(exercises::ch04::X4P3));

        m
    });

static EXAMPLE_REGISTRY: LazyLock<HashMap<&'static str, Box<dyn Example>>> = LazyLock::new(|| {
    let mut m: HashMap<&'static str, Box<dyn Example + 'static>> = HashMap::new();
    // ch02
    m.insert("02.01", Box::new(ch02::EG01));
    m.insert("02.02", Box::new(ch02::EG02));
    // ch03
    m.insert("03.01", Box::new(ch03::EG01));
    m.insert("03.02", Box::new(ch03::EG02));
    m.insert("03.03", Box::new(ch03::EG03));
    m.insert("03.04", Box::new(ch03::EG04));
    m.insert("03.05", Box::new(ch03::EG05));
    m.insert("03.06", Box::new(ch03::EG06));
    m.insert("03.07", Box::new(ch03::EG07));
    m.insert("03.08", Box::new(ch03::EG08));
    m.insert("03.09", Box::new(ch03::EG09));
    m.insert("03.10", Box::new(ch03::EG10));
    m.insert("03.11", Box::new(ch03::EG11));

    // ch04
    m.insert("04.01", Box::new(ch04::EG01));
    m.insert("04.02", Box::new(ch04::EG02));
    m.insert("04.03", Box::new(ch04::EG03));
    m.insert("04.05", Box::new(ch04::EG05));
    m.insert("04.06", Box::new(ch04::EG06));
    m.insert("04.07", Box::new(ch04::EG07));
    m.insert("04.08", Box::new(ch04::EG08));

    // ch05
    m.insert("05.01", Box::new(ch05::EG01));
    m.insert("05.02", Box::new(ch05::EG02));
    m.insert("05.03", Box::new(ch05::EG03));
    m
});

#[allow(dead_code)]
enum RunType {
    EG(String),
    EX(String),
}

fn main() {
    let exercise_registry = &*EXERCISE_REGISTRY;
    let example_registry = &*EXAMPLE_REGISTRY;

    // let run_type = RunType::EX(String::from("4.3"));
    let run_type = RunType::EG(String::from("05.03"));
    match run_type {
        RunType::EX(id) => {
            let ex = exercise_registry.get(id.as_str()).unwrap();
            ex.main();
        }
        RunType::EG(id) => {
            let eg = example_registry.get(id.as_str()).unwrap();
            eg.main();
        }
    }
}
