use regex::{Captures, Regex};
use std::collections::HashMap;

/// Listing 2.3
#[derive(Default, Debug)]
pub struct SimpleTokenizerV1 {
    str2int: HashMap<String, i32>,
    int2str: HashMap<i32, String>,
    unknow_id: i32,
}

impl SimpleTokenizerV1 {
    pub fn from_vocab(vocab: HashMap<&str, i32>) -> Self {
        Self {
            str2int: vocab.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            int2str: vocab.iter().map(|(k, v)| (*v, k.to_string())).collect(),
            unknow_id: vocab.len() as i32 + 10,
        }
    }

    pub fn str_to_int(&self) -> &HashMap<String, i32> {
        &self.str2int
    }

    pub fn int_to_str(&self) -> &HashMap<i32, String> {
        &self.int2str
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        let re = Regex::new(r#"([,.?_!"()']|--|\s)"#).unwrap();
        re.split(text)
            .map(|token| self.str2int.get(token).unwrap_or(&self.unknow_id))
            .cloned()
            .collect::<Vec<i32>>()
    }

    pub fn decode(&self, token_ids: Vec<i32>) -> String {
        let text_vec: Vec<String> = token_ids
            .iter()
            .map(|token_id| self.int2str.get(token_id).unwrap())
            .cloned()
            .collect::<Vec<String>>();
        let text = &text_vec.join(" ");

        // remove space before any punctuanttions
        let re = Regex::new(r#"\s+([,.?!"()\'])"#).unwrap();
        String::from(re.replace_all(text, |caps: &Captures| caps[1].to_string()))
    }
}

/// Listing 2.4
#[derive(Default, Debug)]
pub struct SimpleTokenizerV2 {
    str2int: HashMap<String, i32>,
    int2str: HashMap<i32, String>,
}

impl SimpleTokenizerV2 {
    pub fn from_vocab(vocab: HashMap<&str, i32>) -> Self {
        // add special tokens to vocab if needed
        let vocab_size = vocab.len();
        let unknow = "<|unk|>";
        let insert_unknow = !vocab.contains_key(unknow);
        Self {
            str2int: {
                let mut key2id: HashMap<String, i32> =
                    vocab.iter().map(|(k, v)| (k.to_string(), *v)).collect();
                if insert_unknow {
                    key2id.insert(unknow.to_string(), vocab_size as i32 + 1);
                }
                key2id
            },
            int2str: {
                let mut id2key: HashMap<i32, String> =
                    vocab.iter().map(|(k, v)| (*v, k.to_string())).collect();
                if insert_unknow {
                    id2key.insert(vocab_size as i32 + 1, unknow.to_string());
                }
                id2key
            },
        }
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        let re = Regex::new(r#"([,.?_!"()']|--|\s)"#).unwrap();
        re.split(text)
            .map(|token| {
                self.str2int
                    .get(token)
                    .unwrap_or(self.str2int.get("<|unk|>").unwrap())
            })
            .cloned()
            .collect::<Vec<i32>>()
    }

    pub fn decode(&self, token_ids: Vec<i32>) -> String {
        let text_vec: Vec<String> = token_ids
            .iter()
            .map(|token_id| self.int2str.get(token_id).unwrap())
            .cloned()
            .collect::<Vec<String>>();
        let text = &text_vec.join(" ");

        // remove space before any punctuanttions
        let re = Regex::new(r#"\s+([,.?!"()\'])"#).unwrap();
        String::from(re.replace_all(text, |caps: &Captures| caps[1].to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[fixture]
    pub fn vocab() -> HashMap<&'static str, i32> {
        // arrange
        let mut vocab: HashMap<&str, i32> = HashMap::new();
        vocab.entry("this").or_insert(1);
        vocab.entry("is").or_insert(2);
        vocab.entry("a").or_insert(3);
        vocab.entry("test").or_insert(4);

        vocab
    }

    #[rstest]
    fn test_simple_tokenizer_init(vocab: HashMap<&str, i32>) {
        // act
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);

        // assert
        assert_eq!(Some(&1), tokenizer.str_to_int().get("this"));
        assert_eq!(Some(&2), tokenizer.str_to_int().get("is"));
        assert_eq!(Some(&3), tokenizer.str_to_int().get("a"));
        assert_eq!(Some(&4), tokenizer.str_to_int().get("test"));
    }

    #[rstest]
    fn test_simple_tokenize_encode(vocab: HashMap<&str, i32>) {
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);

        let token_ids = tokenizer.encode("this is a test");
        assert_eq!(1, token_ids[0]);
        assert_eq!(2, token_ids[1]);
        assert_eq!(3, token_ids[2]);
        assert_eq!(4, token_ids[3]);
    }

    #[test]
    fn test_simple_tokenize_decode() {
        let mut vocab: HashMap<&str, i32> = HashMap::new();
        vocab.entry("this").or_insert(1);
        vocab.entry("is").or_insert(2);
        vocab.entry("a").or_insert(3);
        vocab.entry("test").or_insert(4);
        vocab.entry(".").or_insert(5);

        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);

        let token_ids = vec![1, 2, 3, 4, 5];
        let text = tokenizer.decode(token_ids);

        assert_eq!("this is a test.", text);
    }

    #[rstest]
    fn test_simple_tokenizer_v2_encode(vocab: HashMap<&str, i32>) {
        let tokenizer = SimpleTokenizerV2::from_vocab(vocab);
        let token_ids = tokenizer.encode("this is a test!");

        assert_eq!(1, token_ids[0]);
        assert_eq!(2, token_ids[1]);
        assert_eq!(3, token_ids[2]);
        assert_eq!(4, token_ids[3]);
        assert_eq!(5, token_ids[4]);
    }

    #[rstest]
    fn test_simple_tokenzier_v2_decode(vocab: HashMap<&str, i32>) {
        let tokenizer = SimpleTokenizerV2::from_vocab(vocab);

        let token_ids = vec![1, 2, 3, 4, 5];
        let text = tokenizer.decode(token_ids);

        assert_eq!("this is a test <|unk|>", text);
    }
}
