use candle_core::{Device, Result, Tensor};
use candle_datasets::batcher::IterResult2;
use candle_datasets::Batcher;
use rand::{rng, seq::SliceRandom};
use regex::{Captures, Regex};
use std::collections::HashMap;
use std::rc::Rc;
use tiktoken_rs::CoreBPE;

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
        let eof = "<|endoftext|>";
        let insert_unknow = !vocab.contains_key(unknow);
        Self {
            str2int: {
                let mut key2id: HashMap<String, i32> =
                    vocab.iter().map(|(k, v)| (k.to_string(), *v)).collect();
                if insert_unknow {
                    key2id.insert(unknow.to_string(), vocab_size as i32 + 1);
                }
                if !vocab.contains_key(eof) {
                    key2id.insert(eof.to_string(), vocab_size as i32 + 2);
                }
                key2id
            },
            int2str: {
                let mut id2key: HashMap<i32, String> =
                    vocab.iter().map(|(k, v)| (*v, k.to_string())).collect();
                if insert_unknow {
                    id2key.insert(vocab_size as i32 + 1, unknow.to_string());
                }
                if !vocab.contains_key(eof) {
                    id2key.insert(vocab_size as i32 + 2, eof.to_string());
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

/// Listing 2.5 A dataset for batched inputs and targets
pub struct GPTDatasetV1_ {
    input_ids: Vec<Vec<u32>>,
    target_ids: Vec<Vec<u32>>,
}

/// GPTDatasetV1
/// These are refcounted so cloning is chea
#[derive(Clone)]
pub struct GPTDatasetV1(Rc<GPTDatasetV1_>);

impl AsRef<GPTDatasetV1> for GPTDatasetV1 {
    fn as_ref(&self) -> &GPTDatasetV1 {
        self
    }
}

impl std::ops::Deref for GPTDatasetV1 {
    type Target = GPTDatasetV1_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl GPTDatasetV1 {
    pub fn new(input: &str, tokenizer: CoreBPE, max_length: usize, stride: usize) -> Self {
        let token_ids = tokenizer.encode_with_special_tokens(input);

        // println!("token_id:{:?}", token_ids);
        let mut input_ids: Vec<Vec<u32>> = Vec::default();
        let mut target_ids: Vec<Vec<u32>> = Vec::default();

        // get input_ids and target_ids;
        for i in (0..token_ids.len() - max_length).step_by(stride) {
            let input_chunk = &token_ids[i..(i + max_length)];
            let target_chunk = &token_ids[(i + 1_usize)..(i + max_length + 1_usize)];

            input_ids.push(input_chunk.to_vec());
            target_ids.push(target_chunk.to_vec());
        }

        let dataset_ = GPTDatasetV1_ {
            input_ids,
            target_ids,
        };

        Self(Rc::new(dataset_))
    }

    pub fn input_ids(&self) -> &Vec<Vec<u32>> {
        &self.input_ids
    }

    pub fn target_ids(&self) -> &Vec<Vec<u32>> {
        &self.target_ids
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input_ids.len() == 0
    }

    pub fn get_pair_at_index(&self, idx: usize) -> (&Vec<u32>, &Vec<u32>) {
        (&self.input_ids[idx], &self.target_ids[idx])
    }
}

/// Listing 2.6 A data loader to generate batches with input-target pairs
/// We can use `GPTDatasetIter` with `candle_dataset::Batcher` to get desired
/// batches of examples.
pub struct GPTDatasetIter {
    dataset: GPTDatasetV1,
    remaining_indices: Vec<usize>,
}

impl GPTDatasetIter {
    pub fn new(dataset: GPTDatasetV1, shuffle: bool) -> Self {
        // use rev() so that we use Vec::pop() to keep the asc order
        let mut remaining_indices = (0..dataset.len()).rev().collect::<Vec<_>>();
        if shuffle {
            remaining_indices.shuffle(&mut rng());
        }

        Self {
            dataset,
            remaining_indices,
        }
    }
}

impl Iterator for GPTDatasetIter {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        // the remaining indices is reverse order,
        // we use pop here to keep the order asc
        if let Some(idx) = self.remaining_indices.pop() {
            let (input_ids, target_ids) = self.dataset.get_pair_at_index(idx);

            // turn into Tensors and return
            let dev = Device::cuda_if_available(0).unwrap();
            let input_tensor = Tensor::new(&input_ids[..], &dev);
            let target_tensor = Tensor::new(&target_ids[..], &dev);
            Some(candle_core::error::zip(input_tensor, target_tensor))
        } else {
            None
        }
    }
}

pub fn create_dataloader_v1(
    txt: &str,
    batch_size: usize,
    max_length: usize,
    stride: usize,
    shuffle: bool,
    drop_last: bool,
) -> (GPTDatasetV1, Batcher<IterResult2<GPTDatasetIter>>) {
    let tokenizer = tiktoken_rs::get_bpe_from_model("gpt2").unwrap();
    let dataset = GPTDatasetV1::new(txt, tokenizer, max_length, stride);
    let iter = GPTDatasetIter::new(dataset.clone(), shuffle);
    let batch_iter = Batcher::new_r2(iter)
        .batch_size(batch_size)
        .return_last_incomplete_batch(drop_last);
    (dataset, batch_iter)
}

#[cfg(test)]
mod tests {
    use core::panic;

    use super::*;
    use candle_datasets::Batcher;
    use rstest::*;
    use tiktoken_rs::get_bpe_from_model;

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

    #[fixture]
    pub fn txt_tokenizer() -> (String, CoreBPE) {
        let txt = "In the heart of the city";
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        (txt.to_string(), tokenizer)
    }

    #[fixture]
    pub fn gpt_dataset(#[from(txt_tokenizer)] (txt, tokenizer): (String, CoreBPE)) -> GPTDatasetV1 {
        let stride = 1_usize;
        let max_length = 2_usize;
        GPTDatasetV1::new(&txt[..], tokenizer, max_length, stride)
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
        let token_ids = tokenizer.encode("this is a test! <|endoftext|>");

        assert_eq!(1, token_ids[0]);
        assert_eq!(2, token_ids[1]);
        assert_eq!(3, token_ids[2]);
        assert_eq!(4, token_ids[3]);
        assert_eq!(5, token_ids[4]);
        assert_eq!(6, token_ids[5]);
    }

    #[rstest]
    fn test_simple_tokenizer_v2_decode(vocab: HashMap<&str, i32>) {
        let tokenizer = SimpleTokenizerV2::from_vocab(vocab);

        let token_ids = vec![1, 2, 3, 4, 5, 6];
        let text = tokenizer.decode(token_ids);

        assert_eq!("this is a test <|unk|> <|endoftext|>", text);
    }

    #[rstest]
    fn test_gpt_dataset_v1_init(#[from(txt_tokenizer)] (txt, tokenizer): (String, CoreBPE)) {
        let token_ids = tokenizer.encode_with_special_tokens(&txt[..]);
        let stride = 1_usize;
        let max_length = 2_usize;
        let dataset = GPTDatasetV1::new(&txt[..], tokenizer, max_length, stride);

        for mx in 1..max_length {
            // test target alignments
            assert_eq!(dataset.input_ids[0][mx], dataset.target_ids[0][mx - 1]);
        }

        for ix in 1..dataset.input_ids.len() {
            // test max length per input
            assert_eq!(dataset.input_ids[ix].len(), max_length);
            // test stride alignments
            assert_eq!(dataset.input_ids[ix][0], token_ids[ix * stride]);
        }
    }

    #[rstest]
    fn test_gpt_dataset_v1_iter(#[from(txt_tokenizer)] (txt, tokenizer): (String, CoreBPE)) {
        let stride = 1_usize;
        let max_length = 3_usize;
        let dataset = GPTDatasetV1::new(&txt[..], tokenizer, max_length, stride);
        let mut iter = GPTDatasetIter::new(dataset.clone(), false);
        let mut count = 0_usize;

        while let Some(Ok((this_inputs, this_targets))) = iter.next() {
            let this_inputs_vec: Vec<u32> = this_inputs.to_vec1::<u32>().unwrap();
            let this_target_vec: Vec<u32> = this_targets.to_vec1::<u32>().unwrap();

            assert_eq!(this_inputs.shape().dims()[0], max_length);
            assert_eq!(this_targets.shape().dims()[0], max_length);

            for (idx, token_id) in this_inputs_vec.iter().enumerate() {
                assert_eq!(*token_id, dataset.input_ids[count][idx]);
            }
            for (idx, token_id) in this_target_vec.iter().enumerate() {
                assert_eq!(*token_id, dataset.target_ids[count][idx]);
            }

            count += 1;
        }

        assert_eq!(dataset.len(), count);
    }

    #[rstest]
    fn test_gpt_dataset_with_batch(#[from(gpt_dataset)] dataset: GPTDatasetV1) {
        let iter = GPTDatasetIter::new(dataset.clone(), false);
        let batch_size = 2_usize;
        let mut batch_iter = Batcher::new_r2(iter).batch_size(batch_size);

        match batch_iter.next() {
            Some(Ok((inputs, targets))) => {
                println!("inputs: {:?}\n\ntargets: {:?}", inputs, targets);
                assert_eq!(inputs.dims(), targets.dims());
                assert_eq!(inputs.dims()[0], batch_size);
            }
            Some(Err(err)) => panic!("{}", err),
            None => panic!("None"),
        }
    }

    #[rstest]
    fn test_create_dataloader_v1() {
        let txt = "In the heart of the city";
        let batch_size = 2_usize;
        let stride = 1_usize;
        let max_length = 3_usize;
        let shuffle = false;
        let drop_last = false;
        let (_dataset, mut batch_iter) =
            create_dataloader_v1(txt, batch_size, max_length, stride, shuffle, drop_last);

        match batch_iter.next() {
            Some(Ok((inputs, targets))) => {
                assert_eq!(inputs.dims(), targets.dims());
                assert_eq!(inputs.dims()[0], batch_size);
            }
            Some(Err(err)) => panic!("{}", err),
            None => panic!("None"),
        }
    }
}
