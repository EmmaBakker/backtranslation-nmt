from datasets import Dataset

def preprocess_data(dataset_dict, tokenizer, src_lang, tgt_lang, split, max_length=128):
    def preprocess_function(examples):
        inputs = examples[src_lang]
        targets = examples[tgt_lang]

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset_dict[split].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict[split].column_names
    )

    return tokenized_dataset

