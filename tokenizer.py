from transformers import AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("anonymous-german-nlp/german-gpt2")

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128
    )

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    return train_dataset, test_dataset, data_collator


train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)


model = AutoModelWithLMHead.from_pretrained("anonymous-german-nlp/german-gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-gerchef",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

trainer.save_model()