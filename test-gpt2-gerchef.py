from transformers import pipeline


chef = pipeline('text-generation',
                model='./gpt2-gerchef',
                tokenizer='anonymous-german-nlp/german-gpt2',
                )

result = chef('Zuerst Knoblauch')[0]['generated_text']
print(result)