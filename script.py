from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", local_files_only=False) # use_auth_token="Your_HF_TOKEN"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=False)

print("Model and tokenizer loaded successfully!")
