# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="microsoft/phi-2", trust_remote_code=True, device='cuda', max_new_tokens=40)

generated_prompts = []
query = """Give me a sample instruction such as Suppress posts about a topic <X>. Where X is a topic in the news."""
# for _ in range(50):
generated_prompts.append(pipe(query))

print(generated_prompts[0])
