import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer

issues_dataset = load_dataset("lewtun/github-issues", split="train")
print(issues_dataset)

# Filter out the pull requests and rows with no comments
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
print(issues_dataset)

# Drop the columns other than the ones we need for semantic search.
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
print(issues_dataset)

# Create the embeddings by augmenting each comment with the issue title and body
# Since the `comments` column is a list of comments we need to "explode" the column
# so each one becomes its own row with `html_url`, `title`, and `body` info
issues_dataset.set_format("pandas")
df = issues_dataset[:]

print(df["comments"][0].tolist())
comments_df = df.explode("comments", ignore_index=True)
print(comments_df.head(4))

comments_dataset = Dataset.from_pandas(comments_df)
print(comments_dataset)

# Create a `comments length` column to contain number of words per comment
comments_dataset = comments_dataset.map(
    lambda x: {"comments_length": len(x["comments"].split())}
)
# And filter out short comments (anything less than 15 words)
comments_dataset = comments_dataset.filter(lambda x: x["comments_length"] > 15)
print(comments_dataset)


# Now concatenate the issue title, description, and comments into a text column
def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_text)

# Now we create the embeddings with a tokenizer and a model
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)


# To represent each entry in our GitHub issues corpus as a single vector we need
# to `pool` or average our token embeddings. We perform CLS pooling on our models
# outputs where we collect the last hidden state for the `[CLS]` token
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


# Here is the helper function that will tokenize a list of documents, place the
# tensors on the GPU, feed them to the model, and finally apply CLS pooling to
# the outputs:
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


# Test the function by feeding it the first text entry and check output shape:
embedding = get_embeddings(comments_dataset["text"][0])
print(embedding.shape)

# We now add the embeddings and convert them to numpy arrays
"""
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
embeddings_dataset.save_to_disk("github-issues-text-embeddings")
"""

embeddings_dataset = load_from_disk("github-issues-text-embeddings")

# Use the FAISS (Facebook AI Similarity Search) Index to search over embeddings
embeddings_dataset.add_faiss_index(column="embeddings")
# question = "How can I load a dataset offline?"
question = "How can I load offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
print(question_embedding.shape)

# Find similar embeddings with the 5 best matches
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
samples_df: pd.DataFrame = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)
for _, row in samples_df.iterrows():
    print(f"Comment: {row.comments}")
    print(f"Score: {row.scores}")
    print(f"Title: {row.title}")
    print(f"Url: {row.html_url}")
    print("=" * 50)
    print("\n")
