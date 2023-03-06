import collections
import math
from functools import partial

import evaluate
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    GPT2LMHeadModel,
    default_data_collator,
    get_scheduler,
)


# Fine-tune DistilBERT with Accelerate
class AcceleratedDistilBERT:
    def __init__(
        self, dataset, data_collator, model_checkpoint: str, model_name: str, tokenizer
    ):
        self.downsampled_dataset = dataset
        self.data_collator = data_collator
        self.model_checkpoint = model_checkpoint
        self.model_name = model_name
        self.tokenizer = tokenizer

    # To remove randomness during our perplexity scores on training we
    # can apply the masking once on the whole test set and use the default
    # data collator to collect the batches during evaluation
    def _insert_random_mask(self, data_collator, batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    # Now apply the function to our test set and drop the unmasked columns
    # so we can replace them with the masked ones.
    def mask_eval_dataset(self):
        # downsampled_dataset = self.downsampled_dataset.remove_columns(["word_ids"])
        insert_random_mask = partial(self._insert_random_mask, self.data_collator)
        eval_dataset = self.downsampled_dataset["test"].map(
            insert_random_mask,
            batched=True,
            remove_columns=self.downsampled_dataset["test"].column_names,
        )
        return eval_dataset.rename_columns(
            {
                "masked_input_ids": "input_ids",
                "masked_attention_mask": "attention_mask",
                "masked_labels": "labels",
            }
        )

    def execute(self):
        self.downsampled_dataset = self.downsampled_dataset.remove_columns(["word_ids"])
        # batch_size = 64
        # Might even be able to get away with a bigger batch size here.
        # Looks like I am not even hitting 3GB usage on my GPU
        """
        $ nvidia-smi
        +-----------------------------------------------------------------------------+
        |                               |   Memory / Usage     |                      |
        |=============================================================================|
        | N/A   82C    P3    27W /  30W |   2715MiB /  4096MiB |     99%      Default |
        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |    0   N/A  N/A      2462      G   /usr/lib/xorg/Xorg                  4MiB |
        |    0   N/A  N/A    196420      C   ...ience-examples/bin/python     2708MiB |
        +-----------------------------------------------------------------------------+
        """
        batch_size = 8
        train_dataloader = DataLoader(
            self.downsampled_dataset["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=self.data_collator,
        )
        eval_dataset = self.mask_eval_dataset()
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator,
        )
        # Load a fresh pretrained version of the model
        model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint)
        # Specify the optimizer
        optimizer = AdamW(model.parameters(), lr=5e-5)
        # Prepare everything for training
        a = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = a.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_train_epochs = 3
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        output_dir = self.model_name

        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(num_training_steps):
            # Training
            model.train()
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                a.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                # Expands the loss to the given batch size over the batched
                # results from the accelerator
                losses.append(a.gather(loss.repeat(batch_size)))

            losses = torch.cat(losses)
            losses = losses[: len(eval_dataset)]
            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")

            print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

            # Save dataset
            a.wait_for_everyone()
            unwrapped_model = a.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=a.save)
            if a.is_main_process:
                self.tokenizer.save_pretrained(output_dir)
                # Optionally push to hub...


class AcceleratedMarian:
    def __init__(
        self,
        dataset,
        data_collator,
        tokenizer,
        model_checkpoint: str = "Helsinki-NLP/opus-mt-en-fr",
        model_name: str = "marian-finetuned-kde4-en-to-fr-accelerate",
    ):
        self.dataset = dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.model_checkpoint = model_checkpoint
        self.model_name = model_name

    # Takes in predictions and labels and coverts them to the lists of
    # strings our metric object will expect.
    def _postprocess(self, predictions, labels):
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )

        # Replace `-100` in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        return decoded_preds, decoded_labels

    def execute(self):
        # Oh boy this is NOT going to be fast...
        batch_size = 4
        train_dataloader = DataLoader(
            self.dataset["train"],
            shuffle=True,
            batch_size=batch_size,
            collate_fn=self.data_collator,
        )
        eval_dataloader = DataLoader(
            self.dataset["validation"],
            batch_size=batch_size,
            collate_fn=self.data_collator,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_train_epochs = 3
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        output_dir = self.model_name
        metric = evaluate.load("sacrebleu")

        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(num_training_steps):
            # Training
            model.train()
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            model.eval()
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=128,
                    )
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                predictions_gathered = accelerator.gather(generated_tokens)
                labels_gathered = accelerator.gather(labels)

                decoded_preds, decoded_labels = self._postprocess(
                    predictions_gathered, labels_gathered
                )
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            results = metric.compute()
            print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(output_dir)
                # Optionally push to hub...


# In this example we need to explicitly generate our summaries during training
# and define how we compute the ROUGE scores
class AcceleratedMT5:
    def __init__(self, datasets, data_collator, model_checkpoint: str, tokenizer):
        self.datasets = datasets
        self.datasets.set_format("torch")
        self.data_collator = data_collator
        self.model_checkpoint = model_checkpoint
        self.tokenizer = tokenizer

    def _postprocess(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # ROUGE expects a newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def execute(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)
        batch_size = 8
        train_dataloader = DataLoader(
            self.datasets["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=batch_size,
        )
        eval_dataloader = DataLoader(
            self.datasets["validation"],
            collate_fn=self.data_collator,
            batch_size=batch_size,
        )

        rouge_score = evaluate.load("rouge")
        optimizer = AdamW(model.parameters(), lr=2e-5)
        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_train_epochs = 10
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        output_dir = "results-mt5-finetuned-amazon-en-es-accelerate"
        # Training loop:
        # 1) train model by iterating over all examples in train dataloader
        # 2) Generate model summaries at end of each epoch by generating the tokens
        #   and then decoding them into text
        # 3) Compute the ROUGE scores
        # 4) Save the checkpoints
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            model.eval()
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                    )
                    generated_tokens = (
                        accelerator.gather(generated_tokens).cpu().numpy()
                    )
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]

                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(
                        batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id
                    )
                    labels = accelerator.gather(labels).cpu().numpy()
                    # Replace -100 in the labels since we can't decode them
                    labels = np.where(
                        labels != -100, labels, self.tokenizer.pad_token_id
                    )

                    decoded_preds = self.tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )
                    decoded_labels = self.tokenizer.batch_decode(
                        labels, skip_special_tokens=True
                    )
                    decoded_preds, decoded_labels = self._postprocess(
                        decoded_preds, decoded_labels
                    )

                    rouge_score.add_batch(
                        predications=decoded_preds,
                        references=decoded_labels,
                    )

            # Compute metrics
            result = rouge_score.compute()
            # Extract the score
            result = {key: round(value * 100, 4) for key, value in result.items()}
            print(f"Epoch {epoch}: {result}")

            # Save progress
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(output_dir)
                # Optionally push to hub...


# We are interested in autocompletion for data science libraries so we should give
# more weight to training samples that use more of those libraries.
# We find identify with keywords such as `plt`, `pd`, `sk`, `fit`, `predict` which
# are all common import names from the respective libraries
class AcceleratedGPT2:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    @property
    def keytoken_ids(self):
        keytoken_ids = []
        keywords = ["plt", "pd", "sk", "fit", "predict"]
        keywords.extend([f" {k}" for k in keywords])
        # This token should be split into multiple tokens
        keywords.append("testtest")
        for kw in keywords:
            ids = self.tokenizer([kw]).input_ids[0]
            if len(ids) == 1:
                keytoken_ids.append(ids[0])
            else:
                print(f"Keyword has more than one token: {kw}")
        return keytoken_ids

    # We define a loss function that aligns the logits and inputs.
    # The inputs are shifted by one to the right from the labels since
    # the next token is the label for the current token.
    # We cutoff the last logit since we don't have a label for the token
    # that follows the full input sequence.
    # Now we compute loss per sample and count the occurrences of all keywords
    # in each sample. Finally calculate the weighted average over all samples
    # using the occurrences as weights
    # To not throw away samples that have no keywords we add 1 to the weights
    def keytoken_weighted_loss(self, inputs, logits, keytoken_ids, alpha=1.0):
        # Shift so that tokens < n predict n
        shift_labels = inputs[..., 1:].contiguous()
        shift_logits = logits[..., 1:].contiguous()
        # Calculate per-token loss
        loss_fct = CrossEntropyLoss(reduce=False)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        # Resize and average loss per sample
        loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(
            axis=1
        )
        # Calculate and scale weighting
        weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
            axis=[0, 2]
        )
        weights = alpha * (1.0 + weights)
        # Calculate weighted average
        weighted_loss = (loss_per_sample * weights).mean()
        return weighted_loss

    # To use the loss function we need to prepare the following:
    # 1) Dataloaders load the data in batches
    # 2) Set up weight decay params
    # 3) Evaluate with a function
    def get_grouped_params(self, model, no_decay=["bias", "LayerNorm.weight"]):
        weight_decay = 0.1
        params_with_wd, params_without_wd = [], []
        for n, p in model.named_parameters():
            # Skip the weight decay on no_decay params
            if any(nd in n for nd in no_decay):
                params_without_wd.append(p)
            else:
                params_with_wd.append(p)
        return [
            {"params": params_with_wd, "weight_decay": weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    # With this function we can report perplexity and loss at
    # regular intervals
    def evaluate(self, model, eval_dataloader, accelerator):
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(batch["input_ids"], labels=batch["input_ids"])

            losses.append(accelerator.gather(outputs.loss))
        loss = torch.mean(torch.cat(losses))
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return loss.item(), perplexity.item()

    def execute(self, config):
        self.dataset.set_format("torch")
        batch_size = 32
        train_dataloader = DataLoader(
            self.dataset["train"], batch_size=batch_size, shuffle=True
        )
        eval_dataloader = DataLoader(self.dataset["valid"], batch_size=batch_size)

        model = GPT2LMHeadModel(config)
        optimizer = AdamW(self.get_grouped_params(model), lr=5e-4)
        # accelerator = Accelerator(fp16=True)
        accelerator = Accelerator()

        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_train_epochs = 1
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=1_000,
            num_training_steps=num_training_steps,
        )

        # Quick test to see if evaluation works properly
        self.evaluate(model, eval_dataloader, accelerator)

        output_dir = "codeparrot-ds-accelerate"
        gradient_accumulation_steps = 8
        eval_steps = 5_000
        model.train()
        completed_steps = 0
        for epoch in range(num_train_epochs):
            for step, batch in tqdm(
                enumerate(train_dataloader, start=1), total=num_training_steps
            ):
                logits = model(batch["input_ids"]).logits
                loss = self.keytoken_weighted_loss(
                    batch["input_ids"], logits, self.keytoken_ids
                )
                if step % 100 == 0:
                    accelerator.print(
                        {
                            "lr": lr_scheduler.get_lr(),
                            "samples": step * len(batch),
                            "steps": completed_steps,
                            "loss/train": loss.item() * gradient_accumulation_steps,
                        }
                    )
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)
                if step % gradient_accumulation_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1
                if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                    eval_loss, perplexity = self.evaluate(
                        model, eval_dataloader, accelerator
                    )
                    accelerator.print(
                        {"loss/eval": eval_loss, "perplexity": perplexity}
                    )
                    model.train()
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        output_dir, save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        self.tokenizer.save_pretrained(output_dir)
                        # Optionally push to hub...


class AcceleratedBertSquad:
    def __init__(
        self,
        dataset,
        train_dataset,
        raw_validation_dataset,
        tokenizer,
        model_checkpoint: str = "bert-base-cased",
    ):
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.raw_validation_dataset = raw_validation_dataset
        self.model_checkpoint = model_checkpoint
        self.tokenizer = tokenizer
        self.metric = evaluate.load("squad")

    def compute_metrics(self, start_logits, end_logits, features, examples):
        n_best = 20
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-n_best:].tolist()
                end_indexes = np.argsort(end_logit)[-n_best:].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        # if offsets[start_index] is None or offsets[end_index] is None:
                        if any(
                            e is None
                            for e in (offsets[start_index], offsets[end_index])
                        ):
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[
                                offsets[start_index][0] : offsets[end_index][1]
                            ],
                            "logit_score": start_logit[start_index]
                            + end_logit[end_index],
                        }
                        answers.append(answer)

            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in examples
        ]
        return self.metric.compute(
            predictions=predicted_answers, references=theoretical_answers
        )

    def execute(self):
        self.dataset.set_format("torch")
        try:
            validation_set = self.dataset["validation"].remove_columns(
                ["example_id", "offset_mapping"]
            )
        except ValueError:
            print("Columns already removed from validation dataset")
            validation_set = self.dataset["validation"]

        batch_size = 8
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )
        eval_dataloader = DataLoader(
            validation_set,
            batch_size=batch_size,
            collate_fn=default_data_collator,
        )

        model = AutoModelForQuestionAnswering.from_pretrained(self.model_checkpoint)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        # accelerator = Accelerator(fp16=True)
        accelerator = Accelerator()

        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_train_epochs = 3
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        output_dir = "bert-finetuned-squad-accelerate"
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(num_train_epochs):
            # Training
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            model.eval()
            start_logits = []
            end_logits = []
            accelerator.print("Evaluation!")
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                start_logits.append(
                    accelerator.gather(outputs.start_logits).cpu().numpy()
                )
                end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

            start_logits = np.concatenate(start_logits)[
                : len(self.dataset["validation"])
            ]
            end_logits = np.concatenate(end_logits)[: len(self.dataset["validation"])]
            metrics = self.compute_metrics(
                start_logits,
                end_logits,
                self.dataset["validation"],
                self.raw_validation_dataset,
            )
            print(f"Epoch {epoch}: {metrics}")

            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(output_dir)
                # Optionally push to hub...
