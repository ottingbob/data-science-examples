import math
from functools import partial

import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
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
