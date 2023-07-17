# -*- coding: utf-8 -*-
from transformers import FlaxAutoModelForMaskedLM
from tqdm import tqdm
import numpy as np
from flax.training.common_utils import get_metrics, onehot, shard
from flax.training import train_state
import jax.numpy as jnp
import flax
import optax
import jax
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from transformers import AutoConfig
from pathlib import Path
from transformers.utils import send_example_telemetry
import torch
import os
import sys
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".80"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# os.environ['FI_EFA_FORK_SAFE'] = '1'
os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'
os.environ['NCCL_DEBUG'] = 'TRACE'
# os.environ['FI_LOG_LEVEL'] = 'Info'
jax.distributed.initialize(coordinator_address='172.31.45.102:1234', num_processes=2, process_id=0)  
print("device:{}".format(jax.device_count()))

args = sys.argv
language = "wikipedia_it"
model_config = "roberta-base"

model_dir = model_config + f"-pretrained-{language}"


Path(model_dir).mkdir(parents=True, exist_ok=True)

config = AutoConfig.from_pretrained(model_config)
config.save_pretrained(f"{model_dir}")

raw_dataset = load_dataset("wikipedia", "20220301.it")

tokenizer = ByteLevelBPETokenizer()


def batch_iterator(batch_size=1000):
    for i in range(0, len(raw_dataset), batch_size):
        yield raw_dataset["train"][i: i + batch_size]["text"]


tokenizer.train_from_iterator(batch_iterator(), vocab_size=config.vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])


tokenizer.save(f"{model_dir}/tokenizer.json")

max_seq_length = int(args[1])
raw_dataset["train"] = load_dataset("wikipedia", "20220301.it", split="train[5%:]")
raw_dataset["validation"] = load_dataset("wikipedia", "20220301.it", split="train[:5%]")


# these cells should be commented out to run on full dataset
# raw_dataset["train"] = raw_dataset["train"].select(range(10000))
# raw_dataset["validation"] = raw_dataset["validation"].select(range(1000))

tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}")


def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)


tokenized_datasets = raw_dataset.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=raw_dataset["train"].column_names)


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i: i + max_seq_length]
            for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result


tokenized_datasets = tokenized_datasets.map(
    group_texts, batched=True, num_proc=4)


per_device_batch_size = int(args[2])
num_epochs = 3
training_seed = 0
learning_rate = 5e-5

total_batch_size = per_device_batch_size * jax.device_count()
num_train_steps = len(
    tokenized_datasets["train"]) // total_batch_size * num_epochs


model = FlaxAutoModelForMaskedLM.from_config(
    config, seed=training_seed, dtype=jnp.dtype("bfloat16"))
# model = torch.compile(model)
# model = FlaxAutoModelForMaskedLM.from_config(
#     config, seed=training_seed, dtype=jnp.float16)


linear_decay_lr_schedule_fn = optax.linear_schedule(
    init_value=learning_rate, end_value=0, transition_steps=num_train_steps)


adamw = optax.adamw(learning_rate=linear_decay_lr_schedule_fn,
                    b1=0.9, b2=0.98, eps=1e-8, weight_decay=0.01)


state = train_state.TrainState.create(
    apply_fn=model.__call__, params=model.params, tx=adamw)


@flax.struct.dataclass
class FlaxDataCollatorForMaskedLanguageModeling:
    mlm_probability: float = 0.15

    def __call__(self, examples, tokenizer, pad_to_multiple_of=16):
        batch = tokenizer.pad(examples, return_tensors="np",
                              pad_to_multiple_of=pad_to_multiple_of)

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask, tokenizer
        )

        return batch

    def mask_tokens(self, inputs, special_tokens_mask, tokenizer):
        labels = inputs.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(
            1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, np.full(
            labels.shape, 0.8)).astype("bool") & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(
            tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(
            1, np.full(labels.shape, 0.5)).astype("bool")
        indices_random &= masked_indices & ~indices_replaced
        random_words = np.random.randint(
            tokenizer.vocab_size, size=labels.shape, dtype="i4")
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


data_collator = FlaxDataCollatorForMaskedLanguageModeling(mlm_probability=0.15)


def generate_batch_splits(num_samples, batch_size, rng=None):
    samples_idx = jax.numpy.arange(num_samples)

    # if random seed is provided, then shuffle the dataset
    if input_rng is not None:
        samples_idx = jax.random.permutation(input_rng, samples_idx)

    samples_to_remove = num_samples % batch_size

    # throw away incomplete batch
    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]

    batch_idx = np.split(samples_idx, num_samples // batch_size)
    return batch_idx


def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        labels = batch.pop("labels")

        logits = state.apply_fn(**batch, params=params,
                                dropout_rng=dropout_rng, train=True)[0]

        # compute loss, ignore padded input tokens
        label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
        loss = optax.softmax_cross_entropy(
            logits, onehot(labels, logits.shape[-1])) * label_mask

        # take average
        loss = loss.sum() / label_mask.sum()

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
        {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
    )

    return new_state, metrics, new_dropout_rng


parallel_train_step = jax.pmap(train_step, "batch")


def eval_step(params, batch):
    labels = batch.pop("labels")

    logits = model(**batch, params=params, train=False)[0]

    label_mask = jax.numpy.where(labels > 0, 1.0, 0.0)
    loss = optax.softmax_cross_entropy(
        logits, onehot(labels, logits.shape[-1])) * label_mask

    # compute accuracy
    accuracy = jax.numpy.equal(jax.numpy.argmax(
        logits, axis=-1), labels) * label_mask

    # summarize metrics
    metrics = {"loss": loss.sum(), "accuracy": accuracy.sum(),
               "normalizer": label_mask.sum()}
    metrics = jax.lax.psum(metrics, axis_name="batch")

    return metrics


parallel_eval_step = jax.pmap(eval_step, "batch")

state = flax.jax_utils.replicate(state)


def process_eval_metrics(metrics):
    metrics = get_metrics(metrics)
    metrics = jax.tree_map(jax.numpy.sum, metrics)
    normalizer = metrics.pop("normalizer")
    metrics = jax.tree_map(lambda x: x / normalizer, metrics)
    return metrics


rng = jax.random.PRNGKey(training_seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())

import time
for epoch in tqdm(range(1, num_epochs + 1), desc=f"Epoch ...", position=0, leave=True):
    rng, input_rng = jax.random.split(rng)
    epoch_start = time.time()
    # -- Train --
    train_batch_idx = generate_batch_splits(
        len(tokenized_datasets["train"]), total_batch_size, rng=input_rng)

    with tqdm(total=len(train_batch_idx), desc="Training...", leave=False) as progress_bar_train:
        for batch_idx in train_batch_idx:
            model_inputs = data_collator(
                tokenized_datasets["train"][batch_idx], tokenizer=tokenizer, pad_to_multiple_of=16)

            # Model forward
            model_inputs = shard(model_inputs.data)
            state, train_metric, dropout_rngs = parallel_train_step(
                state, model_inputs, dropout_rngs)

            progress_bar_train.update(1)

        progress_bar_train.write(
            f"Train... ({epoch}/{num_epochs} | Loss: {round(train_metric['loss'].mean(), 3)}, Learning Rate: {round(train_metric['learning_rate'].mean(), 6)})"
        )
    print("epoch: %d, time %.4f" % (epoch, time.time()-epoch_start))