from transformers import (
   AutoModelForCausalLM,
   BertTokenizerFast,
   Trainer,
   TextGenerationPipeline,
   CONFIG_MAPPING,
   MODEL_FOR_CAUSAL_LM_MAPPING,
   AutoConfig,
   # AutoTokenizer,
   HfArgumentParser,
   TrainingArguments,
   default_data_collator,
   set_seed,

)
import transformers
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


data_args = DataTrainingArguments(dataset_name='wikitext', dataset_config_name='wikitext-2-raw-v1', train_file=None, validation_file=None, block_size=None, overwrite_cache=False, validation_split_percentage=5, preprocessing_num_workers=None)
training_args = TrainingArguments(output_dir="test-clm", overwrite_output_dir=False, do_train=True,
                                  do_eval=True, do_predict=False, evaluation_strategy="no",
                                  prediction_loss_only=False, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                  gradient_accumulation_steps=1, eval_accumulation_steps=None, learning_rate=5e-05, weight_decay=0.0,
                                  adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=10.0, max_steps=-1,
                                  lr_scheduler_type="linear", warmup_steps=0, logging_dir="runs/model", logging_first_step=False,
                                  logging_steps=500, save_steps=5000, save_total_limit=None, no_cuda=False, seed=42, fp16=False, fp16_opt_level=1,
                                  fp16_backend="auto", local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False, debug=False, dataloader_drop_last=False,
                                  eval_steps=500, dataloader_num_workers=0, past_index=-1, run_name="test-clm", disable_tqdm=False,
                                  remove_unused_columns=True, label_names=None, load_best_model_at_end=False, metric_for_best_model=None,
                                  greater_is_better=None, ignore_data_skip=False, sharded_ddp=False, deepspeed=None, label_smoothing_factor=0.0,
                                  adafactor=False, group_by_length=False, report_to=['tensorboard'], ddp_find_unused_parameters=None,
                                  dataloader_pin_memory=True)
model_args = ModelArguments(model_name_or_path='gpt2-base-chinese', model_type=None, config_name=None, tokenizer_name=None, cache_dir=None, use_fast_tokenizer=True, model_revision='main', use_auth_token=False)


# Set the verbosity to info of the Transformers logger (on main process only):
if is_main_process(training_args.local_rank):
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
logger.info("Training/evaluation parameters %s", training_args)

# Set seed before initializing model.
set_seed(training_args.seed)

# Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
# 'text' is found. You can easily tweak this behavior (see below).
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
if data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    if "validation" not in datasets.keys():
        datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:{data_args.validation_split_percentage}%]",
        )
        datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[{data_args.validation_split_percentage}%:]",
        )
else:
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    extension = (
        data_args.train_file.split(".")[-1]
        if data_args.train_file is not None
        else data_args.validation_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
    datasets = load_dataset(extension, data_files=data_files)


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)


datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[{data_args.validation_split_percentage}%:]",
        )


print(datasets["train"])
print(data_args.dataset_name)
print(data_args.dataset_config_name)


# # data_args
# dataset_name='wikitext'
# dataset_config_name='wikitext-2-raw-v1'
# train_file=None
# validation_file=None
# block_size=None
# overwrite_cache=False
# validation_split_percentage=5
# preprocessing_num_workers=None

# # training_args
# output_dir=/tmp/test-clm
# overwrite_output_dir=False
# do_train=True
# do_eval=True
# do_predict=False
# evaluation_strategy=EvaluationStrategy.NO
# prediction_loss_only=False
# per_device_train_batch_size=1
# per_device_eval_batch_size=1
# gradient_accumulation_steps=1
# eval_accumulation_steps=None
# learning_rate=5e-05
# weight_decay=0.0
# adam_beta1=0.9
# adam_beta2=0.999
# adam_epsilon=1e-08
# max_grad_norm=1.0
# num_train_epochs=3.0
# max_steps=-1
# lr_scheduler_type=SchedulerType.LINEAR
# warmup_steps=0
# logging_dir=runs/Mar08_20-37-03_kw-MS-7B23
# logging_first_step=False
# logging_steps=500
# save_steps=500
# save_total_limit=None
# no_cuda=False
# seed=42
# fp16=False
# fp16_opt_level=O1
# fp16_backend=auto
# local_rank=-1
# tpu_num_cores=None
# tpu_metrics_debug=False
# debug=False
# dataloader_drop_last=False
# eval_steps=500
# dataloader_num_workers=0
# past_index=-1
# run_name=/tmp/test-clm
# disable_tqdm=False
# remove_unused_columns=True
# label_names=None
# load_best_model_at_end=False
# metric_for_best_model=None
# greater_is_better=None
# ignore_data_skip=False
# sharded_ddp=False
# deepspeed=None
# label_smoothing_factor=0.0
# adafactor=False
# group_by_length=False
# report_to=['tensorboard']
# ddp_find_unused_parameters=None
# dataloader_pin_memory=True
# _n_gpu=1

# # model_args
# model_name_or_path='gpt2-base-chinese'
# model_type=None
# config_name=None
# tokenizer_name=None
# cache_dir=None
# use_fast_tokenizer=True
# model_revision='main'
# use_auth_token=False


gpt2_zh = AutoModelForCausalLM.from_pretrained('ckiplab/gpt2-base-chinese')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
gpt2_zh.resize_token_embeddings(len(tokenizer))


do_train = True
if do_train:
    column_names = datasets["train"].column_names
else:
    column_names = datasets["validation"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not data_args.overwrite_cache,
)
if data_args.block_size is None:
    block_size = tokenizer.model_max_length
    if block_size > 1024:
        logger.warn(
            f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            "Picking 1024 instead. You can change that default value by passing --block_size xxx."
        )
    block_size = 1024
else:
    if data_args.block_size > tokenizer.model_max_length:
        logger.warn(
            f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
    block_size = min(data_args.block_size, tokenizer.model_max_length)


# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    load_from_cache_file=not data_args.overwrite_cache,
)

# Initialize our Trainer
trainer = Trainer(
    model=gpt2_zh,
    args=training_args,
    train_dataset=lm_datasets["train"] if training_args.do_train else None,
    eval_dataset=lm_datasets["validation"] if training_args.do_eval else None,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=default_data_collator,
)
last_checkpoint = None
# Training
if training_args.do_train:
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()
trainer.save_model()  # Saves the tokenizer too for easy upload

# Evaluation
results = {}
if training_args.do_eval:
    logger.info("*** Evaluate ***")

    eval_output = trainer.evaluate()

    perplexity = math.exp(eval_output["eval_loss"])
    results["perplexity"] = perplexity

    # trainer.log_metrics("eval", results)
    # trainer.save_metrics("eval", results)

# return results

print(results)
