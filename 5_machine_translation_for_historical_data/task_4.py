from __future__ import annotations
# import os

# os.environ["HF_HOME"] = (
#     "/vol/tensusers/dbrugmans/projects/ru-master-thesis-24-25/models"
# )
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
from abc import ABC
from pathlib import Path

import torch
import datasets
import transformers
import peft
import evaluate
from tqdm import tqdm


path_to_assignment_5 = Path(__file__).parents[0]
path_to_swedish_data = Path(path_to_assignment_5, "data", "swedish")


def set_universal_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False


def get_cuda_device() -> str:
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = _get_cuda_device_with_most_free_memory()
        else:
            device = "cuda"
    else:
        device = "cpu"

    print(torch.device(device))

    if "cuda" in device:
        print(torch.cuda.get_device_name())
        print(
            f"{int(torch.cuda.get_device_properties(0).total_memory / 1000000000)} GB VRAM"
        )

    return device


def _get_cuda_device_with_most_free_memory() -> str:
    device = ""
    device_memory = 0

    for i in range(torch.cuda.device_count()):
        current_device = f"cuda:{i}"
        current_device_memory = torch.cuda.mem_get_info(current_device)[0]

        if current_device_memory >= device_memory:
            device = current_device
            device_memory = current_device_memory

    return device


class NMTModel(ABC):
    def __init__(self, use_custom: bool):
        super().__init__()

        self.use_custom = use_custom

        self.tokenizer = transformers.MBartTokenizer.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt",
            use_fase=False,
        )

        base_model = transformers.MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        )

        if use_custom:
            self.mbart = peft.PeftModel.from_pretrained(
                base_model,
                "/vol/tensusers/dbrugmans/projects/uu-machine-translation-project/models/finetuned_creole_mbart_20",
            )
        else:
            self.mbart = base_model

        self.pad_id = self.tokenizer(
            "<pad>", add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids.item()

        self.bos_id = self.tokenizer(
            "<s>", add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids.item()

        self.eos_id = self.tokenizer(
            "</s>", add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids.item()

    def translate(
        self, source_sentence: str, source_lang: str, target_lang: str
    ) -> str:
        self.mbart.eval()

        tokenizer_input = source_sentence + f" </s> {source_lang}"
        model_input = self.tokenizer(
            tokenizer_input, add_special_tokens=False, return_tensors="pt", padding=True
        ).input_ids

        model_output = self.mbart.generate(
            inputs=model_input,
            use_cache=True,
            num_beams=4,
            max_length=60,
            min_length=1,
            early_stopping=True,
            pad_token_id=self.pad_id,
            bos_token_id=self.bos_id,
            eos_token_id=self.eos_id,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[target_lang],
        )

        translated_sentence = self.tokenizer.decode(
            model_output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return translated_sentence

    def translate_batch(
        self, dataset: datasets.Dataset, source_lang: str, target_lang: str
    ) -> tuple[list[str], list[str], list[str]]:
        sources = []
        targets = []
        predictions = []

        for row in tqdm(dataset["translation"]):
            sources.append(row["src_text"])
            targets.append(row["tgt_text"])
            predictions.append(
                self.translate(row["src_text"], source_lang, target_lang)
            )

        return sources, targets, predictions

    def get_bleu(
        self, dataset: datasets.Dataset, source_lang: str, target_lang: str
    ) -> tuple[float, dict, tuple[list[str], list[str], list[str]]]:
        sources, targets, predictions = self.translate_batch(
            dataset, source_lang, target_lang
        )

        preprocessed_targets = remove_removable_chars(targets)
        preprocessed_predictions = remove_removable_chars(predictions)

        preprocessed_targets = list(map(lambda x: x.lower(), preprocessed_targets))
        preprocessed_targets = list(map(lambda x: [x], preprocessed_targets))

        preprocessed_predictions = list(
            map(lambda x: x.lower(), preprocessed_predictions)
        )

        bleu = evaluate.load("bleu")
        bleu_results = bleu.compute(
            references=preprocessed_targets, predictions=preprocessed_predictions
        )

        return bleu_results["bleu"], bleu_results, (sources, targets, predictions)


def remove_removable_chars(corpus: list[str]) -> list[str]:
    preprocessed_corpus: list[str] = []
    removable_chars = '….,?!:;‘“”()[]{}+=-*_–\|/<>#€%^&*@1234567890"\n│•·†„‘·`‚¬½'

    for text in corpus:
        text = "".join([char for char in text if char not in removable_chars])
        text = text.replace("’", "'")
        text = text.replace("''", "'")
        text = text.strip()

        preprocessed_corpus.append(text)

    return preprocessed_corpus


def create_custom_dataset():
    for split in ["train", "test", "dev"]:
        dataset_dict = {}

        for version in ["hs", "sv"]:
            with open(
                Path(path_to_swedish_data, f"swedish.hs-sv.{split}.{version}"),
                "r",
                encoding="utf-8",
            ) as swedish_text_file:
                swedish_text = swedish_text_file.readlines()

                if split == "test":
                    swedish_text = swedish_text[:2999]

            sentences = []
            current_sentence = []
            for line in swedish_text:
                if not line:
                    current_sentence_str = " ".join(current_sentence)
                    current_sentence_str = current_sentence_str.replace(" .", ".")
                    current_sentence_str = current_sentence_str.strip()

                    sentences.append(current_sentence)
                    current_sentence = []

                if line == "\n":
                    current_sentence_str = " ".join(current_sentence)
                    current_sentence_str = current_sentence_str.replace(" .", ".")
                    current_sentence_str = current_sentence_str.strip()

                    sentences.append(current_sentence_str)
                    current_sentence = []

                line = line.replace("\n", "")
                current_sentence.append(line)

            dataset_dict[f"{version}"] = sentences

        dataset_split = datasets.Dataset.from_dict(dataset_dict)
        # backtranslation_dataset.save_to_disk(
        #     "/vol/tensusers/dbrugmans/projects/uu-machine-translation-project/data/backtranslations/djk-eng-deu"
        # )
        dataset_split.save_to_disk(str(Path(path_to_swedish_data, "nmt", f"{split}")))


def train(
    hs_sv_model: NMTModel, train: datasets.Dataset, validation: datasets.Dataset
) -> None:
    def prepare_dataset_for_training(dataset: datasets.Dataset) -> datasets.Dataset:
        source_sentences = [row for row in dataset["hs"]]
        target_sentences = [row for row in dataset["sv"]]

        tokenized_sources = hs_sv_model.tokenizer(
            source_sentences,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        tokenized_targets = hs_sv_model.tokenizer(
            target_sentences,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).input_ids

        tokenized_sources["labels"] = tokenized_targets

        return tokenized_sources

    lora_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,  # Usually twice the rank `r`
        lora_dropout=0.1,
        bias="none",
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=None,
        eval_strategy="epoch",
        num_train_epochs=5,
        learning_rate=1e-3,
        weight_decay=1e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        save_total_limit=1,
        fp16=True,
        seed=31313131,
    )

    lora_mbart = peft.get_peft_model(hs_sv_model.mbart, lora_config)
    tokenized_train = train.map(prepare_dataset_for_training, batched=True)
    tokenized_validation = validation.map(prepare_dataset_for_training, batched=True)

    trainer = transformers.Seq2SeqTrainer(
        lora_mbart,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        tokenizer=hs_sv_model.tokenizer,
    )

    lora_mbart.train()
    trainer.train()
    lora_mbart.save_pretrained(
        "/vol/tensusers/dbrugmans/projects/uu-machine-translation-project/models/finetuned_creole_mbart_20/"
    )


if __name__ == "__main__":
    set_universal_seed(31313131)
    device = get_cuda_device()

    nmt_model = NMTModel(False)
    print(nmt_model.mbart.__class__)

    # create_custom_dataset()

    # bleu_score, bleu_details, bleu_data = nmt_model.get_bleu(sentence_pairs["test"], "djk", "deu")
    # datasets.load_from_disk("/vol/tensusers/dbrugmans/projects/uu-machine-translation-project/data/backtranslations/djk-eng-deu")
    # print(datasets.load_from_disk(str(Path(path_to_swedish_data, "nmt", "train")))[0])

    train_data = datasets.load_from_disk(
        str(Path(path_to_swedish_data, "nmt", "train"))
    )
    dev_data = datasets.load_from_disk(str(Path(path_to_swedish_data, "nmt", "dev")))
    train(nmt_model, train_data, dev_data)

    # custom_nmt_model = model.NMTModel(True)
    # print(custom_nmt_model.mbart.__class__)
    # bleu_score, bleu_details, bleu_data = custom_nmt_model.get_bleu(sentence_pairs["test"], "djk", "deu")

    # sources, targets, predictions = bleu_data
    # for source, target, prediction in zip(sources, targets, predictions):
    #     print(source)
    #     print(target)
    #     print(prediction)
    #     print()
    # print(f"BLEU: {bleu_score}")
    # print(custom_nmt_model.use_custom)
