# %%
!pip install datasets>=2.6.1
!pip install librosa
!pip install evaluate>=0.30
!pip install jiwer
!pip install gradio
!pip install -q accelerate loralib
!pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git@main
!pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl

# %%
import os, evaluate, torch, gc, numpy as np
from datasets import (
    load_dataset,
    interleave_datasets,
    concatenate_datasets,
    Audio,
    IterableDatasetDict,
)
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    Seq2SeqTrainer,
    AutoTokenizer,
)
from typing import (
    Any,
    Dict,
    List,
    Union,
)
from peft import (
    LoraConfig,
    PeftModel,
    LoraModel,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from dataclasses import dataclass
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from tqdm import tqdm
from torch.utils.data import DataLoader

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name_or_path = ""
language = ""
language_abbr = ""
task = ""
dbn_a = ""
dbn_b = ""
dbn_c = ""

# %%
rd = IterableDatasetDict()

db1 = load_dataset(dbn_a, "", split="train", token=True, trust_remote_code=True, streaming=True)
db2 = load_dataset(dbn_b, "", split="train", token=True, trust_remote_code=True, streaming=True)
db3 = load_dataset(dbn_c, "", split="train", token=True, trust_remote_code=True, streaming=True)

db1 = db1.cast(db2.features)
db3 = db3.cast(db1.features)

rd["train"] = concatenate_datasets([db1, db2, db3])
rd["test"] = load_dataset(dbn_a, "", split="test", token=True, trust_remote_code=True, streaming=True)

# %%
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language_abbr, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

# %%
rd = rd.cast_column("audio", Audio(sampling_rate=16000))

# %%
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()

# %%
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    transcription = batch["sentence"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()

    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

# %%
rd["train"] = rd["train"].shuffle(
    buffer_size=500,
    seed=0,
)

# %%
max_input_length = 30.0
def is_audio_in_length_range(length):
    return length < max_input_length

# %%
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# %%
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# %%
metric = evaluate.load("wer")

# %%
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# %%
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")

# %%
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

# %%
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
model.print_trainable_parameters()

# %%
vd = rd.map(prepare_dataset, remove_columns=list(next(iter(rd.values())).features)).with_format("torch")

# %%
training_args = Seq2SeqTrainingArguments(
    output_dir="",  
    push_to_hub=False,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  
    learning_rate=1e-3,
    weight_decay=0.01,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=2,
    generation_max_length=128,
    logging_steps=15,
    max_steps=1000,
    eval_steps=100,
    save_steps=100,
    remove_unused_columns=False, 
    label_names=["labels"], 
    optim="adafactor",
    report_to=["tensorboard"],
    logging_dir='logs',
)

# %%
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vd["train"],
    eval_dataset=vd["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
)
model.config.use_cache = True

# %%
model.save_pretrained("")
tokenizer.save_pretrained('')
processor.save_pretrained("")

# %%
trainer.train()

# %%
peft_model_id = ""
model.push_to_hub(peft_model_id)

# %%
peft_model_id = "" 
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)

model = PeftModel.from_pretrained(model, peft_model_id)
model.config.use_cache = True

# %%
eval_dataloader = DataLoader(common_voice["test"], batch_size=8, collate_fn=data_collator)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
normalizer = BasicTextNormalizer()

predictions = []
references = []
normalized_predictions = []
normalized_references = []

model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].to("cuda"),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)
            normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
            normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
        del generated_tokens, labels, batch
    gc.collect()
wer = 100 * metric.compute(predictions=predictions, references=references)
normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}

print(f"{wer=} and {normalized_wer=}")
print(eval_metrics)

# %%
# For merging adapter and model

import torch
from peft import PeftModel, PeftConfig
import transformers
import os, time
import tempfile
from torch.utils.data import DataLoader
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    Seq2SeqTrainer,
    AutoTokenizer,
)

BASE_MODEL = ""
LORA_WEIGHTS = ""

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
)

model = PeftModel.from_pretrained(
    model,
    LORA_WEIGHTS,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model")

model.save_pretrained("")
model.push_to_hub("")

# %% [markdown]
# from transformers import (
#     AutomaticSpeechRecognitionPipeline,
#     WhisperForConditionalGeneration,
#     WhisperTokenizer,
#     WhisperProcessor,
# )
# from peft import PeftModel, PeftConfig
# 
# peft_model_id = "sin2piusc/whisper-medium-3L-JP" # Use the same model ID as before.
# language = "ja"
# task = "transcribe"
# peft_config = PeftConfig.from_pretrained(peft_model_id)
# model = WhisperForConditionalGeneration.from_pretrained(
#     peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
# )
# 
# model = PeftModel.from_pretrained(model, peft_model_id)
# tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
# processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
# feature_extractor = processor.feature_extractor
# forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
# pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)
# 
# 
# def transcribe(audio):
#     with torch.cuda.amp.autocast():
#         text = pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
#     return text
# 
# transcribe("a.mp3")

# %% [markdown]
# from datetime import timedelta
# import os
# import whisper
# 
# def transcribe_audio(path):
#     model = whisper.load_model("whisper-medium-3L-JP-MERGED") # Change this to your desired model
#     print("Whisper model loaded.")
#     transcribe = model.transcribe(audio=path)
#     segments = transcribe['segments']
# 
#     for segment in segments:
#         startTime = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
#         endTime = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
#         text = segment['text']
#         segmentId = segment['id']+1
#         segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text[0] is ' ' else text}\n\n"
# 
#         srtFilename = os.path.join("SrtFiles", f"VIDEO_FILENAME.srt")
#         with open(srtFilename, 'a', encoding='utf-8') as srtFile:
#             srtFile.write(segment)
# 
#     return srtFilename


