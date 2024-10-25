import os
import sys
import torch
import wandb
import logging
import datasets
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
import colorlog


@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(
        default=None, metadata={"choices": ["auto", "bfloat16", "float16", "float32"]}
    )
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    block_size: int = field(default=1024)
    num_workers: Optional[int] = field(default=None)
    data_files: Optional[str] = field(
        default=None, metadata={"help": "Path to the local dataset file (JSON format)."}
    )


class CustomLoggerCallback(TrainerCallback):

    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """
        학습 중 로그 발생 시 호출되는 메서드
        """
        if state.is_local_process_zero:
            self.logger.info(f"Logs: {logs}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        """
        평가 후 로그 출력
        """
        if state.is_local_process_zero:
            self.logger.info(f"Evaluation metrics: {metrics}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        체크포인트 저장 시 로그 출력
        """
        if state.is_local_process_zero:
            self.logger.info("Checkpoint saved.")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        학습 시작 시 로그 출력
        """
        if state.is_local_process_zero:
            self.logger.info("Training has started.")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        학습 종료 시 로그 출력
        """
        if state.is_local_process_zero:
            self.logger.info("Training has ended.")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        스텝 종료 시 로그 출력
        """
        # 원한다면 마지막에 한 번 로그로 저장 가능
        if state.is_local_process_zero and state.global_step == state.max_steps:
            logging.info("\n")  # 줄바꿈을 추가해 다음 로그가 새로운 줄에 나오도록 함
            self.logger.info(
                f"Step {state.global_step}/{state.max_steps} training complete."
            )


class CustomWandbLoggerCallback(TrainerCallback):

    def __init__(self, project_name, run_name):
        super().__init__()
        self.project_name = project_name
        self.run_name = run_name
        self.trainer = None  # Trainer 객체를 저장할 변수
        self.init_wandb()

    def init_wandb(self):
        wandb.init(project=self.project_name)
        wandb.run.name = self.run_name
        wandb.alert(
            title="TEST ALERT",
            text="https://www.naver.com",
            level=wandb.AlertLevel.WARN,  # WARN, INFO, or ERROR
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """
        로그 발생 시 wandb에 기록
        """
        if state.is_local_process_zero:
            wandb.log(logs)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        """
        평가 후 메트릭을 wandb에 기록
        """
        if state.is_local_process_zero and metrics is not None:
            wandb.log(metrics)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        학습 종료 시 모델 저장 및 메트릭 로깅을 처리하고, wandb 세션 종료
        """
        if state.is_local_process_zero and self.trainer is not None:
            # 모델 저장
            self.trainer.save_model()

            # 메트릭 저장
            metrics = state.log_history[-1] if state.log_history else {}
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()

            # wandb 종료
            wandb.finish()

    def set_trainer(self, trainer):
        """
        Trainer 객체를 외부에서 설정할 수 있도록 메서드를 추가
        """
        self.trainer = trainer


from transformers import TrainerCallback
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os
from datetime import datetime


class HuggingFaceUploadCallback(TrainerCallback):
    def __init__(self, repo_name, token=None, private=False, organization=None):
        """
        학습 종료 후 Hugging Face Model Hub에 모델을 업로드하는 콜백

        Args:
            repo_name (str): Hugging Face Model Hub에서 생성할 레포지토리 이름
            token (str, optional): Hugging Face API 토큰. 환경 변수로 설정 가능.
            private (bool, optional): 레포지토리를 비공개로 설정할지 여부. 기본값은 False.
            organization (str, optional): 레포지토리가 속할 조직 이름. 기본값은 개인 계정.
        """
        self.repo_name = repo_name
        self.token = token if token else os.getenv("HF_TOKEN")
        self.private = private
        self.organization = organization

        # If organization is provided, include it in the repo_name
        if organization:
            self.repo_name = f"{organization}/{repo_name}"
        else:
            self.repo_name = f"obov/{repo_name}"

    def on_train_end(self, args, state, control, **kwargs):
        """
        학습이 끝나면 Hugging Face Model Hub에 모델을 업로드합니다.
        """
        if state.is_local_process_zero:  # 메인 프로세스에서만 실행
            # 가장 최신 체크포인트 디렉토리를 찾음
            latest_checkpoint = self.get_latest_checkpoint(args.output_dir)
            if latest_checkpoint:
                self.upload_model_to_hub(model_dir=latest_checkpoint)
            else:
                logging.info(f"No checkpoint found in {args.output_dir}")

    def get_latest_checkpoint(self, output_dir):
        """
        가장 최신의 체크포인트 디렉토리를 찾습니다.

        Args:
            output_dir (str): 모델 체크포인트가 저장된 디렉토리

        Returns:
            str: 가장 최근 체크포인트 디렉토리 경로
        """
        checkpoints = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint")
        ]
        if not checkpoints:
            return None
        # 숫자로 된 체크포인트 부분을 기준으로 정렬하여 가장 마지막 체크포인트를 선택
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        logging.info(f"Latest checkpoint found: {latest_checkpoint}")
        return latest_checkpoint

    def create_repo_if_not_exists(self, repo_id, token, private=False):
        """
        레포지토리가 존재하지 않으면 Hugging Face에 레포지토리를 생성합니다.
        """
        api = HfApi()
        try:
            # 레포지토리가 있는지 확인
            api.repo_info(repo_id=repo_id, token=token, repo_type="model")
            logging.info(f"Repository {repo_id} already exists.")
        except RepositoryNotFoundError:
            # 레포지토리가 없으면 생성
            logging.info(f"Creating repository {repo_id} on Hugging Face Hub.")
            create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                exist_ok=True,
                repo_type="model",  # 모델 레포지토리로 생성
            )
            logging.info(f"Repository {repo_id} created successfully.")

    def upload_model_to_hub(self, model_dir):
        """
        Hugging Face Model Hub에 모델을 업로드합니다.
        """
        api = HfApi()

        # 레포지토리가 없다면 생성
        self.create_repo_if_not_exists(self.repo_name, self.token, private=self.private)

        # 현재 시간 추가
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 모델을 업로드
        logging.info(f"Uploading model to Hugging Face Hub: {self.repo_name}")
        api.upload_folder(
            repo_id=self.repo_name,
            folder_path=model_dir,
            commit_message=f"End of training: uploading model: {current_time}",
            token=self.token,
            repo_type="model",  # 모델 레포지토리로 업로드
        )
        logging.info(
            f"Model uploaded successfully to: https://huggingface.co/{self.repo_name}"
        )


def setup_logger(training_args: TrainingArguments):

    # Logger 설정 추가
    logger = logging.getLogger()

    formatter = colorlog.ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname)s%(reset)s: %(message)s",
        log_colors={
            "DEBUG": "blue",
            "INFO": "cyan",
            "WARNING": "yellow",
            "ERROR": "bold_red",
        },
        datefmt="%m/%d/%Y %I:%M:%S %p",  # 날짜 형식 지정
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()  # transformers 라이브러리 로그 레벨 설정

    log_level = training_args.get_process_log_level()

    # 전역 Logger 및 Hugging Face 라이브러리 로그 레벨 설정
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # 기타 Hugging Face logger 옵션 설정
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")
    return logger


def setup_model_and_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, config=config, torch_dtype=args.torch_dtype
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def tokenize_function(examples, tokenizer, text_column_name):
    return tokenizer(examples[text_column_name])


def group_texts(examples, block_size):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    # Argument parsing
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    logger = setup_logger(training_args)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # Model and Tokenizer Setup
    model, tokenizer = setup_model_and_tokenizer(args)

    # Model's max_position_embeddings 확인 및 block_size 조정
    max_pos_embeddings = (
        model.config.max_position_embeddings
        if hasattr(model.config, "max_position_embeddings")
        else 1024
    )
    block_size = min(
        args.block_size, max_pos_embeddings
    )  # block_size를 모델의 max_position_embeddings에 맞춤

    # Dataset Loading and Tokenization
    if args.dataset_type == "json" and args.data_files:
        raw_datasets = load_dataset(
            "json", data_files={"train": args.data_files}, split="train"
        )
    elif args.dataset_type == "huggingface" and args.dataset_name:
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, split="train"
        )
    else:
        logger.error(
            "Invalid dataset configuration. Please provide either dataset_type='json' with data_files or dataset_type='huggingface' with dataset_name."
        )
        sys.exit(1)
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer, text_column_name),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names,
    )

    lm_datasets = tokenized_datasets.map(
        lambda examples: group_texts(examples, block_size),  # 조정된 block_size 사용
        batched=True,
        num_proc=args.num_workers,
    )

    # train_dataset = lm_datasets["train"]
    train_dataset = lm_datasets["train"].select(range(300))
    eval_dataset = lm_datasets["validation"]

    # Check for existing checkpoints
    checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is None and training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

    hf_upload_callback = HuggingFaceUploadCallback(
        repo_name="gpt-finetuned",  # Hugging Face 레포지토리 이름
        organization=None,  # 개인 계정에 업로드하는 경우
        private=False,  # 레포지토리를 공개로 설정
    )

    # Initialize Custom Callbacks
    wandb_callback = CustomWandbLoggerCallback("Hanghae99", "gpt-finetuning")

    # Initialize Trainer with LoggerCallback and WandbLoggerCallback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[
            CustomLoggerCallback(logger),
            wandb_callback,
            hf_upload_callback,
        ],
    )

    # Set trainer instance to the callback
    wandb_callback.set_trainer(trainer)

    # Training
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
