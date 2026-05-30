from dataclasses import dataclass


@dataclass
class DatasetProfile:
    dataset_name: str

    num_samples: int
    num_intents: int

    mean_prompt_length: float
    median_prompt_length: float

    min_prompt_length: int
    max_prompt_length: int

    missing_prompts: int
    missing_intents: int

    duplicate_prompts: int
