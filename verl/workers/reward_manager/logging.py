from collections import defaultdict
from typing import Any
import os
import csv
import json
import torch
from datetime import datetime

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("logging")
class NaiveRewardManagerWithLogging(AbstractRewardManager):
    """
    Reward manager that logs prompt/response/reward with automatic chronological tracking.
    - Tracks first `num_examine` unique prompts
    - Automatically adds timestamp, internal_epoch, and internal_step fields
    - Compatible with verl's PPO trainer without modification
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 10,
        compute_score=None,
        reward_fn_key: str = "data_source",
        save_dir: str = "eval_result",
        save_format: str = "csv",  # "csv" or "json"
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.save_dir = save_dir
        self.save_format = save_format.lower()
        os.makedirs(self.save_dir, exist_ok=True)

        # Log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(self.save_dir, f"reward_log_{timestamp}.{self.save_format}")

        if self.save_format == "csv":
            with open(self.save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "index", "data_source", "prompt", "response", "ground_truth",
                    "reward", "epoch", "step", "timestamp"
                ])

        # Track first N unique prompts
        self.tracked_prompts = set()

        # Internal epoch/step counters (independent of trainer)
        self._internal_step = 0
        self._internal_epoch = 0

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # Auto increment internal step each call
        self._internal_step += 1
        # For simplicity, increase epoch every 100 steps (configurable)
        if self._internal_step % 100 == 0:
            self._internal_epoch += 1

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            # Selective logging
            should_log = False
            if len(self.tracked_prompts) < self.num_examine:
                if prompt_str not in self.tracked_prompts:
                    self.tracked_prompts.add(prompt_str)
                    should_log = True
            elif prompt_str in self.tracked_prompts:
                should_log = True

            if should_log:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                print(f"[{now}] [epoch={self._internal_epoch}] [step={self._internal_step}]")
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", reward)

                record = {
                    "index": i,
                    "data_source": data_source,
                    "prompt": prompt_str,
                    "response": response_str,
                    "ground_truth": ground_truth,
                    "reward": float(reward),
                    "epoch": self._internal_epoch,
                    "step": self._internal_step,
                    "timestamp": now,
                }

                if self.save_format == "csv":
                    with open(self.save_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            record["index"],
                            record["data_source"],
                            record["prompt"],
                            record["response"],
                            record["ground_truth"],
                            record["reward"],
                            record["epoch"],
                            record["step"],
                            record["timestamp"],
                        ])
                elif self.save_format == "json":
                    with open(self.save_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
