"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        AutoModel,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        AutoModel,
    )
import torch

from llmzoo.deploy.webapp.inference import load_model, generate_stream
from llmzoo.deploy.webapp.utils import build_logger, server_error_msg

GB = 1 << 30

logger = build_logger("model_worker", f"model_worker.log")
global_counter = 0

model_semaphore = None


class ModelWorker:
    def __init__(
            self,
            model_path,
            model_name,
            device,
            num_gpus,
            max_gpu_memory,
            load_8bit=False,
            load_4bit=False,
    ):
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.model_name = model_name or model_path.split("/")[-1]
        self.device = device

        logger.info(f"Loading the model {self.model_name}...")
        self.model, self.tokenizer = load_model(model_path, device, num_gpus, max_gpu_memory, load_8bit, load_4bit)

        if hasattr(self.model.config, "max_sequence_length"):
            self.context_len = self.model.config.max_sequence_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.context_len = self.model.config.max_position_embeddings
        else:
            self.context_len = 2048

        self.generate_stream_func = generate_stream

    def get_queue_length(self):
        if (
                model_semaphore is None
                or model_semaphore._value is None
                or model_semaphore._waiters is None
        ):
            return 0
        else:
            return (
                    args.limit_model_concurrency
                    - model_semaphore._value
                    + len(model_semaphore._waiters)
            )

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def generate_stream_gate(self, params):
        try:
            for output in self.generate_stream_func(
                    self.model,
                    self.tokenizer,
                    params,
                    self.device,
                    self.context_len,
                    args.stream_interval,
            ):
                ret = {
                    "text": output,
                    "error_code": 0,
                }
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError:
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"

    async def generate_stream(self, request):
        global model_semaphore, global_counter
        global_counter += 1
        params = await request

        if model_semaphore is None:
            model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
        await model_semaphore.acquire()
        generator = self.generate_stream_gate(params)
        background_tasks = BackgroundTasks()
        background_tasks.add_task(release_model_semaphore)
        return StreamingResponse(generator, background=background_tasks)


def release_model_semaphore():
    if model_semaphore:
        model_semaphore.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21666)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21666")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21888")
    parser.add_argument(
        "--model-path",
        type=str,
        default="facebook/opt-350m",
        help="The path to the weights",
    )
    parser.add_argument("--model-name", type=str, help="Optional name")
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda"
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per gpu. Use a string like '13Gib'",
    )
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # uvicorn.run(app, host=args.host, port=args.port, log_level="info")
