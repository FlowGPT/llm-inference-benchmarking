import asyncio
import argparse
from collections import Counter
import json
import math
import os
import queue
import re
import random
from rich.console import Console
from rich.table import Table
import sys
import threading
import time
import string
import openai
from num2words import num2words
import pandas as pd


def get_prompt_of_length(prompt_length):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    from transformers import LlamaTokenizerFast

    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )

    prompt = "Pick some lines from these poem lines:\n"
    with open(os.path.join(os.path.dirname(__file__), "sonnet.txt"), "r") as f:
        # pick randome lines from the sonnet that total to prompt_length
        lines = f.readlines()
        while len(tokenizer.encode(prompt)) < args.prompt_length:
            prompt += random.choice(lines)
    prompt += "and then tell me a very very long story:"
    return prompt


def rnd_num_generator(num_digits=2) -> str:
    # Step 1: Generate a random number
    # Generate the number of digits specified (e.g. if NUM_DIGITS = 3, then
    # any number between 100 and 1000 is OK).
    rnd_num = random.randrange(10 ** (200 - 1), 10 ** (200))

    # Step 2: convert to words.
    rnd_num_words = num2words(rnd_num)

    return rnd_num, rnd_num_words

def gen_random_string(num_chars=3) -> str:
    return ''.join(random.choices(string.ascii_lowercase, k=num_chars))


async def endpoint_evaluation_request(client, ep_config):
    if args.validate:
        rnd_num, rnd_num_words = rnd_num_generator(args.num_digits)
        prompt = (
            "Convert the following sequence of words into a number:"
            f" {rnd_num_words}.\nPrint the number first, then tell me a very long"
            " story. "
        )
    else:
           rnd_num = None
           random_tokens = args.random_tokens
           random_tokens_str = " ".join(random.choices(args.random_token_list, k=random_tokens))
           prompt = args.prompt + random_tokens_str + ". Give me an extreme long analysis on the previous text"

    words = ""

    try:
        st = time.time()
        ttft = None
        if args.use_chat:
            messages = [
                {"role": "user", "content": prompt},
            ]
            response = await client.chat.completions.create(
                model=ep_config["model"],
                messages=messages,
                max_tokens=args.max_tokens,
                # Please keep temp at 0. Otherwise increases the number of mismatches.
                temperature=0,
                # Do not set to false. You will get bogus results.
                stream=True,
                stream_options={"include_usage": True},
            )
            async for tok in response:
                if not tok.choices:
                    continue
                delta = tok.choices[0].delta
                if delta.content:
                    if ttft is None:
                        ttft = time.time() - st
                    words += delta.content
            tokens_in = tok.usage.prompt_tokens
            tokens_out = tok.usage.completion_tokens
        else:
            response = await client.completions.create(
                model=ep_config["model"],
                prompt=prompt,
                max_tokens=args.max_tokens,
                # Please keep temp at 0. Otherwise increases the number of mismatches.
                temperature=0,
                # Do not set to false. You will get bogus results.
                stream=True,
            )
            async for tok in response:
                if not tok.choices:
                    continue
                delta = tok.choices[0]
                if delta.text:
                    if ttft is None:
                        ttft = time.time() - st
                    words += delta.text
            tokens_in = tok.usage.prompt_tokens
            tokens_out = tok.usage.completion_tokens
        et = time.time()
    except Exception as e:
        return ("Exception", -1, -1, -1, -1, str(e))

    if args.validate:
        nums = re.findall(r"\d+", words)
        if len(nums) > 0:
            retval = int(nums[0])
            valid = "OK"
            cause = ""
            if retval != rnd_num:
                valid = "Mismatch"
                cause = f"input: {rnd_num_words}, expect: {rnd_num}, got: {retval}"
        else:
            valid = "Mismatch"
            cause = f"Output unparseable. Input = {rnd_num}. Output:\n {words}"
    else:
        valid = "OK"
        cause = ""
    return (valid, ttft, et - st, tokens_in, tokens_out, cause)


async def endpoint_evaluation_round(client, concur_requests, ep_config):
    results = await asyncio.gather(*(
        endpoint_evaluation_request(client, ep_config) for _ in range(concur_requests)
    ))
    return results


def endpoint_evaluation_qps(client, ep_config, results_queue, stop_event):
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=run_loop).start()

    time_between_requests = 1 / args.qps

    def task_done_callback(task):
        results_queue.put(task.result())


    while True:
        if stop_event.is_set():
            print("stop event received, stopping loop")
            loop.call_soon_threadsafe(loop.stop)
            return

        st = time.time()
        future = asyncio.run_coroutine_threadsafe(
            endpoint_evaluation_request(client, ep_config), loop
        )
        future.add_done_callback(task_done_callback)
        et = time.time()
        tosleep = time_between_requests - (et - st)
        if tosleep > 0:
            time.sleep(tosleep)

    return results_queue


def endpoint_evaluation(ep_config):
    client = openai.AsyncOpenAI(
        base_url=ep_config["api_base"], api_key=ep_config["api_key"]
    )
    loop = asyncio.new_event_loop()

    if ep_config["model"] is None:
        async def get_model():
            async for model in client.models.list():
                ep_config["model"] = model.id
                break
        loop.run_until_complete(get_model())

    for _ in range(args.warmup):
        loop.run_until_complete(endpoint_evaluation_request(client, ep_config))

    if args.qps is not None:
        num_results_per_round = math.ceil(args.qps * args.round_duration)
        query_results = []
        elts = []
        results_queue = queue.Queue()
        stop_event = threading.Event()
        threading.Thread(
            target=endpoint_evaluation_qps,
            args=(client, ep_config, results_queue, stop_event),
        ).start()

        st = time.time()
        try:
            while True:
                round_results = []
                round_start = time.time()
                while time.time() - round_start < args.round_duration:
                    try:
                        result = results_queue.get(timeout=0.1)
                        round_results.append(result)
                    except queue.Empty:
                        pass
                query_results.extend(round_results)
                et = time.time()
                elapsed_time = et - st
                elts.append(elapsed_time)
                st = et
                
                # Calculate actual QPS
                actual_qps = len(round_results) / elapsed_time
                
                results_analysis(
                    query_results,
                    elts,
                    ep_config["model"],
                    qps=args.qps,
                    actual_qps=actual_qps,
                    json_output=args.json_output,
                )
                query_results = []
                elts = []
        finally:
            stop_event.set()
    else:
        for concur_requests in args.concur_requests:
            query_results = []
            elts = []
            for _ in range(args.rounds):
                st = time.time()
                results = []
                while time.time() - st < args.round_duration:
                    round_results = loop.run_until_complete(
                        endpoint_evaluation_round(client, concur_requests, ep_config)
                    )
                    results.extend(round_results)
                query_results.extend(results)
                et = time.time()
                elt = et - st
                elts.append(elt)
            results_analysis(
                query_results,
                elts,
                ep_config["model"],
                concur_requests,
                json_output=args.json_output,
            )


def results_analysis(
    query_results, elts, model, concur_requests=None, qps=None, actual_qps=None, json_output=None
):
    print("-------------------------")
    if json_output:
        json_output_f = open(json_output, "a")

    df = pd.DataFrame(
        query_results,
        columns=[
            "valid",
            "ttft",
            "total_time",
            "tokens_in",
            "tokens_out",
            "cause",
        ],
    )
    cdf = df[df.valid != "Exception"].copy()
    if len(cdf) > 0:
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        table.add_column("Min")
        table.add_column("P50")
        table.add_column("P90")
        table.add_column("P95")
        table.add_column("P99")
        table.add_column("Max")

        if json_output:
            json_record = {}

        cdf["tokens_per_s"] = cdf.tokens_out / cdf.total_time
        mean_tokens_in = int(cdf["tokens_in"].mean())
        mean_tokens_out = int(cdf["tokens_out"].mean())

        s_per_output_token = (cdf["total_time"] - cdf["ttft"]) / (cdf["tokens_out"] - 1)

        total_input_tokens = cdf['tokens_in'].sum()
        total_output_tokens = cdf['tokens_out'].sum()
        total_time_minutes = sum(elts) / 60
        input_tokens_per_minute = total_input_tokens / total_time_minutes
        output_tokens_per_minute = total_output_tokens / total_time_minutes

        title = f"{model}\n("
        if concur_requests is not None:
            title += f"concurrency={concur_requests}, "
        if qps is not None:
            # if qps is integer, show it as integer, otherwise show it as float
            title += f"target_qps={int(qps) if int(qps) == qps else qps}, "
        if actual_qps is not None:
            title += f"actual_qps={actual_qps:.2f}, "
        title += f"input_tokens={mean_tokens_in}, output_tokens={mean_tokens_out})"
        table.title = title

        if json_output:
            if concur_requests is not None:
                json_record["concurrency"] = concur_requests
            if qps is not None:
                json_record["target_qps"] = qps
            if actual_qps is not None:
                json_record["actual_qps"] = actual_qps
            json_record["input_tokens"] = mean_tokens_in
            json_record["output_tokens"] = mean_tokens_out
            json_record["model"] = model
            json_record["input_tokens_per_minute"] = input_tokens_per_minute
            json_record["output_tokens_per_minute"] = output_tokens_per_minute

        def show_metric(name, unit, val):
            table.add_row(
                f"{name}({unit})",
                f"{val.min():.3f}",
                f"{val.quantile(0.5):.3f}",
                f"{val.quantile(0.9):.3f}",
                f"{val.quantile(0.95):.3f}",
                f"{val.quantile(0.99):.3f}",
                f"{val.max():.3f}",
            )
            if json_output:
                json_record[name] = {
                    "unit": unit,
                    "min": val.min(),
                    "p50": val.quantile(0.5),
                    "p90": val.quantile(0.9),
                    "p95": val.quantile(0.95),
                    "p99": val.quantile(0.99),
                    "max": val.max(),
                }

        show_metric("Latency", "s", cdf["total_time"])
        show_metric("Throughput", "tokens/s", cdf["tokens_per_s"])
        show_metric("TTFT", "s", cdf["ttft"])
        show_metric("TPOT", "ms", s_per_output_token * 1000)
        show_metric("Input Tokens per Minute", "tokens/min", pd.Series([input_tokens_per_minute]))
        show_metric("Output Tokens per Minute", "tokens/min", pd.Series([output_tokens_per_minute]))

        console.print(table)

    def error_analysis(df):
        # Group exceptions based on exceptions cause
        if args.validate:
            exceptions = df[df.valid.isin(["Mismatch", "Exception"])]
        else:
            exceptions = df[df.valid == "Exception"]
        exceptions_by_cause = Counter()
        # Ideally we should group by some error code
        for cause in exceptions["cause"]:
            exceptions_by_cause[cause] += 1

        if exceptions_by_cause:
            print("Exceptions by cause:")
            for cause, count in exceptions_by_cause.items():
                print(f" - {count}: {cause}")

            if json_output:
                json_record["exceptions"] = {}
                for cause, count in exceptions_by_cause.items():
                    json_record["exceptions"][cause] = count

    error_analysis(df)
    print("-------------------------")

    if json_output:
        json.dump(json_record, json_output_f)
        json_output_f.write("\n")
        json_output_f.close()


def read_tokens_to_list(file_path):
    with open(file_path, 'r') as file:
        tokens_list = file.read().strip().splitlines()
    return tokens_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="model name",
    )
    parser.add_argument(
        "--num-digits", type=int, default=200, help="number of digits for mismatch search"
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="sleep between rounds of requests (to deal with rate limiting)",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=1,
        help="number of requests to use for warmup",
    )
    parser.add_argument(
        "-c",
        "--concur-requests",
        default="10",
        help="number of concurrent requests, use comma to separate multiple values",
    )
    parser.add_argument(
        "-r", "--rounds", type=int, default=4, help="number of rounds of requests"
    )
    parser.add_argument(
        "-q",
        "--qps",
        type=float,
        default=None,
        help=(
            "number of requests per second. if set, overrides --concur-requests and"
            " --rounds"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=384,
        help="Upper limit on the number of returned tokens to prevent 'runaway LLMs'.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=117,
        help="Random seed to standardize results. By default fully random.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8080/v1/",
        help="API base url",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="a" * 32,
        help="API key",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Whether to validate the results",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="tell me a very very long story.",
        help="User prompt to send to the model",
    )
    parser.add_argument(
        "-pl",
        "--prompt-length",
        default=None,
        type=int,
        help="Length of the prompt to send to the model (overrides prompt)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="",
        help=(
            "If set, the file that contains the prompt text. Prompt file takes"
            " precedence over prompt."
        ),
    )
    parser.add_argument(
        "--use-chat",
        action="store_true",
        default=False,
        help="Whether to use the chat endpoint",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=False,
        help="If set, the file to save the results in json format",
    )
    parser.add_argument(
        "--round-duration",
        type=int,
        default=60,
        help="Duration of each round in seconds (default: 60)",
    )
    parser.add_argument(
        "--random-tokens",
        type=int,
        default=0,
        help="Number of random tokens after the prompt",
    )
    args = parser.parse_args()
    endpoint_config = {}
    if args.random_seed >= 0:
        random.seed(args.random_seed)
    endpoint_config["api_base"] = args.api_base
    endpoint_config["api_key"] = args.api_key
    endpoint_config["model"] = args.model
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            args.prompt = f.read()
    elif args.prompt_length:
        args.prompt = get_prompt_of_length(args.prompt_length)
    concur_requests = []
    for c in args.concur_requests.split(","):
        try:
            c_int = int(c)
        except ValueError:
            print(f"concurent requests must be integers, got {c}")
            sys.exit(1)

        if c_int <= 0:
            print(f"concurent requests must be positive integers, got {c}")
            sys.exit(1)

        concur_requests.append(c_int)

    args.concur_requests = concur_requests

    args.random_token_list = read_tokens_to_list('english_words.txt')
    
    # Endpoint evaluation
    endpoint_evaluation(endpoint_config)
