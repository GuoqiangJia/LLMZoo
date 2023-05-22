import argparse
import datetime
import json
import os
import time
import uuid

import gradio as gr
import requests

from llmzoo.deploy.webapp.constants import LOGDIR
from llmzoo.deploy.webapp.gradio_css import code_highlight_css
from llmzoo.deploy.webapp.gradio_patch import Chatbot as grChatbot
from llmzoo.deploy.webapp.standalone.model_worker import ModelWorker
from llmzoo.deploy.webapp.utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
)
from llmzoo.utils import (
    get_default_conv_template,
    SeparatorStyle,
)

logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "fastchat Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)

enable_moderation = False
models = []
worker = None

priority = {
    "vicuna-13b": "aaa",
    "koala-13b": "aab",
    "oasst-sft-1-pythia-12b": "aac",
    "dolly-v2-12b": "aad",
    "chatglm-6b": "aae",
    "stablelm-tuned-alpha-7b": "aaf",
}


def set_global_vars(controller_url_, enable_moderation_, models_):
    global controller_url, enable_moderation, models
    controller_url = controller_url_
    enable_moderation = enable_moderation_
    models = models_


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list(controller_url):
    ret = requests.post(controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo_single(url_params):
    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(value=model, visible=True)

    state = None
    return (
        state,
        dropdown_update,
        gr.Chatbot.update(visible=True),
        gr.Textbox.update(visible=True),
        gr.Button.update(visible=True),
        gr.Row.update(visible=True),
        gr.Accordion.update(visible=True),
    )


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return load_demo_single(url_params)


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = None
    return (state, [], "") + (disable_btn,) * 5


def add_text(state, text, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")

    if state is None:
        state = get_default_conv_template().copy()

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5
    if enable_moderation:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg) + (
                no_change_btn,
            ) * 5

    text = text[:1536]  # Hard cut-off
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code


def http_bot(state, model_selector, temperature, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
    temperature = float(temperature)
    max_new_tokens = int(max_new_tokens)

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        new_state = get_default_conv_template(model_name).copy()
        new_state.conv_id = uuid.uuid4().hex
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Construct prompt
    prompt = state.get_prompt()

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else None,
    }
    logger.info(f"==== request ====\n{pload}")

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        response = worker.generate_stream(pload)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"].strip()
                    output = post_process_code(output)
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.02)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg + f" (error_code: 4)"
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "gen_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


block_css = (
        code_highlight_css
        + """
pre {
    white-space: pre-wrap;       /* Since CSS 2.1 */
    white-space: -moz-pre-wrap;  /* Mozilla, since 1999 */
    white-space: -pre-wrap;      /* Opera 4-6 */
    white-space: -o-pre-wrap;    /* Opera 7 */
    word-wrap: break-word;       /* Internet Explorer 5.5+ */
}
#notice_markdown th {
    display: none;
}
"""
)


def build_single_model_ui():
    state = gr.State()

    with gr.Row(elem_id="model_selector_row", visible=False):
        model_selector = gr.Dropdown(
            choices=models,
            value=models[0] if len(models) > 0 else "",
            interactive=True,
            show_label=False,
            visible=False
        ).style(container=False)

    chatbot = grChatbot(elem_id="chatbot", visible=False).style(height=550)
    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
            ).style(container=False)
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False)

    with gr.Row(visible=False) as button_row:
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        max_output_tokens = gr.Slider(
            minimum=0,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # Register listeners
    btn_list = [regenerate_btn, clear_btn]

    regenerate_btn.click(regenerate, state, [state, chatbot, textbox] + btn_list).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

    model_selector.change(clear_history, None, [state, chatbot, textbox] + btn_list)

    textbox.submit(
        add_text, [state, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens],
        [state, chatbot] + btn_list,
    )
    send_btn.click(
        add_text, [state, textbox], [state, chatbot, textbox] + btn_list
    ).then(
        http_bot,
        [state, model_selector, temperature, max_output_tokens],
        [state, chatbot] + btn_list,
    )

    return state, model_selector, chatbot, textbox, send_btn, button_row, parameter_row


def build_demo():
    with gr.Blocks(
            title="ChatGPT like Chatbot",
            theme=gr.themes.Base(),
            css=block_css,
    ) as demo:
        url_params = gr.JSON(visible=False)

        (
            state,
            model_selector,
            chatbot,
            textbox,
            send_btn,
            button_row,
            parameter_row,
        ) = build_single_model_ui()

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [
                    state,
                    model_selector,
                    chatbot,
                    textbox,
                    send_btn,
                    button_row,
                    parameter_row,
                ],
                _js=get_window_url_params,
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model-path", type=str, default="FreedomIntelligence/phoenix-inst-chat-7b")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true", help="Enable content moderation")

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

    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = [args.model_path]
    worker = ModelWorker(model_path=args.model_path, device=args.device, num_gpus=args.num_gpus,
                         max_gpu_memory=args.max_gpu_memory, load_8bit=args.load_8bit, load_4bit=args.load_4bit)
    set_global_vars(args.moderate, models)

    logger.info(args)
    demo = build_demo()
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )
