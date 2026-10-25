#!/usr/bin/env python3
"""Small local chat UI for the repository's held-out-validated steering vectors."""
import argparse
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
VECTOR_MANIFEST = Path(__file__).with_name("confirmed_vectors.json")
SYSTEM_PROMPT = "You are a helpful assistant. Respond naturally and clearly."
# Support both `python playground/steered_chat.py` from the repository root and
# `python /path/to/playground/steered_chat.py` from another working directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Virtue Games · Steering Chat</title><style>
:root{font:15px/1.5 system-ui,-apple-system,sans-serif;color:#20242a;background:#f4f5f7}*{box-sizing:border-box}body{margin:0;display:grid;grid-template-columns:300px 1fr;min-height:100vh}aside{background:#fff;border-right:1px solid #ddd;padding:24px 20px}main{max-width:1000px;width:100%;margin:auto;padding:28px;display:flex;flex-direction:column;height:100vh}h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:24px 0 8px}small,.muted{color:#67717e}.model{font:13px ui-monospace,monospace;overflow-wrap:anywhere}.vector{border:1px solid #ddd;border-radius:10px;padding:12px;margin:12px 0}.vector label{display:flex;align-items:center;gap:9px;font-weight:650}.vector input[type=range]{width:100%;margin:12px 0 0}.value{float:right;font:12px ui-monospace,monospace;color:#555}.row{display:flex;justify-content:space-between;align-items:center;margin-top:8px}.toolbar button,button{border:1px solid #ccd1d8;border-radius:8px;background:white;padding:8px 12px;cursor:pointer}button:hover{background:#f0f3f6}#messages{flex:1;overflow:auto;padding:14px 0}.msg{white-space:pre-wrap;overflow-wrap:anywhere;border-radius:12px;padding:12px 15px;margin:10px 0;max-width:85%}.user{background:#dceaff;margin-left:auto}.assistant{background:white;border:1px solid #e0e2e5}.error{background:#ffe4e4;color:#8c1e1e}.composer{display:flex;gap:10px;align-items:end}.composer textarea{flex:1;resize:vertical;min-height:52px;max-height:180px;border:1px solid #cbd0d7;border-radius:10px;padding:12px;font:inherit}.composer button{height:52px;background:#20242a;color:#fff;padding:0 22px}.status{min-height:24px;margin:5px 0;color:#687380}.controls{display:grid;grid-template-columns:1fr 100px;gap:10px;margin-top:14px}.controls input{width:100%}.toggle{display:flex;align-items:center;gap:6px;font-size:13px}.footer{font-size:12px;margin-top:20px;color:#777}@media(max-width:760px){body{grid-template-columns:1fr}aside{border-right:0;border-bottom:1px solid #ddd;padding:17px}main{height:75vh;padding:18px}.vector{display:inline-block;width:48%;vertical-align:top;margin:8px 1% 0 0}.controls{grid-template-columns:1fr 80px}}
</style></head><body><aside><h1>Virtue Games</h1><div class="muted">Steering chat</div><h2>Model</h2><div id="model" class="model">Loading…</div><div id="device" class="muted"></div><h2>Steering directions</h2><div class="muted">Toggle a direction and set its raw strength. Zero leaves that direction off.</div><div id="vectors"></div><div id="no-vectors" class="muted"></div><div class="row"><label class="toggle"><input id="thinking" type="checkbox"> Enable model thinking</label></div><div class="footer">Runs on this computer at localhost. Directions appear only when a compatible validated checkpoint is selected.</div></aside>
<main><div class="row"><div><h1>Chat</h1><div class="muted">Try a direction, compare its effect, then reset the conversation.</div></div><div class="toolbar"><button id="clear">New chat</button></div></div><div id="messages"></div><div class="status" id="status"></div><form id="form" class="composer"><textarea id="input" placeholder="Write a message… (Enter to send, Shift+Enter for a new line)" autofocus></textarea><button id="send">Send</button></form></main>
<script>
let config, history=[];const $=s=>document.querySelector(s);const messages=$('#messages');
function bubble(role,text,err=false){const d=document.createElement('div');d.className='msg '+(err?'error':role);d.textContent=text;messages.appendChild(d);messages.scrollTop=messages.scrollHeight}
async function init(){const r=await fetch('/api/config');config=await r.json();$('#model').textContent=config.model+' @ '+config.revision;$('#device').textContent='Device: '+config.device;
 for(const v of config.vectors){const box=document.createElement('div');box.className='vector';box.innerHTML=`<label><input type="checkbox" class="enabled"> ${v.label}</label><small>${v.direction}; validation sign agreed on ${(v.validation_positive_family_fraction*100).toFixed(0)}% of ${v.validation_families} families</small><div class="controls"><input class="strength" type="range" min="-2" max="2" step="0.1" value="0"><span class="value">0.0</span></div>`;box.querySelector('.strength').addEventListener('input',e=>box.querySelector('.value').textContent=Number(e.target.value).toFixed(1));box.dataset.id=v.id;$('#vectors').appendChild(box)}if(!config.vectors.length)$('#no-vectors').textContent='No validated vectors match this checkpoint. Chat is available without steering.' }
function settings(){return [...document.querySelectorAll('.vector')].map(b=>({id:b.dataset.id,enabled:b.querySelector('.enabled').checked,alpha:Number(b.querySelector('.strength').value)}))}
$('#clear').onclick=()=>{history=[];messages.replaceChildren();$('#status').textContent='New conversation.'};
$('#form').onsubmit=async e=>{e.preventDefault();const text=$('#input').value.trim();if(!text)return;$('#input').value='';history.push({role:'user',content:text});bubble('user',text);$('#send').disabled=true;$('#status').textContent='Generating…';try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history,steering:settings(),thinking:$('#thinking').checked})});const data=await r.json();if(!r.ok)throw new Error(data.error||'Request failed');history.push({role:'assistant',content:data.reply});bubble('assistant',data.reply);$('#status').textContent=data.steering.length?'Active: '+data.steering.map(x=>`${x.label} ${x.alpha>0?'+':''}${x.alpha}`).join(', '):'Unsteered reply';}catch(err){history.pop();bubble('assistant',err.message,true);$('#status').textContent='Could not generate reply.'}finally{$('#send').disabled=false;$('#input').focus()}};
$('#input').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();$('#form').requestSubmit()}};init().catch(e=>$('#model').textContent='Could not load UI settings: '+e.message);
</script></body></html>'''


class SteeringChat:
    def __init__(self, model_name, revision, device, dtype, max_tokens):
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("Install the chat dependencies with: pip install -r requirements-transformers.txt && pip install transformer-lens==3.9.0") from exc
        from config import RunConfig
        from llm import load_model

        self.torch = torch
        self.lock = threading.Lock()
        manifest = json.loads(VECTOR_MANIFEST.read_text(encoding="utf-8"))
        compatible = model_name == manifest["model"] and revision == manifest["revision"]
        self.config = RunConfig(model=model_name, revision=revision, backend="transformers",
            device=device, dtype=dtype, thinking="off", max_tokens=max_tokens,
            questions=ROOT / "integrity" / "questions.json")
        self.model_context = load_model(self.config)
        self.llm = self.model_context.__enter__()
        self.bridge = None
        self.vectors = {}
        self.vector_info = []
        if compatible:
            try:
                from safetensors.torch import load_file
                from transformer_lens.model_bridge import TransformerBridge
            except ImportError as exc:
                self.close()
                raise SystemExit("Install steering dependencies with: pip install -r requirements-activations.txt") from exc
            self.bridge = TransformerBridge.boot_transformers(model_name, hf_model=self.llm.model,
                tokenizer=self.llm.tokenizer, device=self.llm.model.device, dtype=self.llm.model.dtype,
                revision=revision)
            self.bridge.eval()
        for entry in (manifest["vectors"] if compatible else []):
            path = (VECTOR_MANIFEST.parent / entry["file"]).resolve()
            report_path = (VECTOR_MANIFEST.parent / entry["report"]).resolve()
            if not path.is_file() or not report_path.is_file():
                raise FileNotFoundError(f"Validated {entry['label']} vector or report is missing: {path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            identity = report.get("capture_identity", {})
            if identity.get("model") != model_name or identity.get("resolved_revision") != revision:
                raise ValueError(f"{entry['label']} report model/revision does not match the selected checkpoint")
            tensors = load_file(str(path), device="cpu")
            if entry["tensor"] not in tensors:
                if len(tensors) != 1:
                    raise ValueError(f"Expected tensor {entry['tensor']} in {path}; found {list(tensors)}")
                vector = next(iter(tensors.values()))
            else:
                vector = tensors[entry["tensor"]]
            vector = vector.reshape(-1).to(dtype=torch.float32)
            self.vectors[entry["id"]] = vector
            self.vector_info.append({k: entry[k] for k in ("id", "label", "validation_positive_family_fraction", "validation_families", "direction")})
        self.device = str(self.llm.model.device)
        self.manifest = manifest

    def chat(self, messages, steering, thinking=False):
        if not isinstance(messages, list) or not messages or len(messages) > 40:
            raise ValueError("Send between 1 and 40 user/assistant messages")
        clean = []
        for row in messages:
            if not isinstance(row, dict) or row.get("role") not in ("user", "assistant") or not isinstance(row.get("content"), str):
                raise ValueError("Messages must contain only user or assistant text")
            clean.append({"role": row["role"], "content": row["content"][:12000]})
        active = []
        weighted = []
        for item in steering if isinstance(steering, list) else []:
            if not isinstance(item, dict) or item.get("id") not in self.vectors:
                continue
            alpha = max(-2.0, min(2.0, float(item.get("alpha", 0))))
            if item.get("enabled") and alpha:
                weighted.append((item["id"], alpha, self.vectors[item["id"]]))
                label = next(v["label"] for v in self.vector_info if v["id"] == item["id"])
                active.append({"id": item["id"], "label": label, "alpha": alpha})
        with self.lock:
            if weighted:
                if self.bridge is None:
                    raise ValueError("No validated steering direction matches this checkpoint")
                hook_name = self.manifest["hook"]

                def add_residual(value, hook):
                    if value.ndim != 3:
                        raise ValueError("Expected a batch of residual-stream activations")
                    result = value.clone()
                    addition = torch.zeros(value.shape[-1], device=value.device, dtype=value.dtype)
                    for _, alpha, vector in weighted:
                        if vector.numel() != value.shape[-1]:
                            raise ValueError("Direction width does not match the selected model")
                        addition.add_(vector.to(device=value.device, dtype=value.dtype), alpha=alpha)
                    result[:, -1, :] += addition
                    return result

                context = self.bridge.hooks(fwd_hooks=[(hook_name, add_residual)])
            else:
                context = nullcontext()
            with context:
                template_kwargs = {} if thinking else {"enable_thinking": False}
                full_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *clean]
                # Apply the template to the actual conversation including the fixed system instruction.
                tokenized = self.llm.tokenizer.apply_chat_template(full_messages, tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt",
                    **template_kwargs).to(self.llm.model.device)
                n_ctx = min(self.config.n_ctx, int(getattr(self.llm.model.config, "max_position_embeddings", self.config.n_ctx)))
                if tokenized["input_ids"].shape[-1] + self.config.max_tokens > n_ctx:
                    raise ValueError(f"Conversation is too long for the {n_ctx}-token context. Start a new chat.")
                generation = self.torch.random.fork_rng(devices=[self.llm.model.device.index or 0] if self.llm.model.device.type == "cuda" else [])
                with generation, self.torch.inference_mode():
                    output = self.llm.model.generate(**tokenized, max_new_tokens=self.config.max_tokens,
                        do_sample=False, use_cache=True,
                        pad_token_id=(self.llm.tokenizer.pad_token_id if self.llm.tokenizer.pad_token_id is not None
                                      else self.llm.tokenizer.eos_token_id),
                        eos_token_id=self.llm.tokenizer.eos_token_id)
                answer = self.llm.tokenizer.decode(output[0, tokenized["input_ids"].shape[-1]:], skip_special_tokens=True)
        return answer.strip(), active

    def close(self):
        self.model_context.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--revision", default="1cfa9a7208912126459214e8b04321603b3df60c")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = SteeringChat(args.model, args.revision, args.device, args.dtype, args.max_tokens)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            print("[http] " + fmt % values)

        def send(self, status, value, content_type="application/json; charset=utf-8"):
            body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                return self.send(200, HTML, "text/html; charset=utf-8")
            if self.path == "/api/config":
                return self.send(200, {"model": app.manifest["model"], "revision": app.manifest["revision"], "device": app.device, "vectors": app.vector_info})
            self.send(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/chat":
                return self.send(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100_000:
                    raise ValueError("Request is empty or too large")
                data = json.loads(self.rfile.read(length))
                reply, active = app.chat(data.get("messages"), data.get("steering", []), data.get("thinking", False))
                self.send(200, {"reply": reply, "steering": active})
            except Exception as exc:
                self.send(400, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Steering chat ready: http://127.0.0.1:{args.port}")
    print(f"Loaded {len(app.vector_info)} held-out-validated directions on {app.device}; press Ctrl+C to stop.")
    try:
        if not args.no_browser:
            threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping steering chat.")
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
