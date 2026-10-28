"""ThermalMind backend.

Real deployment pipeline: Hugging Face (GGUF) -> S3 model store -> llama.cpp on this host.
Nothing is hardcoded: region comes from the instance, the bucket name from the AWS
account id + region, the model list from live Hugging Face search, telemetry from psutil.
Optional env vars: THERMALMIND_BUCKET, THERMALMIND_DATA_DIR, AWS_REGION, HF_TOKEN, PORT.
"""
import gc
import importlib.util
import json
import os
import re
import shutil
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import Request, urlopen

import boto3
import httpx
import psutil
import uvicorn
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from huggingface_hub import HfApi, hf_hub_url
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("THERMALMIND_DATA_DIR", BASE_DIR / "model_cache"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "state.json"
HF_TOKEN = os.getenv("HF_TOKEN") or None
PORT = int(os.getenv("PORT", "8000"))

hf = HfApi(token=HF_TOKEN)
s3 = None
AWS = {"ok": False, "region": None, "account": None, "bucket": None, "message": "Not initialised"}
STATE = {"status": "idle", "progress": 0, "stage": "Idle", "error": None, "model": None}
LLM = {"obj": None}
LAST = {"latency_ms": None, "tok_s": None}
HISTORY = deque(maxlen=40)
DEPLOY_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()

REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")
FILE_RE = re.compile(r"^[\w.\-/]+\.gguf$", re.I)
QUANT_RE = re.compile(r"(IQ\d[_A-Z0-9]*|Q\d(?:_K)?(?:_[A-Z0-9]+)?|BF16|F16|F32)", re.I)


# ---------- helpers ----------
def set_state(**kw):
    STATE.update(kw)


def quant_of(name):
    found = QUANT_RE.findall(Path(name).stem)
    return found[-1].upper() if found else "unknown"


def check_repo(repo):
    if not REPO_RE.match(repo):
        raise HTTPException(400, "Invalid Hugging Face repo id (expected owner/name).")


def check_file(name):
    if not FILE_RE.match(name) or ".." in name or name.startswith("/"):
        raise HTTPException(400, "Invalid file name (must be a .gguf file in the repo).")


def loaded_bytes():
    m = STATE.get("model")
    return int(m["size_mb"] * 1e6) if m and LLM["obj"] is not None else 0


def ram_budget():
    vm = psutil.virtual_memory()
    return int(min(vm.total * 0.75, vm.available + loaded_bytes()))


def fits(size):
    return size > 0 and size * 1.15 <= ram_budget()


def read_temp():
    try:
        for readings in psutil.sensors_temperatures().values():
            if readings:
                return round(readings[0].current, 1)
    except Exception:
        pass
    return None


def snapshot():
    vm = psutil.virtual_memory()
    return {
        "t": time.strftime("%H:%M:%S"),
        "cpu": psutil.cpu_percent(None),
        "mem": vm.percent,
        "used_gb": round(vm.used / 1e9, 2),
        "total_gb": round(vm.total / 1e9, 2),
        "model": (STATE["model"] or {}).get("name"),
        "latency_ms": LAST["latency_ms"],
        "tok_s": LAST["tok_s"],
        "temp_c": read_temp(),
    }


def sampler():
    psutil.cpu_percent(None)
    while True:
        time.sleep(3)
        HISTORY.append(snapshot())


# ---------- AWS ----------
def detect_region():
    region = (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
              or boto3.session.Session().region_name)
    if region:
        return region
    try:  # EC2 instance metadata (IMDSv2)
        token = urlopen(Request("http://169.254.169.254/latest/api/token", method="PUT",
                                headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}),
                        timeout=2).read().decode()
        return urlopen(Request("http://169.254.169.254/latest/meta-data/placement/region",
                               headers={"X-aws-ec2-metadata-token": token}),
                       timeout=2).read().decode()
    except Exception:
        return None


def ensure_bucket(bucket, region):
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchBucket", "NotFound"):
            raise
    args = {"Bucket": bucket}
    if region != "us-east-1":
        args["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**args)
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True,
        "BlockPublicPolicy": True, "RestrictPublicBuckets": True})


def init_aws():
    global s3
    try:
        region = detect_region()
        if not region:
            raise RuntimeError("Could not detect the AWS region. Set AWS_REGION for the service.")
        account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
        s3 = boto3.client("s3", region_name=region)
        bucket = os.getenv("THERMALMIND_BUCKET") or f"thermalmind-models-{account}-{region}"
        ensure_bucket(bucket, region)
        AWS.update(ok=True, region=region, account=account, bucket=bucket, message="Ready")
    except NoCredentialsError:
        AWS.update(ok=False, message="No AWS credentials. Attach an IAM role with S3 access to this EC2 instance.")
    except (BotoCoreError, ClientError, RuntimeError) as e:
        AWS.update(ok=False, message=f"{type(e).__name__}: {e}")


def s3_has(key):
    try:
        s3.head_object(Bucket=AWS["bucket"], Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


# ---------- deployment pipeline ----------
def file_size(repo, filename):
    info = hf.model_info(repo, files_metadata=True)
    for s in info.siblings or []:
        if s.rfilename == filename:
            return s.size or 0
    raise RuntimeError(f"{filename} was not found in {repo}.")


def download_hf(repo, filename, dest, total, lo, hi):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with httpx.stream("GET", hf_hub_url(repo, filename), headers=headers,
                      follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_bytes(1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                set_state(progress=round(lo + (hi - lo) * done / max(total, 1), 1),
                          stage=f"Downloading from Hugging Face ({done / 1e6:.0f} of {total / 1e6:.0f} MB)")
    tmp.replace(dest)


def transfer_s3(direction, local, key, total, lo, hi):
    done = [0]

    def cb(n):
        done[0] += n
        verb = "Uploading to" if direction == "up" else "Downloading from"
        set_state(progress=round(lo + (hi - lo) * done[0] / max(total, 1), 1),
                  stage=f"{verb} S3 ({done[0] / 1e6:.0f} of {total / 1e6:.0f} MB)")

    if direction == "up":
        s3.upload_file(str(local), AWS["bucket"], key, Callback=cb)
    else:
        s3.download_file(AWS["bucket"], key, str(local), Callback=cb)


def load_model(path):
    if importlib.util.find_spec("llama_cpp") is None:
        raise RuntimeError("llama-cpp-python is not installed in the venv.")
    from llama_cpp import Llama
    with INFER_LOCK:
        LLM["obj"] = None
        gc.collect()
        LLM["obj"] = Llama(model_path=str(path), n_ctx=2048,
                           n_threads=os.cpu_count() or 2, verbose=False)


def run_deploy(repo, filename):
    try:
        set_state(status="deploying", progress=0, stage="Checking resources", error=None)
        size = file_size(repo, filename)
        if not fits(size):
            raise RuntimeError(
                f"{size / 1e6:.0f} MB model will not fit: about {ram_budget() / 1e6:.0f} MB of RAM is "
                "usable. Pick a smaller quant or a bigger instance.")
        local = DATA_DIR / repo.replace("/", "__") / Path(filename).name
        local.parent.mkdir(parents=True, exist_ok=True)
        key = f"models/{repo}/{filename}"
        in_s3 = s3_has(key)
        cached = local.exists() and local.stat().st_size == size
        if not cached and shutil.disk_usage(DATA_DIR).free < size * 1.1:
            raise RuntimeError("Not enough free disk on the instance for this model.")
        if not cached:
            if in_s3:
                transfer_s3("down", local, key, size, 0, 80)
            else:
                download_hf(repo, filename, local, size, 0, 45)
        if not in_s3:
            transfer_s3("up", local, key, size, 45, 85)
        set_state(progress=88, stage="Loading model into memory")
        load_model(local)
        info = {"name": Path(filename).stem, "repo": repo, "file": filename,
                "quant": quant_of(filename), "size_mb": round(size / 1e6),
                "s3_uri": f"s3://{AWS['bucket']}/{key}", "region": AWS["region"],
                "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        STATE_FILE.write_text(json.dumps(info))
        set_state(status="ready", progress=100, stage="Model deployed and serving", model=info)
    except Exception as e:
        set_state(status="error", stage="Deployment failed", error=str(e))
    finally:
        DEPLOY_LOCK.release()


def restore_last_model():
    try:
        info = json.loads(STATE_FILE.read_text())
        path = DATA_DIR / info["repo"].replace("/", "__") / Path(info["file"]).name
        if path.exists():
            set_state(status="deploying", progress=90, stage="Restoring last model")
            load_model(path)
            set_state(status="ready", progress=100, stage="Model deployed and serving", model=info)
    except Exception:
        set_state(status="idle", progress=0, stage="Idle")


# ---------- inference ----------
def generate(messages, max_tokens=256):
    if LLM["obj"] is None:
        raise HTTPException(409, "No model is loaded. Deploy one first.")
    try:
        with INFER_LOCK:
            t0 = time.time()
            out = LLM["obj"].create_chat_completion(messages=messages, max_tokens=max_tokens,
                                                    temperature=0.7)
            dt = time.time() - t0
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    toks = (out.get("usage") or {}).get("completion_tokens", 0)
    LAST.update(latency_ms=int(dt * 1000), tok_s=round(toks / dt, 1) if dt > 0 and toks else None)
    return out["choices"][0]["message"]["content"].strip()


# ---------- API ----------
@asynccontextmanager
async def lifespan(_app):
    init_aws()
    threading.Thread(target=sampler, daemon=True).start()
    threading.Thread(target=restore_last_model, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


class DeployReq(BaseModel):
    repo: str
    filename: str


class ChatReq(BaseModel):
    message: str


class AskReq(BaseModel):
    query: str


@app.get("/")
def index():
    page = BASE_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(404, "index.html must sit next to run_backend.py")
    return FileResponse(page)


@app.get("/api/system")
def system(refresh: int = 0):
    if refresh or not AWS["ok"] and AWS["message"] == "Not initialised":
        init_aws()
    return {"aws": AWS, "llama": importlib.util.find_spec("llama_cpp") is not None,
            "host": {"vcpu": psutil.cpu_count(), "ram_total": psutil.virtual_memory().total,
                     "disk_free": shutil.disk_usage(DATA_DIR).free, "ram_budget": ram_budget()}}


@app.get("/api/hf/search")
def hf_search(q: str):
    try:
        found = hf.list_models(search=q, filter="gguf", sort="downloads", limit=15)
        return [{"id": m.id, "downloads": getattr(m, "downloads", 0),
                 "likes": getattr(m, "likes", 0)} for m in found]
    except Exception as e:
        raise HTTPException(502, f"Hugging Face search failed: {e}")


@app.get("/api/hf/files")
def hf_files(repo: str):
    check_repo(repo)
    try:
        info = hf.model_info(repo, files_metadata=True)
    except Exception as e:
        raise HTTPException(502, f"Could not read {repo}: {e}")
    files = []
    for s in info.siblings or []:
        n = s.rfilename
        if (not n.lower().endswith(".gguf") or "mmproj" in n.lower()
                or re.search(r"-\d{5}-of-\d{5}", n)):
            continue
        files.append({"filename": n, "size": s.size or 0, "quant": quant_of(n),
                      "fits": fits(s.size or 0)})
    files.sort(key=lambda f: f["size"])
    ok = [f for f in files if f["fits"]]
    pick = next((f for f in ok if f["quant"] == "Q4_K_M"), None) or (ok[-1] if ok else None)
    return {"files": files, "recommended": pick["filename"] if pick else None,
            "ram_budget": ram_budget()}


@app.post("/api/deploy")
def deploy(r: DeployReq):
    check_repo(r.repo)
    check_file(r.filename)
    if not AWS["ok"]:
        init_aws()
    if not AWS["ok"]:
        raise HTTPException(503, AWS["message"])
    if not DEPLOY_LOCK.acquire(blocking=False):
        raise HTTPException(409, "A deployment is already running.")
    threading.Thread(target=run_deploy, args=(r.repo, r.filename), daemon=True).start()
    return {"status": "started"}


@app.get("/api/deploy/status")
def deploy_status():
    return STATE


@app.get("/api/models")
def list_models():
    if not AWS["ok"]:
        return {"bucket": None, "items": [], "message": AWS["message"]}
    try:
        items = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=AWS["bucket"], Prefix="models/"):
            for o in page.get("Contents", []):
                items.append({"key": o["Key"], "size": o["Size"],
                              "modified": o["LastModified"].strftime("%Y-%m-%d %H:%M")})
        return {"bucket": AWS["bucket"], "items": items}
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(502, f"S3 listing failed: {e}")


@app.get("/api/telemetry")
def telemetry():
    return {"now": snapshot(), "history": list(HISTORY)[-12:]}


@app.post("/api/chat")
def chat(r: ChatReq):
    text = generate([{"role": "user", "content": r.message}])
    return {"reply": text, **LAST}


@app.post("/api/supervisor")
def supervisor(r: AskReq):
    s = snapshot()
    temp = f"{s['temp_c']} C" if s["temp_c"] is not None else "not available on this instance"
    facts = (f"Host: {psutil.cpu_count()} vCPU, CPU {s['cpu']}%, RAM {s['used_gb']} of {s['total_gb']} GB "
             f"({s['mem']}%). Temperature sensor: {temp}. Deployment: {STATE['status']}, "
             f"model: {s['model'] or 'none'}. S3 bucket: {AWS['bucket'] or 'unavailable'}.")
    if LLM["obj"] is None:
        return {"response": "No model is loaded yet, so these are the live facts only. " + facts}
    msgs = [{"role": "system", "content": "You are a concise infrastructure supervisor. "
             "Answer only from these live facts and say so if they do not cover the question. " + facts},
            {"role": "user", "content": r.query}]
    return {"response": generate(msgs, 200)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
