# ops/notify.py
import os, time, requests, sys
from contextlib import contextmanager
from urllib.parse import urlencode
from typing import Callable, Optional, Dict, Any

SLACK = os.getenv("NOTIFY_SLACK_WEBHOOK", "").strip()
HC_START = os.getenv("HEALTHCHECKS_STEP_START", "").strip()
HC_SUCCESS = os.getenv("HEALTHCHECKS_STEP_SUCCESS", "").strip()
HC_FAIL = os.getenv("HEALTHCHECKS_STEP_FAIL", "").strip()

DEFAULT_MUTE = set(
    s.strip().lower() for s in os.getenv("NOTIFY_STEP_MUTE", "").split(",") if s.strip()
)

def _post_json(url, payload):
    try: requests.post(url, json=payload, timeout=8)
    except Exception: pass

def _get(url, params=None):
    try:
        if params: url = f"{url}?{urlencode(params)}"
        requests.get(url, timeout=8)
    except Exception: pass

def notify(status: str, title: str, body: str = "", log_tail: Optional[str] = None):
    if SLACK:
        att = {"color": {"start":"#888","ok":"#2EB67D","fail":"#E01E5A"}.get(status,"#888"),
               "blocks":[
                   {"type":"header","text":{"type":"plain_text","text":title}},
                   {"type":"section","text":{"type":"mrkdwn","text":body or ""}}
               ]}
        if log_tail:
            att["blocks"].append({"type":"section","text":{"type":"mrkdwn","text":"```\n"+log_tail+"\n```"}})
        _post_json(SLACK, {"attachments":[att]})
    if status == "start" and HC_START:   _get(HC_START)
    if status == "ok"     and HC_SUCCESS: _get(HC_SUCCESS)
    if status == "fail"   and HC_FAIL:    _get(HC_FAIL, {"msg": (body or "")[:400]})

def read_tail(path: Optional[str], n: int = 60) -> Optional[str]:
    if not path: return None
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size=f.tell(); block=4096; data=b""
            while size>0 and data.count(b"\n")<=n:
                r=min(block,size); size-=r; f.seek(size); data=f.read(r)+data
        return "\n".join(data.decode("utf-8","replace").splitlines()[-n:])
    except Exception:
        return None

# messages spec can be str OR callable(ctx)->str
MessageSpec = Optional[Callable[[Dict[str, Any]], str] | str]

def _render(spec: MessageSpec, ctx: Dict[str, Any], default: str) -> str:
    if spec is None:
        return default
    return spec(ctx) if callable(spec) else str(spec)

@contextmanager
def notify_step(
    title: str,
    log_path: Optional[str] = None,
    messages: Optional[Dict[str, MessageSpec]] = None,
    mute: Optional[set[str] | list[str]] = None,   # <- NEW
):
    """
    messages keys (all optional): 'start', 'ok', 'fail'  (str or callable ctx->str)
    mute: iterable of statuses to suppress, e.g. {'start','ok'} for fail-only
    ctx fields: title, log_path, started_at, duration_s, exit_code, exc
    """
    messages = messages or {}
    mute = set((m.lower() for m in (mute or DEFAULT_MUTE)))
    t0 = time.time()

    ctx: Dict[str, Any] = {
        "title": title, "log_path": log_path, "started_at": t0,
        "duration_s": None, "exit_code": None, "exc": None
    }

    # START
    if "start" not in mute:
        notify("start", title, body=_render(messages.get("start"), ctx, f"Starting: {title}"))

    try:
        yield

    except SystemExit as e:
        code = int(getattr(e, "code", 0) or 0)
        if code != 0:
            ctx.update(exit_code=code, exc=None, duration_s=int(time.time()-t0))
            if "fail" not in mute:
                notify(
                    "fail", f"{title} failed",
                    body=_render(messages.get("fail"), ctx, f"Exit code: {code}"),
                    log_tail=read_tail(log_path),
                )
        raise

    except Exception as e:
        ctx.update(exit_code=None, exc=e, duration_s=int(time.time()-t0))
        if "fail" not in mute:
            pretty = f"{type(e).__name__}: {e}"
            notify(
                "fail", f"{title} crashed",
                body=_render(messages.get("fail"), ctx, pretty),
                log_tail=read_tail(log_path),
            )
        raise

    else:
        ctx.update(duration_s=int(time.time()-t0), exit_code=0, exc=None)
        if "ok" not in mute:
            notify(
                "ok", f"{title} succeeded",
                body=_render(messages.get("ok"), ctx, f"Duration: {ctx['duration_s']}s"),
                log_tail=read_tail(log_path),
            )
