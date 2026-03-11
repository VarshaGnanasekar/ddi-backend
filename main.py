"""
DDI Intelligence Platform — FastAPI Backend
Drop this in your ddi-backend/ folder alongside inference.py, model.py
"""

import os, json, time, asyncio, sqlite3, hashlib, secrets, gc
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt

# ── Model download on startup ─────────────────────────────────────────────────
HF_TOKEN    = os.getenv("HF_TOKEN", "")
HF_REPO     = os.getenv("HF_REPO",  "your-username/ddi-models")
MODEL_DIR       = "/tmp/ddi_models"
REQUIRED_FILES  = ["best_model.pt", "hetero_data_mega.pt", "drug_id_mapping_aux.csv"]
OPTIONAL_FILES  = []   # drug_names.csv not needed — DrugBank IDs used directly

def download_models():
    """Download model files from HuggingFace Hub if not already present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        for fname in REQUIRED_FILES + OPTIONAL_FILES:
            dest = os.path.join(MODEL_DIR, fname)
            optional = fname in OPTIONAL_FILES
            if not os.path.exists(dest):
                print(f"[STARTUP] Downloading {fname}...")
                try:
                    hf_hub_download(
                        repo_id=HF_REPO,
                        filename=fname,
                        repo_type="dataset",
                        token=HF_TOKEN or None,
                        local_dir=MODEL_DIR,
                    )
                    print(f"[STARTUP] ✅ {fname} ready")
                except Exception as e:
                    if optional:
                        print(f"[STARTUP] ⚠️ Optional file {fname} not found, skipping.")
                    else:
                        print(f"[STARTUP] ❌ Download failed: {e}")
                        raise
            else:
                print(f"[STARTUP] ✅ {fname} already cached")
        # Free memory after downloads
        gc.collect()
    except Exception as e:
        print(f"[STARTUP] ❌ Download failed: {e}")
        raise

# Patch the paths in inference.py to point to /tmp/ddi_models
os.environ["DDI_MODEL_DIR"] = MODEL_DIR

# ── SQLite DB ─────────────────────────────────────────────────────────────────
DB_PATH = "/tmp/ddi_history.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            pw_hash  TEXT NOT NULL,
            created  TEXT NOT NULL
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            drug_a      TEXT NOT NULL,
            drug_b      TEXT NOT NULL,
            prediction  TEXT NOT NULL,
            confidence  REAL NOT NULL,
            interaction INTEGER NOT NULL,
            description TEXT NOT NULL,
            created     TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    con.commit(); con.close()

# ── JWT ───────────────────────────────────────────────────────────────────────
JWT_SECRET  = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGO    = "HS256"
JWT_EXPIRY  = 7  # days
security    = HTTPBearer(auto_error=False)

def make_token(user_id: int, username: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRY)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[dict]:
    if not credentials:
        return None
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGO])
    except Exception:
        return None

def require_auth(payload: dict = Depends(verify_token)):
    if not payload:
        raise HTTPException(status_code=401, detail="Authentication required")
    return payload

# ── Lifespan (startup/shutdown) ───────────────────────────────────────────────
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("[STARTUP] Downloading model files...")
    download_models()
    print("[STARTUP] Initialising DDIPredictor...")
    init_db()

    # Set env vars BEFORE importing inference — these are read at __init__ time
    os.environ["DDI_CHECKPOINT"] = os.path.join(MODEL_DIR, "best_model.pt")
    os.environ["DDI_DATA"]       = os.path.join(MODEL_DIR, "hetero_data_mega.pt")
    os.environ["DDI_DRUG_MAP"]   = os.path.join(MODEL_DIR, "drug_id_mapping_aux.csv")

    # Also patch module-level vars (belt-and-suspenders)
    import inference as inf_module
    inf_module.CHECKPOINT_PATH = os.environ["DDI_CHECKPOINT"]
    inf_module.DATA_PATH       = os.environ["DDI_DATA"]
    inf_module.DRUG_MAP_PATH   = os.environ["DDI_DRUG_MAP"]

    from inference import DDIPredictor
    DDIPredictor._instance = None   # force fresh init with correct paths
    predictor = DDIPredictor.get_instance()
    print("[STARTUP] ✅ Ready!")
    yield
    print("[SHUTDOWN] Cleaning up...")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="DDI Intelligence API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Drug name mapping — not used, IDs are canonical ─────────────────────────
_drug_name_map: dict  = {}
_name_to_id_map: dict = {}

def load_drug_names():
    pass  # drug_names.csv not required; DrugBank IDs are used directly

# ── Pydantic models ───────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class PredictRequest(BaseModel):
    drug_a: str
    drug_b: str
    top_k: int = 7

class BatchPair(BaseModel):
    drug_a: str
    drug_b: str

class BatchRequest(BaseModel):
    pairs: List[BatchPair]
    top_k: int = 1

class MultiDrugRequest(BaseModel):
    drugs: List[str]

# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/api/auth/register")
def register(req: RegisterRequest):
    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.execute(
            "INSERT INTO users (username, pw_hash, created) VALUES (?,?,?)",
            (req.username.strip(), pw_hash, datetime.utcnow().isoformat())
        )
        user_id = cur.lastrowid
        con.commit(); con.close()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already taken")
    token = make_token(user_id, req.username)
    return {"token": token, "username": req.username}

@app.post("/api/auth/login")
def login(req: LoginRequest):
    pw_hash = hashlib.sha256(req.password.encode()).hexdigest()
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT id, username FROM users WHERE username=? AND pw_hash=?",
        (req.username.strip(), pw_hash)
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = make_token(row[0], row[1])
    return {"token": token, "username": row[1]}

# ── Drug search ───────────────────────────────────────────────────────────────
@app.get("/api/drugs")
def get_drugs(q: str = "", limit: int = 20):
    """Search drugs by DrugBank ID or common name."""
    all_ids = predictor.get_drug_list()
    if not q:
        results = all_ids[:limit]
    else:
        ql = q.lower()
        # Match by DB ID
        by_id   = [d for d in all_ids if ql in d.lower()]
        # Match by common name
        by_name = [_name_to_id_map[n] for n in _name_to_id_map if ql in n]
        combined = list(dict.fromkeys(by_id + by_name))[:limit]
        results  = combined if combined else by_id[:limit]

    return [
        {"id": did, "name": _drug_name_map.get(did, did)}
        for did in results
    ]

@app.get("/api/drugs/count")
def drug_count():
    return {"count": len(predictor.get_drug_list()), "classes": 105}

# ── Predict (normal) ──────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict(req: PredictRequest, payload: dict = Depends(verify_token)):
    try:
        res = predictor.predict_interaction(req.drug_a, req.drug_b, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Save to history if authenticated
    if payload:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            """INSERT INTO history
               (user_id, drug_a, drug_b, prediction, confidence, interaction, description, created)
               VALUES (?,?,?,?,?,?,?,?)""",
            (int(payload["sub"]), res["drug_a_id"], res["drug_b_id"],
             res["top_prediction"], res["confidence"],
             1 if res["is_interaction"] else 0,
             res["description"], datetime.utcnow().isoformat())
        )
        con.commit(); con.close()

    # Enrich with drug names
    res["drug_a_name"] = _drug_name_map.get(res["drug_a_id"], res["drug_a_id"])
    res["drug_b_name"] = _drug_name_map.get(res["drug_b_id"], res["drug_b_id"])
    return res

# ── Predict (streaming SSE) ───────────────────────────────────────────────────
@app.post("/api/predict/stream")
async def predict_stream(req: PredictRequest, payload: dict = Depends(verify_token)):
    """
    Server-Sent Events stream. Sends progress steps then the result.
    Frontend shows a live progress bar as the GNN runs.
    """
    async def event_stream():
        steps = [
            (0.15, "Loading drug embeddings..."),
            (0.35, "Running R-GCN message passing..."),
            (0.60, "Computing relational convolutions..."),
            (0.80, "Decoding interaction classes..."),
            (0.95, "Ranking predictions..."),
        ]
        for pct, msg in steps:
            yield f"data: {json.dumps({'type':'progress','pct':pct,'msg':msg})}\n\n"
            await asyncio.sleep(0.3)

        try:
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: predictor.predict_interaction(req.drug_a, req.drug_b, top_k=req.top_k)
            )
            if payload:
                con = sqlite3.connect(DB_PATH)
                con.execute(
                    """INSERT INTO history
                       (user_id, drug_a, drug_b, prediction, confidence, interaction, description, created)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (int(payload["sub"]), result["drug_a_id"], result["drug_b_id"],
                     result["top_prediction"], result["confidence"],
                     1 if result["is_interaction"] else 0,
                     result["description"], datetime.utcnow().isoformat())
                )
                con.commit(); con.close()

            result["drug_a_name"] = _drug_name_map.get(result["drug_a_id"], result["drug_a_id"])
            result["drug_b_name"] = _drug_name_map.get(result["drug_b_id"], result["drug_b_id"])
            yield f"data: {json.dumps({'type':'result','data':result})}\n\n"

        except ValueError as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Batch predict ─────────────────────────────────────────────────────────────
@app.post("/api/predict/batch")
def predict_batch(req: BatchRequest, payload: dict = Depends(verify_token)):
    results = []
    for pair in req.pairs:
        try:
            r = predictor.predict_interaction(pair.drug_a, pair.drug_b, top_k=req.top_k)
            r["drug_a_name"] = _drug_name_map.get(r["drug_a_id"], r["drug_a_id"])
            r["drug_b_name"] = _drug_name_map.get(r["drug_b_id"], r["drug_b_id"])
            results.append({"status": "ok", **r})
        except ValueError as e:
            results.append({"status": "error", "drug_a": pair.drug_a,
                            "drug_b": pair.drug_b, "message": str(e)})
    return {"results": results, "total": len(results)}

# ── Multi-drug matrix ─────────────────────────────────────────────────────────
@app.post("/api/predict/matrix")
def predict_matrix(req: MultiDrugRequest, payload: dict = Depends(verify_token)):
    import itertools
    drugs   = list(dict.fromkeys(req.drugs))  # dedupe, preserve order
    results = []
    for da, db in itertools.combinations(drugs, 2):
        try:
            r = predictor.predict_interaction(da, db, top_k=3)
            r["drug_a_name"] = _drug_name_map.get(r["drug_a_id"], r["drug_a_id"])
            r["drug_b_name"] = _drug_name_map.get(r["drug_b_id"], r["drug_b_id"])
            results.append({"status": "ok", **r})
        except ValueError as e:
            results.append({"status": "error", "drug_a": da, "drug_b": db, "message": str(e)})
    return {"drugs": drugs, "pairs": results, "total": len(results)}

# ── Drug profile ──────────────────────────────────────────────────────────────
@app.get("/api/drugs/{drug_id}/profile")
def drug_profile(drug_id: str):
    try:
        p = predictor.get_drug_profile(drug_id)
        p["name"] = _drug_name_map.get(drug_id, drug_id)
        return p
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ── Subgraph (JSON for Three.js) ──────────────────────────────────────────────
@app.get("/api/subgraph")
def subgraph(drug_a: str, drug_b: str, max_nbr: int = 10):
    """Returns nodes + edges as JSON for the React Three.js graph."""
    try:
        _, ia = predictor._resolve(drug_a)
        _, ib = predictor._resolve(drug_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    nodes, edges, seen = {}, [], set()
    sa, sb = predictor._stats.get(ia, {}), predictor._stats.get(ib, {})

    def add_node(nid, label, ntype, meta=""):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "label": label, "type": ntype, "meta": meta}

    def add_edge(fr, to, rel):
        k = (fr, to, rel)
        if k not in seen:
            seen.add(k)
            edges.append({"source": fr, "target": to, "relation": rel})

    add_node(f"d_{ia}", _drug_name_map.get(drug_a, drug_a), "drug_a",
             f"{sa.get('ddi',0)} DDI partners · {sa.get('protein',0)} targets")
    add_node(f"d_{ib}", _drug_name_map.get(drug_b, drug_b), "drug_b",
             f"{sb.get('ddi',0)} DDI partners · {sb.get('protein',0)} targets")

    pfx  = {"drug":"d","protein":"p","atc":"a","disease":"dis","effect":"eff"}
    rlbl = {"ddi":"interacts","drug_drug":"interacts","dpi":"targets",
            "drug_protein":"targets","ppi":"binds","has_atc":"ATC",
            "has_disease":"treats","drug_disease":"treats",
            "has_effect":"causes","drug_effect":"causes","drug_side_effect":"causes"}

    nbrs_a, nbrs_b = set(), set()
    for (st, rel, dt), ei in predictor.data.edge_index_dict.items():
        rl = rlbl.get(rel, rel[:8])
        dp = pfx.get(dt, dt[:2])
        if st == "drug":
            for drug_idx, dnid, nbrs in [(ia, f"d_{ia}", nbrs_a),(ib, f"d_{ib}", nbrs_b)]:
                pos = (ei[0] == drug_idx).nonzero(as_tuple=True)[0][:max_nbr]
                for p in pos:
                    di  = int(ei[1][p])
                    nid = f"{dp}_{di}"
                    lbl = (predictor.drug_ids[di] if dt=="drug" and di < predictor.num_drugs
                           else f"{dt[:3].upper()}-{di}")
                    if dt == "drug":
                        lbl = _drug_name_map.get(lbl, lbl)
                    add_node(nid, lbl, dt)
                    nbrs.add(nid)
                    add_edge(dnid, nid, rl)

    shared = nbrs_a & nbrs_b
    for nid in shared:
        if nid in nodes:
            nodes[nid]["type"] = "shared"

    return {
        "nodes":  list(nodes.values()),
        "edges":  edges,
        "shared": len(shared),
        "drug_a": {"id": drug_a, "name": _drug_name_map.get(drug_a, drug_a)},
        "drug_b": {"id": drug_b, "name": _drug_name_map.get(drug_b, drug_b)},
    }

# ── History ───────────────────────────────────────────────────────────────────
@app.get("/api/history")
def get_history(limit: int = 50, payload: dict = Depends(require_auth)):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        """SELECT drug_a, drug_b, prediction, confidence, interaction, description, created
           FROM history WHERE user_id=? ORDER BY created DESC LIMIT ?""",
        (int(payload["sub"]), limit)
    ).fetchall()
    con.close()
    return [
        {"drug_a": r[0], "drug_b": r[1], "prediction": r[2],
         "confidence": r[3], "is_interaction": bool(r[4]),
         "description": r[5], "created": r[6],
         "drug_a_name": _drug_name_map.get(r[0], r[0]),
         "drug_b_name": _drug_name_map.get(r[1], r[1])}
        for r in rows
    ]

@app.delete("/api/history")
def clear_history(payload: dict = Depends(require_auth)):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM history WHERE user_id=?", (int(payload["sub"]),))
    con.commit(); con.close()
    return {"ok": True}

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model":  predictor.embed_mode if predictor else "not_loaded",
        "drugs":  len(predictor.get_drug_list()) if predictor else 0,
    }

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
