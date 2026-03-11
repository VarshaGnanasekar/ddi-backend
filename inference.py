"""
DDI Inference Engine — NaN-safe + extended features
  1. Drug profile stats
  2. Local subgraph HTML (pure Canvas — no external CDN)
  3. Interaction clinical descriptions
  4. Batch prediction
"""

import gc
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import psutil
from model import MemoryEfficientRGCN

# These are overridden at runtime by main.py before DDIPredictor() is called
CHECKPOINT_PATH = os.environ.get("DDI_CHECKPOINT", "best_model.pt")
DATA_PATH       = os.environ.get("DDI_DATA",       "hetero_data_mega.pt")
DRUG_MAP_PATH   = os.environ.get("DDI_DRUG_MAP",   "drug_id_mapping_aux.csv")

DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 3
DEFAULT_DROPOUT    = 0.2
NUM_TOTAL_CLASSES  = 105
MIN_RAM_GB         = 6.0

# ── Clinical descriptions ─────────────────────────────────────────────────────
_KNOWN = {
    "increase_cns_stimulation":          "Heightened CNS stimulation — risk of agitation, insomnia, tremor, or seizure.",
    "decrease_cns_stimulation":          "Reduced CNS stimulation — may diminish therapeutic alerting effect.",
    "increase_cns_depression":           "Additive CNS depression — risk of excessive sedation or respiratory depression.",
    "decrease_cns_depression":           "Antagonism of CNS depressant effects — reduced sedation or anaesthetic effect.",
    "increase_absorption":               "Enhanced GI absorption — elevated plasma concentrations and toxicity risk.",
    "decrease_absorption":               "Reduced GI absorption — lower drug levels and possible treatment failure.",
    "increase_metabolism":               "Accelerated hepatic metabolism — reduced drug exposure, risk of under-dosing.",
    "decrease_metabolism":               "Inhibited hepatic metabolism — elevated drug levels and toxicity risk.",
    "increase_excretion":                "Increased renal excretion — lower steady-state drug concentrations.",
    "decrease_excretion":                "Decreased excretion — drug accumulation and heightened adverse-effect risk.",
    "increase_serum_concentration":      "Elevated serum drug concentration — toxicity monitoring required.",
    "decrease_serum_concentration":      "Reduced serum drug concentration — possible sub-therapeutic effect.",
    "increase_cardiotoxicity":           "Additive cardiotoxicity — QT prolongation and arrhythmia risk.",
    "increase_nephrotoxicity":           "Combined nephrotoxicity — close renal function monitoring advised.",
    "increase_hepatotoxicity":           "Additive hepatotoxicity — liver function monitoring recommended.",
    "increase_neurotoxicity":            "Additive neurotoxicity — risk of peripheral neuropathy or CNS toxicity.",
    "increase_bleeding":                 "Heightened bleeding risk — may require dose adjustment and INR monitoring.",
    "decrease_bleeding":                 "Reduced anticoagulant effect — risk of thrombosis or treatment failure.",
    "increase_anticoagulant_effect":     "Potentiated anticoagulation — increased haemorrhage risk.",
    "decrease_anticoagulant_effect":     "Antagonised anticoagulation — thromboembolic risk.",
    "increase_hypotensive_effect":       "Additive blood pressure lowering — risk of hypotension or syncope.",
    "decrease_hypotensive_effect":       "Blunted antihypertensive response — blood pressure may be inadequately controlled.",
    "increase_hypoglycemic_effect":      "Enhanced blood glucose lowering — risk of hypoglycaemia.",
    "decrease_hypoglycemic_effect":      "Reduced glycaemic control — blood glucose may rise.",
    "increase_immunosuppressive_effect": "Additive immunosuppression — elevated infection and malignancy risk.",
    "increase_serotonergic_effect":      "Combined serotonergic activity — risk of serotonin syndrome.",
    "increase_qt_prolongation":          "Additive QT interval prolongation — torsades de pointes risk.",
    "increase_constipation":             "Combined anticholinergic activity — increased constipation risk.",
    "increase_congestive_heart_failure": "Worsening cardiac function — fluid retention and decompensation risk.",
    "increase_photosensitivity":         "Heightened photosensitivity — advise sun protection measures.",
    "increase_hypertension":             "Elevated blood pressure — cardiovascular monitoring required.",
    "decrease_diuretic_effect":          "Reduced diuretic efficacy — oedema or fluid retention may worsen.",
    "increase_diuretic_effect":          "Potentiated diuresis — risk of electrolyte imbalance and dehydration.",
    "No Interaction":                    "No clinically significant pharmacokinetic or pharmacodynamic interaction predicted.",
}

def describe_interaction(cls: str) -> str:
    if cls in _KNOWN:
        return _KNOWN[cls]
    k = cls.lower().replace(" ", "_")
    if k in _KNOWN:
        return _KNOWN[k]
    if k.startswith("increase_"):
        return f"May enhance {k[9:].replace('_',' ')} — increased monitoring advised."
    if k.startswith("decrease_"):
        return f"May reduce {k[9:].replace('_',' ')} — efficacy monitoring advised."
    return "Potential pharmacological interaction — clinical review recommended."


def _free_ram():
    return psutil.virtual_memory().available / 1e9

def _sanitize(t, name=""):
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(f"[DDI] WARNING {name} has NaN/Inf — replacing with 0")
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


class DDIPredictor:
    _instance = None

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DDI] device={self.device}")

        # ── Read paths fresh at instantiation time (main.py patches module vars) ──
        ckpt_path = CHECKPOINT_PATH
        data_path = DATA_PATH
        map_path  = DRUG_MAP_PATH
        print(f"[DDI] Loading from: {map_path}")

        self.data = torch.load(data_path, map_location="cpu", weights_only=False)
        self.data["drug"].x    = _sanitize(self.data["drug"].x,    "drug.x")
        self.data["protein"].x = _sanitize(self.data["protein"].x, "protein.x")

        # ── Robust drug ID loading ────────────────────────────────────────────
        df  = pd.read_csv(map_path)
        print(f"[DDI] CSV columns: {list(df.columns)} | rows: {len(df)}")

        # Auto-detect the DrugBank ID column
        col = None
        for candidate in ["drugbank_id", "DrugBank_ID", "drugbankid", "drug_id", "id", "ID"]:
            if candidate in df.columns:
                col = candidate
                break
        if col is None:
            col = df.columns[0]   # fallback: first column
        print(f"[DDI] Using column '{col}' for drug IDs")

        # Strip whitespace + drop blanks / NaN
        raw = [str(x).strip() for x in df[col].tolist()]
        self.drug_ids  = [d for d in raw if d and d.lower() != "nan"]
        self.drug2id   = {d: i for i, d in enumerate(self.drug_ids)}
        self.num_drugs = len(self.drug_ids)

        # Case-insensitive lookup map
        self.lower2drug = {d.lower(): d for d in self.drug_ids}
        print(f"[DDI] {self.num_drugs} drugs loaded. Sample: {self.drug_ids[:5]}")

        self.class_names = (
            list(self.data.class_names) + ["No Interaction"]
            if hasattr(self.data, "class_names")
            else [f"Interaction_type_{i}" for i in range(NUM_TOTAL_CLASSES - 1)] + ["No Interaction"]
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {})
        self.hidden_dim = int(args.get("hidden_dim", DEFAULT_HIDDEN_DIM))
        num_layers      = int(args.get("num_layers", DEFAULT_NUM_LAYERS))
        dropout         = float(args.get("dropout",  DEFAULT_DROPOUT))

        self.model = MemoryEfficientRGCN(
            metadata=self.data.metadata(),
            in_dims={"drug": self.data["drug"].x.size(1), "protein": self.data["protein"].x.size(1)},
            hidden_dim=self.hidden_dim, num_layers=num_layers,
            num_output_classes=NUM_TOTAL_CLASSES, dropout=dropout,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.aux_embeddings = {}
        if "aux_embeddings" in ckpt:
            for key in ["atc", "disease", "effect"]:
                if key in ckpt["aux_embeddings"] and key in self.data.node_types:
                    emb = nn.Embedding(self.data[key].num_nodes, self.hidden_dim)
                    emb.load_state_dict(ckpt["aux_embeddings"][key])
                    self.aux_embeddings[key] = emb

        if _free_ram() >= MIN_RAM_GB:
            try:
                self._drug_embeddings = self._full_embed()
                self.embed_mode = "full_graph"
            except RuntimeError as e:
                if "memory" in str(e).lower():
                    gc.collect()
                    self._drug_embeddings = self._proj_embed()
                    self.embed_mode = "proj_only"
                else:
                    raise
        else:
            self._drug_embeddings = self._proj_embed()
            self.embed_mode = "proj_only"

        self._drug_embeddings = _sanitize(self._drug_embeddings, "embeddings").to(self.device)
        self.edge_mlp         = self.model.edge_mlp.to(self.device)

        print("[DDI] Building drug stats...")
        self._stats = self._build_stats()
        print(f"[DDI] Ready | {self.embed_mode} | {self.num_drugs} drugs")

    @torch.no_grad()
    def _full_embed(self):
        x = {"drug": self.data["drug"].x, "protein": self.data["protein"].x}
        for k, e in self.aux_embeddings.items():
            x[k] = e.weight
        h = self.model(x, dict(self.data.edge_index_dict))["drug"].clone()
        gc.collect(); return h

    @torch.no_grad()
    def _proj_embed(self):
        proj, norm, x = self.model.proj["drug"], self.model.norms["drug"], self.data["drug"].x
        out = []
        for s in range(0, x.size(0), 512):
            h = torch.nan_to_num(F.silu(norm(proj(x[s:s+512]))), nan=0.0, posinf=1.0, neginf=-1.0)
            out.append(h)
        gc.collect(); return torch.cat(out, 0)

    def _build_stats(self):
        stats = {i: {"ddi": 0, "protein": 0, "atc": 0, "disease": 0, "effect": 0}
                 for i in range(self.num_drugs)}
        key_map = {
            ("drug","drug"):    "ddi",
            ("drug","protein"): "protein",
            ("drug","atc"):     "atc",
            ("drug","disease"): "disease",
            ("drug","effect"):  "effect",
        }
        for (st, rel, dt), ei in self.data.edge_index_dict.items():
            sk = (st, dt)
            if sk not in key_map: continue
            k = key_map[sk]
            for s in ei[0].tolist():
                if s < self.num_drugs: stats[s][k] += 1
            if k == "ddi":
                for d in ei[1].tolist():
                    if d < self.num_drugs: stats[d][k] += 1
        return stats

    # ── Robust _resolve ───────────────────────────────────────────────────────
    def _resolve(self, name: str):
        name = str(name).strip()

        # 1. Exact match
        if name in self.drug2id:
            return name, self.drug2id[name]

        # 2. Uppercase (handles "db00316" → "DB00316")
        upper = name.upper()
        if upper in self.drug2id:
            return upper, self.drug2id[upper]

        # 3. Case-insensitive
        low = name.lower()
        if low in self.lower2drug:
            c = self.lower2drug[low]
            return c, self.drug2id[c]

        # 4. Zero-pad short numeric suffix e.g. "DB316" → "DB00316"
        if upper.startswith("DB") and upper[2:].isdigit():
            padded = "DB" + upper[2:].zfill(5)
            if padded in self.drug2id:
                return padded, self.drug2id[padded]

        raise ValueError(
            f"Drug '{name}' not found in the {self.num_drugs}-drug dataset. "
            f"Use the search box to find a valid DrugBank ID."
        )

    def _raw_predict(self, ia, ib, top_k):
        p = torch.tensor([[ia, ib]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            lg = self.edge_mlp(torch.cat([self._drug_embeddings[p[:,0]],
                                          self._drug_embeddings[p[:,1]]], 1))
            pr = torch.softmax(torch.clamp(lg, -50, 50), 1).squeeze(0).cpu().numpy()
        if np.isnan(pr).any(): pr = np.ones(NUM_TOTAL_CLASSES) / NUM_TOTAL_CLASSES
        idx = np.argsort(pr)[::-1][:top_k]
        return pr, idx

    def predict_interaction(self, drug_a, drug_b, top_k=5):
        ida, ia = self._resolve(drug_a)
        idb, ib = self._resolve(drug_b)
        pr, idx = self._raw_predict(ia, ib, top_k)
        top_cls = self.class_names[int(np.argmax(pr))]
        return {
            "drug_a_id":      ida,
            "drug_b_id":      idb,
            "top_prediction": top_cls,
            "confidence":     float(pr.max()),
            "is_interaction": top_cls != "No Interaction",
            "top_k":          [(self.class_names[i], float(pr[i])) for i in idx],
            "embed_mode":     self.embed_mode,
            "description":    describe_interaction(top_cls),
        }

    def get_drug_profile(self, drug_id):
        _, idx = self._resolve(drug_id)
        s = self._stats.get(idx, {})
        return {
            "drug_id":         drug_id,
            "ddi_partners":    s.get("ddi", 0),
            "protein_targets": s.get("protein", 0),
            "atc_codes":       s.get("atc", 0),
            "disease_links":   s.get("disease", 0),
            "side_effects":    s.get("effect", 0),
        }

    def get_subgraph_html(self, drug_a_id, drug_b_id, max_nbr=10):
        _, ia = self._resolve(drug_a_id)
        _, ib = self._resolve(drug_b_id)

        nodes      = {}
        edges      = []
        seen_edges = set()

        COLOR = {
            "drug_a":  "#00B894", "drug_b":  "#3B82F6", "shared":  "#8B5CF6",
            "drug":    "#94A3B8", "protein": "#F59E0B", "atc":     "#10B981",
            "disease": "#EF4444", "effect":  "#CBD5E1",
        }
        RADIUS = {
            "drug_a": 22, "drug_b": 22, "shared": 16,
            "drug": 10, "protein": 9, "atc": 8, "disease": 8, "effect": 7,
        }
        rlbl = {
            "ddi": "interacts", "drug_drug": "interacts",
            "dpi": "targets",   "drug_protein": "targets",
            "ppi": "binds",     "has_atc": "ATC",
            "has_disease": "treats", "drug_disease": "treats",
            "has_effect": "causes", "drug_effect": "causes",
            "drug_side_effect": "causes",
        }
        pfx = {"drug": "d", "protein": "p", "atc": "a", "disease": "dis", "effect": "eff"}

        def add_node(nid, label, ntype, title=""):
            if nid not in nodes:
                nodes[nid] = {
                    "id": nid, "label": label,
                    "color": COLOR.get(ntype, COLOR["effect"]),
                    "radius": RADIUS.get(ntype, 7),
                    "title": title or f"{ntype}: {label}",
                }

        def add_edge(fr, to, label, color="#CBD5E1"):
            k = (fr, to, label)
            if k not in seen_edges:
                seen_edges.add(k)
                edges.append({"from": fr, "to": to, "label": label, "color": color})

        sa = self._stats.get(ia, {})
        sb = self._stats.get(ib, {})
        add_node(f"d_{ia}", drug_a_id, "drug_a",
                 f"Drug A | {sa.get('ddi',0)} DDI partners | {sa.get('protein',0)} targets")
        add_node(f"d_{ib}", drug_b_id, "drug_b",
                 f"Drug B | {sb.get('ddi',0)} DDI partners | {sb.get('protein',0)} targets")

        nbrs_a, nbrs_b = set(), set()

        for (st, rel, dt), ei in self.data.edge_index_dict.items():
            rl   = rlbl.get(rel, rel[:8])
            dp   = pfx.get(dt, dt[:2])
            sp   = pfx.get(st, st[:2])
            ecol = "#64748B" if dt == "drug" else "#94A3B8"

            if st == "drug":
                for drug_idx, dnid, nbrs in [(ia, f"d_{ia}", nbrs_a), (ib, f"d_{ib}", nbrs_b)]:
                    pos = (ei[0] == drug_idx).nonzero(as_tuple=True)[0][:max_nbr]
                    for p in pos:
                        di  = int(ei[1][p])
                        nid = f"{dp}_{di}"
                        if dt == "drug":
                            lbl = self.drug_ids[di] if di < self.num_drugs else f"D-{di}"
                            add_node(nid, lbl, "drug",
                                     f"Drug | {self._stats.get(di,{}).get('ddi',0)} partners")
                        else:
                            add_node(nid, f"{dt[:3].upper()}-{di}", dt, f"{dt} {di}")
                        nbrs.add(nid)
                        add_edge(dnid, nid, rl, ecol)

            if dt == "drug":
                for drug_idx, dnid, nbrs in [(ia, f"d_{ia}", nbrs_a), (ib, f"d_{ib}", nbrs_b)]:
                    pos = (ei[1] == drug_idx).nonzero(as_tuple=True)[0][:max_nbr // 2]
                    for p in pos:
                        si2 = int(ei[0][p])
                        nid = f"{sp}_{si2}"
                        lbl = self.drug_ids[si2] if si2 < self.num_drugs else f"D-{si2}"
                        add_node(nid, lbl, "drug", f"Drug {si2}")
                        nbrs.add(nid)
                        add_edge(nid, dnid, rl, "#64748B")

        shared = nbrs_a & nbrs_b
        for nid in shared:
            if nid in nodes:
                nodes[nid]["color"]  = COLOR["shared"]
                nodes[nid]["radius"] = RADIUS["shared"]

        node_list = list(nodes.values())
        nj = json.dumps(node_list)
        ej = json.dumps(edges)
        sc = len(shared)

        import hashlib, time
        uid  = hashlib.md5(f"{drug_a_id}{drug_b_id}{time.time()}".encode()).hexdigest()[:8]
        id_a = json.dumps(f"d_{ia}")
        id_b = json.dumps(f"d_{ib}")

        return f"""<style>
#wrap-{uid}{{position:relative;width:100%;height:480px;border:1px solid #E2E8F0;border-radius:12px;background:#FAFCFF;overflow:hidden;}}
#canvas-{uid}{{display:block;}}
#tip-{uid}{{position:absolute;background:rgba(15,23,42,.88);color:#fff;font:0.7rem/1.4 monospace;padding:.35rem .65rem;border-radius:6px;pointer-events:none;display:none;white-space:nowrap;z-index:9;}}
</style>
<div style="display:flex;gap:.9rem;flex-wrap:wrap;margin-bottom:.45rem;font:0.68rem monospace;color:#64748B;">
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#00B894;margin-right:4px;vertical-align:middle;"></span>{drug_a_id} (A)</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#3B82F6;margin-right:4px;vertical-align:middle;"></span>{drug_b_id} (B)</span>
  <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#8B5CF6;margin-right:4px;vertical-align:middle;"></span>Shared ({sc})</span>
</div>
<div id="wrap-{uid}"><canvas id="canvas-{uid}"></canvas><div id="tip-{uid}"></div></div>
<script>
(function(){{
  var RAW_NODES={nj};var RAW_EDGES={ej};var MAIN_A={id_a};var MAIN_B={id_b};
  var wrap=document.getElementById('wrap-{uid}');
  var canvas=document.getElementById('canvas-{uid}');
  var tip=document.getElementById('tip-{uid}');
  canvas.width=wrap.clientWidth||860;canvas.height=wrap.clientHeight||480;
  var W=canvas.width,H=canvas.height,ctx=canvas.getContext('2d');
  var nodeMap={{}};
  var nodes=RAW_NODES.map(function(n,i){{
    var angle=(2*Math.PI*i)/RAW_NODES.length,r=Math.min(W,H)*0.33;
    var nd={{id:n.id,label:n.label,color:n.color,radius:n.radius,title:n.title,
             x:W/2+r*Math.cos(angle),y:H/2+r*Math.sin(angle),vx:0,vy:0,fixed:false}};
    nodeMap[n.id]=nd;return nd;
  }});
  if(nodeMap[MAIN_A]){{nodeMap[MAIN_A].x=W/2-95;nodeMap[MAIN_A].y=H/2;}}
  if(nodeMap[MAIN_B]){{nodeMap[MAIN_B].x=W/2+95;nodeMap[MAIN_B].y=H/2;}}
  var edges=RAW_EDGES,frame=0,MAX=260;
  function tick(){{
    if(frame++<MAX)requestAnimationFrame(tick);
    for(var i=0;i<nodes.length;i++)for(var j=i+1;j<nodes.length;j++){{
      var a=nodes[i],b=nodes[j],dx=b.x-a.x||.1,dy=b.y-a.y||.1;
      var d2=dx*dx+dy*dy||1,d=Math.sqrt(d2),f=4000/d2,fx=dx/d*f,fy=dy/d*f;
      if(!a.fixed){{a.vx-=fx;a.vy-=fy;}}if(!b.fixed){{b.vx+=fx;b.vy+=fy;}}
    }}
    edges.forEach(function(e){{
      var a=nodeMap[e.from],b=nodeMap[e.to];if(!a||!b)return;
      var dx=b.x-a.x,dy=b.y-a.y,d=Math.sqrt(dx*dx+dy*dy)||1,f=(d-115)*.035;
      var fx=dx/d*f,fy=dy/d*f;
      if(!a.fixed){{a.vx+=fx;a.vy+=fy;}}if(!b.fixed){{b.vx-=fx;b.vy-=fy;}}
    }});
    nodes.forEach(function(n){{
      if(n.fixed)return;n.vx+=(W/2-n.x)*.004;n.vy+=(H/2-n.y)*.004;
      n.x+=n.vx;n.y+=n.vy;n.vx*=.8;n.vy*=.8;
      var pad=n.radius+4;n.x=Math.max(pad,Math.min(W-pad,n.x));n.y=Math.max(pad,Math.min(H-pad,n.y));
    }});
    ctx.clearRect(0,0,W,H);
    edges.forEach(function(e){{
      var a=nodeMap[e.from],b=nodeMap[e.to];if(!a||!b)return;
      ctx.save();ctx.globalAlpha=.55;ctx.strokeStyle=e.color||'#CBD5E1';ctx.lineWidth=1.3;
      ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.restore();
    }});
    nodes.forEach(function(n){{
      if(n.id===MAIN_A||n.id===MAIN_B){{ctx.save();ctx.shadowColor=n.color;ctx.shadowBlur=14;}}
      ctx.beginPath();ctx.arc(n.x,n.y,n.radius,0,2*Math.PI);
      ctx.fillStyle=n.color;ctx.fill();ctx.strokeStyle='rgba(255,255,255,.7)';ctx.lineWidth=1.8;ctx.stroke();
      if(n.id===MAIN_A||n.id===MAIN_B)ctx.restore();
      ctx.font=(n.id===MAIN_A||n.id===MAIN_B?'bold ':'')+((n.radius>=18?11:9))+'px monospace';
      ctx.fillStyle='#1E293B';ctx.textAlign='center';ctx.textBaseline='top';
      var lbl=n.label.length>11?n.label.slice(0,10)+'…':n.label;
      ctx.fillText(lbl,n.x,n.y+n.radius+3);
    }});
  }}
  canvas.addEventListener('mousemove',function(ev){{
    var rc=canvas.getBoundingClientRect(),mx=ev.clientX-rc.left,my=ev.clientY-rc.top,hit=null;
    for(var i=0;i<nodes.length;i++){{var n=nodes[i],dx=mx-n.x,dy=my-n.y;if(dx*dx+dy*dy<=(n.radius+3)*(n.radius+3)){{hit=n;break;}}}}
    if(hit){{tip.style.display='block';tip.style.left=(mx+14)+'px';tip.style.top=(my-8)+'px';tip.textContent=hit.title;}}
    else tip.style.display='none';
  }});
  var drag=null;
  canvas.addEventListener('mousedown',function(ev){{
    var rc=canvas.getBoundingClientRect(),mx=ev.clientX-rc.left,my=ev.clientY-rc.top;
    for(var i=0;i<nodes.length;i++){{var n=nodes[i],dx=mx-n.x,dy=my-n.y;if(dx*dx+dy*dy<=(n.radius+3)*(n.radius+3)){{drag=n;n.fixed=true;if(frame>=MAX){{frame=MAX-40;tick();}}break;}}}}
  }});
  window.addEventListener('mousemove',function(ev){{if(!drag)return;var rc=canvas.getBoundingClientRect();drag.x=ev.clientX-rc.left;drag.y=ev.clientY-rc.top;drag.vx=drag.vy=0;}});
  window.addEventListener('mouseup',function(){{if(drag){{drag.fixed=false;drag=null;}}}});
  tick();
}})();
</script>"""

    def predict_batch(self, pairs, top_k=1):
        rows = []
        for drug_a, drug_b in pairs:
            try:
                r = self.predict_interaction(str(drug_a).strip(), str(drug_b).strip(), top_k=top_k)
                row = {
                    "Drug A":      r["drug_a_id"],
                    "Drug B":      r["drug_b_id"],
                    "Prediction":  r["top_prediction"],
                    "Confidence":  f"{r['confidence']*100:.1f}%",
                    "Interaction": "Yes" if r["is_interaction"] else "No",
                    "Description": r["description"],
                }
                for rank, (cls, prob) in enumerate(r["top_k"], 1):
                    if rank > 1:
                        row[f"Top-{rank}"] = f"{cls} ({prob*100:.1f}%)"
            except ValueError as e:
                row = {"Drug A": drug_a, "Drug B": drug_b, "Prediction": "ERROR",
                       "Confidence": "-", "Interaction": "-", "Description": str(e)}
            rows.append(row)
        return pd.DataFrame(rows)

    def get_drug_list(self):  return sorted(self.drug_ids)
    def get_embed_mode(self): return self.embed_mode

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
