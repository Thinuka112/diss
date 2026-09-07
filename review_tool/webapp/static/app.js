"use strict";

// ---- key translation: profile key names (Tkinter-style) -> browser KeyboardEvent.key
const KEYMAP = { Right:"ArrowRight", Left:"ArrowLeft", Up:"ArrowUp", Down:"ArrowDown",
                 space:" ", BackSpace:"Backspace" };
function toKey(name){ return KEYMAP[name] || name; }

let profile = null;        // {param, prompt, bindings, blind, skip_key, undo_key}
let current = null;        // current image_id on screen
let lastJudged = null;     // last image the reviewer judged (for undo)
let viewer = null;         // OpenSeadragon instance
const $ = (id) => document.getElementById(id);

async function api(path, opts){
  const r = await fetch(path, Object.assign({ headers:{ "Content-Type":"application/json" } }, opts));
  if(!r.ok){ const e = await r.json().catch(()=>({detail:r.statusText})); throw new Error(e.detail || r.statusText); }
  return r.json();
}

function show(view){ for(const v of ["signin","review","done"]) $(v).classList.toggle("hidden", v!==view); }

// ---------- sign-in ----------
async function initSignin(){
  const info = await api("/api/profiles");
  const sel = $("profile"); sel.innerHTML = "";
  for(const p of info.profiles){
    const o = document.createElement("option"); o.value = p.param; o.textContent = p.param; sel.appendChild(o);
  }
  $("profile").closest("label").style.display = info.profiles.length > 1 ? "" : "none";
  $("setMeta").textContent = `${info.images} slides to review`;
  show("signin");
}

async function signin(){
  $("signinErr").textContent = "";
  try{
    const res = await api("/api/signin", { method:"POST", body: JSON.stringify({
      passcode: $("passcode").value, name: $("name").value, profile: $("profile").value }) });
    profile = res.profile;
    $("who").textContent = `${res.name} · ${profile.param}`;
    startReview();
  }catch(e){ $("signinErr").textContent = e.message; }
}

// ---------- review ----------
function ensureViewer(){
  if(viewer) return viewer;
  viewer = OpenSeadragon({
    id: "osd",
    prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/images/",
    showNavigator: true, navigatorPosition: "TOP_RIGHT",
    gestureSettingsMouse: { clickToZoom:false, dblClickToZoom:true, scrollToZoom:true },
    maxZoomPixelRatio: 2.5, animationTime: 0.3, visibilityRatio: 0.9,
  });
  return viewer;
}

function renderButtons(){
  const box = $("buttons"); box.innerHTML = "";
  profile.bindings.forEach((b, i) => {
    const btn = document.createElement("button");
    const keyName = b.key;
    btn.innerHTML = `${b.label}<span class="k">${keyName} · ${i+1}</span>`;
    btn.onclick = () => judge(b.label, keyName);
    box.appendChild(btn);
  });
  // Undo lives in the same row, same styling, so it is as findable as yes/no.
  const u = document.createElement("button");
  u.id = "undoBtn";
  u.innerHTML = `Undo<span class="k">${profile.undo_key || "BackSpace"}</span>`;
  u.onclick = undo;
  box.appendChild(u);
}

// small transient message (used only to explain undo, so it never fails silently)
let toastTimer = null;
function toast(msg){
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 3200);
}

async function startReview(){
  ensureViewer();
  $("prompt").textContent = profile.prompt;
  renderButtons();
  show("review");
  await loadNext();
}

async function loadNext(){
  const n = await api("/api/next");
  updateProgress(n.progress);
  if(n.done){ $("doneProgress").textContent = summary(n.progress); show("done"); return; }
  current = n.image_id;
  viewer.open({ type:"image", url: n.image_url });
}

function updateProgress(p){
  $("progress").textContent = `${p.judged_by_me} / ${p.total} judged`;
}
function summary(p){ return `You judged ${p.judged_by_me} of ${p.total} slides.`; }

async function judge(label, key){
  if(!current) return;
  try{
    await api("/api/judge", { method:"POST", body: JSON.stringify({ image_id: current, judgment: label, key }) });
    lastJudged = current;
    await loadNext();
  }catch(e){ alert(e.message); }
}

async function undo(){
  if(!lastJudged){
    // Explain instead of silently doing nothing (the old behaviour confused reviewers).
    toast("Nothing to undo — no answer given yet since this page opened.");
    return;
  }
  try{
    await api("/api/undo", { method:"POST", body: JSON.stringify({ image_id: lastJudged }) });
    current = lastJudged; lastJudged = null;      // bring that image back to re-judge
    viewer.open({ type:"image", url:"/api/image/" + current });
  }catch(e){ alert(e.message); }
}

async function skip(){ await loadNext(); }

// ---------- keyboard ----------
document.addEventListener("keydown", (ev) => {
  if($("review").classList.contains("hidden")) return;
  if(!profile) return;
  // number keys 1-4 map to bindings
  const num = parseInt(ev.key, 10);
  if(num >= 1 && num <= profile.bindings.length){ ev.preventDefault(); const b = profile.bindings[num-1]; judge(b.label, b.key); return; }
  for(const b of profile.bindings){ if(ev.key === toKey(b.key)){ ev.preventDefault(); judge(b.label, b.key); return; } }
  if(ev.key === toKey(profile.skip_key)){ ev.preventDefault(); skip(); return; }
  if(ev.key === toKey(profile.undo_key)){ ev.preventDefault(); undo(); return; }
});

// ---------- wiring ----------
$("signinBtn").onclick = signin;
$("name").addEventListener("keydown", e => { if(e.key==="Enter") signin(); });
$("passcode").addEventListener("keydown", e => { if(e.key==="Enter") $("name").focus(); });
$("skipBtn").onclick = skip;
// (undoBtn is created inside renderButtons, wired there)
$("signout").onclick = async () => { await api("/api/signout", { method:"POST" }); location.reload(); };
$("reviewMore").onclick = () => startReview();

// ---------- boot ----------
(async () => {
  try{
    const w = await api("/api/whoami");
    if(w.signed_in){ profile = w.profile; $("who").textContent = `${w.name} · ${profile.param}`; startReview(); }
    else { await initSignin(); }
  }catch(e){ await initSignin(); }
})();
