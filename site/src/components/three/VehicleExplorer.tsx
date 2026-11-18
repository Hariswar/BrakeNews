import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import VehicleScene, { type CameraCommand } from "./VehicleScene";
import ComplaintTimeline from "./ComplaintTimeline";
import { SYSTEMS, displayValue } from "../../lib/visualization/mapping";
import type { CategoryKey, DisplayMode, ModelFamily, VisualizationData } from "../../lib/visualization/types";

export interface ExplorerProps { data: VisualizationData | null; variant: "home" | "vehicle" | "report"; }
const NUMBER: Partial<Record<CategoryKey, number>> = { engine: 1, brakes: 2, restraints: 3, steering: 4, body: 5 };
const MODES: { value: DisplayMode; label: string; explanation: string }[] = [
  { value: "neutral", label: "Neutral", explanation: "Select a numbered region or use the system list. A highlight represents a complaint category, not a confirmed defective part." },
  { value: "volume", label: "Complaint counts", explanation: "Brighter blue indicates the highest individual problem count listed for a system. Counts are owner reports, not failure rates, severity or a safety score." },
  { value: "flags", label: "Statistical flags", explanation: "Blue means at least one listed problem has a statistical flag in the current snapshot. Flags are not official recalls and do not establish a defect." },
  { value: "rising", label: "Rising reports", explanation: "Blue means at least one listed problem has an existing surge signal. Reporting changes alone do not establish a change in failure rates." },
  { value: "recalls", label: "Matched recalls", explanation: "Blue means an existing record links a listed problem to an official recall. Check your VIN for applicability; uncolored regions are not a safety assurance." },
];
const labelMonth = (date: string | null) => date ? new Date(`${date.slice(0, 7)}-01T12:00:00Z`).toLocaleDateString("en-US", { month: "long", year: "numeric", timeZone: "UTC" }) : "date unavailable";

export default function VehicleExplorer({ data, variant }: ExplorerProps) {
  const id = useId();
  const host = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [family, setFamily] = useState<ModelFamily>("sedan");
  const featured = data?.problems.find((p) => p.id === data.featuredProblem);
  const [selected, setSelected] = useState<CategoryKey | null>(featured?.category ?? null);
  const [mode, setMode] = useState<DisplayMode>("neutral");
  const [command, setCommand] = useState<CameraCommand>({ kind: "reset", serial: 0 });
  const [reducedMotion, setReducedMotion] = useState(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [rotating, setRotating] = useState(false);
  const [visible, setVisible] = useState(true);
  const [sceneVisible, setSceneVisible] = useState(true);
  const [historyId, setHistoryId] = useState<number | null>(null);
  const [month, setMonth] = useState<{ month: string; complaints: number; maximum: number } | null>(null);
  const stopRotation = useCallback(() => setRotating(false), []);
  const move = useCallback((kind: CameraCommand["kind"]) => { setRotating(false); setCommand((c) => ({ kind, serial: c.serial + 1 })); }, []);
  const select = useCallback((key: CategoryKey) => { setSelected(key); setHistoryId(null); setMonth(null); move("reset"); }, [move]);
  useEffect(() => {
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    const change = () => { setReducedMotion(preference.matches); if (preference.matches) setRotating(false); };
    preference.addEventListener("change", change);
    return () => preference.removeEventListener("change", change);
  }, []);
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => { setSceneVisible(entry.isIntersecting); if (!entry.isIntersecting) setRotating(false); });
    if (stage.current) observer.observe(stage.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    let intersecting = true;
    const update = () => { const active = intersecting && !document.hidden; setVisible(active); if (!active) setRotating(false); };
    const observer = new IntersectionObserver(([entry]) => { intersecting = entry.isIntersecting; update(); }, { threshold: .05 });
    if (host.current) observer.observe(host.current);
    document.addEventListener("visibilitychange", update);
    return () => { observer.disconnect(); document.removeEventListener("visibilitychange", update); };
  }, []);
  useEffect(() => {
    if (variant !== "home") return;
    const focus = (e: Event) => { if ((e.target as HTMLElement).closest("[data-picker]")) move("inspect"); };
    document.addEventListener("focusin", focus);
    return () => document.removeEventListener("focusin", focus);
  }, [variant, move]);
  const history = data?.problems.find((p) => p.id === historyId);
  const histories = data?.problems.filter((p) => p.monthly.length > 0) ?? [];
  const values = useMemo(() => Object.fromEntries(SYSTEMS.map((s) => [s.key, displayValue(data?.problems ?? [], s.key, mode)])) as Record<CategoryKey, number>, [data, mode]);
  const tones = useMemo(() => {
    if (history && month) return { [history.category]: month.complaints / month.maximum };
    const max = Math.max(1, ...Object.values(values));
    return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, value / max]));
  }, [history, month, values]);
  const system = SYSTEMS.find((s) => s.key === selected);
  const problems = data?.problems.filter((p) => !selected || p.category === selected) ?? [];
  const showHistory = (value: string) => {
    const problem = value ? data?.problems.find((p) => p.id === Number(value)) : undefined;
    setMonth(null); setHistoryId(problem?.id ?? null);
    if (problem) setSelected(problem.category);
    move("reset");
  };
  return <div ref={host} className={`vx-explorer vx-explorer-${variant}`}>
    <div className="vx-visual">
    <div ref={stage} className="vx-stage" tabIndex={0} role="group" aria-label="3D vehicle view" aria-describedby={`${id}-keys`} onKeyDown={(e) => {
      if (e.target !== e.currentTarget) return;
      const key = ({ ArrowLeft: "left", ArrowRight: "right", "+": "in", "=": "in", "-": "out", Home: "reset" } as Record<string, CameraCommand["kind"]>)[e.key];
      if (key) { e.preventDefault(); if (key === "reset") setSelected(null); move(key); }
    }}>
      <VehicleScene family={family} selected={history || mode !== "neutral" ? null : selected} focusCategory={selected} tones={tones} onSelect={select} decorative={!data} command={command} reducedMotion={reducedMotion} rotating={rotating} visible={visible && sceneVisible} onInteract={stopRotation} />
      <span className="vx-stage-caption">Generic {family === "suv" ? "SUV" : family} · illustrative</span>
    </div>
    <div className="vx-controls">
      <div className="vx-camera" role="group" aria-label="Camera controls">
        <button type="button" onClick={() => move("left")} aria-label="Rotate left">↶</button>
        <button type="button" onClick={() => move("right")} aria-label="Rotate right">↷</button>
        <button type="button" onClick={() => move("in")} aria-label="Zoom in">+</button>
        <button type="button" onClick={() => move("out")} aria-label="Zoom out">−</button>
        <button type="button" onClick={() => { setSelected(null); move("reset"); }}>Reset view</button>
        {!reducedMotion && <button type="button" aria-pressed={rotating} onClick={() => setRotating(!rotating)}>{rotating ? "Stop rotation" : "Auto rotate"}</button>}
      </div>
      <p id={`${id}-keys`} className="vx-small">Drag to rotate. Focus the view and use ← / →, + / −, or Home. The buttons work without dragging.</p>
      <div className="vx-family" role="group" aria-label="Illustrative body style">
        <span className="vx-small">Illustration:</span>
        {(["sedan", "suv", "pickup"] as ModelFamily[]).map((f) => <button type="button" key={f} aria-pressed={family === f} onClick={() => { setFamily(f); move("reset"); }}>{f === "suv" ? "SUV" : f === "sedan" ? "Sedan" : "Pickup"}</button>)}
      </div>
      <p className="vx-small">Generic geometry, not a model of {data ? `the ${data.vehicle}` : "a specific vehicle"}. Choose a body style for illustration only.</p>
    </div>
    </div>
    {data && <div className="vx-data">
      <div className="vx-source"><strong>{data.vehicle}</strong><span>Published data through {labelMonth(data.snapshot)}</span></div>
      <div className="vx-mode">
        <label htmlFor={`${id}-mode`}>Color by</label>
        <select id={`${id}-mode`} value={mode} disabled={!!history} onChange={(e) => setMode(e.target.value as DisplayMode)}>{MODES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}</select>
      </div>
      <p className="vx-legend">{history ? "Monthly replay is active. End replay to return to the current snapshot’s colors." : MODES.find((m) => m.value === mode)?.explanation}</p>
      <div className="vx-system-list" role="group" aria-label="Explore reported systems">
        <button type="button" aria-pressed={!selected} onClick={() => { setSelected(null); setHistoryId(null); setMonth(null); move("reset"); }}>All listed problems</button>
        {SYSTEMS.map((s) => <button type="button" key={s.key} aria-pressed={selected === s.key} onClick={() => select(s.key)}>
          <span className="vx-system-name">{NUMBER[s.key] && <span className="vx-number">{NUMBER[s.key]}</span>}{s.label}</span>
          {s.location === "unlocalized" && <span className="vx-system-meta">Text only · no exact location</span>}
          {mode !== "neutral" && !history && <span className="vx-system-meta">{mode === "volume" ? data.problems.some((p) => p.category === s.key && p.complaints !== null) ? `Largest listed count: ${values[s.key].toLocaleString("en-US")}` : "No count available" : values[s.key] ? MODES.find((m) => m.value === mode)?.label : "None listed in this snapshot"}</span>}
        </button>)}
      </div>
      <div className="vx-selection" aria-live="polite" aria-atomic="true"><h3>{system?.label ?? "All listed problems"}</h3><p className="vx-small">{system?.note ?? "Choose a broad system to narrow the list. All evidence remains available in the page’s original tables and reports."}</p><span className="vx-small">{problems.length} listed {problems.length === 1 ? "problem" : "problems"}</span></div>
      {histories.length > 0 && <div className="vx-history-choice">
        <label htmlFor={`${id}-history`}>Replay an available problem history</label>
        <select id={`${id}-history`} value={historyId ?? ""} onChange={(e) => showHistory(e.target.value)}><option value="">Current snapshot · replay off</option>{histories.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
        {history && <button type="button" onClick={() => { setHistoryId(null); setMonth(null); }}>End replay</button>}
      </div>}
      {history && <ComplaintTimeline key={history.id} problem={history} active={visible} reducedMotion={reducedMotion} onMonth={setMonth} />}
      {data.symptom && <details className="vx-symptom"><summary>Reported symptom · AI-written summary</summary><p>{data.symptom}</p><a href="/method/#checks">How AI reports are checked</a></details>}
      <div className="vx-problems">
        {problems.length === 0 ? <p>No problems in this system are listed in this snapshot. This does not establish that the system is free of defects.</p> : problems.map((p) => <details key={p.id} className="vx-problem" open={problems.length === 1}>
          <summary><span className="vx-problem-title">{p.name}</span><span className="vx-problem-summary">{p.complaints === null ? "Complaint count unavailable" : `${p.complaints.toLocaleString("en-US")} owner complaints`}{p.flagged && " · Flagged"}{p.recallIds.length > 0 && " · Matched recall"}</span></summary>
          <h4><a href={p.evidenceHref}>{p.name}</a></h4>
          <dl>
            <div><dt>Owner complaints</dt><dd>{p.complaints === null ? "Unavailable" : p.complaints.toLocaleString("en-US")}</dd></div>
            <div><dt>Statistical flag</dt><dd>{p.flagged === null ? "Unavailable" : p.flagged ? "Flagged" : "Not flagged"}</dd></div>
            <div><dt>Reporting surge</dt><dd>{p.rising === null ? "Unavailable" : p.rising ? "Surge signal" : "No surge signal"}</dd></div>
            <div><dt>Matched official recalls</dt><dd>{p.recallIds.length ? p.recallIds.map((r) => <a key={r} href={`https://www.nhtsa.gov/recalls?nhtsaId=${encodeURIComponent(r)}`}>#{r}</a>) : "None linked in this record"}</dd></div>
            {p.recallRisk !== null && <div><dt>Existing 24-month recall estimate</dt><dd>{new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 }).format(p.recallRisk)}<span className="vx-small">Model estimate for this problem; not a safety rating or guarantee. <a href="/method/#model">About the estimate</a></span></dd></div>}
          </dl>
          <div className="vx-evidence"><a href={p.evidenceHref}>Detailed evidence →</a>{p.reportHref && <a href={p.reportHref}>AI-written investigation →</a>}</div>
        </details>)}
      </div>
      <p className="vx-small vx-disclosure">Current values are from the published snapshot, including during replay. Small complaint counts can vary sharply; these voluntary reports are not failure rates. Complaint categories are not confirmed defects. <a href={data.sourceHref}>NHTSA source</a> · <a href="https://www.nhtsa.gov/recalls">Check your VIN</a> · <a href="/method/">Method and limitations</a></p>
    </div>}
  </div>;
}
