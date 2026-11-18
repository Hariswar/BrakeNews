import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { VisualProblem } from "../../lib/visualization/types";

interface Props {
  problem: VisualProblem;
  active: boolean;
  reducedMotion: boolean;
  onMonth: (month: { month: string; complaints: number; maximum: number } | null) => void;
}
const monthLabel = (date: string) => new Date(`${date.slice(0, 7)}-01T12:00:00Z`).toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });

export default function ComplaintTimeline({ problem, active, reducedMotion, onMonth }: Props) {
  const id = useId();
  const host = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(true);
  const months = useMemo(() => [...problem.monthly].sort((a, b) => a.month.localeCompare(b.month)), [problem]);
  const maximum = Math.max(1, ...months.map((m) => m.complaints));
  const [index, setIndex] = useState(months.length - 1);
  const [playing, setPlaying] = useState(false);
  const current = months[index];
  useEffect(() => {
    onMonth(current ? { ...current, maximum } : null);
    return () => onMonth(null);
  }, [current, maximum, onMonth]);
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting));
    if (host.current) observer.observe(host.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => { if (!active || !visible || reducedMotion) setPlaying(false); }, [active, visible, reducedMotion]);
  useEffect(() => {
    if (!playing || !active || !visible || reducedMotion) return;
    const timer = window.setInterval(() => setIndex((i) => {
      if (i >= months.length - 2) { setPlaying(false); return months.length - 1; }
      return i + 1;
    }), 750);
    return () => window.clearInterval(timer);
  }, [playing, active, visible, reducedMotion, months.length]);
  if (!current) return null;
  return <section ref={host} className="vx-timeline" aria-labelledby={`${id}-title`}>
    <h4 id={`${id}-title`}>Monthly complaint replay</h4>
    <p className="vx-small">{problem.name}. Uses this published report’s grouping of complaints, including past months. This is a retrospective view, not a record of predictions issued at the time.</p>
    <div className="vx-timeline-value" aria-live={playing ? "off" : "polite"}>
      <strong>{monthLabel(current.month)}</strong><span>{current.complaints.toLocaleString("en-US")} owner complaints received</span>
    </div>
    <svg viewBox={`0 0 ${Math.max(months.length * 6, 60)} 48`} preserveAspectRatio="none" className="vx-bars" aria-hidden="true">
      {months.map((m, i) => <rect key={m.month} x={i * Math.max(6, 60 / months.length)} y={46 - m.complaints / maximum * 42} width={Math.max(4, 60 / months.length - 2)} height={Math.max(1, m.complaints / maximum * 42)} fill={i === index ? "var(--brand)" : "var(--border-strong)"} />)}
      {problem.events.map((event, i) => {
        const at = months.findIndex((m) => m.month.slice(0, 7) === event.date.slice(0, 7));
        if (at < 0) return null;
        const x = (at + .5) * Math.max(6, 60 / months.length);
        return <path key={i} d={`M${x - 2},0 L${x + 2},0 L${x},4 Z`} fill="var(--ink)" />;
      })}
    </svg>
    <label htmlFor={`${id}-month`} className="vx-small">Choose a month</label>
    <input id={`${id}-month`} type="range" min={0} max={months.length - 1} value={index} aria-valuetext={`${monthLabel(current.month)}: ${current.complaints} complaints`} onChange={(e) => { setPlaying(false); setIndex(Number(e.target.value)); }} />
    <div className="vx-timeline-buttons">
      <button type="button" disabled={index === 0} onClick={() => { setPlaying(false); setIndex((i) => i - 1); }}>Previous month</button>
      {!reducedMotion && <button type="button" aria-pressed={playing} onClick={() => { if (!playing && index === months.length - 1) setIndex(0); setPlaying(!playing); }}>{playing ? "Pause replay" : "Play replay"}</button>}
      <button type="button" disabled={index === months.length - 1} onClick={() => { setPlaying(false); setIndex((i) => i + 1); }}>Next month</button>
    </div>
    <p className="vx-small">Only this problem’s broad system is colored, using its count for the selected month. Other regions have no historical value displayed. Current flags and recall estimates below do not change with the month.</p>
    {problem.events.length > 0 && <details className="vx-events">
      <summary>Signal and matched recall dates · triangle markers</summary>
      <ul>{problem.events.map((event, i) => <li key={`${event.date}-${i}`}>
        <time dateTime={event.date}>{monthLabel(event.date)}</time>{" — "}
        {event.href ? <a href={event.href}>{event.label}</a> : event.label}
        {event.date.slice(0, 7) > current.month.slice(0, 7) && <span> · After selected month</span>}
      </li>)}</ul>
      <p className="vx-small">Signal dates come from the existing retrospective analysis. Recall dates appear only when the existing report matches a campaign and its date is available in the same snapshot.</p>
    </details>}
  </section>;
}
