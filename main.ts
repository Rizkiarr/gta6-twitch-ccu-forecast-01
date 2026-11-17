// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only): shape the day from the comparables.
//
// The house baseline stays flat (every hour = hour 1's print, widening spread with sqrt(h)) and scores
// ~0.387 on the practice. This entry instead LEARNS the typical launch-day hourly shape from the case's
// comparables (past big launches' hourly counts): each comparable's own hourly curve, normalized to its
// own hour-1, gives a relative log-ratio t_h for hour h (1..24). We aggregate that shape across all
// comparables (median for stability, then mean for the spread) and set each hour's five quantiles as a
// log-normal around the case's hour-1 print with that hour-dependent log-spread:
//
//   q_h(tau) = p1 * exp(z_tau * t_h),   t_h = log-relative shape of hour h
//
// z_tau is the standard normal's quantile at the scoring levels (0.05, 0.25, 0.5, 0.75, 0.95). The shape is
// learned from the comparables, so a big launch's evening peak (when the US is awake) and overnight fall are
// reproduced rather than assumed, instead of the baseline's flat scaling.
//
// What the case gives us:
//   - known: hour 1's print (p1), viewers, reads
//   - comparables: each launch's hourly averages (day_start to release day + 3) from SullyGnome
//   - history / history_hourly: the case's own history before hour 1 (the final is the recorder's minute
//     reads; a practice case the historical source's points)
//
// The run is deterministic: no randomness, so two executions are byte for byte identical.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level (half-up rounded). */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Hourly {
  from: string;
  viewers: number;
  channels: number | null;
  reads: number;
}
interface Comparable {
  name: string;
  day_start: string;
  unlock: string;
  platforms: string;
  source: string;
  hourly: Hourly[];
}
interface Case {
  levels: number[];
  known: Array<{ hour: number; from: string; until: string; viewers: number; channels: number; reads: number }>;
  forecast: Array<{ hour: number; from: string; until: string }>;
  history: Array<{ at: string; viewers: number; channels: number | null }>;
  history_hourly: Array<{ from: string; viewers: number; channels: number | null; reads: number }>;
  streams: unknown;
  categories_hourly: unknown;
  comparables: Comparable[];
  notes: string;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The standard normal's quantile at tau, rounded to 3 decimals. */
const zAt = (tau: number): number => {
  const m = tau;
  return Math.round((zFromNormal(m) * 1000) / 1000);
};

// Normal CDF approximation via the standard error function; returns the standard normal quantile.
function zFromNormal(p: number): number {
  const phi = (x: number): number => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  // Geyer-Smith-Blatchett inverse: quantile of standard normal
  const a = Math.abs(p - 0.5);
  const b = Math.sqrt(-2 * Math.log(1 - a * a));
  const d = 0.5 - a - b;
  return d;
}

/** A comparable's hourly curve as log-relative spreads of its hourly mean around its own hour 1: s_h = ln(v_h)/ln(v_1). */
function comparableLogSpread(comp: Comparable): number[] {
  const day = comp.day_start.slice(0, 10);
  const byHour = new Map<number, number[]>();
  for (const p of comp.hourly) {
    if (!p.from.slice(0, 10).startsWith(day)) continue;
    const hh = Number(p.from.slice(11, 13));
    if (!byHour.has(hh)) byHour.set(hh, []);
    byHour.get(hh)!.push(p.viewers);
  }
  const vals: number[] = [];
  for (let h = 1; h <= 24; h++) {
    const a = byHour.get(h);
    if (!a || a.length === 0) continue;
    const v = Math.max(a.reduce((x, y) => x + y, 0), 1) / a.length;
    const v1 = Math.max(byHour.get(1)?.reduce((x, y) => x + y, 0) ?? 1, 1) / (byHour.get(1)?.length ?? 1);
    vals.push(Math.log(v) - Math.log(v1));
  }
  return vals;
}

/** Median or mean of a slice of numbers. */
function agg(xs: number[], mode: 'median' | 'mean'): number[] {
  if (xs.length === 0) return [];
  const s = [...xs].sort((a, b) => a - b);
  if (mode === 'median') {
    const m = Math.floor((s.length - 1) / 2);
    return s.length % 2 ? [s[m], s[m + 1]] : [s[m]];
  }
  return s.reduce((a, b) => a + b, 0) / s.length;
}

/** Aggregate comparable shapes into the target's per-hour log-spread t_h: each comparable's log-ratio of its
 * hourly mean to its own hour 1 (ln(v_h)/ln(v_1)), median per hour then mean over comparables. */
function learnShape(comps: Comparable[]): number[] {
  const slists = comps.map((c) => comparableLogSpread(c));
  const nH = Math.max(...slists.map((r) => r.length));
  const t: number[] = new Array(nH + 1);
  for (let h = 1; h <= 24; h++) {
    const ss = slists.filter((r) => r[h - 1] !== undefined).map((r) => r[h - 1]);
    const med = agg(ss, 'median');
    const a = agg(med, 'mean');
    t[h] = a.length ? a[0] : (med.length ? med[0] : 0);
  }
  // clamp tiny negative log-spreads (a comparable whose hour 1 is tiny) away from the floor.
  for (let h = 1; h <= 24; h++) {
    if (t[h] < -5) t[h] = -5;
  }
  return t;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  if (c.comparables.length === 0) {
    // No comparables (unexpected): fall back to the baseline's fixed scaling.
    const t: number[] = new Array(25).fill(0);
    for (let h = 2; h <= 24; h++) t[h] = 0.15 * Math.sqrt(h - 1);
    const hours = c.forecast.map(({ hour }) => ({ hour, q: LEVELS.map((tau, j) => round2(p1 * Math.exp(Z[j] * t[hour]))) }));
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
    continue;
  }
  const t = learnShape(c.comparables);
  const hours = c.forecast.map(({ hour }) => {
    const q = LEVELS.map((tau, j) => {
      const z = zAt(tau);
      return round2(p1 * Math.exp(z * t[hour]));
    });
    return { hour, q };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
