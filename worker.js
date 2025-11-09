/* worker.js — content-based + MMR + filters (categories/brands/price/budget/brandBoost) */

self.onmessage = (e) => {
  const { products, selIndices, lambda, topk, budget, filters = {} } = e.data || {};
  try {
    const recs = recommend(products, selIndices, {
      lambda: clamp(lambda ?? 0.8, 0, 1),
      topk: Math.max(1, Math.min(50, parseInt(topk || 10))),
      budget: parseFloat(budget || 0),
      filters
    });
    self.postMessage(recs);
  } catch (err) {
    console.error('Worker error:', err);
    self.postMessage([]);
  }
};

/* ----------------------- Core recommender ----------------------- */

function recommend(products, selIdxs, opts) {
  const { lambda, topk, budget, filters = {} } = opts;

  // normalize filters
  const fCats = new Set((filters.categories || []).map(normalizeString));
  const fBrands = new Set((filters.brands || []).map(normalizeString));
  const minP = toNum(filters.minPrice || 0);
  const maxP = toNum(filters.maxPrice || 0);
  const brandBoost = Math.max(0, Math.min(1, parseFloat(filters.brandBoost || 0)));

  // 0) Guards
  if (!Array.isArray(products) || !products.length) return [];
  const seeds = selIdxs.map(i => products[i]).filter(Boolean);
  if (!seeds.length && !fCats.size && !fBrands.size) return []; // need at least one driver

  // If no seeds, synthesize a "seed centroid" from filters (categories/brands) so it still works
  const syntheticSeed = (!seeds.length) ? [{
    product_name: Array.from(fCats).join(' ') || Array.from(fBrands).join(' ') || 'seed',
    product_category_tree: Array.from(fCats).join(' '),
    brand: Array.from(fBrands).join(' '),
    description: ''
  }] : null;
  const drivingSeeds = syntheticSeed || seeds;

  // 1) Tokenize seeds and build candidate pool
  const seedTokens = new Set();
  drivingSeeds.forEach(p => tokensOf(p).forEach(t => seedTokens.add(t)));

  const MAX_CAND = 30000;
  const candidatesSet = new Set();

  // Prefilter pass: budget/price band + category/brand includes + token overlap
  for (let i = 0; i < products.length; i++) {
    if (selIdxs.includes(i)) continue;
    const p = products[i];
    if (!p || !p.product_name) continue;

    const price = toNum(p.price);
    if (budget > 0 && price > budget) continue;
    if (minP > 0 && price > 0 && price < minP) continue;
    if (maxP > 0 && price > 0 && price > maxP) continue;

    if (fCats.size) {
      // use first-level category if tree like "Electronics > Cameras > ..."
      const cat = normalizeString(String(p.product_category_tree || '').split('>')[0]);
      if (!fCats.has(cat)) continue;
    }
    if (fBrands.size) {
      const br = normalizeString(p.brand);
      if (!fBrands.has(br)) continue;
    }

    const toks = tokensOf(p);
    if (!toks.length) continue;

    // If user provided seeds: require token overlap with seed tokens
    // If no seeds (synthetic), allow all after filter gates
    if (seeds.length) {
      let intersects = false;
      for (const t of toks) { if (seedTokens.has(t)) { intersects = true; break; } }
      if (!intersects) continue;
    }

    candidatesSet.add(i);
    if (candidatesSet.size >= MAX_CAND) break;
  }

  if (candidatesSet.size === 0) {
    // fallback: take a slice that passes only budget/price gates so we can still recommend something
    for (let i = 0; i < products.length && candidatesSet.size < 5000; i++) {
      if (selIdxs.includes(i)) continue;
      const p = products[i];
      if (!p || !p.product_name) continue;
      const price = toNum(p.price);
      if (budget > 0 && price > budget) continue;
      if (minP > 0 && price > 0 && price < minP) continue;
      if (maxP > 0 && price > 0 && price > maxP) continue;
      candidatesSet.add(i);
    }
  }

  const candidates = Array.from(candidatesSet);
  const N = candidates.length;
  if (!N) return [];

  // 2) TF-IDF in candidate space
  const df = new Map();
  const candTokens = new Array(N);
  for (let k = 0; k < N; k++) {
    const toks = tokensOf(products[candidates[k]]);
    candTokens[k] = toks;
    const uniq = new Set(toks);
    for (const t of uniq) df.set(t, (df.get(t) || 0) + 1);
  }
  const idf = new Map();
  for (const [t, d] of df) idf.set(t, Math.log((N + 1) / (d + 1)) + 1);

  const candVecs = new Array(N);
  const candNorms = new Float32Array(N);
  for (let k = 0; k < N; k++) {
    const v = new Map();
    for (const t of candTokens[k]) {
      const w = idf.get(t);
      if (!w) continue;
      v.set(t, (v.get(t) || 0) + w); // tf * idf
    }
    candVecs[k] = v;
    candNorms[k] = l2(v);
  }

  // 2.3 seed centroid (use drivingSeeds which may be synthetic)
  const centroid = new Map();
  drivingSeeds.forEach(p => {
    for (const t of tokensOf(p)) {
      const w = idf.get(t);
      if (!w) continue;
      centroid.set(t, (centroid.get(t) || 0) + w);
    }
  });
  const centroidNorm = l2(centroid) || 1e-8;

  // 3) Base relevance: cosine to centroid
  const rel = new Float32Array(N);
  for (let k = 0; k < N; k++) rel[k] = cosine(centroid, centroidNorm, candVecs[k], candNorms[k]);

  // 4) Price affinity to seed median (true seeds only; if synthetic, keep neutral)
  const seedPrices = seeds.map(p => toNum(p.price)).filter(x => x > 0);
  const medianSeedPrice = seedPrices.length ? median(seedPrices) : 0;
  const priceAff = new Float32Array(N);
  const SIGMA = 0.35;
  for (let k = 0; k < N; k++) {
    const price = toNum(products[candidates[k]].price);
    if (medianSeedPrice > 0 && price > 0) {
      const dist = Math.log(price / medianSeedPrice);
      priceAff[k] = Math.exp(-0.5 * (dist * dist) / (SIGMA * SIGMA));
    } else {
      priceAff[k] = 0.5; // neutral-ish
    }
  }

  // 5) Combine base relevance + price
  const baseScore = new Float32Array(N);
  for (let k = 0; k < N; k++) baseScore[k] = 0.7 * rel[k] + 0.3 * priceAff[k];

  // 5.1 Brand boost (small, controlled nudge)
  if (brandBoost > 0 && fBrands.size) {
    for (let k = 0; k < N; k++) {
      const br = normalizeString(products[candidates[k]].brand);
      if (fBrands.has(br)) baseScore[k] += 0.15 * brandBoost;
    }
  }

  // 6) MMR diversity selection
  const out = [];
  const chosenVecs = [];
  const chosenNorms = [];
  const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => baseScore[b] - baseScore[a]);
  const L = clamp(topk, 1, 50);

  for (let pick = 0; pick < L && pick < N; pick++) {
    let bestIdx = -1, bestMMR = -1;

    if (pick === 0) {
      bestIdx = order[0];
    } else {
      for (let ii = 0; ii < order.length; ii++) {
        const i = order[ii];
        const candV = candVecs[i], candN = candNorms[i];
        let maxSim = 0;
        for (let j = 0; j < chosenVecs.length; j++) {
          const sim = cosine(chosenVecs[j], chosenNorms[j], candV, candN);
          if (sim > maxSim) maxSim = sim;
        }
        const mmr = lambda * baseScore[i] - (1 - lambda) * maxSim;
        if (mmr > bestMMR) { bestMMR = mmr; bestIdx = i; }
      }
    }

    if (bestIdx < 0) break;
    const chosenGlobal = candidates[bestIdx];
    out.push(withWhy(products, chosenGlobal, drivingSeeds, budget, medianSeedPrice, fCats, fBrands));
    chosenVecs.push(candVecs[bestIdx]);
    chosenNorms.push(candNorms[bestIdx]);
    order.splice(order.indexOf(bestIdx), 1);
  }

  return out;
}

/* ----------------------- Helpers ----------------------- */

function withWhy(products, idx, seeds, budget, medPrice, fCats, fBrands) {
  const p = products[idx];

  // choose nearest seed for explanation (if seeds exist)
  let bestSeed = null, bestSim = -1;
  if (seeds && seeds.length) {
    const seedCentroid = new Map();
    // quick reuse: centroid-like over seed tokens (unit weights for text)
    seeds.forEach(s => tokensOf(s).forEach(t => seedCentroid.set(t, (seedCentroid.get(t) || 0) + 1)));
    const sNorm = l2(seedCentroid) || 1e-8;

    const v = new Map();
    tokensOf(p).forEach(t => v.set(t, (v.get(t) || 0) + 1));
    const vNorm = l2(v) || 1e-8;

    // measure similarity to each seed crudely (good enough for “why”)
    for (const s of seeds) {
      const sVec = new Map();
      tokensOf(s).forEach(t => sVec.set(t, (sVec.get(t) || 0) + 1));
      const sim = cosine(sVec, l2(sVec) || 1e-8, v, vNorm);
      if (sim > bestSim) { bestSim = sim; bestSeed = s; }
    }
  }

  const bits = [];
  if (bestSeed?.product_name) bits.push(`similar to ${trim(bestSeed.product_name, 40)}`);
  if (p.product_category_tree) bits.push(`category: ${p.product_category_tree}`);
  if (p.brand) bits.push(`brand: ${p.brand}`);
  if (p.price) bits.push(`price ₹${num(p.price)}`);
  if (budget > 0) bits.push(p.price <= budget ? 'within budget' : 'over budget');
  if (fCats && fCats.size) {
    const cat = String(p.product_category_tree || '').split('>')[0].trim().toLowerCase();
    if (fCats.has(cat)) bits.push('matches chosen category');
  }
  if (fBrands && fBrands.size) {
    const br = String(p.brand || '').trim().toLowerCase();
    if (fBrands.has(br)) bits.push('matches chosen brand');
  }
  p.reason = bits.join(' • ');
  return p;
}

function tokensOf(p) {
  const parts = [
    p.product_name || '',
    (p.product_category_tree || '').split('>')[0] || '',
    p.brand || '',
    p.description || ''
  ];
  return tokenize(parts.join(' '));
}

function tokenize(s) {
  s = (s || '').toLowerCase();
  const toks = s.replace(/[^a-z0-9\s]+/g, ' ').split(/\s+/).filter(t => t.length > 1);
  const stop = new Set(['the','and','for','with','from','into','your','you','this','that','have','has','are','was','were','of','in','on','to','a','an']);
  return toks.filter(t => !stop.has(t));
}

function l2(v) { let s = 0; for (const w of v.values()) s += w * w; return Math.sqrt(s) || 0; }

function cosine(a, aN, b, bN) {
  if (!aN || !bN) return 0;
  let dot = 0;
  const [small, big] = a.size < b.size ? [a, b] : [b, a];
  for (const [t, w] of small) { const ww = big.get(t); if (ww) dot += w * ww; }
  return dot / (aN * bN);
}

function toNum(x) { const n = parseFloat(String(x).replace(/[^\d.]/g, '')); return isFinite(n) ? n : 0; }
function median(arr) { const a = arr.slice().sort((x,y)=>x-y); const m = a.length>>1; return a.length%2?a[m]:(a[m-1]+a[m])/2; }
function num(x) { return Number(toNum(x)).toLocaleString(); }
function trim(s, n) { s = String(s || ''); return s.length > n ? s.slice(0, n - 1) + '…' : s; }
function normalizeString(s) { return String(s || '').trim().toLowerCase(); }
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
