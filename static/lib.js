export function uint32_to_rgba(color) {
  var a = (color >> 24) & 0xff;
  var b = (color >> 16) & 0xff;
  var g = (color >> 8) & 0xff;
  var r = color & 0xff;
  return [r, g, b, a];
}

export function rgba_to_uint32(r, g, b, a) {
  return (a << 24) | (b << 16) | (g << 8) | r;
}

export function rgb888_to_rgb565(r, g, b) {
  return ((r << 8) & 0xf800) | ((g << 2) & 0x03e0) | (b >> 3);
}

export function rgba8888_to_rgba4444(r, g, b, a) {
  return (r >> 4) | (g & 0xf0) | ((b & 0xf0) << 4) | ((a & 0xf0) << 8);
}

export function rgb888_to_rgb444(r, g, b) {
  return ((r >> 4) << 8) | (g & 0xf0) | (b >> 4);
}

// Alternative 565 ?
// return ((r & 0xf8) << 8) + ((g & 0xfc) << 3) + (b >> 3);

// Alternative 4444 ?
// ((a & 0xf0) << 8) | ((r & 0xf0) << 4) | (g & 0xf0) | (b >> 4);

export function nearestColorIndexRGBA(r, g, b, a, palette) {
  let k = 0;
  let mindist = 1e100;
  for (let i = 0; i < palette.length; i++) {
    const px2 = palette[i];
    const a2 = px2[3];
    let curdist = sqr(a2 - a);
    if (curdist > mindist) continue;
    const r2 = px2[0];
    curdist += sqr(r2 - r);
    if (curdist > mindist) continue;
    const g2 = px2[1];
    curdist += sqr(g2 - g);
    if (curdist > mindist) continue;
    const b2 = px2[2];
    curdist += sqr(b2 - b);
    if (curdist > mindist) continue;
    mindist = curdist;
    k = i;
  }
  return k;
}

export function nearestColorIndexRGB(r, g, b, palette) {
  let k = 0;
  let mindist = 1e100;
  for (let i = 0; i < palette.length; i++) {
    const px2 = palette[i];
    const r2 = px2[0];
    let curdist = sqr(r2 - r);
    if (curdist > mindist) continue;
    const g2 = px2[1];
    curdist += sqr(g2 - g);
    if (curdist > mindist) continue;
    const b2 = px2[2];
    curdist += sqr(b2 - b);
    if (curdist > mindist) continue;
    mindist = curdist;
    k = i;
  }
  return k;
}

export function sqr(a) {
  return a * a;
}

export function clamp(value, min, max) {
  return value < min ? min : value > max ? max : value;
}

export function ColorCache(format = "rgb565") {
  const hasAlpha = format === "rgba4444" || format === "rgba8888";

  let storage;
  let get;
  let find;

  if (format === "rgb888" || format === "rgba8888") {
    storage = new Map();

    find =
      format === "rgb888"
        ? (color, palette) => {
            const b = (color >> 16) & 0xff;
            const g = (color >> 8) & 0xff;
            const r = color & 0xff;
            return nearestColorIndexRGB(r, g, b, palette);
          }
        : (color, palette) => {
            const a = (color >> 24) & 0xff;
            const b = (color >> 16) & 0xff;
            const g = (color >> 8) & 0xff;
            const r = color & 0xff;
            return nearestColorIndexRGBA(r, g, b, apalette);
          };

    get = (color) => {
      if (storage.has(color)) return storage.get(color);
      else storage.set(color, find(color));
    };
  } else if (
    format === "rgba4444" ||
    format === "rgb565" ||
    format === "rgb444"
  ) {
    const bincount = format === "rgb444" ? 4096 : 65536;
    storage = new Array(bincount);
  }

  return {
    search,
    has(colorUint32) {},
    dispose() {
      storage = null;
    },
  };
}

export function dither(pixels, width, height, palette, format, kernel) {
  const data = new Uint8ClampedArray(pixels);
  const view = new Uint32Array(data.buffer);
  const length = view.length;
  const index = new Uint8Array(length);
  const bincount = format === "rgb444" ? 4096 : 65536;
  const cache = new Array(bincount);
  const error = [0, 0, 0];
  const rgb888_to_key =
    format === "rgb444" ? rgb888_to_rgb444 : rgb888_to_rgb565;

  // Floyd-Steinberg kernel
  kernel = kernel || [
    [7 / 16, 1, 0],
    [3 / 16, -1, 1],
    [5 / 16, 0, 1],
    [1 / 16, 1, 1],
  ];

  const palette32 = palette.map((rgba) => {
    const r = rgba[0];
    const g = rgba[1];
    const b = rgba[2];
    const a = rgba.length === 4 ? rgba[3] : 0xff;
    return (a << 24) | (b << 16) | (g << 8) | r;
  });

  const hasAlpha = format === "rgba4444";

  for (let i = 0; i < length; i++) {
    // get pixel x, y
    const x = Math.floor(i % width);
    const y = Math.floor(i / width);

    // get pixel r,g,b
    const color = view[i];

    // get a pixel index
    let idx;
    let r, g, b;

    if (hasAlpha) {
      a = (color >> 24) & 0xff;
      b = (color >> 16) & 0xff;
      g = (color >> 8) & 0xff;
      r = color & 0xff;
      const key = rgba8888_to_rgba4444(r, g, b, a);
      idx =
        key in cache
          ? cache[key]
          : (cache[key] = nearestColorIndexRGBA(r, g, b, a, palette));
    } else {
      b = (color >> 16) & 0xff;
      g = (color >> 8) & 0xff;
      r = color & 0xff;
      const key = rgb888_to_key(r, g, b);
      idx =
        key in cache
          ? cache[key]
          : (cache[key] = nearestColorIndexRGB(r, g, b, palette));
    }

    // the palette index for this pixel is now set
    index[i] = idx;

    const newRGB = palette[idx];

    // compute error from target
    error[0] = r - newRGB[0];
    error[1] = g - newRGB[1];
    error[2] = b - newRGB[2];

    // assign paletted colour to view
    view[i] = palette32[idx];

    // diffuse error to other pixels
    for (let i = 0; i < kernel.length; i++) {
      const K = kernel[i];
      const kx = K[1] + x;
      const ky = K[2] + y;
      if (kx >= 0 && kx < width && ky >= 0 && ky < height) {
        const kidx = (kx + ky * width) * 4;
        const diffusion = K[0];
        for (let c = 0; c < 3; c++) {
          const dst = c + kidx;
          data[dst] = data[dst] + error[c] * diffusion;
        }
      }
    }
  }
  return index;
}

// Modified from:
// https://github.com/mcychan/PnnQuant.js/blob/master/src/pnnquant.js

/* Fast pairwise nearest neighbor based algorithm for multilevel thresholding
Copyright (C) 2004-2019 Mark Tyler and Dmitry Groshev
Copyright (c) 2018-2021 Miller Cy Chan
* error measure; time used is proportional to number of bins squared - WJ */

function find_nn(bins, idx, hasAlpha) {
  var nn = 0;
  var err = 1e100;

  const bin1 = bins[idx];
  const n1 = bin1.cnt;
  const wa = bin1.ac;
  const wr = bin1.rc;
  const wg = bin1.gc;
  const wb = bin1.bc;
  for (var i = bin1.fw; i != 0; i = bins[i].fw) {
    const bin = bins[i];
    const n2 = bin.cnt;
    const nerr2 = (n1 * n2) / (n1 + n2);
    if (nerr2 >= err) continue;

    var nerr = 0;
    if (hasAlpha) {
      nerr += nerr2 * sqr(bin.ac - wa);
      if (nerr >= err) continue;
    }

    nerr += nerr2 * sqr(bin.rc - wr);
    if (nerr >= err) continue;

    nerr += nerr2 * sqr(bin.gc - wg);
    if (nerr >= err) continue;

    nerr += nerr2 * sqr(bin.bc - wb);
    if (nerr >= err) continue;
    err = nerr;
    nn = i;
  }
  bin1.err = err;
  bin1.nn = nn;
}

function create_bin() {
  return {
    ac: 0,
    rc: 0,
    gc: 0,
    bc: 0,
    cnt: 0,
    nn: 0,
    fw: 0,
    bk: 0,
    tm: 0,
    mtm: 0,
    err: 0,
  };
}

function bin_add_rgb(bin, r, g, b) {
  bin.rc += r;
  bin.gc += g;
  bin.bc += b;
  bin.cnt++;
}

function create_bin_list(data, format) {
  const bincount = format === "rgb444" ? 4096 : 65536;
  const bins = new Array(bincount);
  const size = data.length;

  /* Build histogram */
  // Note: Instead of introducing branching/conditions
  // within a very hot per-pixel iteration, we just duplicate the code
  // for each new condition
  if (format === "rgba4444") {
    for (let i = 0; i < size; ++i) {
      const color = data[i];
      const a = (color >> 24) & 0xff;
      const b = (color >> 16) & 0xff;
      const g = (color >> 8) & 0xff;
      const r = color & 0xff;

      // reduce to rgb4444 16-bit uint
      const index = rgba8888_to_rgba4444(r, g, b, a);
      let bin = index in bins ? bins[index] : (bins[index] = create_bin());
      bin.rc += r;
      bin.gc += g;
      bin.bc += b;
      bin.ac += a;
      bin.cnt++;
    }
  } else if (format === "rgb444") {
    for (let i = 0; i < size; ++i) {
      const color = data[i];
      const b = (color >> 16) & 0xff;
      const g = (color >> 8) & 0xff;
      const r = color & 0xff;

      // reduce to rgb444 12-bit uint
      const index = rgb888_to_rgb444(r, g, b);
      let bin = index in bins ? bins[index] : (bins[index] = create_bin());
      bin.rc += r;
      bin.gc += g;
      bin.bc += b;
      bin.cnt++;
    }
  } else {
    for (let i = 0; i < size; ++i) {
      const color = data[i];
      const b = (color >> 16) & 0xff;
      const g = (color >> 8) & 0xff;
      const r = color & 0xff;

      // reduce to rgb565 16-bit uint
      const index = rgb888_to_rgb565(r, g, b);
      let bin = index in bins ? bins[index] : (bins[index] = create_bin());
      bin.rc += r;
      bin.gc += g;
      bin.bc += b;
      bin.cnt++;
    }
  }
  return bins;
}

export function quantize(rgba, maxColors, opts = {}) {
  const {
    format = "rgb565",
    clearAlpha = true,
    clearAlphaColor = 0x00,
    clearAlphaThreshold = 0,
    oneBitAlpha = false,
  } = opts;

  if (!rgba || !rgba.buffer) {
    throw new Error("quantize() expected RGBA Uint8Array data");
  }
  if (!(rgba instanceof Uint8Array) && !(rgba instanceof Uint8ClampedArray)) {
    throw new Error("quantize() expected RGBA Uint8Array data");
  }

  const data = new Uint32Array(rgba.buffer);

  let useSqrt = opts.useSqrt !== false;

  // format can be:
  // rgb565 (default)
  // rgb444
  // rgba4444

  const hasAlpha = format === "rgba4444";
  const bins = create_bin_list(data, format);
  const bincount = bins.length;
  const bincountMinusOne = bincount - 1;
  const heap = new Uint32Array(bincount + 1);

  /* Cluster nonempty bins at one end of array */
  var maxbins = 0;
  for (var i = 0; i < bincount; ++i) {
    const bin = bins[i];
    if (bin != null) {
      var d = 1.0 / bin.cnt;
      if (hasAlpha) bin.ac *= d;
      bin.rc *= d;
      bin.gc *= d;
      bin.bc *= d;
      bins[maxbins++] = bin;
    }
  }

  if (sqr(maxColors) / maxbins < 0.022) {
    useSqrt = false;
  }

  var i = 0;
  for (; i < maxbins - 1; ++i) {
    bins[i].fw = i + 1;
    bins[i + 1].bk = i;
    if (useSqrt) bins[i].cnt = Math.sqrt(bins[i].cnt);
  }
  if (useSqrt) bins[i].cnt = Math.sqrt(bins[i].cnt);

  var h, l, l2;
  /* Initialize nearest neighbors and build heap of them */
  for (i = 0; i < maxbins; ++i) {
    find_nn(bins, i, false);
    /* Push slot on heap */
    var err = bins[i].err;
    for (l = ++heap[0]; l > 1; l = l2) {
      l2 = l >> 1;
      if (bins[(h = heap[l2])].err <= err) break;
      heap[l] = h;
    }
    heap[l] = i;
  }

  /* Merge bins which increase error the least */
  var extbins = maxbins - maxColors;
  for (i = 0; i < extbins; ) {
    var tb;
    /* Use heap to find which bins to merge */
    for (;;) {
      var b1 = heap[1];
      tb = bins[b1]; /* One with least error */
      /* Is stored error up to date? */
      if (tb.tm >= tb.mtm && bins[tb.nn].mtm <= tb.tm) break;
      if (tb.mtm == bincountMinusOne)
        /* Deleted node */ b1 = heap[1] = heap[heap[0]--];
      /* Too old error value */ else {
        find_nn(bins, b1, false);
        tb.tm = i;
      }
      /* Push slot down */
      var err = bins[b1].err;
      for (l = 1; (l2 = l + l) <= heap[0]; l = l2) {
        if (l2 < heap[0] && bins[heap[l2]].err > bins[heap[l2 + 1]].err) l2++;
        if (err <= bins[(h = heap[l2])].err) break;
        heap[l] = h;
      }
      heap[l] = b1;
    }

    /* Do a merge */
    var nb = bins[tb.nn];
    var n1 = tb.cnt;
    var n2 = nb.cnt;
    var d = 1.0 / (n1 + n2);
    if (hasAlpha) tb.ac = d * (n1 * tb.ac + n2 * nb.ac);
    tb.rc = d * (n1 * tb.rc + n2 * nb.rc);
    tb.gc = d * (n1 * tb.gc + n2 * nb.gc);
    tb.bc = d * (n1 * tb.bc + n2 * nb.bc);
    tb.cnt += nb.cnt;
    tb.mtm = ++i;

    /* Unchain deleted bin */
    bins[nb.bk].fw = nb.fw;
    bins[nb.fw].bk = nb.bk;
    nb.mtm = bincountMinusOne;
  }

  // let palette = new Uint32Array(maxColors);
  let palette = [];

  /* Fill palette */
  var k = 0;
  for (i = 0; ; ++k) {
    let r = clamp(Math.round(bins[i].rc), 0, 0xff);
    let g = clamp(Math.round(bins[i].gc), 0, 0xff);
    let b = clamp(Math.round(bins[i].bc), 0, 0xff);

    let a = 0xff;
    if (hasAlpha) {
      a = clamp(Math.round(bins[i].ac), 0, 0xff);
      if (oneBitAlpha) {
        const threshold = typeof oneBitAlpha === "number" ? oneBitAlpha : 127;
        a = a <= threshold ? 0x00 : 0xff;
      }
      if (clearAlpha && a <= clearAlphaThreshold) {
        r = g = b = clearAlphaColor;
        a = 0x00;
      }
    }

    const color = hasAlpha ? [r, g, b, a] : [r, g, b];
    const exists = existsInPalette(palette, color);
    if (!exists) palette.push(color);
    if ((i = bins[i].fw) == 0) break;
  }

  return palette;
}

function existsInPalette(palette, color) {
  for (let i = 0; i < palette.length; i++) {
    const p = palette[i];
    let matchesRGB =
      p[0] === color[0] && p[1] === color[1] && p[2] === color[2];
    let matchesAlpha =
      p.length >= 4 && color.length >= 4 ? p[3] === color[3] : true;
    if (matchesRGB && matchesAlpha) return true;
  }
  return false;
}

// TODO: Further 'clean' palette by merging nearly-identical colors?
