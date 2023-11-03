export function roundBounds(bounds) {
  const [min, max] = bounds;
  return [min.map((n) => Math.round(n)), max.map((n) => Math.round(n))];
}

export function splitBounds(
  bounds,
  fract = 0.5,
  horizontal = true,
  inverse = false
) {
  const [min, max] = bounds;
  const [x1, y1] = min;
  const [x2, y2] = max;
  const width = x2 - x1;
  const height = y2 - y1;

  fract = Math.max(0, Math.min(1, fract));
  fract = inverse ? 1 - fract : fract;

  const dim = horizontal ? width : height;
  const off = dim * fract;

  let a, b;
  if (horizontal) {
    a = [
      [x1, y1],
      [x1 + off, y2],
    ];
    b = [
      [x1 + off, y1],
      [x2, y2],
    ];
  } else {
    a = [
      [x1, y1],
      [x2, y1 + off],
    ];
    b = [
      [x1, y1 + off],
      [x2, y2],
    ];
  }
  return [a, b];
}

export function copyBounds(bounds) {
  return bounds.map((b) => b.slice());
}

export function boundsIntersect(boundsA, boundsB) {
  const [minA, maxA] = boundsA;
  const [minB, maxB] = boundsB;

  const [aMinX, aMinY] = minA;
  const [aMaxX, aMaxY] = maxA;
  const [bMinX, bMinY] = minB;
  const [bMaxX, bMaxY] = maxB;

  const noOverlap =
    aMinX > bMaxX || bMinX > aMaxX || aMinY > bMaxY || bMinY > aMaxY;
  return !noOverlap;
}

export function boundsInsideOther(boundsA, boundsB) {
  const [minA, maxA] = boundsA;
  const [minB, maxB] = boundsB;

  const width = maxA[0] - minA[0];
  const height = maxA[1] - minA[1];
  const x = minA[0];
  const y = minA[1];

  const owidth = maxB[0] - minB[0];
  const oheight = maxB[1] - minB[1];
  const ox = minB[0];
  const oy = minB[1];

  return (
    width > 0 &&
    height > 0 &&
    owidth > 0 &&
    oheight > 0 &&
    ox >= x &&
    ox + owidth <= x + width &&
    oy >= y &&
    oy + oheight <= y + height
  );
}

// Note: the right and bottom edge boundaries are exclusive
export function pointInsideBounds(point, bounds) {
  const [min, max] = bounds;
  const left = min[0];
  const top = min[1];
  const right = max[0];
  const bottom = max[1];
  const [x, y] = point;
  return x >= left && y >= top && x < right && y < bottom;
}
