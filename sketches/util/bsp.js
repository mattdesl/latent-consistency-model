import defaultRandom from "canvas-sketch-util/random";
import { copyBounds, splitBounds, roundBounds } from "./bounds.js";
import { lerp } from "canvas-sketch-util/math";
import { getLeafNodes } from "./tree-util.js";

export class Partition {
  constructor(bounds, parent, depth = 0) {
    this.bounds = copyBounds(bounds);
    this.depth = depth;
    this.children = [];
    this.parent = parent;
  }

  get x() {
    const [min] = this.bounds;
    return min[0];
  }

  set x(v) {
    this.bounds[0][0] = v;
  }

  get y() {
    const [min] = this.bounds;
    return min[1];
  }

  set y(v) {
    this.bounds[0][1] = v;
  }

  get width() {
    const [min, max] = this.bounds;
    return max[0] - min[0];
  }

  get height() {
    const [min, max] = this.bounds;
    return max[1] - min[1];
  }

  get leaf() {
    return this.parent && this.children.length === 0;
  }

  detach() {
    if (this.parent) {
      const idx = this.parent.children.indexOf(this);
      this.parent.children.splice(idx, 1);
    }
  }
}

export default function bsp(bounds, opts = {}) {
  const root = new Partition(bounds, null, 0);
  const { splitCount = Infinity, maxDepth = 2, random = defaultRandom } = opts;
  let i = 0;
  const tooSmallSet = new Set();
  while (true) {
    if (i >= splitCount) break;
    const nodes = getLeafNodes(root).filter(
      (n) => n.depth < maxDepth && !tooSmallSet.has(n)
    );
    if (nodes.length === 0) break;
    const leaf = random.pick(nodes);
    const r = split(leaf, opts);
    if (!r) {
      tooSmallSet.add(leaf);
    }
    i++;
  }
  return root;
}

export const randomChoice = (bias = 1, exp = 1, rng = Math.random) => {
  return rng() / Math.pow(bias, exp) < 0.5;
};

export function split(node, opts = {}) {
  const {
    variance = 0.5,
    squariness = 0.5,
    random = defaultRandom,
    fract = random.range(0.5 - variance * 0.5, 0.5 + variance * 0.5),
    horizontal = randomChoice(
      node.width / node.height,
      squariness,
      random.value
    ),
    inverse = random.boolean(),
    minDimension = 0,
  } = opts;
  const childBounds = splitBounds(node.bounds, fract, horizontal, inverse);
  const children = childBounds.map(
    (b) => new Partition(b, node, node.depth + 1)
  );
  if (children.every((child) => largeEnough(child, minDimension))) {
    node.children = children;
    return { children, fract, horizontal, inverse };
  } else {
    return null;
  }
}

export function largeEnough(node, minDimension) {
  if (!isFinite(minDimension)) return true;
  return node.width >= minDimension && node.height >= minDimension;
}

export function getVerts(node) {
  const [min, max] = node.bounds;
  return [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ].map((n) => {
    return [lerp(min[0], max[0], n[0]), lerp(min[1], max[1], n[1])];
  });
}

export function getRandomEdge(node, rng = Math.random) {
  const n = Math.floor(rng() * 4);
  const verts = getVerts(node);
  const v0 = verts[n];
  const v1 = verts[(n + 1) % verts.length];
  return [v0, v1];
}
