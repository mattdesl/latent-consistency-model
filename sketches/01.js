import canvasSketch from "canvas-sketch";
import * as random from "canvas-sketch-util/random.js";
import bsp, { split, Partition } from "./util/bsp.js";
import { traverseDepthFirst, getLeafNodes } from "./util/tree-util";

random.setSeed("" || random.getRandomSeed());
console.log("Seed:", random.getSeed());

const settings = {
  suffix: random.getSeed(),
  dimensions: "A4",
  pixelsPerInch: 300,
  // dimensions: [256, 256],
};

const imageSeed = () => random.rangeFloor(0, 1e10);

async function generate(opts = {}) {
  const seed = opts.seed || imageSeed();
  opts = { ...opts, seed };
  const host = location.hostname;
  const port = 8000;
  const protocol = location.protocol;
  const queryString = new URLSearchParams(opts).toString();
  const resp = await fetch(`${protocol}//${host}:${port}/image?${queryString}`);
  const blob = await resp.blob();
  return {
    image: await createImageBitmap(blob),
    seed: parseInt(resp.headers.get("x-seed"), 10),
    steps: parseInt(resp.headers.get("x-steps"), 10),
  };
}

const sketch = async ({ width, height, update }) => {
  const dim = Math.min(width, height);
  const marginFactor = 0.05;
  const margin = dim * marginFactor;
  const rootBounds = [
    [margin, margin],
    [width - margin, height - margin],
  ];

  const rootWidth = width - margin * 2;
  const rootHeight = height - margin * 2;

  const maxImageSize = 512;
  const ratio = Math.min(
    1,
    maxImageSize / rootWidth,
    maxImageSize / rootHeight
  );
  const imageWidth = Math.floor(rootWidth * ratio);
  const imageHeight = Math.floor(rootHeight * ratio);
  console.log("New Size: %s x %s", imageWidth, imageHeight);

  // const { image } = await generate({
  //   prompt: "black and white portrait, old man",
  //   steps: 6,
  //   width: imageWidth,
  //   height: imageHeight,
  // });

  const tree = bsp(rootBounds, {
    minDimension: width * 0.01,
    // splitCount: 30,
    // maxDepth: Infinity,

    splitCount: Infinity,
    maxDepth: 6,
    // squariness: 1,
  });

  const generationSet = new Set();

  const nodes = getLeafNodes(tree);
  const maxGenerations = 7;
  // const maxGenerations = nodes.length;
  const prompts = [
    // "zebra",
    "old man",
    "old woman",
    "mature woman",
    "business woman",
    "astronaut",
    "vintage submarine sea man",
  ];

  for (const node of nodes) {
    const [ox, oy] = random.insideCircle(dim * 0.00005);
    node.x += ox;
    node.y += oy;
    node.rotation = random.gaussian(0, (0.5 * Math.PI) / 180);
  }

  const startFetch = async () => {
    const list = nodes.sort((a, b) => b.width * b.height - a.width * a.height);

    let steps = 5;
    for (const node of list) {
      const inject = random.pick(prompts);
      // const prompt = `wildflower meadow, bokeh, pastel colors, motion from the wind, drifting petals, dusk sunset, wide shot, 8k hd`;
      const prompt = `cloud photo crop, high contrast, dusk, colourful`;
      // const prompt = `sonia delaunay, geometric abstraction, detailed landscape of a field, pastel colors`;
      // const prompt = `skateboarding, photography, skater, fisheye`;
      // const prompt = `portrait of a ${inject}, bokeh, depth of field, 8k hd, zoomed out, pastel colors`;

      let generation;
      if (generationSet.size >= maxGenerations) {
        generation = random.pick([...generationSet]);
      } else {
        generation = await generate({
          prompt,
          steps,
          width: imageWidth,
          height: imageHeight,
        });
        generationSet.add(generation);
      }
      node.generation = generation;

      update();
    }
  };
  startFetch();

  return ({ context, width, height }) => {
    const lineWidth = dim * 0.00175;
    context.lineJoin = context.lineCap = "round";
    context.fillStyle = "black";
    context.fillRect(0, 0, width, height);

    nodes.forEach((node) => {
      context.save();
      const nx = node.x + node.width / 2;
      const ny = node.y + node.height / 2;
      context.translate(nx, ny);
      context.rotate(node.rotation);
      context.translate(-nx, -ny);

      if (node.generation) {
        context.save();

        context.beginPath();
        context.rect(node.x, node.y, node.width, node.height);
        context.clip();
        context.drawImage(
          node.generation.image,
          margin,
          margin,
          rootWidth,
          rootHeight
        );
        context.restore();
      } else {
        context.lineWidth = lineWidth;
        context.strokeStyle = "white";
        context.strokeRect(node.x, node.y, node.width, node.height);
      }
      context.restore();
    });
  };
};

canvasSketch(sketch, settings);
