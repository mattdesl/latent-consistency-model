import canvasSketch from "canvas-sketch";
import * as random from "canvas-sketch-util/random.js";
import bsp, { split, Partition } from "./util/bsp.js";
import { traverseDepthFirst, getLeafNodes } from "./util/tree-util";

random.setSeed("" || random.getRandomSeed());
console.log("Seed:", random.getSeed());

const settings = {
  suffix: random.getSeed(),
  dimensions: "A4",
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
  const marginFactor = 0.1;
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
    minDimension: width * 0.1,
    splitCount: 15,
    maxDepth: Infinity,
  });

  const nodes = getLeafNodes(tree);
  const prompt =
    "black and white portrait of an old man, bokeh, depth of field, 8k hd, zoomed out";

  for (const node of nodes) {
    const [ox, oy] = random.insideCircle(dim * 0.01);
    node.x += ox;
    node.y += oy;
  }

  const startFetch = async () => {
    const list = nodes.sort((a, b) => a.width * a.height - b.width * b.height);

    let steps = 4;
    for (let step = 1; step <= steps; step++) {
      for (const node of list) {
        let seed = node.generation ? node.generation.seed : imageSeed();
        node.generation = await generate({
          prompt,
          seed,
          steps: step,
          width: imageWidth,
          height: imageHeight,
        });
        update();
      }
    }
  };
  startFetch();

  return ({ context, width, height }) => {
    context.fillStyle = "black";
    context.fillRect(0, 0, width, height);

    nodes.forEach((node) => {
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
        context.strokeRect(node.x, node.y, node.width, node.height);
      }
    });
  };
};

canvasSketch(sketch, settings);
