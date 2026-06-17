import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { finished } from "node:stream/promises";

const BASE_URL =
  "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights";
const TARGET_DIR = path.join(process.cwd(), "public", "models");

const FILES = [
  // SSD Mobilenet v1 (Default detection)
  "ssd_mobilenetv1_model-weights_manifest.json",
  "ssd_mobilenetv1_model-shard1",
  "ssd_mobilenetv1_model-shard2",

  // Tiny Face Detector (Faster detection option)
  "tiny_face_detector_model-weights_manifest.json",
  "tiny_face_detector_model-shard1",

  // Face Landmark 68 (68 landmarks detection)
  "face_landmark_68_model-weights_manifest.json",
  "face_landmark_68_model-shard1",

  // Face Recognition (descriptor computation)
  "face_recognition_model-weights_manifest.json",
  "face_recognition_model-shard1",
  "face_recognition_model-shard2",
];

async function downloadFile(url, targetPath) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to download ${url}: ${res.statusText}`);
  }
  const fileStream = fs.createWriteStream(targetPath);
  await finished(Readable.fromWeb(res.body).pipe(fileStream));
  console.log(`Downloaded: ${path.basename(targetPath)}`);
}

async function main() {
  try {
    if (!fs.existsSync(TARGET_DIR)) {
      fs.mkdirSync(TARGET_DIR, { recursive: true });
      console.log(`Created directory: ${TARGET_DIR}`);
    }

    console.log("Starting model downloads from GitHub...");
    for (const filename of FILES) {
      const targetPath = path.join(TARGET_DIR, filename);
      // Skip download if file already exists and is non-empty
      if (fs.existsSync(targetPath) && fs.statSync(targetPath).size > 0) {
        console.log(`Skipping (already exists): ${filename}`);
        continue;
      }

      const fileUrl = `${BASE_URL}/${filename}`;
      await downloadFile(fileUrl, targetPath);
    }
    console.log("All models downloaded successfully!");
  } catch (error) {
    console.error("Error downloading models:", error);
    process.exit(1);
  }
}

main();
