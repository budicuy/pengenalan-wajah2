"use server";

import fs from "node:fs";
import path from "node:path";

interface Student {
  filename: string;
  label: string;
  url: string;
}

export async function getDataset() {
  try {
    const datasetDir = path.join(process.cwd(), "public", "dataset");

    // Check if dataset directory exists
    if (!fs.existsSync(datasetDir)) {
      return { error: "Dataset directory not found", data: [] as Student[] };
    }

    const files = fs.readdirSync(datasetDir);

    // Filter only image files (e.g., .jpg, .jpeg, .png, .webp)
    const imageExtensions = [".jpg", ".jpeg", ".png", ".webp"];
    const imageFiles = files.filter((file) => {
      const ext = path.extname(file).toLowerCase();
      return imageExtensions.includes(ext);
    });

    // Parse files to create labels and asset URLs
    const dataset = imageFiles.map((file) => {
      const ext = path.extname(file);
      const baseName = path.basename(file, ext);
      // Remove patterns like " (1)" or " (2)" from the filename to get the clean name
      const label = baseName.replace(/\s*\(\d+\)\s*$/, "").trim();

      return {
        filename: file,
        label: label,
        url: `/dataset/${encodeURIComponent(file)}`,
      };
    });

    // Sort dataset alphabetically by label
    dataset.sort((a, b) => a.label.localeCompare(b.label));

    return { data: dataset };
  } catch (error: any) {
    console.error("Error reading dataset directory:", error);
    return {
      error: error.message || "Internal Server Error",
      data: [] as Student[],
    };
  }
}
