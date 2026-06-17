import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";

export async function GET() {
  try {
    const datasetDir = path.join(process.cwd(), "public", "dataset");

    // Check if dataset directory exists
    if (!fs.existsSync(datasetDir)) {
      return NextResponse.json(
        { error: "Dataset directory not found" },
        { status: 404 },
      );
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

    return NextResponse.json(dataset);
  } catch (error: any) {
    console.error("Error reading dataset directory:", error);
    return NextResponse.json(
      { error: "Internal Server Error", details: error.message },
      { status: 500 },
    );
  }
}
