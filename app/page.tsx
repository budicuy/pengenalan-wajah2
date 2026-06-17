"use client";

import dynamic from "next/dynamic";

// Dynamically import FaceRecognition to avoid Server-Side Rendering (SSR) issues
const FaceRecognition = dynamic(
  () => import("@/app/components/FaceRecognition"),
  {
    ssr: false,
    loading: () => (
      <div className="min-h-screen bg-zinc-950 flex flex-col items-center justify-center text-zinc-400 gap-4 font-mono">
        <div className="w-12 h-12 border-4 border-zinc-800 border-t-cyan-500 rounded-full animate-spin" />
        <span className="text-sm">Inisialisasi Modul Pengenalan Wajah...</span>
      </div>
    ),
  },
);

export default function Home() {
  return (
    <main className="min-h-screen bg-zinc-950 flex flex-col justify-start">
      <FaceRecognition />
    </main>
  );
}
