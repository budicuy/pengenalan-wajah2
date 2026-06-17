"use client";

import * as faceapi from "@vladmandic/face-api";
import { useEffect, useRef, useState } from "react";

interface Student {
  filename: string;
  label: string;
  url: string;
}

interface LogItem {
  id: string;
  name: string;
  timestamp: string;
  confidence: number;
  photoUrl?: string;
}

type ProcessStatus =
  | "idle"
  | "loading-models"
  | "fetching-dataset"
  | "processing-dataset"
  | "ready"
  | "error";
type StudentStatus = "pending" | "success" | "failed";

export default function FaceRecognition() {
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number | null>(null);
  const lastLogged = useRef<Record<string, number>>({});

  // States
  const [status, setStatus] = useState<ProcessStatus>("idle");
  const [statusMessage, setStatusMessage] =
    useState<string>("Memulai sistem...");
  const [progress, setProgress] = useState({ current: 0, total: 0, label: "" });

  const [dataset, setDataset] = useState<Student[]>([]);
  const [studentStatuses, setStudentStatuses] = useState<
    Record<string, StudentStatus>
  >({});
  const [faceMatcher, setFaceMatcher] = useState<faceapi.FaceMatcher | null>(
    null,
  );

  const [logs, setLogs] = useState<LogItem[]>([]);
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [cameraActive, setCameraActive] = useState<boolean>(false);

  // Settings States
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState<string>("");
  const [detectorType, setDetectorType] = useState<"ssd" | "tiny">("tiny"); // tiny is much smoother for real-time
  const [threshold, setThreshold] = useState<number>(0.5); // Euclidean distance (lower = stricter match)
  const [showLandmarks, setShowLandmarks] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [selectedPreviewStudent, setSelectedPreviewStudent] =
    useState<Student | null>(null);

  // Load list of cameras
  useEffect(() => {
    if (typeof window !== "undefined" && navigator.mediaDevices) {
      navigator.mediaDevices
        .enumerateDevices()
        .then((deviceList) => {
          const videoDevices = deviceList.filter(
            (d) => d.kind === "videoinput",
          );
          setDevices(videoDevices);
          if (videoDevices.length > 0) {
            setSelectedDevice(videoDevices[0].deviceId);
          }
        })
        .catch((err) => {
          console.error("Error listing camera devices:", err);
        });
    }
  }, []);

  // Initialize models and dataset
  // biome-ignore lint/correctness/useExhaustiveDependencies: run once on mount
  useEffect(() => {
    initSystem();

    // Cleanup webcam stream on unmount
    return () => {
      stopCamera();
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, []);

  // Rebuild FaceMatcher when threshold or faceMatcher references change
  const rebuildFaceMatcherWithThreshold = (
    descriptors: faceapi.LabeledFaceDescriptors[],
  ) => {
    // FaceMatcher accepts maxDistance in the constructor
    // maxDistance: 1 - threshold (since threshold is confidence, distance threshold is 1 - confidence)
    // Actually face-api distance is Euclidean distance:
    // 0 is identical, 1 is completely different.
    // Slider threshold is confidence (0.4 - 0.8). Let's convert confidence to distance threshold.
    // Distance threshold = 1 - confidence. E.g., 60% confidence threshold = 0.4 distance threshold.
    const distanceThreshold = 1 - threshold;
    const matcher = new faceapi.FaceMatcher(descriptors, distanceThreshold);
    setFaceMatcher(matcher);
  };

  const initSystem = async () => {
    try {
      setStatus("loading-models");
      setStatusMessage("Memuat model AI (Deteksi & Pengenalan Wajah)...");

      // Load model weights from public folder
      const MODEL_URL = "/models";
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

      setStatus("fetching-dataset");
      setStatusMessage("Mengambil data mahasiswa dari server...");

      // Fetch list of students in dataset
      const res = await fetch("/api/dataset");
      if (!res.ok) {
        throw new Error("Gagal memuat list dataset gambar dari API");
      }
      const studentList: Student[] = await res.json();
      setDataset(studentList);

      // Setup initial status as pending for all
      const initialStatuses: Record<string, StudentStatus> = {};
      studentList.forEach((s) => {
        initialStatuses[s.label] = "pending";
      });
      setStudentStatuses(initialStatuses);

      // Check if we have cached descriptors in localStorage
      setStatus("processing-dataset");
      setStatusMessage("Memproses dataset mahasiswa...");

      const cacheKey = "face-recognition-dataset-cache-v2";
      const cachedData = localStorage.getItem(cacheKey);

      let loadedDescriptors: faceapi.LabeledFaceDescriptors[] = [];

      if (cachedData) {
        try {
          setStatusMessage("Memuat deskriptor wajah dari cache lokal...");
          const parsed = JSON.parse(cachedData);

          loadedDescriptors = parsed.map((item: any) => {
            return new faceapi.LabeledFaceDescriptors(
              item.label,
              item.descriptors.map((d: number[]) => new Float32Array(d)),
            );
          });

          // Set all cached students as success
          const cachedStatuses: Record<string, StudentStatus> = {};
          studentList.forEach((s) => {
            const hasCache = parsed.some(
              (p: any) => p.label === s.label && p.descriptors.length > 0,
            );
            cachedStatuses[s.label] = hasCache ? "success" : "failed";
          });
          setStudentStatuses(cachedStatuses);

          console.log("Successfully loaded face descriptors from cache");
        } catch (cacheErr) {
          console.error(
            "Error parsing descriptor cache, recalculating...",
            cacheErr,
          );
          localStorage.removeItem(cacheKey);
        }
      }

      // If cache was empty or invalid, extract descriptors now
      if (loadedDescriptors.length === 0) {
        loadedDescriptors = await extractDescriptors(studentList);
      }

      if (loadedDescriptors.length === 0) {
        throw new Error(
          "Tidak ada data wajah mahasiswa yang berhasil diekstraksi dari dataset.",
        );
      }

      const distanceThreshold = 1 - threshold;
      const matcher = new faceapi.FaceMatcher(
        loadedDescriptors,
        distanceThreshold,
      );
      setFaceMatcher(matcher);

      setStatus("ready");
      setStatusMessage("Sistem Siap. Silakan aktifkan kamera.");
    } catch (error: any) {
      console.error("Initialization error:", error);
      setStatus("error");
      setStatusMessage(
        `Gagal menginisialisasi sistem: ${error.message || error}`,
      );
    }
  };

  const extractDescriptors = async (studentList: Student[]) => {
    const loadedDescriptors: faceapi.LabeledFaceDescriptors[] = [];
    const statuses: Record<string, StudentStatus> = {};
    const serializedCache: any[] = [];

    // Group images by student label to allow multiple pictures per student
    const studentGroups: Record<string, Student[]> = {};
    studentList.forEach((student) => {
      if (!studentGroups[student.label]) {
        studentGroups[student.label] = [];
      }
      studentGroups[student.label].push(student);
    });

    const labels = Object.keys(studentGroups);
    let currentIdx = 0;

    for (const label of labels) {
      currentIdx++;
      setProgress({
        current: currentIdx,
        total: labels.length,
        label: label,
      });

      const photos = studentGroups[label];
      const descriptorsForLabel: Float32Array[] = [];

      for (const photo of photos) {
        try {
          // Load image using face-api utility
          const img = await faceapi.fetchImage(photo.url);

          // Detect single face and compute features
          // SSD Mobilenet v1 is more accurate for extracting database features
          const detection = await faceapi
            .detectSingleFace(
              img,
              new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }),
            )
            .withFaceLandmarks()
            .withFaceDescriptor();

          if (detection) {
            descriptorsForLabel.push(detection.descriptor);
          } else {
            console.warn(`Wajah tidak terdeteksi pada foto: ${photo.filename}`);
          }
        } catch (photoErr) {
          console.error(`Gagal memproses gambar ${photo.filename}:`, photoErr);
        }
      }

      if (descriptorsForLabel.length > 0) {
        const labeledDesc = new faceapi.LabeledFaceDescriptors(
          label,
          descriptorsForLabel,
        );
        loadedDescriptors.push(labeledDesc);
        statuses[label] = "success";

        serializedCache.push({
          label: label,
          descriptors: descriptorsForLabel.map((d) => Array.from(d)),
        });
      } else {
        statuses[label] = "failed";
        console.error(
          `Gagal mendeteksi wajah sama sekali untuk mahasiswa: ${label}`,
        );
      }

      // Live update statuses
      setStudentStatuses((prev) => ({
        ...prev,
        [label]: descriptorsForLabel.length > 0 ? "success" : "failed",
      }));
    }

    // Save extracted descriptors to cache
    const cacheKey = "face-recognition-dataset-cache-v2";
    localStorage.setItem(cacheKey, JSON.stringify(serializedCache));

    return loadedDescriptors;
  };

  const resetCache = async () => {
    if (
      confirm(
        "Apakah Anda yakin ingin menghapus cache deskriptor wajah dan mengekstrak ulang dari dataset gambar?",
      )
    ) {
      const cacheKey = "face-recognition-dataset-cache-v2";
      localStorage.removeItem(cacheKey);
      stopCamera();
      initSystem();
    }
  };

  // Start Webcam
  const startCamera = async (deviceId?: string) => {
    try {
      stopCamera();

      const constraints: MediaStreamConstraints = {
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
        setIsDetecting(true);
        setStatusMessage("Kamera aktif. Melakukan deteksi...");
      }
    } catch (err: any) {
      console.error("Error starting webcam:", err);
      alert(`Gagal mengakses kamera: ${err.message || err}`);
    }
  };

  // Stop Webcam
  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => {
        track.stop();
      });
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setIsDetecting(false);

    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    if (status === "ready") {
      setStatusMessage("Kamera dimatikan.");
    }
  };

  // Triggered when video starts playing
  const handleVideoPlay = () => {
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }
    requestRef.current = requestAnimationFrame(detectFrame);
  };

  // Detection Loop
  const detectFrame = async () => {
    if (
      !videoRef.current ||
      videoRef.current.paused ||
      videoRef.current.ended ||
      !isDetecting ||
      !faceMatcher
    ) {
      requestRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.videoWidth && video.videoHeight && canvas) {
      // Ensure dimensions match
      if (
        canvas.width !== video.videoWidth ||
        canvas.height !== video.videoHeight
      ) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const displaySize = {
        width: video.videoWidth,
        height: video.videoHeight,
      };

      // Choose detection options
      let detectionOptions:
        | faceapi.TinyFaceDetectorOptions
        | faceapi.SsdMobilenetv1Options;
      if (detectorType === "tiny") {
        detectionOptions = new faceapi.TinyFaceDetectorOptions({
          inputSize: 224,
          scoreThreshold: 0.4,
        });
      } else {
        detectionOptions = new faceapi.SsdMobilenetv1Options({
          minConfidence: 0.5,
        });
      }

      // Perform detections
      const detections = await faceapi
        .detectAllFaces(video, detectionOptions)
        .withFaceLandmarks()
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      const ctx = canvas.getContext("2d");

      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Landmarks if enabled (futuristic blue points)
        if (showLandmarks && resizedDetections.length > 0) {
          resizedDetections.forEach((det) => {
            const landmarks = det.landmarks;
            const points = landmarks.positions;

            ctx.fillStyle = "rgba(6, 182, 212, 0.7)"; // Cyan
            points.forEach((pt) => {
              ctx.beginPath();
              ctx.arc(pt.x, pt.y, 2, 0, 2 * Math.PI);
              ctx.fill();
            });
          });
        }

        // Draw Bounding Boxes and Recognition Results
        resizedDetections.forEach((detection) => {
          const { descriptor, detection: det } = detection;

          // Re-evaluate distance threshold inside matching logic
          // faceMatcher was constructed with 1 - threshold
          const match = faceMatcher.findBestMatch(descriptor);

          const box = det.box;
          const isUnknown = match.label === "unknown";

          // Emerald Green for recognized, Red for unknown
          const color = isUnknown
            ? "rgba(239, 68, 68, 0.95)"
            : "rgba(16, 185, 129, 0.95)";
          const glowColor = isUnknown
            ? "rgba(239, 68, 68, 0.4)"
            : "rgba(16, 185, 129, 0.4)";

          // Draw Glow Box (cyberpunk aesthetic)
          ctx.shadowColor = glowColor;
          ctx.shadowBlur = 10;
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;

          const x = box.x;
          const y = box.y;
          const w = box.width;
          const h = box.height;
          const r = 6; // corner radius

          // Main rect
          ctx.beginPath();
          ctx.roundRect(x, y, w, h, r);
          ctx.stroke();

          // Reset shadows for details
          ctx.shadowBlur = 0;

          // Futuristic corner ticks
          const tickLen = Math.min(20, w * 0.2);
          ctx.lineWidth = 4;
          // Top-left
          ctx.beginPath();
          ctx.moveTo(x, y + tickLen);
          ctx.lineTo(x, y);
          ctx.lineTo(x + tickLen, y);
          ctx.stroke();
          // Top-right
          ctx.beginPath();
          ctx.moveTo(x + w, y + tickLen);
          ctx.lineTo(x + w, y);
          ctx.lineTo(x + w - tickLen, y);
          ctx.stroke();
          // Bottom-left
          ctx.beginPath();
          ctx.moveTo(x, y + h - tickLen);
          ctx.lineTo(x, y + h);
          ctx.lineTo(x + tickLen, y + h);
          ctx.stroke();
          // Bottom-right
          ctx.beginPath();
          ctx.moveTo(x + w, y + h - tickLen);
          ctx.lineTo(x + w, y + h);
          ctx.lineTo(x + w - tickLen, y + h);
          ctx.stroke();

          // Draw floating label card
          const confidence = isUnknown
            ? 0
            : Math.round((1 - match.distance) * 100);
          const nameText = isUnknown ? "TIDAK DIKENAL" : match.label;
          const confidenceText = isUnknown ? "" : ` [${confidence}%]`;

          ctx.font = "bold 16px monospace";
          const labelText = `${nameText}${confidenceText}`;
          const textWidth = ctx.measureText(labelText).width;

          // Save context, translate to middle of label block, and flip horizontally.
          // This cancels out the CSS scale-x(-1) mirror for the text!
          ctx.save();
          ctx.translate(x + textWidth / 2 + 8, y - 12);
          ctx.scale(-1, 1);

          // Draw label background centered at (0, 0)
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.roundRect(-textWidth / 2 - 8, -16, textWidth + 16, 24, 4);
          ctx.fill();

          // Draw label text centered at (0, 0)
          ctx.fillStyle = "#ffffff";
          ctx.fillText(labelText, -textWidth / 2, 2);

          ctx.restore();

          // Add to log if recognized
          if (!isUnknown && confidence >= threshold * 100) {
            logRecognition(match.label, confidence);
          }
        });
      }
    }

    requestRef.current = requestAnimationFrame(detectFrame);
  };

  const logRecognition = (name: string, confidence: number) => {
    const now = Date.now();
    const lastTime = lastLogged.current[name] || 0;

    // Cooldown 8 seconds to prevent flooding log table
    if (now - lastTime > 8000) {
      lastLogged.current[name] = now;

      const matchedStudent = dataset.find((s) => s.label === name);
      const timestamp = new Date().toLocaleTimeString("id-ID", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });

      setLogs((prev) => [
        {
          id: Math.random().toString(36).substring(2, 9),
          name,
          timestamp,
          confidence,
          photoUrl: matchedStudent?.url,
        },
        ...prev.slice(0, 49), // Limit to 50 logs
      ]);
    }
  };

  // Filter students based on search query
  const filteredDataset = dataset.filter((s) =>
    s.label.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-8 text-zinc-100 font-sans">
      {/* Page Title */}
      <header className="flex flex-col md:flex-row md:items-center justify-between border-b border-zinc-800 pb-6 mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">
            FACIAL RECOGNITION SYSTEM
          </h1>
          <p className="text-sm text-zinc-400 mt-1 font-mono">
            Sistem Deteksi Kehadiran Mahasiswa Real-Time Berbasis AI
          </p>
        </div>

        {/* Status Indicator */}
        <div className="flex items-center gap-3 bg-zinc-900/80 border border-zinc-800 px-4 py-2 rounded-xl backdrop-blur-md">
          <span
            className={`w-3 h-3 rounded-full animate-pulse ${
              status === "ready" && cameraActive
                ? "bg-emerald-500 shadow-[0_0_8px_#10b981]"
                : status === "ready"
                  ? "bg-amber-500 shadow-[0_0_8px_#f59e0b]"
                  : status === "error"
                    ? "bg-red-500 shadow-[0_0_8px_#ef4444]"
                    : "bg-cyan-500 shadow-[0_0_8px_#06b6d4]"
            }`}
          />
          <div className="text-xs font-mono">
            <span className="text-zinc-500 block uppercase tracking-wider text-[9px]">
              SISTEM STATUS
            </span>
            <span className="text-zinc-300 font-semibold">{statusMessage}</span>
          </div>
        </div>
      </header>

      {/* Main Content Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* LEFT COLUMN: Camera Feed & Controls (7 Cols) */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          {/* CAMERA FEED BOX */}
          <div className="relative aspect-video w-full rounded-2xl overflow-hidden bg-zinc-950 border border-zinc-800 shadow-2xl flex flex-col items-center justify-center group">
            {/* Hologram/Scanner lines */}
            {cameraActive && (
              <div className="absolute inset-0 pointer-events-none border border-emerald-500/20 rounded-2xl overflow-hidden z-10">
                <div className="w-full h-[2px] bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent absolute animate-[scan_3s_linear_infinite]" />
              </div>
            )}

            {/* Video Feed */}
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              onPlay={handleVideoPlay}
              className={`w-full h-full object-cover transform scale-x-[-1] ${cameraActive ? "block" : "hidden"}`}
            />

            {/* Drawing Canvas Overlays */}
            <canvas
              ref={canvasRef}
              className={`absolute top-0 left-0 w-full h-full transform scale-x-[-1] z-20 ${cameraActive ? "block" : "hidden"}`}
            />

            {/* Camera Offline Placeholder */}
            {!cameraActive && (
              <div className="flex flex-col items-center justify-center p-8 text-center z-10 transition-all duration-300">
                {status === "processing-dataset" ? (
                  <div className="flex flex-col items-center gap-4">
                    {/* Spinning loader */}
                    <div className="w-16 h-16 border-4 border-cyan-500/20 border-t-cyan-400 rounded-full animate-spin" />
                    <div className="text-sm font-mono max-w-sm">
                      <span className="text-cyan-400 font-bold block mb-1">
                        Mengekstrak Fitur Wajah...
                      </span>
                      <span className="text-zinc-400 block text-xs">
                        Mengindeks foto mahasiswa ke database AI lokal. Proses
                        ini hanya berjalan satu kali.
                      </span>
                      {progress.total > 0 && (
                        <div className="mt-3 w-full bg-zinc-800 rounded-full h-2 overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-cyan-400 to-blue-500 h-full transition-all duration-300"
                            style={{
                              width: `${(progress.current / progress.total) * 100}%`,
                            }}
                          />
                        </div>
                      )}
                      <span className="text-[10px] text-zinc-500 font-mono mt-1 block">
                        {progress.current} / {progress.total} : {progress.label}
                      </span>
                    </div>
                  </div>
                ) : status === "loading-models" ? (
                  <div className="flex flex-col items-center gap-4">
                    <div className="w-12 h-12 border-4 border-zinc-800 border-t-amber-400 rounded-full animate-spin" />
                    <span className="text-sm font-mono text-zinc-400">
                      Memuat Model Pembelajaran Mesin...
                    </span>
                  </div>
                ) : status === "error" ? (
                  <div className="flex flex-col items-center text-red-400 gap-3">
                    <svg
                      className="w-12 h-12"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                      />
                    </svg>
                    <span className="text-sm font-mono">{statusMessage}</span>
                    <button
                      onClick={initSystem}
                      className="mt-2 px-4 py-2 bg-zinc-800 border border-zinc-700 hover:bg-zinc-700 rounded-xl text-xs font-semibold"
                    >
                      Coba Lagi
                    </button>
                  </div>
                ) : (
                  <div
                    className="flex flex-col items-center gap-4 cursor-pointer"
                    onClick={() => startCamera(selectedDevice)}
                  >
                    <div className="w-20 h-20 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-600 group-hover:text-emerald-400 group-hover:border-emerald-500/50 group-hover:shadow-[0_0_15px_rgba(16,185,129,0.1)] transition-all duration-300">
                      <svg
                        className="w-10 h-10"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                        />
                      </svg>
                    </div>
                    <span className="text-sm font-semibold tracking-wider font-mono text-zinc-500 group-hover:text-zinc-300 transition-colors">
                      KLIK UNTUK MENGAKTIFKAN KAMERA
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Grid overlay for cyberpunk feel */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.01)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.01)_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />
          </div>

          {/* CONTROLS CARD */}
          <div className="bg-zinc-900/60 border border-zinc-800 rounded-2xl p-6 backdrop-blur-md">
            <h2 className="text-lg font-bold font-mono tracking-wider border-b border-zinc-800 pb-3 mb-4 text-zinc-300">
              KONTROL & PARAMETER
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Left Column Controls */}
              <div className="flex flex-col gap-4">
                {/* Camera Toggle */}
                <div className="flex flex-col gap-2">
                  <label className="text-xs font-mono text-zinc-400 uppercase tracking-wider">
                    Kamera Feed
                  </label>
                  <div className="flex gap-2">
                    <button
                      onClick={() =>
                        cameraActive
                          ? stopCamera()
                          : startCamera(selectedDevice)
                      }
                      disabled={status !== "ready"}
                      className={`flex-1 py-2.5 px-4 rounded-xl font-bold font-mono transition-all duration-300 flex items-center justify-center gap-2 border text-sm ${
                        cameraActive
                          ? "bg-red-500/10 border-red-500/30 text-red-400 hover:bg-red-500/20"
                          : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20"
                      }`}
                    >
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        {cameraActive ? (
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636"
                          />
                        ) : (
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                          />
                        )}
                      </svg>
                      {cameraActive ? "NONAKTIFKAN" : "AKTIFKAN KAMERA"}
                    </button>

                    <button
                      onClick={resetCache}
                      disabled={status !== "ready"}
                      title="Reset database wajah lokal (hapus cache)"
                      className="p-2.5 bg-zinc-800 border border-zinc-700 hover:bg-zinc-700 rounded-xl transition-colors text-zinc-300 disabled:opacity-50"
                    >
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 7.89M9 11l3-3 3 3m-3-3v12"
                        />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* Camera Input Selection */}
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-mono text-zinc-400 uppercase tracking-wider">
                    Pilih Kamera
                  </label>
                  <select
                    value={selectedDevice}
                    onChange={(e) => {
                      setSelectedDevice(e.target.value);
                      if (cameraActive) startCamera(e.target.value);
                    }}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded-xl px-3 py-2 text-sm text-zinc-300 focus:outline-none focus:border-cyan-500 font-mono"
                  >
                    {devices.length === 0 ? (
                      <option value="">Default Camera</option>
                    ) : (
                      devices.map((device, idx) => (
                        <option key={device.deviceId} value={device.deviceId}>
                          {device.label || `Camera ${idx + 1}`}
                        </option>
                      ))
                    )}
                  </select>
                </div>
              </div>

              {/* Right Column Controls */}
              <div className="flex flex-col gap-4">
                {/* Detector Type Selector */}
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs font-mono text-zinc-400 uppercase tracking-wider">
                    Model Detektor Wajah
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => setDetectorType("tiny")}
                      className={`py-2 px-3 border rounded-xl text-xs font-mono font-semibold transition-all duration-300 ${
                        detectorType === "tiny"
                          ? "bg-cyan-500/10 border-cyan-500 text-cyan-400"
                          : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                      }`}
                    >
                      TINY DETECTOR (Cepat)
                    </button>
                    <button
                      onClick={() => setDetectorType("ssd")}
                      className={`py-2 px-3 border rounded-xl text-xs font-mono font-semibold transition-all duration-300 ${
                        detectorType === "ssd"
                          ? "bg-cyan-500/10 border-cyan-500 text-cyan-400"
                          : "bg-zinc-950 border-zinc-800 text-zinc-400 hover:border-zinc-700"
                      }`}
                    >
                      SSD MOBILENET (Akurat)
                    </button>
                  </div>
                </div>

                {/* Matcher Threshold */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex justify-between items-center">
                    <label className="text-xs font-mono text-zinc-400 uppercase tracking-wider">
                      Akurasi Kemiripan Minimum
                    </label>
                    <span className="text-xs font-bold font-mono text-cyan-400">
                      {Math.round(threshold * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.3"
                    max="0.8"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => {
                      const newThreshold = parseFloat(e.target.value);
                      setThreshold(newThreshold);
                      if (faceMatcher) {
                        rebuildFaceMatcherWithThreshold(
                          faceMatcher.labeledDescriptors,
                        );
                      }
                    }}
                    className="w-full accent-cyan-500 bg-zinc-950 rounded-lg appearance-none h-2 border border-zinc-800 cursor-pointer"
                  />
                  <span className="text-[10px] text-zinc-500 font-mono">
                    *Makin tinggi persen, pencocokan makin ketat (menghindari
                    salah deteksi).
                  </span>
                </div>

                {/* Checkbox for Landmark display */}
                <div className="flex items-center gap-2 mt-1">
                  <input
                    type="checkbox"
                    id="showLandmarks"
                    checked={showLandmarks}
                    onChange={(e) => setShowLandmarks(e.target.checked)}
                    className="w-4 h-4 accent-cyan-500 bg-zinc-950 border border-zinc-800 rounded"
                  />
                  <label
                    htmlFor="showLandmarks"
                    className="text-xs font-semibold text-zinc-300 select-none cursor-pointer"
                  >
                    Tampilkan Titik Landmark Wajah (Visualizer AI)
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Dataset List & Logs (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6">
          {/* TAB SYSTEM: DATASET MAHASISWA & LOG RIWAYAT */}
          <div className="flex flex-col flex-1 bg-zinc-900/60 border border-zinc-800 rounded-2xl overflow-hidden backdrop-blur-md min-h-[550px] max-h-[600px]">
            {/* Headers / Navigation */}
            <div className="flex bg-zinc-950 border-b border-zinc-800 p-2 gap-2">
              <div className="flex-1 text-center py-2 px-3 bg-zinc-900 rounded-xl border border-zinc-800/80">
                <span className="text-xs font-bold font-mono tracking-wider text-cyan-400 block">
                  DATABASE DATASET ({dataset.length})
                </span>
              </div>
            </div>

            {/* Content Body */}
            <div className="flex flex-col flex-1 p-5 overflow-hidden">
              {/* Search Bar for dataset */}
              <div className="relative mb-4">
                <input
                  type="text"
                  placeholder="Cari nama mahasiswa..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 hover:border-zinc-700 rounded-xl pl-10 pr-4 py-2 text-sm text-zinc-300 focus:outline-none focus:border-cyan-500 font-sans"
                />
                <div className="absolute left-3.5 top-2.5 text-zinc-500">
                  <svg
                    className="w-4.5 h-4.5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                  </svg>
                </div>
              </div>

              {/* Dataset Scroll Area */}
              <div className="flex-1 overflow-y-auto pr-1 flex flex-col gap-2.5">
                {filteredDataset.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-zinc-500 font-mono text-xs">
                    <span>Data mahasiswa tidak ditemukan</span>
                  </div>
                ) : (
                  filteredDataset.map((student) => {
                    const studentStatus =
                      studentStatuses[student.label] || "pending";
                    return (
                      <div
                        key={student.filename}
                        onClick={() => setSelectedPreviewStudent(student)}
                        className="flex items-center justify-between p-2.5 bg-zinc-950/80 border border-zinc-800 hover:border-zinc-700/80 rounded-xl transition-all duration-200 cursor-pointer group"
                      >
                        <div className="flex items-center gap-3">
                          {/* Mini Thumbnail */}
                          <div className="w-10 h-10 rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800 flex-shrink-0 relative">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={student.url}
                              alt={student.label}
                              className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                              onError={(e) => {
                                // Fallback icon
                                (e.target as HTMLElement).style.display =
                                  "none";
                              }}
                            />
                          </div>

                          <div>
                            <span className="text-sm font-semibold text-zinc-200 block group-hover:text-cyan-400 transition-colors">
                              {student.label}
                            </span>
                            <span className="text-[10px] text-zinc-500 font-mono block">
                              {student.filename}
                            </span>
                          </div>
                        </div>

                        {/* Status Badge */}
                        <div className="flex items-center gap-2">
                          {studentStatus === "success" && (
                            <span className="flex items-center gap-1 text-[10px] font-bold font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                              READY
                            </span>
                          )}
                          {studentStatus === "failed" && (
                            <span className="flex items-center gap-1 text-[10px] font-bold font-mono px-2 py-0.5 rounded-full bg-red-500/10 border border-red-500/20 text-red-400">
                              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                              NO FACE
                            </span>
                          )}
                          {studentStatus === "pending" && (
                            <span className="flex items-center gap-1 text-[10px] font-bold font-mono px-2 py-0.5 rounded-full bg-zinc-800 border border-zinc-700 text-zinc-400 animate-pulse">
                              WAITING
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* PRESENCE LOGS CARD */}
          <div className="flex flex-col flex-1 bg-zinc-900/60 border border-zinc-800 rounded-2xl overflow-hidden backdrop-blur-md min-h-[350px] max-h-[400px]">
            <div className="bg-zinc-950 border-b border-zinc-800 p-4 flex justify-between items-center">
              <h2 className="text-sm font-bold font-mono tracking-wider text-emerald-400 uppercase flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping" />
                RIWAYAT PRESENSI KEHADIRAN (LIVE)
              </h2>
              {logs.length > 0 && (
                <button
                  onClick={() => setLogs([])}
                  className="text-[10px] font-bold font-mono hover:text-red-400 text-zinc-500 uppercase transition-colors"
                >
                  CLEAR LOG
                </button>
              )}
            </div>

            <div className="flex-1 p-4 overflow-y-auto flex flex-col gap-2">
              {logs.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 text-zinc-500 font-mono text-xs h-full">
                  <span>Belum ada wajah mahasiswa terdeteksi</span>
                  <span className="text-[10px] text-zinc-600 mt-1">
                    Nyalakan kamera & dekatkan wajah ke lensa
                  </span>
                </div>
              ) : (
                logs.map((log) => (
                  <div
                    key={log.id}
                    className="flex items-center justify-between p-2.5 bg-zinc-950/40 border border-zinc-800/60 rounded-xl animate-[fadeIn_0.3s_ease-out]"
                  >
                    <div className="flex items-center gap-3">
                      {/* Student Profile Thumbnail */}
                      <div className="w-8 h-8 rounded-lg overflow-hidden bg-zinc-900 border border-zinc-800 flex-shrink-0">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={log.photoUrl || ""}
                          alt={log.name}
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            (e.target as HTMLElement).style.display = "none";
                          }}
                        />
                      </div>

                      <div>
                        <span className="text-xs font-semibold text-zinc-200 block">
                          {log.name}
                        </span>
                        <span className="text-[9px] text-zinc-500 font-mono block">
                          Kemiripan:{" "}
                          <span className="text-emerald-400 font-bold">
                            {log.confidence}%
                          </span>
                        </span>
                      </div>
                    </div>

                    <div className="text-right">
                      <span className="text-[10px] font-bold font-mono px-2 py-0.5 rounded-lg bg-emerald-500/10 border border-emerald-500/10 text-emerald-400 block">
                        HADIR
                      </span>
                      <span className="text-[9px] text-zinc-500 font-mono block mt-0.5">
                        {log.timestamp}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* STUDENT PREVIEW MODAL */}
      {selectedPreviewStudent && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-[fadeIn_0.2s_ease-out]">
          <div className="bg-zinc-900 border border-zinc-800 w-full max-w-md rounded-2xl overflow-hidden shadow-2xl relative">
            {/* Modal Header */}
            <div className="p-4 border-b border-zinc-800 flex justify-between items-center bg-zinc-950">
              <span className="text-sm font-bold font-mono tracking-wider text-cyan-400">
                DETAIL MAHASISWA
              </span>
              <button
                onClick={() => setSelectedPreviewStudent(null)}
                className="text-zinc-500 hover:text-zinc-300 p-1 transition-colors"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 flex flex-col items-center gap-4">
              <div className="w-64 h-64 rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950 relative shadow-inner">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={selectedPreviewStudent.url}
                  alt={selectedPreviewStudent.label}
                  className="w-full h-full object-cover"
                />
              </div>

              <div className="text-center">
                <h3 className="text-xl font-bold text-zinc-100">
                  {selectedPreviewStudent.label}
                </h3>
                <p className="text-xs font-mono text-zinc-500 mt-1">
                  {selectedPreviewStudent.filename}
                </p>
              </div>

              <div className="w-full grid grid-cols-2 gap-3 text-center mt-2 text-xs font-mono">
                <div className="bg-zinc-950 p-2.5 rounded-xl border border-zinc-800">
                  <span className="text-zinc-500 block text-[9px] uppercase">
                    Status AI
                  </span>
                  <span
                    className={`font-bold mt-0.5 block ${
                      studentStatuses[selectedPreviewStudent.label] ===
                      "success"
                        ? "text-emerald-400"
                        : "text-amber-500"
                    }`}
                  >
                    {studentStatuses[selectedPreviewStudent.label] === "success"
                      ? "FITUR SIAP"
                      : "BELUM SCAN"}
                  </span>
                </div>

                <div className="bg-zinc-950 p-2.5 rounded-xl border border-zinc-800">
                  <span className="text-zinc-500 block text-[9px] uppercase">
                    Jumlah Foto
                  </span>
                  <span className="text-cyan-400 font-bold mt-0.5 block">
                    1 Gambar
                  </span>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 bg-zinc-950 border-t border-zinc-800 flex justify-end">
              <button
                onClick={() => setSelectedPreviewStudent(null)}
                className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-xs font-semibold rounded-xl transition-colors text-zinc-200"
              >
                Tutup
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Dynamic scan anim style */}
      <style
        dangerouslySetInnerHTML={{
          __html: `
        @keyframes scan {
          0% { top: 0%; }
          50% { top: 100%; }
          100% { top: 0%; }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: scale(0.95); }
          to { opacity: 1; transform: scale(1); }
        }
      `,
        }}
      />
    </div>
  );
}
