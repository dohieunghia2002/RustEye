"use client"
import { useEffect, useRef, useState } from "react";
import NextImage from "next/image";

/**
 * RustEye — Hệ thống phát hiện và đánh giá hao mòn bề mặt trên cột trụ viễn thông
 * Frontend Next.js (pages/index.tsx)
 *
 * Back-end API (Flask) mặc định: http://localhost:5000/predict
 */

const BACKEND_URL: string = "http://localhost:5000/predict";

export default function RustEye() {
  const [file, setFile] = useState<File | null>(null);
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [opacity, setOpacity] = useState<number>(0.2);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [compositedReady, setCompositedReady] = useState<boolean>(false);

  const onFileSelected = (f: File) => {
    if (!f) return;
    setError("");
    setFile(f);
    const url = URL.createObjectURL(f);
    setImgUrl(url);
    setMaskUrl(null);
    setCompositedReady(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleBrowse = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelected(e.target.files[0]);
    }
  };

  const requestPredict = async () => {
    if (!file) {
      setError("Vui lòng chọn ảnh trước.");
      return;
    }
    setIsLoading(true);
    setError("");
    setMaskUrl(null);
    setCompositedReady(false);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(BACKEND_URL, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`API lỗi: ${res.status} ${txt}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setMaskUrl(url);
    } catch (err: any) {
      setError(err.message || "Có lỗi khi gọi API.");
    } finally {
      setIsLoading(false);
    }
  };

  const redrawComposite = async () => {
    if (!imgUrl || !maskUrl) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const [img, mask] = await Promise.all([
      loadImage(imgUrl),
      loadImage(maskUrl),
    ]);

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tctx = tempCanvas.getContext("2d");
    if (!tctx) return;

    tctx.drawImage(mask, 0, 0, canvas.width, canvas.height);
    const maskData = tctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = maskData.data;

    const overlay = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const od = overlay.data;

    for (let i = 0; i < od.length; i += 4) {
      const m = data[i];
      if (m > 0) {
        const a = Math.min(255, Math.floor(m * opacity));
        od[i] = Math.max(od[i], 255);
        od[i + 1] = od[i + 1] * (1 - opacity);
        od[i + 2] = od[i + 2] * (1 - opacity);
        od[i + 3] = Math.max(od[i + 3], a);
      }
    }

    ctx.putImageData(overlay, 0, 0);
    setCompositedReady(true);
  };

  useEffect(() => {
    redrawComposite();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [maskUrl, imgUrl, opacity]);

  const downloadComposite = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = `rusteye_result_${Date.now()}.png`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900">
      <header className="sticky top-0 bg-white/80 backdrop-blur z-10 border-b">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 flex items-center justify-center shadow">
              <NextImage
                src="/ctu-logo.png"
                alt="CTU Logo"
                width={44}
                height={44}
              />
            </div>
            <div>
              <h1 className="font-bold text-xl">RustEye</h1>
              <p className="text-xs text-neutral-500">
                Hệ thống phát hiện & đánh giá hao mòn bề mặt trên cột trụ viễn thông
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-neutral-500">
            <NextImage
              src="/vnpt-logo.jpg"
              alt="VNPT Logo"
              width={20}
              height={20}
            />
            <span className="text-[#006db6]">VNPT AN GIANG</span>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">
        <section
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="border-2 border-dashed rounded-2xl p-6 bg-white shadow-sm flex flex-col items-center justify-center gap-3"
        >
          <p className="text-base font-medium">
            Kéo thả ảnh cột trụ vào đây hoặc chọn từ máy
          </p>
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleBrowse}
            className="hidden"
          />
          <div className="flex items-center gap-3">
            <label
              htmlFor="file-input"
              className="px-4 py-2 rounded-2xl bg-neutral-900 text-white cursor-pointer shadow"
            >
              Chọn ảnh
            </label>
            <button
              onClick={requestPredict}
              disabled={!file || isLoading}
              className="px-4 py-2 rounded-2xl bg-orange-500 text-white disabled:opacity-50 shadow"
            >
              {isLoading ? "Đang phân tích…" : "Phân tích với RustEye"}
            </button>
          </div>
          {file && (
            <p className="text-xs text-neutral-500">
              Đã chọn: {file.name} ({Math.round(file.size / 1024)} KB)
            </p>
          )}
          {error && (
            <div className="text-sm text-red-600 font-medium">{error}</div>
          )}
        </section>

        <section className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-2xl shadow-sm p-4 border">
            <h2 className="font-semibold mb-3">Ảnh gốc</h2>
            {imgUrl ? (
              <img
                src={imgUrl}
                alt="original"
                className="w-full h-auto rounded-xl border"
              />
            ) : (
              <div className="h-64 grid place-content-center text-neutral-400">
                Chưa có ảnh
              </div>
            )}
          </div>

          <div className="bg-white rounded-2xl shadow-sm p-4 border">
            <h2 className="font-semibold mb-3">Mask từ mô hình</h2>
            {maskUrl ? (
              <img
                src={maskUrl}
                alt="mask"
                className="w-full h-auto rounded-xl border"
              />
            ) : (
              <div className="h-64 grid place-content-center text-neutral-400">
                Chưa có mask
              </div>
            )}
          </div>
        </section>

        <section className="mt-8 bg-white rounded-2xl shadow-sm p-4 border">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Overlay (ảnh gốc + mask)</h2>
            <div className="flex items-center gap-3">
              <label className="text-sm text-neutral-600">
                Độ đậm vùng hư hại: {Math.round(opacity * 100)}%
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
              />
              <button
                onClick={downloadComposite}
                disabled={!compositedReady}
                className="px-3 py-1.5 rounded-xl bg-neutral-900 text-white disabled:opacity-40"
              >
                Tải ảnh kết quả
              </button>
            </div>
          </div>
          <div className="w-full overflow-auto">
            <canvas ref={canvasRef} className="max-w-full rounded-xl border" />
          </div>
          {!imgUrl && (
            <p className="text-sm text-neutral-400 mt-3">
              Chọn ảnh và chạy phân tích để hiển thị overlay.
            </p>
          )}
        </section>

        <footer className="text-center text-xs text-neutral-500 mt-8">
          Đại học Cần Thơ 2025
        </footer>
      </main>
    </div>
  );
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
