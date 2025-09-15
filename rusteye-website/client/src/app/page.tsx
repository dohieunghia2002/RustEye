"use client";
import { useEffect, useRef, useState } from "react";
import NextImage from "next/image";

const BACKEND_URL = "http://localhost:5000";

interface PredictResult {
  mask_rust_url: string;
  mask_pole_url: string;
  mask_sam2_url: string;
  isnet: {
    rust_percentage: number;
    fuzzy_level: number;
    severity: number;
    action: string;
  };
  sam2: {
    rust_percentage: number;
    fuzzy_level: number;
    severity: number;
    action: string;
  };
}

export default function RustEye() {
  const [file, setFile] = useState<File | null>(null);
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [maskRustUrl, setMaskRustUrl] = useState<string | null>(null);
  const [maskPoleUrl, setMaskPoleUrl] = useState<string | null>(null);
  const [maskSam2Url, setMaskSam2Url] = useState<string | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResult | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const canvasIsnetRef = useRef<HTMLCanvasElement | null>(null);
  const canvasSam2Ref = useRef<HTMLCanvasElement | null>(null);
  const [compositedIsnetReady, setCompositedIsnetReady] = useState<boolean>(false);
  const [compositedSam2Ready, setCompositedSam2Ready] = useState<boolean>(false);
  const [opacity, setOpacity] = useState<number>(0.2);

  const onFileSelected = (f: File) => {
    if (!f) return;
    setError("");
    setFile(f);
    const url = URL.createObjectURL(f);
    setImgUrl(url);
    setMaskRustUrl(null);
    setMaskPoleUrl(null);
    setMaskSam2Url(null);
    setPredictResult(null);
    setCompositedIsnetReady(false);
    setCompositedSam2Ready(false);
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
    setMaskRustUrl(null);
    setMaskPoleUrl(null);
    setMaskSam2Url(null);
    setPredictResult(null);
    setCompositedIsnetReady(false);
    setCompositedSam2Ready(false);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`API lỗi: ${res.status} ${txt}`);
      }

      const data: PredictResult = await res.json();

      setMaskRustUrl(`${BACKEND_URL}${data.mask_rust_url}`);
      setMaskPoleUrl(`${BACKEND_URL}${data.mask_pole_url}`);
      setMaskSam2Url(`${BACKEND_URL}${data.mask_sam2_url}`);
      setPredictResult(data);
    } catch (err: any) {
      setError(err.message || "Có lỗi khi gọi API.");
    } finally {
      setIsLoading(false);
    }
  };

  const redrawComposite = async (
    canvasRef: React.RefObject<HTMLCanvasElement | null>,
    setReady: React.Dispatch<React.SetStateAction<boolean>>,
    maskUrl: string | null,
    maskRustUrl: string | null,
    isSam2: boolean
  ) => {
    if (!imgUrl || !maskUrl || !maskRustUrl || !canvasRef.current) return;
    const canvas = canvasRef.current;

    const [img, mask, maskRust] = await Promise.all([
      loadImage(imgUrl),
      loadImage(maskUrl),
      loadImage(maskRustUrl),
    ]);

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = canvas.width;
    maskCanvas.height = canvas.height;
    const mctx = maskCanvas.getContext("2d")!;
    mctx.drawImage(mask, 0, 0, canvas.width, canvas.height);
    const maskData = mctx.getImageData(0, 0, canvas.width, canvas.height);

    const rustCanvas = document.createElement("canvas");
    rustCanvas.width = canvas.width;
    rustCanvas.height = canvas.height;
    const rctx = rustCanvas.getContext("2d")!;
    rctx.drawImage(maskRust, 0, 0, canvas.width, canvas.height);
    const rustData = rctx.getImageData(0, 0, canvas.width, canvas.height);

    const overlay = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const od = overlay.data;

    for (let i = 0; i < od.length; i += 4) {
      const mValue = maskData.data[i]; // Kênh R của mask cột
      const rValue = rustData.data[i]; // Kênh R của mask hao mòn

      if (mValue > 0) {
        if (isSam2) {
          // SAM2: Xanh dương cho cột
          od[i] = od[i] * (1 - opacity); // Red
          od[i + 1] = od[i + 1] * (1 - opacity); // Green
          od[i + 2] = Math.max(od[i + 2], 255); // Blue
        } else {
          // ISNet: Xanh dương cho cột
          od[i] = od[i] * (1 - opacity); // Red
          od[i + 1] = od[i + 1] * (1 - opacity); // Green
          od[i + 2] = Math.max(od[i + 2], 255); // Blue
        }
        od[i + 3] = 255; // Alpha
      }

      if (rValue > 0) {
        // Đỏ cho hao mòn (YOLO), ưu tiên hơn cột nếu chồng lấp
        od[i] = 255; // Red
        od[i + 1] = od[i + 1] * (1 - opacity); // Green
        od[i + 2] = od[i + 2] * (1 - opacity); // Blue
        od[i + 3] = 255; // Alpha
      }
    }

    ctx.putImageData(overlay, 0, 0);
    setReady(true);
  };

  useEffect(() => {
    redrawComposite(canvasIsnetRef, setCompositedIsnetReady, maskPoleUrl, maskRustUrl, false); // ISNet
    redrawComposite(canvasSam2Ref, setCompositedSam2Ready, maskSam2Url, maskRustUrl, true); // SAM2
  }, [maskPoleUrl, maskSam2Url, maskRustUrl, imgUrl, opacity]);

  const downloadComposite = (canvasRef: React.RefObject<HTMLCanvasElement | null>, prefix: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = `${prefix}_result_${Date.now()}.png`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900">
      <header className="sticky top-0 bg-white/80 backdrop-blur z-10 border-b">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 flex items-center justify-center shadow">
              <NextImage src="/ctu-logo.png" alt="CTU Logo" width={44} height={44} />
            </div>
            <div>
              <h1 className="font-bold text-xl">RustEye</h1>
              <p className="text-xs text-neutral-500">
                Hỗ Trợ Quyết Định Bảo Trì Trụ Cột Viễn Thông Với Thị Giác Máy Tính và Fuzzy-Contraints
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-neutral-500">
            <NextImage src="/vnpt-logo.jpg" alt="VNPT Logo" width={20} height={20} />
            <span className="text-[#006db6]">VNPT AN GIANG</span>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">
        {/* Upload Section */}
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
          {error && <div className="text-sm text-red-600 font-medium">{error}</div>}
        </section>

        {/* Hàng 1: Ảnh gốc, Mask hao mòn (YOLO) */}
        <section className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-2xl shadow-sm p-4 border">
            <h2 className="font-semibold mb-3">Ảnh gốc</h2>
            {imgUrl ? (
              <img src={imgUrl} alt="original" className="w-full h-auto rounded-xl border" />
            ) : (
              <div className="h-64 grid place-content-center text-neutral-400">
                Chưa có ảnh
              </div>
            )}
          </div>
          <div className="bg-white rounded-2xl shadow-sm p-4 border">
            <h2 className="font-semibold mb-3">Mask hao mòn (YOLO)</h2>
            {maskRustUrl ? (
              <img src={maskRustUrl} alt="mask_rust" className="w-full h-auto rounded-xl border" />
            ) : (
              <div className="h-64 grid place-content-center text-neutral-400">
                Chưa có mask
              </div>
            )}
          </div>
        </section>

        {/* Hàng 2: Mask cột (ISNet), Mask cột (SAM2) */}
        <section className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-2xl shadow-sm p-4 border">
            <h2 className="font-semibold mb-3">Mask cột (ISNet)</h2>
            {maskPoleUrl ? (
              <img src={maskPoleUrl} alt="mask_pole_isnet" className="w-full h-auto rounded-xl border" />
            ) : (
              <div className="h-64 grid place-content-center text-neutral-400">
                Chưa có mask
              </div>
            )}
          </div>
          <div className="bg-white rounded-2xl shadow-sm p-4 border">
            <h2 className="font-semibold mb-3">Mask cột (SAM2)</h2>
            {maskSam2Url ? (
              <img src={maskSam2Url} alt="mask_pole_sam2" className="w-full h-auto rounded-xl border" />
            ) : (
              <div className="h-64 grid place-content-center text-neutral-400">
                Chưa có mask
              </div>
            )}
          </div>
        </section>

        {/* Hàng 3: Kết quả từ ISNet, Kết quả từ SAM2 */}
        {predictResult && (
          <section className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl shadow-sm p-4 border">
              <h2 className="font-semibold mb-3">Kết quả từ ISNet</h2>
              <ul className="text-sm space-y-1">
                <li><span className="font-medium">Phần trăm hao mòn:</span> {predictResult.isnet.rust_percentage.toFixed(2)}%</li>
                <li><span className="font-medium">Fuzzy level:</span> {predictResult.isnet.fuzzy_level.toFixed(2)}</li>
                <li><span className="font-medium">Mức độ hư hại:</span> {predictResult.isnet.severity}</li>
                <li><span className="font-medium">Khuyến nghị:</span> <span className="text-orange-600 font-semibold">{predictResult.isnet.action}</span></li>
              </ul>
            </div>
            <div className="bg-white rounded-2xl shadow-sm p-4 border">
              <h2 className="font-semibold mb-3">Kết quả từ SAM2</h2>
              <ul className="text-sm space-y-1">
                <li><span className="font-medium">Phần trăm hao mòn:</span> {predictResult.sam2.rust_percentage.toFixed(2)}%</li>
                <li><span className="font-medium">Fuzzy level:</span> {predictResult.sam2.fuzzy_level.toFixed(2)}</li>
                <li><span className="font-medium">Mức độ hư hại:</span> {predictResult.sam2.severity}</li>
                <li><span className="font-medium">Khuyến nghị:</span> <span className="text-orange-600 font-semibold">{predictResult.sam2.action}</span></li>
              </ul>
            </div>
          </section>
        )}

        {/* Hàng 4: Overlay (ảnh gốc + mask ISNet) */}
        <section className="mt-8 bg-white rounded-2xl shadow-sm p-4 border">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Overlay (ảnh gốc + mask ISNet)</h2>
            <button
              onClick={() => downloadComposite(canvasIsnetRef, "isnet")}
              disabled={!compositedIsnetReady}
              className="px-3 py-1.5 rounded-xl bg-neutral-900 text-white disabled:opacity-40"
            >
              Tải ảnh kết quả
            </button>
          </div>
          <div className="flex items-center gap-6 mb-2 text-sm">
            <div className="flex items-center gap-1">
              <span className="w-4 h-4 rounded-sm inline-block" style={{ backgroundColor: "blue" }}></span>
              <span>Cột (ISNet)</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-4 h-4 rounded-sm inline-block" style={{ backgroundColor: "red" }}></span>
              <span>Hư hại (YOLO)</span>
            </div>
          </div>
          <div className="w-full overflow-auto">
            <canvas ref={canvasIsnetRef} className="max-w-full rounded-xl border" />
          </div>
          {!imgUrl && (
            <p className="text-sm text-neutral-400 mt-3">
              Chọn ảnh và chạy phân tích để hiển thị overlay.
            </p>
          )}
        </section>

        {/* Hàng 5: Overlay (ảnh gốc + mask SAM2) */}
        <section className="mt-8 bg-white rounded-2xl shadow-sm p-4 border">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold">Overlay (ảnh gốc + mask SAM2)</h2>
            <button
              onClick={() => downloadComposite(canvasSam2Ref, "sam2")}
              disabled={!compositedSam2Ready}
              className="px-3 py-1.5 rounded-xl bg-neutral-900 text-white disabled:opacity-40"
            >
              Tải ảnh kết quả
            </button>
          </div>
          <div className="flex items-center gap-6 mb-2 text-sm">
            <div className="flex items-center gap-1">
              <span className="w-4 h-4 rounded-sm inline-block" style={{ backgroundColor: "blue" }}></span>
              <span>Cột (SAM2)</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="w-4 h-4 rounded-sm inline-block" style={{ backgroundColor: "red" }}></span>
              <span>Hư hại (YOLO)</span>
            </div>
          </div>
          <div className="w-full overflow-auto">
            <canvas ref={canvasSam2Ref} className="max-w-full rounded-xl border" />
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
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}