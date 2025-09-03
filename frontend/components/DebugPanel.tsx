'use client';

interface DebugPanelProps {
  similarity: number;
  rank: number;
}

export default function DebugPanel({ similarity, rank }: DebugPanelProps) {
  return (
    <div className="absolute top-2 right-2 bg-black/80 text-white text-xs p-2 rounded-lg shadow-lg backdrop-blur-sm z-10">
      <div className="flex flex-col gap-1">
        <div className="flex justify-between items-center gap-2">
          <span className="text-gray-300">Score:</span>
          <span className="font-mono font-bold text-green-400">{similarity.toFixed(3)}</span>
        </div>
        <div className="flex justify-between items-center gap-2">
          <span className="text-gray-300">Rank:</span>
          <span className="font-mono font-bold text-blue-400">#{rank}</span>
        </div>
      </div>
    </div>
  );
}