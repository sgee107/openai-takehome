'use client';

import { useEffect } from 'react';
import { EyeIcon, EyeSlashIcon } from '@heroicons/react/24/outline';

interface DebugToggleProps {
  debugMode: boolean;
  onToggle: (enabled: boolean) => void;
}

export default function DebugToggle({ debugMode, onToggle }: DebugToggleProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'd') {
        e.preventDefault();
        onToggle(!debugMode);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [debugMode, onToggle]);

  return (
    <div className="fixed bottom-6 right-6 z-40">
      <button
        onClick={() => onToggle(!debugMode)}
        className={`p-3 rounded-full shadow-lg backdrop-blur-sm transition-all duration-200 ${
          debugMode 
            ? 'bg-blue-500 text-white hover:bg-blue-600' 
            : 'bg-white/80 text-gray-700 hover:bg-white border border-gray-300'
        }`}
        title={`${debugMode ? 'Disable' : 'Enable'} debug mode (Cmd/Ctrl + D)`}
      >
        {debugMode ? (
          <EyeSlashIcon className="h-6 w-6" />
        ) : (
          <EyeIcon className="h-6 w-6" />
        )}
      </button>
      
      {debugMode && (
        <div className="absolute bottom-full right-0 mb-2 px-3 py-1 bg-black/80 text-white text-xs rounded-lg whitespace-nowrap">
          Debug mode active
        </div>
      )}
    </div>
  );
}