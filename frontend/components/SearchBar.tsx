'use client';

import { useState, FormEvent } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';

interface SearchBarProps {
  onSearch: (query: string) => void;
  loading: boolean;
  initialQuery?: string;
}

export default function SearchBar({ onSearch, loading, initialQuery = '' }: SearchBarProps) {
  const [query, setQuery] = useState(initialQuery);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative flex items-center">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for fashion items... (e.g., 'blue shirts', 'running shoes')"
            className="w-full pl-4 pr-12 py-4 text-lg border-2 border-gray-300 rounded-full 
                     focus:border-blue-500 focus:ring-2 focus:ring-blue-200 focus:outline-none
                     bg-white backdrop-blur-sm shadow-lg transition-all duration-200
                     text-gray-900 placeholder-gray-500
                     disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={loading}
          />
          
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="absolute right-2 p-2 rounded-full bg-blue-500 text-white 
                     hover:bg-blue-600 focus:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-200
                     disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors duration-200"
          >
            {loading ? (
              <div className="animate-spin w-6 h-6 border-2 border-white border-t-transparent rounded-full" />
            ) : (
              <MagnifyingGlassIcon className="w-6 h-6" />
            )}
          </button>
        </div>
        
        {/* Search suggestions */}
        <div className="mt-4 flex flex-wrap gap-2 justify-center">
          {['blue shirts', 'running shoes', 'dresses', 'jackets', 'casual wear'].map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => {
                setQuery(suggestion);
                onSearch(suggestion);
              }}
              disabled={loading}
              className="px-3 py-1 text-sm bg-white backdrop-blur-sm border border-gray-300 
                       rounded-full hover:bg-white hover:border-blue-300 transition-all duration-200
                       text-gray-700 hover:text-gray-900
                       disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </form>
    </div>
  );
}