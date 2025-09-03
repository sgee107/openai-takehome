'use client';

import { useState, useCallback } from 'react';
import { ChatResponse, ProductResult } from '../lib/types';
import SearchBar from '../components/SearchBar';
import ProductGrid from '../components/ProductGrid';
import ProductModal from '../components/ProductModal';
import DebugToggle from '../components/DebugToggle';

export default function Home() {
  const [searchState, setSearchState] = useState({
    query: '',
    results: [] as ProductResult[],
    loading: false,
    debugMode: false,
  });
  const [selectedProduct, setSelectedProduct] = useState<ProductResult | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const handleSearch = useCallback(async (query: string) => {
    setSearchState(prev => ({ ...prev, loading: true, query }));
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query, limit: 20 }),
      });
      
      if (!response.ok) {
        throw new Error('Search failed');
      }
      
      const data: ChatResponse = await response.json();
      setSearchState(prev => ({ 
        ...prev, 
        results: data.results, 
        loading: false 
      }));
    } catch (error) {
      console.error('Search error:', error);
      setSearchState(prev => ({ 
        ...prev, 
        results: [], 
        loading: false 
      }));
    }
  }, []);

  const handleProductClick = useCallback((product: ProductResult) => {
    setSelectedProduct(product);
    setModalOpen(true);
  }, []);

  const handleFindSimilar = useCallback((product: ProductResult) => {
    setModalOpen(false);
    // Extract keywords from the product for similarity search
    const keywords = [
      product.main_category,
      product.store,
      ...product.title.split(' ').slice(0, 3)
    ].filter(Boolean).join(' ');
    
    handleSearch(keywords);
  }, [handleSearch]);

  const toggleDebugMode = useCallback((enabled: boolean) => {
    setSearchState(prev => ({ ...prev, debugMode: enabled }));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 relative">
      {/* Fashion gradient background */}
      <div className="absolute inset-0 fashion-gradient" />
      
      {/* Animated fashion elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Floating fashion icons */}
        <div className="absolute top-20 left-10 text-6xl opacity-20 fashion-float">👗</div>
        <div className="absolute top-40 right-20 text-5xl opacity-15 fashion-float">👕</div>
        <div className="absolute bottom-32 left-20 text-7xl opacity-10 fashion-float">👠</div>
        <div className="absolute bottom-20 right-10 text-6xl opacity-25 fashion-float">👜</div>
        <div className="absolute top-60 left-1/2 text-4xl opacity-20 fashion-float">🧥</div>
        <div className="absolute top-32 left-1/3 text-5xl opacity-15 fashion-float">👖</div>
        <div className="absolute bottom-40 right-1/3 text-6xl opacity-20 fashion-float">⌚</div>
        
        {/* Blur circles for depth */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-200/20 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-200/20 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-pink-200/15 rounded-full blur-3xl" />
      </div>

      {/* Main Content */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Header */}
        <header className="text-center py-12 px-4">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-800 mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
            Fashion Search
          </h1>
          <p className="text-lg md:text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Discover your perfect style with AI-powered fashion search
          </p>
          
          {/* Search Bar */}
          <SearchBar 
            onSearch={handleSearch}
            loading={searchState.loading}
            initialQuery={searchState.query}
          />
        </header>

        {/* Results */}
        <main className="flex-1 pb-20">
          {searchState.query && (
            <div className="max-w-7xl mx-auto">
              <ProductGrid 
                products={searchState.results}
                loading={searchState.loading}
                onProductClick={handleProductClick}
                showDebug={searchState.debugMode}
              />
            </div>
          )}
          
          {!searchState.query && !searchState.loading && (
            <div className="text-center py-16 px-4">
              <div className="text-gray-400 text-8xl mb-8">👗</div>
              <h2 className="text-2xl md:text-3xl font-semibold text-gray-700 mb-4">
                Start your fashion journey
              </h2>
              <p className="text-gray-500 text-lg mb-8 max-w-md mx-auto">
                Search for clothing, accessories, or describe your style to find the perfect match
              </p>
              <div className="flex flex-wrap justify-center gap-2 text-sm">
                <span className="text-gray-400">Try:</span>
                {['vintage dresses', 'casual sneakers', 'formal shirts', 'summer accessories'].map((term) => (
                  <button
                    key={term}
                    onClick={() => handleSearch(term)}
                    className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200 transition-colors"
                  >
                    {term}
                  </button>
                ))}
              </div>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="text-center py-8 px-4 text-gray-500 text-sm">
          <p>Fashion Search Demo - Built with Next.js, TypeScript & Tailwind CSS</p>
          {searchState.debugMode && (
            <p className="mt-2 text-blue-600">Debug mode active - Use Cmd/Ctrl+D to toggle</p>
          )}
        </footer>
      </div>

      {/* Product Modal */}
      <ProductModal
        product={selectedProduct}
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onFindSimilar={handleFindSimilar}
        showDebug={searchState.debugMode}
      />

      {/* Debug Toggle */}
      <DebugToggle
        debugMode={searchState.debugMode}
        onToggle={toggleDebugMode}
      />
    </div>
  );
}