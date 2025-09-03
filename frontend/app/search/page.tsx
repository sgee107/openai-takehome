'use client';

import { useState, useCallback } from 'react';
import { 
  ChatSearchResponse, 
  ProductResultVNext, 
  SearchStateVNext, 
  AgentDecision,
  FacetGroup,
  FollowupPrompt
} from '../../lib/types';
import SearchBar from '../../components/SearchBar';
import ProductGrid from '../../components/ProductGrid';
import ProductModal from '../../components/ProductModal';
import DebugToggle from '../../components/DebugToggle';

export default function SearchPage() {
  const [searchState, setSearchState] = useState<SearchStateVNext>({
    query: '',
    results: [],
    loading: false,
    debugMode: false,
    agent: undefined,
    facets: undefined,
    followups: undefined,
    debug: undefined,
  });
  const [selectedProduct, setSelectedProduct] = useState<ProductResultVNext | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const handleSearch = useCallback(async (query: string) => {
    setSearchState(prev => ({ 
      ...prev, 
      loading: true, 
      query,
      // Clear previous results
      results: [],
      agent: undefined,
      facets: undefined,
      followups: undefined,
      debug: undefined
    }));
    
    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query, 
          topK: 20, 
          lambda: 0.85,
          debug: searchState.debugMode
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Search failed');
      }
      
      const data: ChatSearchResponse = await response.json();
      
      // Log for debugging
      console.log('Search response:', data);
      
      setSearchState(prev => ({ 
        ...prev, 
        results: data.results, 
        loading: false,
        agent: data.agent,
        facets: data.facets,
        followups: data.followups,
        debug: data.debug
      }));
    } catch (error) {
      console.error('Search error:', error);
      setSearchState(prev => ({ 
        ...prev, 
        results: [], 
        loading: false 
      }));
      
      // Show error to user (you could add a toast notification here)
      alert(error instanceof Error ? error.message : 'Search failed. Please try again.');
    }
  }, [searchState.debugMode]);

  const handleProductClick = useCallback((product: ProductResultVNext) => {
    setSelectedProduct(product);
    setModalOpen(true);
  }, []);

  const handleFindSimilar = useCallback((product: ProductResultVNext) => {
    setModalOpen(false);
    // Create a similarity search query from the product title
    const keywords = product.title.split(' ').slice(0, 4).join(' ');
    handleSearch(`similar to ${keywords}`);
  }, [handleSearch]);

  const handleFollowupClick = useCallback((followup: FollowupPrompt) => {
    const enhancedQuery = `${searchState.query} ${followup.text.toLowerCase().replace('?', '')}`;
    handleSearch(enhancedQuery);
  }, [searchState.query, handleSearch]);

  const toggleDebugMode = useCallback((enabled: boolean) => {
    setSearchState(prev => ({ ...prev, debugMode: enabled }));
  }, []);

  const getComplexityColor = (complexity: 1 | 2 | 3) => {
    switch (complexity) {
      case 1: return 'text-green-600 bg-green-100';  // Direct
      case 2: return 'text-blue-600 bg-blue-100';    // Filtered  
      case 3: return 'text-orange-600 bg-orange-100'; // Ambiguous
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getComplexityLabel = (complexity: 1 | 2 | 3) => {
    switch (complexity) {
      case 1: return 'Direct';
      case 2: return 'Filtered';  
      case 3: return 'Ambiguous';
      default: return 'Unknown';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 relative">
      {/* Fashion gradient background */}
      <div className="absolute inset-0 fashion-gradient" />
      
      {/* Animated fashion elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 text-6xl opacity-20 fashion-float">👗</div>
        <div className="absolute top-40 right-20 text-5xl opacity-15 fashion-float">👕</div>
        <div className="absolute bottom-32 left-20 text-7xl opacity-10 fashion-float">👠</div>
        <div className="absolute bottom-20 right-10 text-6xl opacity-25 fashion-float">👜</div>
        <div className="absolute top-60 left-1/2 text-4xl opacity-20 fashion-float">🧥</div>
        <div className="absolute top-32 left-1/3 text-5xl opacity-15 fashion-float">👖</div>
        <div className="absolute bottom-40 right-1/3 text-6xl opacity-20 fashion-float">⌚</div>
        
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-200/20 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-200/20 rounded-full blur-3xl" />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-pink-200/15 rounded-full blur-3xl" />
      </div>

      {/* Main Content */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Header */}
        <header className="text-center py-12 px-4">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-800 mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
            Fashion Search vNext
          </h1>
          <p className="text-lg md:text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Enhanced AI search with intent classification and image-enriched embeddings
          </p>
          
          {/* Search Bar */}
          <SearchBar 
            onSearch={handleSearch}
            loading={searchState.loading}
            initialQuery={searchState.query}
          />

          {/* Agent Decision Display */}
          {searchState.agent && (
            <div className="mt-6 max-w-2xl mx-auto">
              <div className="bg-white/80 backdrop-blur-sm rounded-lg p-4 shadow-sm border">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getComplexityColor(searchState.agent.complexity)}`}>
                      Complexity: {getComplexityLabel(searchState.agent.complexity)}
                    </span>
                    {searchState.agent.parsed?.category && (
                      <span className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm">
                        Category: {searchState.agent.parsed.category}
                      </span>
                    )}
                  </div>
                  {searchState.debug && (
                    <span className="text-xs text-gray-500">
                      {searchState.debug.timings?.total?.toFixed(0)}ms
                    </span>
                  )}
                </div>
                {searchState.agent.reason && (
                  <p className="text-sm text-gray-600 mt-2">{searchState.agent.reason}</p>
                )}
              </div>
            </div>
          )}
        </header>

        {/* Results */}
        <main className="flex-1 pb-20">
          <div className="max-w-7xl mx-auto px-4">
            {/* Follow-up Questions */}
            {searchState.followups && searchState.followups.length > 0 && (
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Need help refining your search?</h3>
                <div className="flex flex-wrap gap-2">
                  {searchState.followups.map((followup, index) => (
                    <button
                      key={index}
                      onClick={() => handleFollowupClick(followup)}
                      className="px-4 py-2 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg text-sm
                               border border-blue-200 transition-colors duration-200"
                      title={followup.rationale}
                    >
                      {followup.text}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Facets */}
            {searchState.facets && searchState.facets.length > 0 && (
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Filter Results</h3>
                <div className="space-y-4">
                  {searchState.facets.map((facetGroup, groupIndex) => (
                    <div key={groupIndex} className="bg-white/70 backdrop-blur-sm rounded-lg p-4">
                      <h4 className="font-medium text-gray-800 mb-2">{facetGroup.name}</h4>
                      <div className="flex flex-wrap gap-2">
                        {facetGroup.options.map((option, optionIndex) => (
                          <button
                            key={optionIndex}
                            className="px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full text-sm
                                     border border-gray-300 transition-colors duration-200"
                          >
                            {option.value} ({option.count})
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Product Grid */}
            {searchState.query && (
              <ProductGrid 
                products={searchState.results.map(p => ({
                  // Convert vNext results to legacy format for existing component
                  parent_asin: p.id,
                  title: p.title,
                  main_category: 'Fashion', // Default since not in vNext response
                  store: undefined,
                  images: p.image ? [{ hi_res: p.image }] : [],
                  price: p.price?.toString(),
                  rating: p.rating,
                  rating_number: p.ratingCount,
                  features: [],
                  details: {},
                  categories: [],
                  description: [],
                  similarity_score: p.match.final,
                  rank: searchState.results.indexOf(p) + 1
                }))}
                loading={searchState.loading}
                onProductClick={(product) => {
                  // Find the original vNext product
                  const vNextProduct = searchState.results.find(p => p.id === product.parent_asin);
                  if (vNextProduct) {
                    handleProductClick(vNextProduct);
                  }
                }}
                showDebug={searchState.debugMode}
              />
            )}
            
            {!searchState.query && !searchState.loading && (
              <div className="text-center py-16 px-4">
                <div className="text-gray-400 text-8xl mb-8">🔍</div>
                <h2 className="text-2xl md:text-3xl font-semibold text-gray-700 mb-4">
                  Try the enhanced search experience
                </h2>
                <p className="text-gray-500 text-lg mb-8 max-w-md mx-auto">
                  Our vNext API features intent classification, Bayesian ranking, and image-enriched embeddings
                </p>
                <div className="flex flex-wrap justify-center gap-2 text-sm">
                  <span className="text-gray-400">Try:</span>
                  {['blue casual shirt', 'running shoes under $100', 'work clothes', 'summer dress'].map((term) => (
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
          </div>
        </main>

        {/* Footer */}
        <footer className="text-center py-8 px-4 text-gray-500 text-sm">
          <p>Fashion Search vNext - Enhanced with Intent Classification & Bayesian Ranking</p>
          {searchState.debugMode && (
            <div className="mt-2 space-y-1">
              <p className="text-blue-600">Debug mode active - Use Cmd/Ctrl+D to toggle</p>
              {searchState.debug && (
                <div className="text-xs bg-black/10 rounded p-2 inline-block">
                  Plan: {searchState.debug.plan}
                </div>
              )}
            </div>
          )}
        </footer>
      </div>

      {/* Product Modal - needs adaptation for vNext */}
      {selectedProduct && (
        <ProductModal
          product={{
            parent_asin: selectedProduct.id,
            title: selectedProduct.title,
            main_category: 'Fashion',
            store: undefined,
            images: selectedProduct.image ? [{ hi_res: selectedProduct.image }] : [],
            price: selectedProduct.price?.toString(),
            rating: selectedProduct.rating,
            rating_number: selectedProduct.ratingCount,
            features: [`Match Score: ${selectedProduct.match.final.toFixed(3)}`],
            details: { 
              'Semantic Score': selectedProduct.match.semantic.toFixed(3),
              'Rating Score': selectedProduct.match.rating.toFixed(3),
              'Lambda Used': selectedProduct.match.lambda_used.toFixed(3)
            },
            categories: [],
            description: selectedProduct.match.explanation ? [selectedProduct.match.explanation] : [],
            similarity_score: selectedProduct.match.final,
            rank: searchState.results.indexOf(selectedProduct) + 1
          }}
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          onFindSimilar={() => handleFindSimilar(selectedProduct)}
          showDebug={searchState.debugMode}
        />
      )}

      {/* Debug Toggle */}
      <DebugToggle
        debugMode={searchState.debugMode}
        onToggle={toggleDebugMode}
      />
    </div>
  );
}