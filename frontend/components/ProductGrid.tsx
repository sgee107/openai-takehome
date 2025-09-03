'use client';

import { ProductResult } from '../lib/types';
import ProductCard from './ProductCard';

interface ProductGridProps {
  products: ProductResult[];
  loading: boolean;
  onProductClick: (product: ProductResult) => void;
  showDebug?: boolean;
}

function ProductSkeleton() {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-200">
      <div className="animate-pulse">
        {/* Image skeleton */}
        <div className="aspect-square bg-gray-300"></div>
        
        {/* Content skeleton */}
        <div className="p-3 space-y-2">
          {/* Title skeleton */}
          <div className="h-4 bg-gray-300 rounded w-full"></div>
          <div className="h-4 bg-gray-300 rounded w-3/4"></div>
          
          {/* Store skeleton */}
          <div className="h-3 bg-gray-300 rounded w-1/2"></div>
          
          {/* Rating skeleton */}
          <div className="flex items-center gap-1">
            <div className="flex gap-1">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="w-3 h-3 bg-gray-300 rounded"></div>
              ))}
            </div>
            <div className="h-3 bg-gray-300 rounded w-8"></div>
          </div>
          
          {/* Price skeleton */}
          <div className="h-6 bg-gray-300 rounded w-20"></div>
          
          {/* Button skeleton */}
          <div className="h-8 bg-gray-300 rounded w-full"></div>
        </div>
      </div>
    </div>
  );
}

export default function ProductGrid({ 
  products, 
  loading, 
  onProductClick, 
  showDebug = false 
}: ProductGridProps) {
  // Show skeletons while loading
  if (loading) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 p-4">
        {[...Array(12)].map((_, index) => (
          <ProductSkeleton key={index} />
        ))}
      </div>
    );
  }

  // Show message if no products
  if (!products || products.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 px-4">
        <div className="text-gray-400 text-6xl mb-4">🔍</div>
        <h3 className="text-xl font-semibold text-gray-700 mb-2">No products found</h3>
        <p className="text-gray-500 text-center max-w-md">
          Try searching for different terms or browse our suggested categories above.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Results header */}
      <div className="px-4 py-2 bg-white/70 backdrop-blur-sm rounded-lg border border-gray-200">
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-600">
            Showing <span className="font-medium">{products.length}</span> results
          </p>
          {showDebug && (
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              Debug mode active
            </div>
          )}
        </div>
      </div>

      {/* Products grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 p-4">
        {products.map((product) => (
          <ProductCard
            key={product.parent_asin}
            product={product}
            onClick={onProductClick}
            showDebug={showDebug}
          />
        ))}
      </div>
    </div>
  );
}