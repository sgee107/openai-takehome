'use client';

import Image from 'next/image';
import { ProductResult } from '../lib/types';
import { getProductImageUrl } from '../lib/mockData';
import DebugPanel from './DebugPanel';
import { StarIcon } from '@heroicons/react/24/solid';
import { StarIcon as StarOutlineIcon } from '@heroicons/react/24/outline';

interface ProductCardProps {
  product: ProductResult;
  onClick: (product: ProductResult) => void;
  showDebug?: boolean;
}

export default function ProductCard({ product, onClick, showDebug = false }: ProductCardProps) {
  const imageUrl = getProductImageUrl(product, 'large');
  
  const renderStars = (rating: number) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;
    
    for (let i = 0; i < fullStars; i++) {
      stars.push(<StarIcon key={i} className="w-3 h-3 text-yellow-400" />);
    }
    
    if (hasHalfStar) {
      stars.push(
        <div key="half" className="relative w-3 h-3">
          <StarOutlineIcon className="absolute w-3 h-3 text-yellow-400" />
          <div className="absolute w-1.5 h-3 overflow-hidden">
            <StarIcon className="w-3 h-3 text-yellow-400" />
          </div>
        </div>
      );
    }
    
    const emptyStars = 5 - Math.ceil(rating);
    for (let i = 0; i < emptyStars; i++) {
      stars.push(<StarOutlineIcon key={`empty-${i}`} className="w-3 h-3 text-gray-300" />);
    }
    
    return stars;
  };

  return (
    <div 
      className="group relative cursor-pointer bg-white rounded-lg shadow-md hover:shadow-xl 
                 transform hover:scale-105 transition-all duration-300 overflow-hidden
                 border border-gray-200 hover:border-blue-300"
      onClick={() => onClick(product)}
    >
      {/* Debug Panel */}
      {showDebug && <DebugPanel similarity={product.similarity_score} rank={product.rank} />}
      
      {/* Product Image */}
      <div className="relative aspect-square bg-gray-100">
        <Image
          src={imageUrl}
          alt={product.title}
          fill
          className="object-cover group-hover:scale-110 transition-transform duration-300"
          sizes="(max-width: 768px) 50vw, (max-width: 1200px) 33vw, 16vw"
          onError={(e) => {
            const target = e.target as HTMLImageElement;
            target.src = '/placeholder-product.jpg';
          }}
        />
        
        {/* Overlay gradient */}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors duration-300" />
      </div>
      
      {/* Product Info */}
      <div className="p-3 space-y-2">
        {/* Title */}
        <h3 className="text-sm font-medium text-gray-900 line-clamp-2 group-hover:text-blue-600 transition-colors">
          {product.title}
        </h3>
        
        {/* Store/Brand */}
        {product.store && (
          <p className="text-xs text-gray-500 font-medium">
            {product.store}
          </p>
        )}
        
        {/* Rating */}
        {product.rating && (
          <div className="flex items-center gap-1">
            <div className="flex">
              {renderStars(product.rating)}
            </div>
            <span className="text-xs text-gray-500">
              ({product.rating_number || 0})
            </span>
          </div>
        )}
        
        {/* Price */}
        <div className="flex items-center justify-between">
          {product.price ? (
            <span className="text-lg font-bold text-green-600">
              ${product.price}
            </span>
          ) : (
            <span className="text-sm text-gray-500">Price not available</span>
          )}
        </div>
        
        {/* Coming Soon Button */}
        <button
          disabled
          className="w-full py-2 px-3 bg-gray-300 text-gray-500 text-sm font-medium rounded-md
                   cursor-not-allowed transition-colors relative overflow-hidden"
          title="This is a demo - purchasing not available"
        >
          <span className="relative z-10">Coming Soon</span>
          <div className="absolute inset-0 bg-gradient-to-r from-blue-400/20 to-purple-400/20 opacity-50" />
        </button>
      </div>
    </div>
  );
}