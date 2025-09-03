'use client';

import { Fragment, useState } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { XMarkIcon, ChevronLeftIcon, ChevronRightIcon, ShareIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { StarIcon } from '@heroicons/react/24/solid';
import { StarIcon as StarOutlineIcon } from '@heroicons/react/24/outline';
import Image from 'next/image';
import { ProductResult } from '../lib/types';
import DebugPanel from './DebugPanel';

interface ProductModalProps {
  product: ProductResult | null;
  isOpen: boolean;
  onClose: () => void;
  onFindSimilar?: (product: ProductResult) => void;
  showDebug?: boolean;
}

export default function ProductModal({ 
  product, 
  isOpen, 
  onClose, 
  onFindSimilar,
  showDebug = false 
}: ProductModalProps) {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  if (!product) return null;

  const images = product.images && product.images.length > 0 ? product.images : [];
  const currentImage = images[currentImageIndex];

  const nextImage = () => {
    setCurrentImageIndex((prev) => (prev + 1) % images.length);
  };

  const prevImage = () => {
    setCurrentImageIndex((prev) => (prev - 1 + images.length) % images.length);
  };

  const renderStars = (rating: number) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;
    
    for (let i = 0; i < fullStars; i++) {
      stars.push(<StarIcon key={i} className="w-4 h-4 text-yellow-400" />);
    }
    
    if (hasHalfStar) {
      stars.push(
        <div key="half" className="relative w-4 h-4">
          <StarOutlineIcon className="absolute w-4 h-4 text-yellow-400" />
          <div className="absolute w-2 h-4 overflow-hidden">
            <StarIcon className="w-4 h-4 text-yellow-400" />
          </div>
        </div>
      );
    }
    
    const emptyStars = 5 - Math.ceil(rating);
    for (let i = 0; i < emptyStars; i++) {
      stars.push(<StarOutlineIcon key={`empty-${i}`} className="w-4 h-4 text-gray-300" />);
    }
    
    return stars;
  };

  const shareProduct = async () => {
    try {
      await navigator.clipboard.writeText(`${product.title} - ${window.location.origin}`);
      // You could add a toast notification here
      alert('Product link copied to clipboard!');
    } catch (error) {
      console.error('Failed to copy to clipboard', error);
    }
  };

  return (
    <Transition.Root show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black bg-opacity-75 transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 z-10 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4 text-center sm:p-0">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
              enterTo="opacity-100 translate-y-0 sm:scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 translate-y-0 sm:scale-100"
              leaveTo="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
            >
              <Dialog.Panel className="relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-4xl">
                {/* Debug Panel */}
                {showDebug && (
                  <div className="absolute top-4 right-16 z-20">
                    <DebugPanel similarity={product.similarity_score} rank={product.rank} />
                  </div>
                )}

                {/* Close Button */}
                <button
                  onClick={onClose}
                  className="absolute top-4 right-4 z-20 p-2 bg-white/80 backdrop-blur-sm rounded-full hover:bg-white transition-colors"
                >
                  <XMarkIcon className="h-6 w-6 text-gray-600" />
                </button>

                <div className="grid grid-cols-1 lg:grid-cols-2">
                  {/* Image Carousel */}
                  <div className="relative bg-gray-100 aspect-square">
                    {images.length > 0 ? (
                      <>
                        <Image
                          src={currentImage?.hi_res || currentImage?.large || currentImage?.thumb || '/placeholder-product.jpg'}
                          alt={product.title}
                          fill
                          className="object-cover"
                          sizes="(max-width: 1024px) 100vw, 50vw"
                        />
                        
                        {/* Navigation arrows */}
                        {images.length > 1 && (
                          <>
                            <button
                              onClick={prevImage}
                              className="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 bg-black/50 text-white rounded-full hover:bg-black/70 transition-colors"
                            >
                              <ChevronLeftIcon className="h-6 w-6" />
                            </button>
                            <button
                              onClick={nextImage}
                              className="absolute right-4 top-1/2 transform -translate-y-1/2 p-2 bg-black/50 text-white rounded-full hover:bg-black/70 transition-colors"
                            >
                              <ChevronRightIcon className="h-6 w-6" />
                            </button>
                          </>
                        )}

                        {/* Image indicators */}
                        {images.length > 1 && (
                          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex gap-2">
                            {images.map((_, index) => (
                              <button
                                key={index}
                                onClick={() => setCurrentImageIndex(index)}
                                className={`w-2 h-2 rounded-full transition-colors ${
                                  index === currentImageIndex ? 'bg-white' : 'bg-white/50'
                                }`}
                              />
                            ))}
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <div className="text-gray-400 text-center">
                          <div className="text-4xl mb-2">📷</div>
                          <p>No image available</p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Product Details */}
                  <div className="p-6 space-y-6">
                    <div>
                      <h2 className="text-2xl font-bold text-gray-900 mb-2">
                        {product.title}
                      </h2>
                      
                      {product.store && (
                        <p className="text-lg text-gray-600 font-medium">
                          by {product.store}
                        </p>
                      )}
                    </div>

                    {/* Category breadcrumb */}
                    {product.categories && product.categories.length > 0 && (
                      <div className="text-sm text-gray-500">
                        {product.categories[0]?.join(' › ')}
                      </div>
                    )}

                    {/* Rating */}
                    {product.rating && (
                      <div className="flex items-center gap-2">
                        <div className="flex">
                          {renderStars(product.rating)}
                        </div>
                        <span className="text-lg font-medium">
                          {product.rating}
                        </span>
                        <span className="text-gray-500">
                          ({product.rating_number || 0} reviews)
                        </span>
                      </div>
                    )}

                    {/* Price */}
                    <div className="text-3xl font-bold text-green-600">
                      {product.price ? `$${product.price}` : 'Price not available'}
                    </div>

                    {/* Features */}
                    {product.features && product.features.length > 0 && (
                      <div>
                        <h3 className="text-lg font-semibold mb-3">Features</h3>
                        <ul className="space-y-2">
                          {product.features.slice(0, 5).map((feature, index) => (
                            <li key={index} className="flex items-start gap-2 text-gray-700">
                              <div className="w-1 h-1 bg-gray-400 rounded-full mt-2 flex-shrink-0" />
                              {feature}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Details */}
                    {product.details && Object.keys(product.details).length > 0 && (
                      <div>
                        <h3 className="text-lg font-semibold mb-3">Product Details</h3>
                        <div className="grid grid-cols-1 gap-2">
                          {Object.entries(product.details).slice(0, 6).map(([key, value]) => (
                            <div key={key} className="flex justify-between py-1 border-b border-gray-100">
                              <span className="text-gray-600 font-medium">{key}:</span>
                              <span className="text-gray-800 text-right flex-1 ml-4">
                                {String(value).slice(0, 50)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex gap-3">
                      <button
                        disabled
                        className="flex-1 py-3 px-4 bg-gray-300 text-gray-500 font-medium rounded-md
                                 cursor-not-allowed relative overflow-hidden"
                        title="This is a demo - purchasing not available"
                      >
                        <span className="relative z-10">Coming Soon</span>
                        <div className="absolute inset-0 bg-gradient-to-r from-blue-400/20 to-purple-400/20 opacity-50" />
                      </button>
                      
                      {onFindSimilar && (
                        <button
                          onClick={() => onFindSimilar(product)}
                          className="px-4 py-3 bg-blue-500 text-white font-medium rounded-md hover:bg-blue-600 transition-colors flex items-center gap-2"
                        >
                          <MagnifyingGlassIcon className="h-5 w-5" />
                          Find Similar
                        </button>
                      )}
                      
                      <button
                        onClick={shareProduct}
                        className="px-4 py-3 bg-gray-100 text-gray-700 font-medium rounded-md hover:bg-gray-200 transition-colors"
                      >
                        <ShareIcon className="h-5 w-5" />
                      </button>
                    </div>
                  </div>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
}